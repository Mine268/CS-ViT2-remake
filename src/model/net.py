"""Top-level model wiring for stage1/stage2 training and inference."""

from __future__ import annotations

import enum
import itertools
from typing import Dict, List, Optional, Tuple

import einops as eps
import kornia
import numpy as np
import smplx
import torch
import torch.nn as nn
from accelerate.logging import get_logger
from safetensors.torch import load_file

from ..constant import MANO_J_REGRESSOR_PATH, MANO_JOINT_COUNT, MANO_ROOT
from ..utils.metric import MetricMeter, build_dataset_group_mask
from ..utils.rot import rotation6d_to_rotation_matrix
from .backbone import ViTBackbone
from .heads import MANOTransformerDecoderHead
from .loss import RemakeLoss
from .perspective import PerspInfoEmbedderCrossAttn
from .temporal import TemporalEncoder
from .ti import TITokenTransform, invert_ti_camera, invert_ti_pose_root

logger = get_logger(__name__)


class PoseNet(nn.Module):
    """End-to-end hand pose model used by both training stages."""

    class Stage(enum.Enum):
        """Explicit stage enum so call sites do not rely on magic strings."""

        STAGE1 = "stage1"
        STAGE2 = "stage2"

    def __init__(
        self,
        stage: str,
        stage1_weight_path: Optional[str],
        backbone_str: str,
        img_size: Optional[int],
        img_mean: List[float],
        img_std: List[float],
        infusion_feats_lyr: Optional[List[int]],
        drop_cls: bool,
        backbone_kwargs: Optional[Dict],
        num_handec_layer: int,
        num_handec_head: int,
        ndim_handec_mlp: int,
        ndim_handec_head: int,
        prob_handec_dropout: float,
        prob_handec_emb_dropout: float,
        handec_emb_dropout_type: str,
        handec_norm: str,
        ndim_handec_norm_cond_dim: int,
        ndim_handec_ctx: Optional[int],
        handec_skip_token_embed: bool,
        handec_mean_init: bool,
        handec_denorm_output: bool,
        handec_heatmap_resulotion,
        pie_type: str,
        num_pie_sample: int,
        pie_fusion: str,
        num_temporal_head: int,
        num_temporal_layer: int,
        trope_scalar: float,
        zero_linear: bool,
        ti_enabled: bool,
        ti_num_layer: int,
        ti_num_head: int,
        ti_angle_range_deg: float,
        ti_scale_range: List[float],
        ti_loss_weight: float,
        ti_apply_stage1: bool,
        ti_apply_stage2: bool,
        joint_rep_type: str,
        freeze_backbone: bool,
        norm_by_hand: bool,
        handec_cam_head_type: str,
        root_z_num_bins: int,
        root_z_d_min: float,
        root_z_d_max: float,
        root_z_prior_k: float,
        root_z_geom_hidden_dim: int,
        root_z_dropout: float,
        root_z_use_data_source_embed: bool,
        lambda_theta: float,
        lambda_shape: float,
        lambda_trans: float,
        lambda_rel: float,
        lambda_img: float,
        lambda_uv_patch: float,
        lambda_root_z_cls: float,
        lambda_root_z_res: float,
        hm_sigma: float,
        pred_joint_z_min_mm: float,
        reproj_loss_type: str,
        reproj_loss_delta: float,
        ego_datasets: List[str],
        aux_datasets: List[str],
        root_min_valid_joints_2d: int,
        root_min_hand_bbox_edge_px: float,
    ):
        """
        Construct the full model graph.

        The constructor is intentionally verbose because Hydra resolves directly into it. This
        keeps all stage-dependent behavior explicit at model creation time.
        """
        super().__init__()
        if norm_by_hand:
            raise NotImplementedError("CS-ViT2-remake only supports norm_by_hand=false")
        self.stage = PoseNet.Stage(stage)
        self.cam_head_type = handec_cam_head_type
        if self.cam_head_type != "patch_uv_rho_multibin":
            raise ValueError(f"Only patch_uv_rho_multibin is supported, got {self.cam_head_type}")

        backbone_kwargs = backbone_kwargs or {}
        self.backbone = ViTBackbone(
            backbone_str=backbone_str,
            img_size=img_size,
            infusion_feats_lyr=infusion_feats_lyr,
            backbone_kwargs=dict(backbone_kwargs),
        )
        # We never run masked-image modeling in this repo, so the pretrained mask token
        # would otherwise stay trainable while remaining unused in every iteration.
        mask_token = getattr(
            getattr(self.backbone.backbone, "embeddings", None), "mask_token", None
        )
        if isinstance(mask_token, nn.Parameter):
            mask_token.requires_grad_(False)
        self.register_buffer("img_mean", torch.Tensor(img_mean))
        self.register_buffer("img_std", torch.Tensor(img_std))
        self.has_cls_token = self.backbone.get_has_cls_token()
        self.drop_cls = drop_cls and self.has_cls_token
        self.patch_size = self.backbone.get_patch_size()
        self.hidden_size = self.backbone.get_hidden_size()
        self.img_size = self.backbone.get_img_size()
        self.num_patch = self.backbone.get_num_patch()

        if pie_type != "ca":
            raise ValueError(f"Only pie_type='ca' is supported, got {pie_type}")
        del pie_fusion
        self.persp_info_embedder = PerspInfoEmbedderCrossAttn(
            hidden_size=self.hidden_size,
            num_sample=num_pie_sample,
            num_token=self.num_patch**2 + int(self.has_cls_token and not self.drop_cls),
        )

        self.register_buffer(
            "J_regressor_mano",
            torch.from_numpy(np.load(MANO_J_REGRESSOR_PATH)).type(torch.float32),
        )
        self.rmano_layer = smplx.create(MANO_ROOT, "mano", is_rhand=True, use_pca=False)
        self.rmano_layer.requires_grad_(False)
        self.rmano_layer.eval()

        self.joint_rep_type = joint_rep_type
        self.ti_enabled = bool(ti_enabled)
        self.ti_apply_stage1 = bool(ti_apply_stage1)
        self.ti_apply_stage2 = bool(ti_apply_stage2)
        self.ti_loss_weight = float(ti_loss_weight)
        self.ti_loss_terms = frozenset({"theta", "shape", "trans", "joint_rel"})
        if len(ti_scale_range) != 2:
            raise ValueError(f"ti_scale_range must have length 2, got {ti_scale_range}")
        self.ti_scale_range = (float(ti_scale_range[0]), float(ti_scale_range[1]))
        self.ti_angle_range_rad = float(np.deg2rad(ti_angle_range_deg))
        if self.ti_scale_range[0] <= 0.0 or self.ti_scale_range[0] > self.ti_scale_range[1]:
            raise ValueError(f"Invalid ti_scale_range={self.ti_scale_range}")
        if self.ti_enabled and self.ti_apply_stage2:
            raise NotImplementedError(
                "TI apply_stage2 is reserved for a later clip-shared implementation"
            )
        self.handec = MANOTransformerDecoderHead(
            joint_rep_type=joint_rep_type,
            dim=self.hidden_size,
            depth=num_handec_layer,
            heads=num_handec_head,
            mlp_dim=ndim_handec_mlp,
            dim_head=ndim_handec_head,
            dropout=prob_handec_dropout,
            emb_dropout=prob_handec_emb_dropout,
            emb_dropout_type=handec_emb_dropout_type,
            norm=handec_norm,
            norm_cond_dim=ndim_handec_norm_cond_dim,
            context_dim=ndim_handec_ctx,
            skip_token_embedding=handec_skip_token_embed,
            use_mean_init=handec_mean_init,
            denorm_output=handec_denorm_output,
            norm_by_hand=False,
            heatmap_resolution=handec_heatmap_resulotion,
            patch_size=self.img_size,
            cam_head_type=handec_cam_head_type,
            root_z_num_bins=root_z_num_bins,
            root_z_d_min=root_z_d_min,
            root_z_d_max=root_z_d_max,
            root_z_prior_k=root_z_prior_k,
            root_z_geom_hidden_dim=root_z_geom_hidden_dim,
            root_z_dropout=root_z_dropout,
            root_z_use_data_source_embed=root_z_use_data_source_embed,
        )
        self.ti_transform = (
            TITokenTransform(
                dim=self.hidden_size,
                depth=ti_num_layer,
                heads=ti_num_head,
                dropout=prob_handec_dropout,
            )
            if self.ti_enabled
            else None
        )

        self.temporal_refiner = TemporalEncoder(
            dim=self.hidden_size,
            num_head=num_temporal_head,
            num_layer=num_temporal_layer,
            dropout=prob_handec_dropout,
            trope_scalar=trope_scalar,
            zero_linear=zero_linear,
        )
        if self.stage == PoseNet.Stage.STAGE1:
            # Stage 1 decodes each frame independently and never enters the temporal path.
            self.temporal_refiner.requires_grad_(False)

        self.loss_fn = RemakeLoss(
            lambda_theta=lambda_theta,
            lambda_shape=lambda_shape,
            lambda_trans=lambda_trans,
            lambda_rel=lambda_rel,
            lambda_img=lambda_img,
            lambda_uv_patch=lambda_uv_patch,
            lambda_root_z_cls=lambda_root_z_cls,
            lambda_root_z_res=lambda_root_z_res,
            hm_centers=self.handec.get_centers(),
            hm_sigma=hm_sigma,
            pred_joint_z_min_mm=pred_joint_z_min_mm,
            reproj_loss_type=reproj_loss_type,
            reproj_loss_delta=reproj_loss_delta,
            ego_datasets=ego_datasets,
            aux_datasets=aux_datasets,
            min_valid_joints_2d=root_min_valid_joints_2d,
            min_hand_bbox_edge_px=root_min_hand_bbox_edge_px,
            rho_d_min=root_z_d_min,
            rho_d_max=root_z_d_max,
        )
        self.metric_meter = MetricMeter()
        self.freeze_backbone = freeze_backbone

        if self.stage == PoseNet.Stage.STAGE2 and stage1_weight_path is not None:
            self.load_pretrained(stage1_weight_path)

    def load_pretrained(self, path: str):
        """Load a saved checkpoint directory or a raw `.safetensors` file."""
        model_path = path
        if not model_path.endswith(".safetensors"):
            model_path = f"{path}/model.safetensors"
        logger.info("Loading pretrained weights from %s", model_path)
        state_dict = load_file(model_path)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("Missing keys when loading pretrained weights: %s", missing[:10])
        if unexpected:
            logger.warning("Unexpected keys when loading pretrained weights: %s", unexpected[:10])

    def decode_hand_param(
        self,
        img: torch.Tensor,
        bbox: torch.Tensor,
        focal: torch.Tensor,
        princpt: torch.Tensor,
        hand_bbox: torch.Tensor,
    ):
        """Run the visual backbone, perspective embedder, and hand decoder for one frame batch."""
        feats = self.encode_visual_tokens(img=img, bbox=bbox, focal=focal, princpt=princpt)
        return self.decode_from_tokens(
            feats=feats,
            bbox=bbox,
            focal=focal,
            princpt=princpt,
            hand_bbox=hand_bbox,
        )

    def encode_visual_tokens(
        self,
        img: torch.Tensor,
        bbox: torch.Tensor,
        focal: torch.Tensor,
        princpt: torch.Tensor,
    ) -> torch.Tensor:
        """Encode image patches into perspective-aware tokens."""
        feats = self.backbone(img)
        if self.drop_cls:
            feats = feats[:, 1:]
        return self.persp_info_embedder(feats=feats, bbox=bbox, focal=focal, princpt=princpt)

    def decode_from_tokens(
        self,
        feats: torch.Tensor,
        bbox: torch.Tensor,
        focal: torch.Tensor,
        princpt: torch.Tensor,
        hand_bbox: torch.Tensor,
    ):
        """Decode MANO parameters from already encoded tokens."""
        return self.handec(
            feats,
            patch_bbox=bbox,
            hand_bbox=hand_bbox,
            focal=focal,
            princpt=princpt,
        )

    def _prepare_frame_inputs(
        self,
        img: torch.Tensor,
        bbox: torch.Tensor,
        focal: torch.Tensor,
        princpt: torch.Tensor,
        hand_bbox: torch.Tensor,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize and flatten clip-shaped inputs into per-frame batches."""
        num_frame = img.shape[1]
        img = (img - self.img_mean[None, None, :, None, None]) / self.img_std[
            None, None, :, None, None
        ]
        img, bbox, focal, princpt, hand_bbox = map(
            lambda t: eps.rearrange(t, "b t ... -> (b t) ..."),
            [img, bbox, focal, princpt, hand_bbox],
        )
        return num_frame, img, bbox, focal, princpt, hand_bbox

    def _should_run_ti_branch(self) -> bool:
        """Decide whether the TI regularization branch participates in this forward pass."""
        if (
            not self.training
            or not self.ti_enabled
            or self.ti_transform is None
            or self.ti_loss_weight <= 0.0
        ):
            return False
        if self.stage == PoseNet.Stage.STAGE1:
            return self.ti_apply_stage1
        return self.ti_apply_stage2

    def _sample_ti_params(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample per-frame TI scale and rotation parameters."""
        scale = torch.empty(batch_size, device=device, dtype=dtype).uniform_(
            self.ti_scale_range[0],
            self.ti_scale_range[1],
        )
        angle_rad = torch.empty(batch_size, device=device, dtype=dtype).uniform_(
            -self.ti_angle_range_rad,
            self.ti_angle_range_rad,
        )
        return scale, angle_rad

    def _apply_ti_tokens(
        self, feats: torch.Tensor, scale: torch.Tensor, angle_rad: torch.Tensor
    ) -> torch.Tensor:
        """Transform token features with the TI module."""
        if self.ti_transform is None:
            raise RuntimeError("TI branch requested while ti_transform is not initialized")
        return self.ti_transform(feats, scale=scale, angle_rad=angle_rad)

    def _invert_ti_predictions(
        self,
        pose: torch.Tensor,
        shape: torch.Tensor,
        cam_aux: Dict[str, torch.Tensor],
        scale: torch.Tensor,
        angle_rad: torch.Tensor,
    ):
        """Map transformed-branch predictions back to the original camera frame."""
        pose_inv = invert_ti_pose_root(
            pose=pose,
            angle_rad=angle_rad,
            joint_rep_type=self.joint_rep_type,
        )
        trans_inv, ray_inv, rho_inv = invert_ti_camera(
            pred_ray_unit=cam_aux["pred_ray_unit"],
            pred_rho=cam_aux["pred_rho"],
            scale=scale,
            angle_rad=angle_rad,
        )
        cam_aux_inv = dict(cam_aux)
        cam_aux_inv["pred_ray_unit"] = ray_inv
        cam_aux_inv["pred_rho"] = rho_inv
        return pose_inv, shape, trans_inv, cam_aux_inv

    def _predict_ti_stage1(
        self,
        img: torch.Tensor,
        bbox: torch.Tensor,
        focal: torch.Tensor,
        princpt: torch.Tensor,
        hand_bbox: torch.Tensor,
    ):
        """Run the Stage1 TI branch and inverse-transform its predictions."""
        num_frame, img, bbox, focal, princpt, hand_bbox = self._prepare_frame_inputs(
            img=img,
            bbox=bbox,
            focal=focal,
            princpt=princpt,
            hand_bbox=hand_bbox,
        )
        if num_frame != 1:
            raise NotImplementedError("TI Stage1 currently expects single-frame batches")
        feats = self.encode_visual_tokens(img=img, bbox=bbox, focal=focal, princpt=princpt)
        scale, angle_rad = self._sample_ti_params(feats.shape[0], feats.device, feats.dtype)
        ti_feats = self._apply_ti_tokens(feats=feats, scale=scale, angle_rad=angle_rad)
        (pose_t, shape_t, _), cam_aux_t, _ = self.decode_from_tokens(
            feats=ti_feats,
            bbox=bbox,
            focal=focal,
            princpt=princpt,
            hand_bbox=hand_bbox,
        )
        pose_inv, shape_inv, trans_inv, cam_aux_inv = self._invert_ti_predictions(
            pose=pose_t,
            shape=shape_t,
            cam_aux=cam_aux_t,
            scale=scale,
            angle_rad=angle_rad,
        )
        pose_inv, shape_inv, trans_inv = map(
            lambda t: eps.rearrange(t, "(b t) d -> b t d", t=1),
            [pose_inv, shape_inv, trans_inv],
        )
        cam_aux_inv = {
            key: (
                eps.rearrange(value, "(b t) ... -> b t ...", t=1)
                if torch.is_tensor(value)
                else value
            )
            for key, value in cam_aux_inv.items()
        }
        return pose_inv, shape_inv, trans_inv, cam_aux_inv

    def predict_mano_param(
        self,
        img: torch.Tensor,
        bbox: torch.Tensor,
        focal: torch.Tensor,
        princpt: torch.Tensor,
        timestamp: Optional[torch.Tensor] = None,
        hand_bbox: Optional[torch.Tensor] = None,
    ):
        """
        Predict MANO parameters and camera translation from an image clip.

        Stage behavior:
        - stage1: decode each frame independently and only keep the current frame output
        - stage2: decode frame tokens first, refine them temporally, then decode final outputs
        """
        if hand_bbox is None:
            hand_bbox = bbox
        num_frame, img, bbox, focal, princpt, hand_bbox = self._prepare_frame_inputs(
            img=img,
            bbox=bbox,
            focal=focal,
            princpt=princpt,
            hand_bbox=hand_bbox,
        )

        if self.stage == PoseNet.Stage.STAGE1:
            (pose, shape, trans), cam_aux, _ = self.decode_hand_param(
                img=img,
                bbox=bbox,
                focal=focal,
                princpt=princpt,
                hand_bbox=hand_bbox,
            )
            # Stage1 only supervises/predicts the current frame, even if the loader shape is clip-like.
            out_frames = 1
        else:
            _, _, tokens_out = self.decode_hand_param(
                img=img,
                bbox=bbox,
                focal=focal,
                princpt=princpt,
                hand_bbox=hand_bbox,
            )
            tokens_out = eps.rearrange(tokens_out, "(b t) d -> b t d", t=num_frame)
            tokens_out = self.temporal_refiner(tokens_out, timestamp)
            (pose, shape, trans), cam_aux = self.handec.decode_token(
                eps.rearrange(tokens_out, "b t d -> (b t) d"),
                patch_bbox=bbox,
                hand_bbox=hand_bbox,
                focal=focal,
                princpt=princpt,
            )
            out_frames = num_frame

        # Rebuild `[B, T, ...]` structure so downstream loss code does not care which stage produced it.
        pose, shape, trans = map(
            lambda t: eps.rearrange(t, "(b t) d -> b t d", t=out_frames),
            [pose, shape, trans],
        )
        cam_aux = {
            key: (
                eps.rearrange(value, "(b t) ... -> b t ...", t=out_frames)
                if torch.is_tensor(value)
                else value
            )
            for key, value in cam_aux.items()
        }
        return pose, shape, trans, cam_aux

    def mano_to_pose(self, pose, shape):
        """Run MANO forward kinematics and return root-relative joints/vertices in millimeters."""
        batch_size, _, _ = pose.shape
        njoint_hand = self.J_regressor_mano.shape[0]
        shape = eps.rearrange(shape, "b t d -> (b t) d")
        pose = eps.rearrange(pose, "b t d -> (b t) d")
        if self.joint_rep_type == "6d":
            pose_aa = eps.rearrange(pose, "b (j d) -> (b j) d", j=MANO_JOINT_COUNT)
            pose_aa = rotation6d_to_rotation_matrix(pose_aa)
            pose_aa = kornia.geometry.conversions.rotation_matrix_to_axis_angle(pose_aa)
            pose = eps.rearrange(pose_aa, "(b j) d -> b (j d)", j=MANO_JOINT_COUNT)
        elif self.joint_rep_type == "quat":
            pose_aa = eps.rearrange(pose, "b (j d) -> (b j) d", j=MANO_JOINT_COUNT)
            pose_aa = kornia.geometry.conversions.quaternion_to_axis_angle(pose_aa)
            pose = eps.rearrange(pose_aa, "(b j) d -> b (j d)", j=MANO_JOINT_COUNT)
        elif self.joint_rep_type != "3":
            raise NotImplementedError(f"Unsupported rotation type={self.joint_rep_type}")

        mano_output = self.rmano_layer(
            betas=shape,
            global_orient=pose[:, :3],
            hand_pose=pose[:, 3:],
            transl=torch.zeros(size=(pose.shape[0], 3), device=pose.device),
        )
        joints = torch.einsum("nvd,jv->njd", mano_output.vertices, self.J_regressor_mano)
        joint_root_detach = joints[:, :1].detach()
        verts_rel = eps.rearrange(
            (mano_output.vertices - joint_root_detach) * 1e3, "(b t) v d -> b t v d", b=batch_size
        )
        joint_rel = eps.rearrange(
            (joints - joint_root_detach) * 1e3, "(b t) j d -> b t j d", b=batch_size, j=njoint_hand
        )
        return joint_rel, verts_rel

    @torch.inference_mode()
    def predict_full(
        self,
        img: torch.Tensor,
        bbox: torch.Tensor,
        focal: torch.Tensor,
        princpt: torch.Tensor,
        timestamp: Optional[torch.Tensor] = None,
        hand_bbox: Optional[torch.Tensor] = None,
        joint_cam_gt: Optional[torch.Tensor] = None,
        joint_3d_valid_gt: Optional[torch.Tensor] = None,
    ):
        """Inference helper returning absolute/relative joints and vertices for evaluation/export."""
        del joint_cam_gt, joint_3d_valid_gt
        pose_pred, shape_pred, trans_pred, _ = self.predict_mano_param(
            img=img,
            bbox=bbox,
            focal=focal,
            princpt=princpt,
            timestamp=timestamp,
            hand_bbox=hand_bbox,
        )
        pose_pred = pose_pred[:, -1:]
        shape_pred = shape_pred[:, -1:]
        trans_pred = trans_pred[:, -1:]
        joint_rel_pred, vert_rel_pred = self.mano_to_pose(pose_pred, shape_pred)
        joint_cam_pred = joint_rel_pred + trans_pred[:, :, None, :]
        vert_cam_pred = vert_rel_pred + trans_pred[:, :, None, :]
        return {
            "mano_pose_pred": pose_pred,
            "mano_shape_pred": shape_pred,
            "trans_pred": trans_pred,
            "trans_pred_denorm": trans_pred,
            "joint_cam_pred": joint_cam_pred,
            "vert_cam_pred": vert_cam_pred,
            "joint_rel_pred": joint_rel_pred,
            "vert_rel_pred": vert_rel_pred,
            "norm_scale": torch.ones((trans_pred.shape[0], 1), device=trans_pred.device),
            "norm_valid": torch.ones((trans_pred.shape[0], 1), device=trans_pred.device),
        }

    def forward(self, batch):
        """Training/eval forward returning scalar loss, logging state, and detached predictions."""
        pose_pred, shape_pred, trans_pred, cam_aux = self.predict_mano_param(
            img=batch["patches"],
            bbox=batch["patch_bbox"],
            focal=batch["focal"],
            princpt=batch["princpt"],
            timestamp=batch["timestamp"],
            hand_bbox=batch["hand_bbox"],
        )
        loss, loss_state, result = self.loss_fn(pose_pred, shape_pred, trans_pred, cam_aux, batch)
        if self._should_run_ti_branch():
            ti_pose_pred, ti_shape_pred, ti_trans_pred, ti_cam_aux = self._predict_ti_stage1(
                img=batch["patches"],
                bbox=batch["patch_bbox"],
                focal=batch["focal"],
                princpt=batch["princpt"],
                hand_bbox=batch["hand_bbox"],
            )
            ti_loss, ti_loss_state, _ = self.loss_fn(
                ti_pose_pred,
                ti_shape_pred,
                ti_trans_pred,
                ti_cam_aux,
                batch,
                enabled_terms=self.ti_loss_terms,
            )
            loss = loss + self.ti_loss_weight * ti_loss
            loss_state["loss_ti_total"] = ti_loss.detach()
            for key, value in ti_loss_state.items():
                loss_state[f"ti_{key}"] = value.detach()
        metric_state = self.metric_meter(
            batch["joint_cam"][:, -1:],
            batch["joint_cam"][:, -1:] - batch["joint_cam"][:, -1:, :1],
            result["verts_cam_gt"][:, -1:],
            result["verts_rel_gt"][:, -1:],
            result["joint_cam_pred"][:, -1:],
            result["joint_rel_pred"][:, -1:],
            result["verts_cam_pred"][:, -1:],
            result["verts_rel_pred"][:, -1:],
            batch["has_mano"][:, -1:],
            batch["joint_3d_valid"][:, -1:],
            torch.ones_like(batch["has_mano"][:, -1:]),
            build_dataset_group_mask(
                batch.get("data_source"),
                self.loss_fn.ego_datasets,
                device=pose_pred.device,
                dtype=pose_pred.dtype,
            ),
            build_dataset_group_mask(
                batch.get("data_source"),
                self.loss_fn.aux_datasets,
                device=pose_pred.device,
                dtype=pose_pred.dtype,
            ),
        )
        return {
            "loss": loss,
            "state": loss_state | metric_state,
            "result": {
                "joint_cam_pred": result["joint_cam_pred"].detach(),
                "verts_cam_pred": result["verts_cam_pred"].detach(),
                "verts_cam_gt": result["verts_cam_gt"].detach(),
            },
        }

    def get_optim_param_dict(self, lr: float, backbone_lr: Optional[float]):
        """
        Build optimizer parameter groups for the current stage.

        Stage1 optimizes the perspective embedder + decoder, and optionally the backbone.
        Stage2 additionally optimizes the temporal refiner because that path becomes active.
        """
        ret = []
        if self.stage == PoseNet.Stage.STAGE1:
            ret.append(
                {
                    "params": filter(
                        lambda p: p.requires_grad,
                        itertools.chain(
                            self.persp_info_embedder.parameters(),
                            self.handec.parameters(),
                            (
                                ()
                                if self.ti_transform is None or not self.ti_apply_stage1
                                else self.ti_transform.parameters()
                            ),
                        ),
                    ),
                    "lr": lr,
                }
            )
            if backbone_lr is not None:
                ret.append(
                    {
                        "params": filter(lambda p: p.requires_grad, self.backbone.parameters()),
                        "lr": backbone_lr,
                    }
                )
        else:
            ret.append(
                {
                    "params": filter(
                        lambda p: p.requires_grad,
                        itertools.chain(
                            self.temporal_refiner.parameters(),
                            (
                                ()
                                if self.ti_transform is None or not self.ti_apply_stage2
                                else self.ti_transform.parameters()
                            ),
                        ),
                    ),
                    "lr": lr,
                }
            )
        return ret

    def set_dropout_rate(self, dropout_rate: float):
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be in range [0.0, 1.0], got {dropout_rate}")
        for module in self.handec.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_rate
        if hasattr(self, "temporal_refiner") and self.temporal_refiner is not None:
            for module in self.temporal_refiner.modules():
                if isinstance(module, nn.Dropout):
                    module.p = dropout_rate
        if self.ti_transform is not None:
            for module in self.ti_transform.modules():
                if isinstance(module, nn.Dropout):
                    module.p = dropout_rate

    def train(self, mode=True):
        if self.stage == PoseNet.Stage.STAGE1:
            super().train(mode)
            self.backbone.train(mode and not self.freeze_backbone)
            if self.ti_transform is not None:
                self.ti_transform.train(mode and self.ti_apply_stage1)
        else:
            super().train(mode)
            self.backbone.train(False)
            self.persp_info_embedder.train(False)
            self.handec.train(False)
            self.temporal_refiner.train(mode)
            if self.ti_transform is not None:
                self.ti_transform.train(mode and self.ti_apply_stage2)
        return self

    def eval(self):
        return self.train(False)
