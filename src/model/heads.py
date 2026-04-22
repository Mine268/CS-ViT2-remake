from __future__ import annotations

"""Prediction heads for MANO parameters, UV heatmaps, and rho multibin depth."""

from typing import Optional, Tuple, Union

import einops as eps
import kornia
import numpy as np
import torch
import torch.nn as nn

from ..constant import (
    JOINT_DIM_DICT,
    MANO_JOINT_COUNT,
    MANO_MEAN_NPZ,
    MANO_SHAPE_DIM,
    NORM_STAT_NPZ,
)
from ..utils.proj import image_uv_to_camera_ray, patch_uv_to_image_uv
from ..utils.rot import rotation6d_to_rotation_matrix
from .common import TransformerDecoder
from .root_z import (
    RHO_GEOM_DIM,
    compute_rho_prior_and_geom,
    decode_delta_log_rho_predictions,
)


class SoftargmaxHead2DJoint(nn.Module):
    """Predict one 2D point as a spatial distribution plus softargmax expectation."""

    def __init__(self, dim: int, resolution: Tuple[int, int], x_range, y_range):
        super().__init__()
        self.height = int(resolution[0])
        self.width = int(resolution[1])
        self.decuv = nn.Linear(dim, self.height * self.width, bias=False)
        self.register_buffer("x_centers", torch.linspace(x_range[0], x_range[1], self.width))
        self.register_buffer("y_centers", torch.linspace(y_range[0], y_range[1], self.height))

    def forward(self, token: torch.Tensor):
        """Decode logits on a fixed grid and recover expected `(x, y)` coordinates."""
        prefix_shape = token.shape[:-1]
        logits_flat = self.decuv(token)
        log_hm_uv = torch.nn.functional.log_softmax(logits_flat, dim=-1).view(
            *prefix_shape, self.height, self.width
        )
        hm_uv = torch.exp(log_hm_uv)
        hm_x = hm_uv.sum(dim=-2)
        hm_y = hm_uv.sum(dim=-1)
        pred_x = torch.sum(hm_x * self.x_centers, dim=-1, keepdim=True)
        pred_y = torch.sum(hm_y * self.y_centers, dim=-1, keepdim=True)
        return torch.cat([pred_x, pred_y], dim=-1), log_hm_uv

    def get_centers(self):
        """Return the x/y coordinate centers used by the heatmap grid."""
        return self.x_centers, self.y_centers


class RhoMultiBinHead(nn.Module):
    """
    Predict camera distance `rho` as prior-centered multibin classification + residual.

    This head combines:
    - token features from the decoder
    - explicit geometry features derived from hand bbox and intrinsics

    The output dictionary keeps both logits and decoded geometric quantities so losses and
    diagnostics can share one forward pass.
    """

    def __init__(
        self,
        dim: int,
        num_bins: int,
        d_min: float,
        d_max: float,
        prior_k: float,
        geom_hidden_dim: int = 256,
        dropout: float = 0.0,
        use_data_source_embed: bool = False,
    ):
        super().__init__()
        if use_data_source_embed:
            raise NotImplementedError("data_source embedding is not implemented in RhoMultiBinHead")
        if num_bins <= 0:
            raise ValueError(f"num_bins must be positive, got {num_bins}")
        if d_max <= d_min:
            raise ValueError(f"d_max must be larger than d_min, got {d_min} >= {d_max}")

        self.num_bins = num_bins
        self.d_min = float(d_min)
        self.d_max = float(d_max)
        self.prior_k = float(prior_k)

        self.geom_proj = nn.Sequential(
            nn.Linear(RHO_GEOM_DIM, geom_hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(geom_hidden_dim, geom_hidden_dim, bias=True),
            nn.GELU(),
        )
        self.fusion_norm = nn.LayerNorm(dim + geom_hidden_dim)
        self.cls_head = nn.Linear(dim + geom_hidden_dim, num_bins, bias=True)
        self.res_head = nn.Linear(dim + geom_hidden_dim, num_bins, bias=True)

    def forward(
        self,
        token: torch.Tensor,
        hand_bbox: torch.Tensor,
        focal: torch.Tensor,
        princpt: torch.Tensor,
    ):
        """Fuse token/geometric features and decode a scalar `rho` prediction plus auxiliaries."""
        rho_prior, log_rho_prior, geom_feat = compute_rho_prior_and_geom(
            hand_bbox=hand_bbox,
            focal=focal,
            princpt=princpt,
            prior_k=self.prior_k,
        )
        geom_hidden = self.geom_proj(geom_feat)
        fused = self.fusion_norm(torch.cat([token, geom_hidden], dim=-1))
        rho_cls_logits = self.cls_head(fused)
        rho_residuals = self.res_head(fused)
        decoded = decode_delta_log_rho_predictions(
            rho_cls_logits=rho_cls_logits,
            rho_residuals=rho_residuals,
            log_rho_prior=log_rho_prior,
            d_min=self.d_min,
            d_max=self.d_max,
        )
        aux = {
            "cam_head_type": "patch_uv_rho_multibin",
            "rho_cls_logits": rho_cls_logits,
            "rho_residuals": rho_residuals,
            "pred_rho": decoded["pred_rho"][..., None],
            "rho_prior": rho_prior[..., None],
            "log_rho_prior": log_rho_prior[..., None],
            "rho_geom_feat": geom_feat,
            "pred_rho_bin": decoded["pred_bin"],
            "pred_rho_residual": decoded["pred_residual"],
            "pred_delta_log_rho": decoded["pred_delta_log_rho"][..., None],
            "pred_log_rho": decoded["pred_log_rho"][..., None],
        }
        return decoded["pred_rho"][..., None], aux


class MANOTransformerDecoderHead(nn.Module):
    """Decode backbone tokens into MANO pose/shape and camera parameters."""

    def __init__(
        self,
        joint_rep_type: str,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        emb_dropout_type: str = "drop",
        norm: str = "layer",
        norm_cond_dim: int = -1,
        context_dim: Optional[int] = None,
        skip_token_embedding: bool = False,
        use_mean_init: bool = True,
        denorm_output: bool = False,
        norm_by_hand: bool = False,
        heatmap_resolution: Union[int, Tuple[int]] = (512, 512, 1024),
        patch_size: Optional[Union[int, Tuple[int, int]]] = None,
        cam_head_type: str = "patch_uv_rho_multibin",
        root_z_num_bins: int = 8,
        root_z_d_min: float = -0.71,
        root_z_d_max: float = 0.75,
        root_z_prior_k: float = 121.0,
        root_z_geom_hidden_dim: int = 256,
        root_z_dropout: float = 0.0,
        root_z_use_data_source_embed: bool = False,
    ):
        super().__init__()
        if norm_by_hand:
            raise NotImplementedError("CS-ViT2-remake only supports norm_by_hand=false")
        if denorm_output:
            raise NotImplementedError("denorm_output is removed in the remake repo")
        if cam_head_type != "patch_uv_rho_multibin":
            raise ValueError(f"Only patch_uv_rho_multibin is supported, got {cam_head_type}")
        if isinstance(heatmap_resolution, int):
            heatmap_resolution = (heatmap_resolution, heatmap_resolution, heatmap_resolution)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if patch_size is None:
            raise ValueError("patch_uv_rho_multibin requires patch_size")

        self.joint_rep_type = joint_rep_type
        self.joint_dim = JOINT_DIM_DICT[joint_rep_type]
        self.npose = self.joint_dim * MANO_JOINT_COUNT
        self.patch_size = patch_size
        self.cam_head_type = cam_head_type

        self.transformer = TransformerDecoder(
            num_tokens=1,
            token_dim=(self.npose + MANO_SHAPE_DIM + 3),
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout,
            emb_dropout_type=emb_dropout_type,
            norm=norm,
            norm_cond_dim=norm_cond_dim,
            context_dim=context_dim,
            skip_token_embedding=skip_token_embedding,
        )

        self.decpose = nn.Linear(dim, self.npose)
        self.decshape = nn.Linear(dim, MANO_SHAPE_DIM)

        norm_stats = np.load(NORM_STAT_NPZ)
        mean = norm_stats["mean"]
        std = norm_stats["std"]
        x_range = [mean[0] - std[0] * 5, mean[0] + std[0] * 5]
        y_range = [mean[1] - std[1] * 5, mean[1] + std[1] * 5]
        del x_range, y_range

        patch_h, patch_w = self.patch_size
        uv_h = min(int(heatmap_resolution[0]), max(8, int(patch_h) // 4))
        uv_w = min(int(heatmap_resolution[1]), max(8, int(patch_w) // 4))
        self.deccam_uv = SoftargmaxHead2DJoint(
            dim,
            (uv_h, uv_w),
            [0.0, float(patch_w)],
            [0.0, float(patch_h)],
        )
        self.decrho = RhoMultiBinHead(
            dim=dim,
            num_bins=root_z_num_bins,
            d_min=root_z_d_min,
            d_max=root_z_d_max,
            prior_k=root_z_prior_k,
            geom_hidden_dim=root_z_geom_hidden_dim,
            dropout=root_z_dropout,
            use_data_source_embed=root_z_use_data_source_embed,
        )

        if use_mean_init:
            mean_params = np.load(MANO_MEAN_NPZ)
            init_hand_pose = torch.from_numpy(mean_params["pose"].astype(np.float32))
            if joint_rep_type == "6d":
                pass
            elif joint_rep_type == "3":
                init_hand_pose = rotation6d_to_rotation_matrix(init_hand_pose.reshape(-1, 6))
                init_hand_pose = kornia.geometry.conversions.rotation_matrix_to_axis_angle(init_hand_pose)
                init_hand_pose = torch.flatten(init_hand_pose).unsqueeze(0)
            elif joint_rep_type == "quat":
                init_hand_pose = rotation6d_to_rotation_matrix(init_hand_pose.reshape(-1, 6))
                init_hand_pose = kornia.geometry.conversions.rotation_matrix_to_quaternion(init_hand_pose)
                init_hand_pose = torch.flatten(init_hand_pose).unsqueeze(0)
            else:
                raise NotImplementedError(f"Unsupported joint_rep_type={joint_rep_type}")
            init_betas = torch.from_numpy(mean_params["shape"].astype("float32")).unsqueeze(0)
            init_cam = torch.from_numpy(mean_params["cam"].astype(np.float32)).unsqueeze(0)
        else:
            init_hand_pose = torch.randn(size=(1, self.npose))
            init_betas = torch.randn(size=(1, MANO_SHAPE_DIM))
            init_cam = torch.randn(size=(1, 3))

        self.register_buffer("init_hand_pose", init_hand_pose)
        self.register_buffer("init_betas", init_betas)
        self.register_buffer("init_cam", init_cam)

    def encode_img(self, x: torch.Tensor) -> torch.Tensor:
        """Run the token decoder starting from learned mean MANO/camera initialization."""
        batch_size = x.shape[0]
        init_hand_pose = self.init_hand_pose.expand(batch_size, -1)
        init_betas = self.init_betas.expand(batch_size, -1)
        init_cam = self.init_cam.expand(batch_size, -1)
        token = torch.cat([init_hand_pose, init_betas, init_cam], dim=1)[:, None, :]
        return self.transformer(token, context=x).squeeze(1)

    def decode_token(
        self,
        token_out: torch.Tensor,
        patch_bbox: torch.Tensor,
        hand_bbox: torch.Tensor,
        focal: torch.Tensor,
        princpt: torch.Tensor,
    ):
        """Decode one latent token into pose, shape, camera, and dense auxiliary outputs."""
        pred_hand_pose = self.decpose(token_out)
        pred_betas = self.decshape(token_out)
        pred_uv_patch, pred_log_heatmaps_uv = self.deccam_uv(token_out)
        pred_rho, rho_aux = self.decrho(
            token=token_out,
            hand_bbox=hand_bbox,
            focal=focal,
            princpt=princpt,
        )
        pred_uv_img = patch_uv_to_image_uv(
            pred_uv_patch,
            patch_bbox=patch_bbox,
            patch_size=self.patch_size,
        )
        pred_q, pred_ray_unit, pred_q_norm = image_uv_to_camera_ray(
            pred_uv_img,
            focal=focal,
            princpt=princpt,
        )
        # `pred_rho` is a scalar distance along the image ray, so multiplying by the unit ray
        # directly recovers the 3D camera translation/root position.
        pred_cam = pred_ray_unit * pred_rho
        cam_aux = {
            "cam_head_type": self.cam_head_type,
            "log_hm_uv_patch": pred_log_heatmaps_uv,
            "pred_uv_patch": pred_uv_patch,
            "pred_uv_img": pred_uv_img,
            "pred_q": pred_q,
            "pred_q_norm": pred_q_norm[..., None],
            "pred_ray_unit": pred_ray_unit,
        } | rho_aux
        return (pred_hand_pose, pred_betas, pred_cam), cam_aux

    def forward(
        self,
        x: torch.Tensor,
        patch_bbox: torch.Tensor,
        hand_bbox: torch.Tensor,
        focal: torch.Tensor,
        princpt: torch.Tensor,
    ):
        """Full decoder forward used by the image pathway."""
        token_out = self.encode_img(x)
        (pred_hand_pose, pred_betas, pred_cam), cam_aux = self.decode_token(
            token_out,
            patch_bbox=patch_bbox,
            hand_bbox=hand_bbox,
            focal=focal,
            princpt=princpt,
        )
        return (pred_hand_pose, pred_betas, pred_cam), cam_aux, token_out

    def get_centers(self):
        """Expose UV grid centers so the loss can build Gaussian heatmap targets."""
        return (*self.deccam_uv.get_centers(), None)
