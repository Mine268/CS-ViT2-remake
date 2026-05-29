"""Loss functions and supervision routing for the remake training pipeline."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.mano import RMANOLayer
from ..utils.proj import proj_points_3d
from .root_z import encode_delta_log_rho_targets


class RobustL1Loss(nn.Module):
    """L1 near zero, logarithmically softened for large residuals."""

    def __init__(self, delta: float = 100.0, reduction: str = "none"):
        super().__init__()
        self.delta = float(delta)
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        abs_diff = torch.abs(pred - target)
        inside_mask = abs_diff < self.delta
        loss_l1 = abs_diff
        outside_diff = torch.clamp(abs_diff - self.delta, min=0.0)
        loss_log = self.delta * (1.0 + torch.log1p(outside_diff / self.delta))
        loss = torch.where(inside_mask, loss_l1, loss_log)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def robust_masked_mean(loss: torch.Tensor, mask: torch.Tensor):
    """Average an elementwise loss over valid entries, returning zero when nothing is valid."""
    mask = mask.to(dtype=loss.dtype)
    if mask.shape != loss.shape:
        mask = torch.broadcast_to(mask, loss.shape)
    loss_masked = loss * mask
    total_loss = loss_masked.sum()
    total_valid = mask.sum()
    if total_valid.item() > 1e-6:
        return total_loss / total_valid
    return total_loss * 0.0


def build_dataset_group_mask(
    data_sources: Optional[Sequence[str]],
    target_sources: Sequence[str],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a `[B, 1]` mask selecting which batch items belong to a dataset group."""
    if data_sources is None:
        return torch.zeros((0, 1), device=device, dtype=dtype)
    target_set = {str(x) for x in target_sources}
    mask = [str(source) in target_set for source in data_sources]
    return torch.tensor(mask, device=device, dtype=dtype).view(-1, 1)


def joint_img_to_patch_resized(
    joint_img: torch.Tensor,
    patch_bbox: torch.Tensor,
    patch_size: Tuple[int, int],
) -> torch.Tensor:
    """Convert image-space joint coordinates into resized patch coordinates."""
    patch_h, patch_w = patch_size
    patch_width = torch.clamp(patch_bbox[..., 2] - patch_bbox[..., 0], min=1e-6)
    patch_height = torch.clamp(patch_bbox[..., 3] - patch_bbox[..., 1], min=1e-6)
    joint_patch_resized = torch.empty_like(joint_img)
    joint_patch_resized[..., 0] = (
        (joint_img[..., 0] - patch_bbox[..., 0, None]) * patch_w / patch_width[..., None]
    )
    joint_patch_resized[..., 1] = (
        (joint_img[..., 1] - patch_bbox[..., 1, None]) * patch_h / patch_height[..., None]
    )
    return joint_patch_resized


class RemakeLoss(nn.Module):
    """
    Central supervision module for MANO, camera, and reprojection losses.

    The key routing rule is:
    - `ego_datasets`: receive absolute/root supervision
    - `ego_datasets + aux_datasets`: receive local/MANO supervision
    """

    def __init__(
        self,
        lambda_theta: float,
        lambda_shape: float,
        lambda_trans: float,
        lambda_rel: float,
        lambda_img: float,
        lambda_uv_patch: float,
        lambda_root_z_cls: float,
        lambda_root_z_res: float,
        hm_centers,
        hm_sigma: float,
        pred_joint_z_min_mm: float,
        reproj_loss_type: str,
        reproj_loss_delta: float,
        ego_datasets: List[str],
        aux_datasets: List[str],
        min_valid_joints_2d: int = 0,
        min_hand_bbox_edge_px: float = 0.0,
        rho_d_min: float = -0.71,
        rho_d_max: float = 0.75,
    ):
        super().__init__()
        self.l1 = nn.L1Loss(reduction="none")
        self.lambda_theta = lambda_theta
        self.lambda_shape = lambda_shape
        self.lambda_trans = lambda_trans
        self.lambda_rel = lambda_rel
        self.lambda_img = lambda_img
        self.lambda_uv_patch = lambda_uv_patch
        self.lambda_root_z_cls = lambda_root_z_cls
        self.lambda_root_z_res = lambda_root_z_res
        self.hm_sigma = hm_sigma
        self.pred_joint_z_min_mm = float(pred_joint_z_min_mm)
        self.ego_datasets = list(ego_datasets)
        self.aux_datasets = list(aux_datasets)
        self.min_valid_joints_2d = int(min_valid_joints_2d)
        self.min_hand_bbox_edge_px = float(min_hand_bbox_edge_px)
        self.rho_d_min = float(rho_d_min)
        self.rho_d_max = float(rho_d_max)
        self.register_buffer("x_centers", hm_centers[0])
        self.register_buffer("y_centers", hm_centers[1])
        if reproj_loss_type == "l1":
            self.reproj_loss_fn = self.l1
        elif reproj_loss_type == "robust_l1":
            self.reproj_loss_fn = RobustL1Loss(delta=reproj_loss_delta, reduction="none")
        else:
            raise ValueError(f"Unsupported reproj_loss_type: {reproj_loss_type}")
        self.rmano_layer = RMANOLayer()

    def _zero_like(self, tensor: torch.Tensor) -> torch.Tensor:
        """Create a scalar zero that stays on the same graph/device as `tensor`."""
        return tensor.sum() * 0.0

    def _masked_mean_scalar(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute a scalar masked mean for already-reduced values."""
        mask = mask.to(dtype=values.dtype)
        total_valid = mask.sum()
        if total_valid.item() <= 1e-6:
            return values.sum() * 0.0
        return (values * mask).sum() / total_valid

    def _resolve_enabled_terms(self, enabled_terms: Optional[Iterable[str]]) -> set[str]:
        valid_terms = {
            "theta",
            "shape",
            "trans",
            "joint_rel",
            "joint_img",
            "uv_patch",
            "rho_cls",
            "rho_res",
        }
        if enabled_terms is None:
            return valid_terms
        resolved = {str(term) for term in enabled_terms}
        unknown = resolved - valid_terms
        if unknown:
            raise ValueError(f"Unknown loss terms: {sorted(unknown)}")
        return resolved

    def compute_root_frame_filter_mask(self, batch) -> torch.Tensor:
        """Reject frames that are too weak for stable absolute root/rho supervision."""
        frame_mask = torch.ones_like(batch["has_intr"], dtype=torch.float32)
        if self.min_valid_joints_2d > 0:
            valid_joint_count = torch.sum(batch["joint_2d_valid"] > 0.5, dim=-1)
            frame_mask = frame_mask * (valid_joint_count >= self.min_valid_joints_2d).float()
        if self.min_hand_bbox_edge_px > 0:
            hand_bbox = batch["hand_bbox"]
            bbox_min_edge = torch.minimum(
                hand_bbox[..., 2] - hand_bbox[..., 0],
                hand_bbox[..., 3] - hand_bbox[..., 1],
            )
            frame_mask = frame_mask * (bbox_min_edge >= self.min_hand_bbox_edge_px).float()
        return frame_mask

    def compute_hm_ce_2d(
        self,
        pred_log_probs: torch.Tensor,
        gt_coords: torch.Tensor,
        x_grid_positions: torch.Tensor,
        y_grid_positions: torch.Tensor,
    ):
        """Cross-entropy between predicted log-heatmaps and Gaussian targets centered on GT UV."""
        prefix_ndim = gt_coords.dim() - 1
        gt_x = gt_coords[..., 0].unsqueeze(-1).unsqueeze(-1)
        gt_y = gt_coords[..., 1].unsqueeze(-1).unsqueeze(-1)
        x_view_shape = [1] * prefix_ndim + [1, -1]
        y_view_shape = [1] * prefix_ndim + [-1, 1]
        x_grid = x_grid_positions.view(x_view_shape)
        y_grid = y_grid_positions.view(y_view_shape)
        squared_diff = (gt_x - x_grid) ** 2 + (gt_y - y_grid) ** 2
        target_unnormalized = torch.exp(-squared_diff / (2 * self.hm_sigma**2))
        target_probs = target_unnormalized / (
            target_unnormalized.sum(dim=(-2, -1), keepdim=True) + 1e-9
        )
        return -(target_probs * pred_log_probs).sum(dim=(-2, -1))

    def forward(
        self,
        pose_pred,
        shape_pred,
        trans_pred,
        cam_aux,
        batch,
        enabled_terms: Optional[Iterable[str]] = None,
    ):
        """
        Compute all supervised losses for one training batch.

        Important routing:
        - MANO pose/shape and relative joints use `all_supervised_mask`
        - translation, rho, and reprojection use `ego_mask`
        """
        with torch.no_grad():
            _, vert_rel_gt = self.rmano_layer(batch["mano_pose"], batch["mano_shape"])
        joint_rel_pred, vert_rel_pred = self.rmano_layer(pose_pred, shape_pred.detach())
        enabled_terms = self._resolve_enabled_terms(enabled_terms)

        pose_gt = batch["mano_pose"]
        shape_gt = batch["mano_shape"]
        has_mano = batch["has_mano"]
        joint_2d_valid = batch["joint_2d_valid"]
        joint_3d_valid = batch["joint_3d_valid"]
        has_intr = batch["has_intr"]
        trans_gt = batch["joint_cam"][:, :, 0]

        ego_mask = build_dataset_group_mask(
            batch.get("data_source"),
            self.ego_datasets,
            device=pose_pred.device,
            dtype=pose_pred.dtype,
        )
        all_supervised_mask = build_dataset_group_mask(
            batch.get("data_source"),
            self.ego_datasets + self.aux_datasets,
            device=pose_pred.device,
            dtype=pose_pred.dtype,
        )
        if all_supervised_mask.numel() == 0:
            # Evaluation or synthetic tests may omit dataset names entirely; fall back to "all on".
            all_supervised_mask = torch.ones(
                (pose_pred.shape[0], 1), device=pose_pred.device, dtype=pose_pred.dtype
            )

        if "theta" in enabled_terms:
            loss_theta = robust_masked_mean(
                self.l1(pose_pred, pose_gt), has_mano[..., None] * all_supervised_mask[:, :, None]
            )
        else:
            loss_theta = self._zero_like(pose_pred)
        if "shape" in enabled_terms:
            loss_shape = robust_masked_mean(
                self.l1(shape_pred, shape_gt), has_mano[..., None] * all_supervised_mask[:, :, None]
            )
        else:
            loss_shape = self._zero_like(shape_pred)

        root_valid_mask = joint_3d_valid[:, :, 0]
        root_uv_patch_gt = batch["joint_patch_resized"][:, :, 0]
        root_uv_valid = joint_2d_valid[:, :, 0]
        range_frame_filter = self.compute_root_frame_filter_mask(batch)
        uv_patch_valid = root_uv_valid * range_frame_filter

        if "uv_patch" in enabled_terms:
            loss_uv_patch = self.compute_hm_ce_2d(
                cam_aux["log_hm_uv_patch"],
                root_uv_patch_gt,
                self.x_centers,
                self.y_centers,
            )
            loss_uv_patch = robust_masked_mean(loss_uv_patch, uv_patch_valid)
        else:
            loss_uv_patch = self._zero_like(trans_pred)

        ego_root_valid = (
            ego_mask
            * root_valid_mask
            * has_intr
            * (trans_gt[..., 2] > 0.0).float()
            * range_frame_filter
        )
        if "rho_cls" in enabled_terms or "rho_res" in enabled_terms:
            rho_gt = torch.linalg.norm(trans_gt, dim=-1)
            encoded_rho = encode_delta_log_rho_targets(
                rho=rho_gt,
                log_rho_prior=cam_aux["log_rho_prior"].squeeze(-1),
                d_min=self.rho_d_min,
                d_max=self.rho_d_max,
                num_bins=cam_aux["rho_cls_logits"].shape[-1],
            )
            valid_bool = ego_root_valid > 0.5
            if torch.any(valid_bool):
                if "rho_cls" in enabled_terms:
                    loss_rho_cls = F.cross_entropy(
                        cam_aux["rho_cls_logits"][valid_bool],
                        encoded_rho["bin_idx"][valid_bool],
                        reduction="mean",
                    )
                else:
                    loss_rho_cls = self._zero_like(cam_aux["rho_cls_logits"])
                pred_rho_res = (
                    cam_aux["rho_residuals"]
                    .gather(
                        dim=-1,
                        index=encoded_rho["bin_idx"].unsqueeze(-1),
                    )
                    .squeeze(-1)
                )
                if "rho_res" in enabled_terms:
                    loss_rho_res = F.smooth_l1_loss(
                        pred_rho_res[valid_bool],
                        encoded_rho["residual"][valid_bool],
                        reduction="mean",
                        beta=0.1,
                    )
                else:
                    loss_rho_res = self._zero_like(cam_aux["rho_residuals"])
                pred_rho_bin = torch.argmax(cam_aux["rho_cls_logits"], dim=-1)
                rho_bin_acc = (
                    (pred_rho_bin[valid_bool] == encoded_rho["bin_idx"][valid_bool]).float().mean()
                )
                rho_mae_mm = torch.abs(cam_aux["pred_rho"].squeeze(-1) - rho_gt)
                rho_mae_mm = self._masked_mean_scalar(rho_mae_mm, ego_root_valid)
            else:
                loss_rho_cls = cam_aux["rho_cls_logits"].sum() * 0.0
                loss_rho_res = cam_aux["rho_residuals"].sum() * 0.0
                rho_bin_acc = cam_aux["rho_cls_logits"].sum() * 0.0
                rho_mae_mm = cam_aux["pred_rho"].sum() * 0.0
        else:
            loss_rho_cls = self._zero_like(trans_pred)
            loss_rho_res = self._zero_like(trans_pred)
            rho_bin_acc = self._zero_like(trans_pred)
            rho_mae_mm = self._zero_like(trans_pred)

        if "trans" in enabled_terms:
            loss_trans = robust_masked_mean(
                self.l1(trans_pred, trans_gt), ego_root_valid[..., None]
            )
        else:
            loss_trans = self._zero_like(trans_pred)
        if "joint_rel" in enabled_terms:
            loss_joint_rel = robust_masked_mean(
                self.l1(joint_rel_pred, batch["joint_rel"]),
                joint_3d_valid[..., None] * all_supervised_mask[:, :, None, None],
            )
        else:
            loss_joint_rel = self._zero_like(joint_rel_pred)

        # Absolute 3D joints are recovered by adding the predicted root translation back to the
        # MANO-relative joints produced by the kinematic layer.
        joint_cam_pred = joint_rel_pred + trans_pred[:, :, None, :]
        pred_joint_z = joint_cam_pred[..., 2]
        pred_joint_z_min = torch.min(pred_joint_z)
        pred_joint_z_valid = (pred_joint_z > self.pred_joint_z_min_mm).to(
            dtype=joint_3d_valid.dtype
        )
        reproj_valid = (
            joint_3d_valid
            * has_intr[..., None]
            * pred_joint_z_valid
            * ego_mask[:, :, None]
            * range_frame_filter[..., None]
        )
        if "joint_img" in enabled_terms:
            joint_img_gt = proj_points_3d(batch["joint_cam"], batch["focal"], batch["princpt"])
            joint_img_pred = proj_points_3d(joint_cam_pred, batch["focal"], batch["princpt"])
            loss_joint_img = robust_masked_mean(
                self.reproj_loss_fn(joint_img_pred, joint_img_gt),
                reproj_valid[..., None],
            )
        else:
            loss_joint_img = self._zero_like(joint_cam_pred)

        loss = (
            self.lambda_theta * loss_theta
            + self.lambda_shape * loss_shape
            + self.lambda_uv_patch * loss_uv_patch
            + self.lambda_trans * loss_trans
            + self.lambda_root_z_cls * loss_rho_cls
            + self.lambda_root_z_res * loss_rho_res
            + self.lambda_rel * loss_joint_rel
            + self.lambda_img * loss_joint_img
        )

        loss_state = {
            "loss_theta": loss_theta.detach(),
            "loss_shape": loss_shape.detach(),
            "loss_trans": loss_trans.detach(),
            "loss_uv_patch": loss_uv_patch.detach(),
            "loss_rho_cls": loss_rho_cls.detach(),
            "loss_rho_res": loss_rho_res.detach(),
            "rho_bin_acc": rho_bin_acc.detach(),
            "rho_mae_mm": rho_mae_mm.detach(),
            "pred_joint_z_min": pred_joint_z_min.detach(),
            "reproj_valid_frac": reproj_valid.mean().detach(),
            "ego_root_valid_frac": ego_root_valid.mean().detach(),
            "loss_joint_rel": loss_joint_rel.detach(),
            "loss_joint_img": loss_joint_img.detach(),
        }

        fk_result = {
            "verts_cam_gt": vert_rel_gt + batch["joint_cam"][:, :, :1],
            "verts_rel_gt": vert_rel_gt,
            "joint_cam_pred": joint_cam_pred,
            "joint_rel_pred": joint_rel_pred,
            "verts_cam_pred": vert_rel_pred + trans_pred[..., None, :],
            "verts_rel_pred": vert_rel_pred,
        }
        return loss, loss_state, fk_result
