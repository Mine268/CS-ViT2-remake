"""Metric helpers used during online training logs and streaming validation."""

from typing import Any, Iterable, Optional, Set

import numpy as np
import torch

from .data_source import stringify_data_source_value


def _normalize_data_source_value(value: Any) -> str:
    """Normalize heterogeneous data_source payloads before metric-side filtering."""
    return stringify_data_source_value(value)


def build_dataset_group_mask(
    data_sources: Optional[Iterable[Any]],
    target_sources: Iterable[str],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Build a `[B, 1]` tensor mask selecting which batch items belong to a named dataset group.

    Args:
        data_sources: Per-sample dataset names from the current batch.
        target_sources: Dataset names belonging to one logical group, e.g. ego or aux.
        device: Target device for the returned mask tensor.
        dtype: Target dtype for the returned mask tensor.
    """
    if data_sources is None:
        return torch.zeros((0, 1), device=device, dtype=dtype)
    target_set = {str(name) for name in target_sources}
    values = [float(_normalize_data_source_value(value) in target_set) for value in data_sources]
    return torch.tensor(values, device=device, dtype=dtype).view(-1, 1)


def build_excluded_data_source_mask(
    data_sources: Optional[Iterable[Any]],
    excluded_sources: Optional[Set[str]] = None,
):
    """
    为 data_source 序列构造保留掩码，默认排除 COCO-WholeBody。

    Returns:
        np.ndarray[bool] 或 None
    """
    if data_sources is None:
        return None

    if excluded_sources is None:
        excluded_sources = {"COCO-WholeBody"}

    normalized = [_normalize_data_source_value(value) for value in data_sources]
    return np.array(
        [value not in excluded_sources for value in normalized],
        dtype=bool,
    )


def compute_mpjpe_stats(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor = None,
):
    """
    计算 MPJPE 的累积统计量。

    Args:
        pred: 预测的关节坐标 [B, T, J, 3] 或 [..., J, 3]
        gt: 真实的关节坐标，形状同 pred
        mask: 关节级有效性掩码 [B, T, J] 或 [..., J]
    """
    error_per_joint = torch.norm(pred - gt, p=2, dim=-1)

    if mask is None:
        return error_per_joint.sum(), torch.tensor(error_per_joint.numel(), device=pred.device)

    mask_bool = mask > 0.5
    if mask_bool.any():
        return error_per_joint[mask_bool].sum(), mask_bool.sum()

    return torch.tensor(0.0, device=pred.device), torch.tensor(0.0, device=pred.device)


def compute_mpvpe_stats(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor = None,
):
    """
    计算 MPVPE 的累积统计量。

    Args:
        pred: 预测的顶点坐标 [B, T, V, 3]
        gt: 真实的顶点坐标 [B, T, V, 3]
        mask: 帧级有效性掩码 [B, T]
    """
    if pred is None or gt is None:
        zero = torch.tensor(0.0)
        return zero, zero

    error_per_vertex = torch.norm(pred - gt, p=2, dim=-1)
    error_per_vertex = torch.mean(error_per_vertex, dim=-1)

    if mask is None:
        return error_per_vertex.sum(), torch.tensor(error_per_vertex.numel(), device=pred.device)

    mask_bool = mask > 0.5
    if mask_bool.any():
        return (error_per_vertex * mask).sum(), mask_bool.sum()

    return torch.tensor(0.0, device=pred.device), torch.tensor(0.0, device=pred.device)


def compute_rte_stats(pred, gt, mask):
    """
    pred, gt: [b,t,3]
    mask: [b,t]
    """
    if pred is None or gt is None:
        zero = torch.tensor(0.0)
        return zero, zero

    error_per = torch.norm(pred - gt, p=2, dim=-1)
    if mask is None:
        return error_per.sum(), torch.tensor(error_per.numel(), device=pred.device)

    mask_bool = mask > 0.5
    if mask_bool.any():
        valid_errors = error_per[mask_bool]
        return valid_errors.sum(), torch.tensor(valid_errors.numel(), device=pred.device)

    return torch.tensor(0.0, device=pred.device), torch.tensor(0.0, device=pred.device)


def _safe_ratio(total_error: torch.Tensor, total_count: torch.Tensor) -> torch.Tensor:
    """Avoid NaNs when a metric has zero valid samples in the current aggregation window."""
    if total_count.item() <= 0:
        return torch.tensor(0.0, device=total_error.device)
    return total_error / total_count


class MetricMeter:
    """Stateless metric computer used on per-step model outputs."""

    def __init__(self, *args, **kwargs):
        pass

    @torch.no_grad()
    def __call__(
        self,
        joint_cam_gt,
        joint_rel_gt,
        verts_cam_gt,
        verts_rel_gt,
        joint_cam_pred,
        joint_rel_pred,
        verts_cam_pred,
        verts_rel_pred,
        has_mano,
        joint_3d_valid,
        norm_valid,
        ego_mask,
        aux_mask,
    ):
        """Compute the scalar metrics logged from a single batched model forward."""
        joint_mask_all = joint_3d_valid * norm_valid[:, :, None]
        joint_mask_ego = joint_mask_all * ego_mask[:, :, None]
        joint_mask_aux = joint_mask_all * aux_mask[:, :, None]

        vert_mask_all = has_mano * norm_valid
        vert_mask_ego = vert_mask_all * ego_mask
        vert_mask_aux = vert_mask_all * aux_mask

        rte_mask_all = joint_3d_valid[:, :, 0] * norm_valid
        rte_mask_ego = rte_mask_all * ego_mask
        rte_mask_aux = rte_mask_all * aux_mask

        cs_mpjpe_stats = compute_mpjpe_stats(
            joint_cam_pred,
            joint_cam_gt,
            joint_mask_all,
        )
        cs_mpjpe_stats_ego = compute_mpjpe_stats(
            joint_cam_pred,
            joint_cam_gt,
            joint_mask_ego,
        )
        cs_mpjpe_stats_aux = compute_mpjpe_stats(
            joint_cam_pred,
            joint_cam_gt,
            joint_mask_aux,
        )
        rs_mpjpe_stats = compute_mpjpe_stats(
            joint_rel_pred,
            joint_rel_gt,
            joint_mask_all,
        )
        rs_mpjpe_stats_ego = compute_mpjpe_stats(
            joint_rel_pred,
            joint_rel_gt,
            joint_mask_ego,
        )
        rs_mpjpe_stats_aux = compute_mpjpe_stats(
            joint_rel_pred,
            joint_rel_gt,
            joint_mask_aux,
        )
        cs_mpvpe_stats = compute_mpvpe_stats(
            verts_cam_pred,
            verts_cam_gt,
            vert_mask_all,
        )
        cs_mpvpe_stats_ego = compute_mpvpe_stats(
            verts_cam_pred,
            verts_cam_gt,
            vert_mask_ego,
        )
        cs_mpvpe_stats_aux = compute_mpvpe_stats(
            verts_cam_pred,
            verts_cam_gt,
            vert_mask_aux,
        )
        rs_mpvpe_stats = compute_mpvpe_stats(
            verts_rel_pred,
            verts_rel_gt,
            vert_mask_all,
        )
        rs_mpvpe_stats_ego = compute_mpvpe_stats(
            verts_rel_pred,
            verts_rel_gt,
            vert_mask_ego,
        )
        rs_mpvpe_stats_aux = compute_mpvpe_stats(
            verts_rel_pred,
            verts_rel_gt,
            vert_mask_aux,
        )
        rte_stats_all = compute_rte_stats(
            joint_cam_pred[:, :, 0],
            joint_cam_gt[:, :, 0],
            rte_mask_all,
        )
        rte_stats_ego = compute_rte_stats(
            joint_cam_pred[:, :, 0],
            joint_cam_gt[:, :, 0],
            rte_mask_ego,
        )
        rte_stats_aux = compute_rte_stats(
            joint_cam_pred[:, :, 0],
            joint_cam_gt[:, :, 0],
            rte_mask_aux,
        )

        return {
            "micro_mpjpe_all": _safe_ratio(*cs_mpjpe_stats),
            "micro_mpjpe_ego": _safe_ratio(*cs_mpjpe_stats_ego),
            "micro_mpjpe_aux": _safe_ratio(*cs_mpjpe_stats_aux),
            "micro_mpjpe_rel_all": _safe_ratio(*rs_mpjpe_stats),
            "micro_mpjpe_rel_ego": _safe_ratio(*rs_mpjpe_stats_ego),
            "micro_mpjpe_rel_aux": _safe_ratio(*rs_mpjpe_stats_aux),
            "micro_mpvpe_all": _safe_ratio(*cs_mpvpe_stats),
            "micro_mpvpe_ego": _safe_ratio(*cs_mpvpe_stats_ego),
            "micro_mpvpe_aux": _safe_ratio(*cs_mpvpe_stats_aux),
            "micro_mpvpe_rel_all": _safe_ratio(*rs_mpvpe_stats),
            "micro_mpvpe_rel_ego": _safe_ratio(*rs_mpvpe_stats_ego),
            "micro_mpvpe_rel_aux": _safe_ratio(*rs_mpvpe_stats_aux),
            "micro_rte_all": _safe_ratio(*rte_stats_all),
            "micro_rte_ego": _safe_ratio(*rte_stats_ego),
            "micro_rte_aux": _safe_ratio(*rte_stats_aux),
        }


class StreamingMetricMeter:
    """Accumulate validation metrics across many steps before producing a final summary."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Clear all running sums and counts."""
        self.accumulators = {
            "cs_mpjpe_all": [0.0, 0.0],
            "cs_mpjpe_ego": [0.0, 0.0],
            "cs_mpjpe_aux": [0.0, 0.0],
            "rs_mpjpe_all": [0.0, 0.0],
            "rs_mpjpe_ego": [0.0, 0.0],
            "rs_mpjpe_aux": [0.0, 0.0],
            "cs_mpvpe_all": [0.0, 0.0],
            "cs_mpvpe_ego": [0.0, 0.0],
            "cs_mpvpe_aux": [0.0, 0.0],
            "rs_mpvpe_all": [0.0, 0.0],
            "rs_mpvpe_ego": [0.0, 0.0],
            "rs_mpvpe_aux": [0.0, 0.0],
            "rte_all": [0.0, 0.0],
            "rte_ego": [0.0, 0.0],
            "rte_aux": [0.0, 0.0],
        }

    @torch.no_grad()
    def update(
        self,
        joint_cam_gt,
        joint_rel_gt,
        verts_cam_gt,
        verts_rel_gt,
        joint_cam_pred,
        joint_rel_pred,
        verts_cam_pred,
        verts_rel_pred,
        has_mano,
        joint_3d_valid,
        norm_valid,
        ego_mask,
        aux_mask,
    ):
        """Accumulate one validation batch into the running metric totals."""
        joint_mask_all = joint_3d_valid * norm_valid[:, :, None]
        joint_mask_ego = joint_mask_all * ego_mask[:, :, None]
        joint_mask_aux = joint_mask_all * aux_mask[:, :, None]

        vert_mask_all = has_mano * norm_valid
        vert_mask_ego = vert_mask_all * ego_mask
        vert_mask_aux = vert_mask_all * aux_mask

        rte_mask_all = joint_3d_valid[:, :, 0] * norm_valid
        rte_mask_ego = rte_mask_all * ego_mask
        rte_mask_aux = rte_mask_all * aux_mask

        self._accumulate(
            "cs_mpjpe_all",
            compute_mpjpe_stats(
                joint_cam_pred,
                joint_cam_gt,
                joint_mask_all,
            ),
        )
        self._accumulate(
            "cs_mpjpe_ego",
            compute_mpjpe_stats(
                joint_cam_pred,
                joint_cam_gt,
                joint_mask_ego,
            ),
        )
        self._accumulate(
            "cs_mpjpe_aux",
            compute_mpjpe_stats(
                joint_cam_pred,
                joint_cam_gt,
                joint_mask_aux,
            ),
        )
        self._accumulate(
            "rs_mpjpe_all",
            compute_mpjpe_stats(
                joint_rel_pred,
                joint_rel_gt,
                joint_mask_all,
            ),
        )
        self._accumulate(
            "rs_mpjpe_ego",
            compute_mpjpe_stats(
                joint_rel_pred,
                joint_rel_gt,
                joint_mask_ego,
            ),
        )
        self._accumulate(
            "rs_mpjpe_aux",
            compute_mpjpe_stats(
                joint_rel_pred,
                joint_rel_gt,
                joint_mask_aux,
            ),
        )
        self._accumulate(
            "cs_mpvpe_all",
            compute_mpvpe_stats(
                verts_cam_pred,
                verts_cam_gt,
                vert_mask_all,
            ),
        )
        self._accumulate(
            "cs_mpvpe_ego",
            compute_mpvpe_stats(
                verts_cam_pred,
                verts_cam_gt,
                vert_mask_ego,
            ),
        )
        self._accumulate(
            "cs_mpvpe_aux",
            compute_mpvpe_stats(
                verts_cam_pred,
                verts_cam_gt,
                vert_mask_aux,
            ),
        )
        self._accumulate(
            "rs_mpvpe_all",
            compute_mpvpe_stats(
                verts_rel_pred,
                verts_rel_gt,
                vert_mask_all,
            ),
        )
        self._accumulate(
            "rs_mpvpe_ego",
            compute_mpvpe_stats(
                verts_rel_pred,
                verts_rel_gt,
                vert_mask_ego,
            ),
        )
        self._accumulate(
            "rs_mpvpe_aux",
            compute_mpvpe_stats(
                verts_rel_pred,
                verts_rel_gt,
                vert_mask_aux,
            ),
        )
        self._accumulate(
            "rte_all",
            compute_rte_stats(
                joint_cam_pred[:, :, 0],
                joint_cam_gt[:, :, 0],
                rte_mask_all,
            ),
        )
        self._accumulate(
            "rte_ego",
            compute_rte_stats(
                joint_cam_pred[:, :, 0],
                joint_cam_gt[:, :, 0],
                rte_mask_ego,
            ),
        )
        self._accumulate(
            "rte_aux",
            compute_rte_stats(
                joint_cam_pred[:, :, 0],
                joint_cam_gt[:, :, 0],
                rte_mask_aux,
            ),
        )

    def _accumulate(self, key, stats_tuple):
        """Add a `(total_error, total_count)` pair into one named accumulator."""
        self.accumulators[key][0] += stats_tuple[0].item()
        self.accumulators[key][1] += stats_tuple[1].item()

    def compute(self):
        """Convert accumulated sums/counts into final scalar validation metrics."""
        results = {}
        key_map = {
            "cs_mpjpe_all": "micro_mpjpe_all",
            "cs_mpjpe_ego": "micro_mpjpe_ego",
            "cs_mpjpe_aux": "micro_mpjpe_aux",
            "rs_mpjpe_all": "micro_mpjpe_rel_all",
            "rs_mpjpe_ego": "micro_mpjpe_rel_ego",
            "rs_mpjpe_aux": "micro_mpjpe_rel_aux",
            "cs_mpvpe_all": "micro_mpvpe_all",
            "cs_mpvpe_ego": "micro_mpvpe_ego",
            "cs_mpvpe_aux": "micro_mpvpe_aux",
            "rs_mpvpe_all": "micro_mpvpe_rel_all",
            "rs_mpvpe_ego": "micro_mpvpe_rel_ego",
            "rs_mpvpe_aux": "micro_mpvpe_rel_aux",
            "rte_all": "micro_rte_all",
            "rte_ego": "micro_rte_ego",
            "rte_aux": "micro_rte_aux",
        }

        for internal_key, output_key in key_map.items():
            total_error, total_count = self.accumulators[internal_key]
            if total_count > 0:
                results[output_key] = total_error / total_count
            else:
                results[output_key] = 0.0

        return results
