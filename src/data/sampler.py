from __future__ import annotations

"""Sampling-policy helpers that sit between Hydra config and WebDataset iteration."""

from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np


def build_clip_sample_filter_fn(
    filter_cfg: Optional[Mapping[str, Any]],
) -> Optional[Callable[[Dict[str, Any]], bool]]:
    """
    Compile frame-quality thresholds from config into a clip-level acceptance predicate.

    The returned function is applied before expensive image decoding and augmentation. It decides
    whether a candidate clip is trainable based on per-frame 2D validity and hand-bbox size.
    """
    if filter_cfg is None or not bool(filter_cfg.get("enabled", False)):
        return None

    min_valid_joints_2d = int(filter_cfg.get("min_valid_joints_2d", 0))
    min_hand_bbox_edge_px = float(filter_cfg.get("min_hand_bbox_edge_px", 0.0))
    frame_policy = str(filter_cfg.get("frame_policy", "all"))
    if frame_policy not in {"all", "last", "any"}:
        raise ValueError(f"Unsupported frame_policy: {frame_policy}")

    def _filter(clip_sample: Dict[str, Any]) -> bool:
        """Check whether a normalized clip sample satisfies the configured frame policy."""
        hand_bbox = clip_sample.get("hand_bbox")
        joint_2d_valid = clip_sample.get("joint_2d_valid", clip_sample.get("joint_valid"))
        if hand_bbox is None or joint_2d_valid is None:
            return True

        hand_bbox_np = np.asarray(hand_bbox, dtype=np.float32)
        joint_2d_valid_np = np.asarray(joint_2d_valid, dtype=np.float32)

        bbox_min_edge = np.minimum(
            hand_bbox_np[..., 2] - hand_bbox_np[..., 0],
            hand_bbox_np[..., 3] - hand_bbox_np[..., 1],
        )
        valid_joint_count = np.sum(joint_2d_valid_np > 0.5, axis=-1)

        frame_ok = np.ones_like(bbox_min_edge, dtype=bool)
        if min_valid_joints_2d > 0:
            frame_ok &= valid_joint_count >= min_valid_joints_2d
        if min_hand_bbox_edge_px > 0:
            frame_ok &= bbox_min_edge >= min_hand_bbox_edge_px

        if frame_policy == "all":
            return bool(np.all(frame_ok))
        if frame_policy == "last":
            return bool(frame_ok[-1])
        return bool(np.any(frame_ok))

    return _filter
