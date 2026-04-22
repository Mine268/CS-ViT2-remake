from __future__ import annotations

"""Sampling-policy helpers that sit between Hydra config and WebDataset iteration."""

from collections import OrderedDict
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from omegaconf import DictConfig

from ..utils.misc import as_list, expand_glob_patterns


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


def compute_dataset_reweight_probs(
    dataset_sources: Mapping[str, Sequence[str]],
    dataset_weights: Mapping[str, float],
) -> "OrderedDict[str, float]":
    """Legacy helper that normalizes one-level dataset weights into probabilities."""
    if len(dataset_sources) == 0:
        raise ValueError("dataset_sources must not be empty")
    if len(dataset_weights) == 0:
        raise ValueError("dataset_weights must not be empty")

    missing_weights = [name for name in dataset_sources.keys() if name not in dataset_weights]
    if missing_weights:
        raise ValueError(f"Missing reweight weights for datasets: {missing_weights}")

    missing_sources = [name for name in dataset_weights.keys() if name not in dataset_sources]
    if missing_sources:
        raise ValueError(f"Missing dataset sources for weights: {missing_sources}")

    empty_datasets = [name for name, urls in dataset_sources.items() if len(urls) == 0]
    if empty_datasets:
        raise ValueError(f"Empty dataset sources: {empty_datasets}")

    total_weight = float(sum(float(weight) for weight in dataset_weights.values()))
    if total_weight <= 0.0:
        raise ValueError(f"dataset_weights sum must be positive, got {dict(dataset_weights)}")

    normalized: "OrderedDict[str, float]" = OrderedDict()
    for dataset_name in dataset_weights.keys():
        weight = float(dataset_weights[dataset_name])
        if weight <= 0.0:
            raise ValueError(f"dataset_weights[{dataset_name}] must be positive, got {weight}")
        normalized[dataset_name] = weight / total_weight
    return normalized


def collect_reweight_dataset_config(
    reweight_cfg: DictConfig,
) -> Tuple["OrderedDict[str, List[str]]", "OrderedDict[str, float]"]:
    """Legacy parser for the pre-registry `DATA.train.reweight.datasets` layout."""
    dataset_entries = reweight_cfg.get("datasets", [])
    if len(dataset_entries) == 0:
        raise ValueError("DATA.train.reweight.datasets must be non-empty")

    dataset_sources: "OrderedDict[str, List[str]]" = OrderedDict()
    dataset_weights: "OrderedDict[str, float]" = OrderedDict()
    for entry in dataset_entries:
        dataset_name = str(entry.get("name", "")).strip()
        if dataset_name == "":
            raise ValueError("Each reweight dataset entry must provide a non-empty name")
        if dataset_name in dataset_sources:
            raise ValueError(f"Duplicate reweight dataset entry: {dataset_name}")

        matched_files = expand_glob_patterns([str(x) for x in as_list(entry.get("source", []))])
        if len(matched_files) == 0:
            raise ValueError(f"reweight dataset {dataset_name} matched no files")

        dataset_sources[dataset_name] = matched_files
        dataset_weights[dataset_name] = float(entry.get("weight", 0.0))

    return dataset_sources, dataset_weights
