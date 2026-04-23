from __future__ import annotations

"""Configuration parsing utilities for dataset registry and training sampling plans."""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Mapping, MutableMapping, Sequence

from omegaconf import DictConfig

from ..utils.data_source import normalize_data_source_alias_key
from ..utils.misc import as_list, expand_glob_patterns


TRAIN_GROUP_NAMES = ("ego", "aux")


@dataclass(frozen=True)
class TrainDataPlan:
    """
    Fully validated training data plan derived from config.

    `group_weights` controls the expected sample ratio between ego and aux streams.
    `group_dataset_weights[group][dataset]` controls the expected ratio inside each group.
    The resulting training stream is implemented as a two-level RandomMix:
    1. sample a group according to `group_weights`
    2. sample a dataset inside that group according to `group_dataset_weights`
    """

    dataset_alias_map: Dict[str, str]
    group_weights: "OrderedDict[str, float]"
    group_dataset_weights: Dict[str, "OrderedDict[str, float]"]
    group_dataset_sources: Dict[str, "OrderedDict[str, List[str]]"]
    ego_datasets: List[str]
    aux_datasets: List[str]


def build_data_source_alias_map(datasets_cfg: Mapping[str, Mapping]) -> Dict[str, str]:
    """
    Build the alias lookup table declared in `DATA.datasets`.

    Each dataset implicitly aliases itself by its canonical config key, then extends that with any
    extra entries from `aliases`. The map is normalized through `normalize_data_source_alias_key`
    so naming variants like `InterHand2.6M`, `interhand26m`, and `interhand_26m` can be routed to
    the same canonical dataset name.
    """
    alias_map: Dict[str, str] = {}
    for dataset_name, dataset_cfg in datasets_cfg.items():
        aliases = [str(dataset_name), *[str(alias) for alias in as_list(dataset_cfg.get("aliases", []))]]
        for alias in aliases:
            alias_key = normalize_data_source_alias_key(alias)
            previous = alias_map.get(alias_key)
            if previous is not None and previous != dataset_name:
                raise ValueError(
                    f"Alias '{alias}' is ambiguous between datasets '{previous}' and '{dataset_name}'"
                )
            alias_map[alias_key] = str(dataset_name)
    return alias_map


def _normalize_named_weights(
    weights: MutableMapping[str, float],
    context: str,
) -> "OrderedDict[str, float]":
    """Validate a name->weight mapping and normalize it into probabilities."""
    if len(weights) == 0:
        raise ValueError(f"{context} must not be empty")

    total_weight = 0.0
    normalized: "OrderedDict[str, float]" = OrderedDict()
    for name, weight in weights.items():
        numeric = float(weight)
        if numeric <= 0.0:
            raise ValueError(f"{context}[{name}] must be positive, got {numeric}")
        normalized[str(name)] = numeric
        total_weight += numeric

    if total_weight <= 0.0:
        raise ValueError(f"{context} sum must be positive")

    for name in normalized.keys():
        normalized[name] = normalized[name] / total_weight
    return normalized


def _get_training_groups_cfg(data_cfg: DictConfig):
    """Return the declared training groups and verify that the required groups exist."""
    groups_cfg = data_cfg.train.get("groups")
    if groups_cfg is None:
        raise ValueError("DATA.train.groups must be configured")

    unknown_groups = [name for name in groups_cfg.keys() if str(name) not in TRAIN_GROUP_NAMES]
    if unknown_groups:
        raise ValueError(
            f"Unsupported DATA.train.groups entries: {unknown_groups}. Expected only {list(TRAIN_GROUP_NAMES)}"
        )

    missing_groups = [group_name for group_name in TRAIN_GROUP_NAMES if group_name not in groups_cfg]
    if missing_groups:
        raise ValueError(f"DATA.train.groups is missing required groups: {missing_groups}")
    return groups_cfg


def collect_supervision_dataset_groups(data_cfg: DictConfig) -> Dict[str, List[str]]:
    """
    Return canonical dataset names used by each training supervision group.

    This only validates config structure and dataset membership. It does not touch the filesystem,
    so model construction can depend on it without expanding dataset globs.
    """

    datasets_cfg = data_cfg.get("datasets")
    if datasets_cfg is None or len(datasets_cfg) == 0:
        raise ValueError("DATA.datasets must define at least one dataset")

    groups_cfg = _get_training_groups_cfg(data_cfg)
    membership: Dict[str, List[str]] = {}
    seen_dataset_to_group: Dict[str, str] = {}
    for group_name in TRAIN_GROUP_NAMES:
        group_cfg = groups_cfg[group_name]
        dataset_weights_cfg = group_cfg.get("datasets")
        if dataset_weights_cfg is None or len(dataset_weights_cfg) == 0:
            raise ValueError(f"DATA.train.groups.{group_name}.datasets must be non-empty")

        dataset_names: List[str] = []
        for dataset_name in dataset_weights_cfg.keys():
            dataset_name = str(dataset_name)
            if dataset_name not in datasets_cfg:
                raise ValueError(
                    f"DATA.train.groups.{group_name}.datasets references unknown dataset '{dataset_name}'"
                )
            previous_group = seen_dataset_to_group.get(dataset_name)
            if previous_group is not None:
                raise ValueError(
                    f"Dataset '{dataset_name}' is assigned to both groups '{previous_group}' and '{group_name}'"
                )
            seen_dataset_to_group[dataset_name] = group_name
            dataset_names.append(dataset_name)
        membership[group_name] = dataset_names
    return membership


def _resolve_dataset_split_sources(
    datasets_cfg: Mapping[str, Mapping],
    dataset_name: str,
    split_name: str,
) -> List[str]:
    """Expand one dataset/split entry from the registry into concrete shard paths."""
    dataset_cfg = datasets_cfg[dataset_name]
    split_cfg = dataset_cfg.get("splits", {})
    patterns = [str(x) for x in as_list(split_cfg.get(split_name, []))]
    if len(patterns) == 0:
        raise ValueError(
            f"DATA.datasets.{dataset_name}.splits.{split_name} must provide at least one source pattern"
        )

    matched_files = expand_glob_patterns(patterns)
    if len(matched_files) == 0:
        raise ValueError(
            f"DATA.datasets.{dataset_name}.splits.{split_name} matched no files for patterns {patterns}"
        )
    return matched_files


def _resolve_dataset_split_sources_if_available(
    datasets_cfg: Mapping[str, Mapping],
    dataset_name: str,
    split_name: str,
) -> List[str] | None:
    """
    Resolve one dataset/split entry if that split exists for the dataset.

    Returning `None` means the dataset is intentionally unavailable for the requested split. This is
    used by the clip-native training setup where some datasets only provide `train_stage1` but not
    `train_stage2`.
    """
    split_cfg = datasets_cfg[dataset_name].get("splits", {})
    if split_name not in split_cfg:
        return None
    return _resolve_dataset_split_sources(datasets_cfg, dataset_name, split_name)


def build_train_data_plan(data_cfg: DictConfig) -> TrainDataPlan:
    """
    Resolve the complete training sampling plan from `DATA`.

    This performs all expensive and failure-prone validation in one place:
    - dataset registry must exist
    - ego/aux group membership must be explicit and non-overlapping
    - every dataset weight and group weight must be positive
    - every dataset split must expand to at least one file on disk
    """
    datasets_cfg = data_cfg.get("datasets")
    if datasets_cfg is None or len(datasets_cfg) == 0:
        raise ValueError("DATA.datasets must define at least one dataset")

    membership = collect_supervision_dataset_groups(data_cfg)
    groups_cfg = _get_training_groups_cfg(data_cfg)
    split_name = str(data_cfg.train.get("split", "train"))

    group_weight_inputs: "OrderedDict[str, float]" = OrderedDict()
    group_dataset_weights: Dict[str, "OrderedDict[str, float]"] = {}
    group_dataset_sources: Dict[str, "OrderedDict[str, List[str]]"] = {}

    for group_name in TRAIN_GROUP_NAMES:
        group_cfg = groups_cfg[group_name]
        group_weight_inputs[group_name] = float(group_cfg.get("weight", 0.0))

        dataset_weight_inputs: "OrderedDict[str, float]" = OrderedDict()
        dataset_sources: "OrderedDict[str, List[str]]" = OrderedDict()
        for dataset_name, weight in group_cfg.get("datasets").items():
            dataset_name = str(dataset_name)
            resolved_sources = _resolve_dataset_split_sources_if_available(
                datasets_cfg=datasets_cfg,
                dataset_name=dataset_name,
                split_name=split_name,
            )
            if resolved_sources is None:
                continue
            dataset_weight_inputs[dataset_name] = float(weight)
            dataset_sources[dataset_name] = resolved_sources

        group_dataset_weights[group_name] = _normalize_named_weights(
            dataset_weight_inputs,
            context=f"DATA.train.groups.{group_name}.datasets",
        )
        group_dataset_sources[group_name] = dataset_sources

    return TrainDataPlan(
        dataset_alias_map=build_data_source_alias_map(datasets_cfg),
        group_weights=_normalize_named_weights(group_weight_inputs, context="DATA.train.groups"),
        group_dataset_weights=group_dataset_weights,
        group_dataset_sources=group_dataset_sources,
        ego_datasets=list(membership["ego"]),
        aux_datasets=list(membership["aux"]),
    )
