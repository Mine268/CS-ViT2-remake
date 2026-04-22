"""Unit tests for the config-driven training data plan builder."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.data.config import build_data_source_alias_map
from src.data.config import build_train_data_plan
from src.data.config import collect_supervision_dataset_groups


def _make_data_cfg(tmp_path: Path):
    """Construct a tiny on-disk dataset registry for sampling-plan tests."""
    ego_tar = tmp_path / "ego_000000.tar"
    aux_tar = tmp_path / "aux_000000.tar"
    ego_tar.touch()
    aux_tar.touch()
    return OmegaConf.create(
        {
            "datasets": {
                "EgoSet": {
                    "aliases": ["egoset", "ego_set"],
                    "splits": {"train": [str(tmp_path / "ego_*.tar")]},
                },
                "AuxSet": {
                    "aliases": ["auxset"],
                    "splits": {"train": [str(tmp_path / "aux_*.tar")]},
                },
            },
            "train": {
                "split": "train",
                "groups": {
                    "ego": {
                        "weight": 0.3,
                        "datasets": {"EgoSet": 1.0},
                    },
                    "aux": {
                        "weight": 0.7,
                        "datasets": {"AuxSet": 1.0},
                    },
                },
            },
        }
    )


def test_build_data_source_alias_map_comes_from_config(tmp_path: Path):
    """Aliases should be derived entirely from config rather than hardcoded Python tables."""
    data_cfg = _make_data_cfg(tmp_path)
    alias_map = build_data_source_alias_map(data_cfg.datasets)
    assert alias_map["egoset"] == "EgoSet"
    assert alias_map["egoset"] == alias_map["ego_set".replace("_", "")]
    assert alias_map["auxset"] == "AuxSet"


def test_build_train_data_plan_resolves_group_weights_and_sources(tmp_path: Path):
    """The plan builder should normalize weights and expand split paths into concrete shards."""
    data_cfg = _make_data_cfg(tmp_path)
    plan = build_train_data_plan(data_cfg)
    assert plan.group_weights["ego"] == 0.3
    assert plan.group_weights["aux"] == 0.7
    assert plan.group_dataset_sources["ego"]["EgoSet"] == [str(tmp_path / "ego_000000.tar")]
    assert plan.group_dataset_sources["aux"]["AuxSet"] == [str(tmp_path / "aux_000000.tar")]
    assert plan.ego_datasets == ["EgoSet"]
    assert plan.aux_datasets == ["AuxSet"]


def test_collect_supervision_dataset_groups_rejects_duplicate_dataset_across_groups(tmp_path: Path):
    """A dataset may belong to exactly one supervision group."""
    data_cfg = _make_data_cfg(tmp_path)
    data_cfg.train.groups.aux.datasets.EgoSet = 1.0
    with pytest.raises(ValueError, match="assigned to both groups"):
        collect_supervision_dataset_groups(data_cfg)
