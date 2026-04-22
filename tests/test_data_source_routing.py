"""Tests for config-driven dataset-name routing and supervision masks."""

from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

from src.data.config import build_data_source_alias_map
from src.data.schema import normalize_decoded_clip_sample
from src.model.loss import build_dataset_group_mask


def _make_minimal_decoded_sample(data_source: str):
    """Create the smallest decoded sample that still exercises data_source normalization."""
    return {
        "__key__": "sample_0000",
        "imgs_path.json": ["frame_0000.webp"],
        "img_bytes.pickle": [b""],
        "data_source.json": data_source,
    }


def _load_stage1_data_cfg():
    """Load `cfg.DATA` from the stage1 Hydra config for config-driven tests."""
    config_dir = str((Path(__file__).resolve().parents[1] / "config").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="stage1")
    return cfg.DATA


def test_normalize_decoded_clip_sample_uses_config_driven_alias_map():
    """Observed shard aliases should resolve to the canonical dataset names declared in config."""
    data_cfg = _load_stage1_data_cfg()
    alias_map = build_data_source_alias_map(data_cfg.datasets)
    alias_to_canonical = {
        "assemblyhands": "AssemblyHands",
        "dexycb": "DexYCB",
        "freihand": "FreiHAND",
        "ho3d": "HO3D_v3",
        "hot3d": "HOT3D",
        "ih26m": "InterHand2.6M",
        "mtc": "MTC",
        "rhd": "RHD",
    }
    for alias, canonical in alias_to_canonical.items():
        normalized = normalize_decoded_clip_sample(
            _make_minimal_decoded_sample(alias),
            data_source_alias_map=alias_map,
        )
        assert normalized["data_source"] == canonical


def test_normalize_decoded_clip_sample_can_force_registry_dataset_name():
    """Training streams may stamp samples with the registry dataset name regardless of shard metadata."""
    data_cfg = _load_stage1_data_cfg()
    alias_map = build_data_source_alias_map(data_cfg.datasets)
    normalized = normalize_decoded_clip_sample(
        _make_minimal_decoded_sample("totally-wrong-source-name"),
        default_data_source="AssemblyHands",
        data_source_alias_map=alias_map,
        force_data_source=True,
    )
    assert normalized["data_source"] == "AssemblyHands"


def test_build_dataset_group_mask_matches_canonical_training_dataset_names():
    """Loss-side routing should operate on canonical dataset names emitted by training loaders."""
    mask = build_dataset_group_mask(
        data_sources=["HOT3D", "InterHand2.6M", "AssemblyHands", "unknown"],
        target_sources=["HOT3D", "AssemblyHands"],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert mask.tolist() == [[1.0], [0.0], [1.0], [0.0]]
