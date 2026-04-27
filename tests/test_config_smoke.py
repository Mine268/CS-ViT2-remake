"""Smoke tests for Hydra config parsing and model/runtime wiring assumptions."""

from pathlib import Path

from hydra import compose, initialize_config_dir

from src.data.config import build_train_data_plan
from src.data.config import collect_supervision_dataset_groups
from src.train.engine import create_accelerator
from src.train.engine import setup_model


def test_stage1_config_parses():
    """Stage1 config should parse and expose the expected registry/group structure."""
    config_dir = str((Path(__file__).resolve().parents[1] / "config").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="stage1")
    assert cfg.MODEL.handec.cam_head_type == "patch_uv_rho_multibin"
    assert cfg.MODEL.norm_by_hand is False
    assert cfg.DATA.train.split == "train_stage1"
    assert cfg.DATA.val.source == cfg.DATA.datasets.AssemblyHands.splits.val_stage1
    assert cfg.DATA.val.batch_size == 16
    assert "HOT3D" in cfg.DATA.datasets
    assert cfg.DATA.datasets.AssemblyHands.splits.val_stage1[0].endswith("/AssemblyHands/val_stage1/*.tar")
    assert cfg.DATA.datasets.AssemblyHands.splits.val_stage2[0].endswith("/AssemblyHands/val_stage2/*.tar")
    supervision_groups = collect_supervision_dataset_groups(cfg.DATA)
    assert supervision_groups["ego"] == ["HOT3D", "AssemblyHands"]
    assert "MTC" in supervision_groups["aux"]


def test_stage2_config_extends_stage1():
    """Stage2 should inherit stage1 defaults and only override the temporal settings."""
    config_dir = str((Path(__file__).resolve().parents[1] / "config").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="stage2")
    assert cfg.MODEL.stage == "stage2"
    assert cfg.MODEL.num_frame == 7
    assert cfg.DATA.train.split == "train_stage2"
    assert cfg.DATA.val.source == cfg.DATA.datasets.AssemblyHands.splits.val_stage2
    assert cfg.DATA.val.batch_size == 6
    plan = build_train_data_plan(cfg.DATA)
    assert "FreiHAND" not in plan.group_dataset_weights["aux"]
    assert "RHD" not in plan.group_dataset_weights["aux"]


def test_train_data_plan_matches_current_sampling_intent():
    """Current data.yaml should still represent the intended ego/aux sampling ratios."""
    config_dir = str((Path(__file__).resolve().parents[1] / "config").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="stage1")

    plan = build_train_data_plan(cfg.DATA)
    assert plan.group_weights["ego"] == 0.2
    assert plan.group_weights["aux"] == 0.8
    assert plan.group_dataset_weights["ego"]["HOT3D"] == 0.5
    assert plan.group_dataset_weights["ego"]["AssemblyHands"] == 0.5
    assert plan.group_dataset_weights["aux"]["InterHand2.6M"] == 0.25


def test_accelerator_disables_dispatch_batches_for_iterable_wds():
    """Iterable WebDataset batches must not be concatenated across ranks by Accelerate."""
    config_dir = str((Path(__file__).resolve().parents[1] / "config").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="stage1")

    accelerator = create_accelerator(cfg)
    assert accelerator.dataloader_config.dispatch_batches is False


def test_stage1_freezes_temporal_refiner_and_unused_mask_token():
    """Stage1 should not keep parameters trainable that never participate in its forward path."""
    config_dir = str((Path(__file__).resolve().parents[1] / "config").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="stage1")

    net = setup_model(cfg)
    temporal_params = list(net.temporal_refiner.parameters())
    assert len(temporal_params) > 0
    assert all(not param.requires_grad for param in temporal_params)

    mask_token = getattr(getattr(net.backbone.backbone, "embeddings", None), "mask_token", None)
    if mask_token is not None:
        assert mask_token.requires_grad is False
