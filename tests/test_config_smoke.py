"""Smoke tests for Hydra config parsing and model/runtime wiring assumptions."""

from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from src.data.config import build_train_data_plan, collect_supervision_dataset_groups
from src.model.backbone import strip_register_tokens
from src.train.engine import create_accelerator, get_linear_warmup_constant_schedule, setup_model


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
    assert cfg.DATA.datasets.AssemblyHands.splits.val_stage1[0].endswith(
        "/AssemblyHands/val_stage1/*.tar"
    )
    assert cfg.DATA.datasets.AssemblyHands.splits.val_stage2[0].endswith(
        "/AssemblyHands/val_stage2/*.tar"
    )
    assert "cosine_cycle" not in cfg.GENERAL
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


def test_dinog_configs_parse_with_expected_overrides():
    """DINO-G variants should compose cleanly and override the backbone-dependent dimensions."""
    config_dir = str((Path(__file__).resolve().parents[1] / "config").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        stage1 = compose(config_name="stage1_dinog")
        stage2 = compose(config_name="stage2_dinog")

    assert stage1.MODEL.backbone.backbone_str == "model/facebook/dinov2-giant"
    assert stage1.MODEL.handec.context_dim == 1536
    assert stage1.TRAIN.sample_per_device == 16
    assert stage1.TRAIN.backbone_lr is None
    assert stage2.MODEL.stage == "stage2"
    assert stage2.MODEL.backbone.backbone_str == "model/facebook/dinov2-giant"
    assert stage2.MODEL.handec.context_dim == 1536
    assert stage2.TRAIN.sample_per_device == 3
    assert stage2.TRAIN.backbone_lr is None
    assert stage2.DATA.train.split == "train_stage2"


def test_dinov3_configs_parse_with_expected_overrides():
    """DINOv3 H+ variants should keep full fine-tuning enabled and use the 1280-dim context."""
    config_dir = str((Path(__file__).resolve().parents[1] / "config").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        stage1 = compose(config_name="stage1_dinov3")
        stage2 = compose(config_name="stage2_dinov3")

    assert stage1.MODEL.backbone.backbone_str == "model/facebook/dinov3-vith16plus"
    assert stage1.MODEL.handec.context_dim == 1280
    assert stage1.TRAIN.sample_per_device == 2
    assert stage1.TRAIN.backbone_lr == 1e-5
    assert stage2.MODEL.stage == "stage2"
    assert stage2.MODEL.backbone.backbone_str == "model/facebook/dinov3-vith16plus"
    assert stage2.MODEL.handec.context_dim == 1280
    assert stage2.TRAIN.sample_per_device == 1
    assert stage2.TRAIN.backbone_lr == 1e-5
    assert stage2.DATA.train.split == "train_stage2"


def test_dinov3_large_configs_parse_with_expected_overrides():
    """DINOv3-L variants should use the 1024-dim context and keep full fine-tuning enabled."""
    config_dir = str((Path(__file__).resolve().parents[1] / "config").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        stage1 = compose(config_name="stage1_dinov3_large")
        stage2 = compose(config_name="stage2_dinov3_large")

    assert stage1.MODEL.backbone.backbone_str == "model/facebook/dinov3-vitl16-pretrain-lvd1689m"
    assert stage1.MODEL.handec.context_dim == 1024
    assert stage1.TRAIN.sample_per_device == 4
    assert stage1.TRAIN.backbone_lr == 1e-5
    assert stage2.MODEL.stage == "stage2"
    assert stage2.MODEL.backbone.backbone_str == "model/facebook/dinov3-vitl16-pretrain-lvd1689m"
    assert stage2.MODEL.handec.context_dim == 1024
    assert stage2.TRAIN.sample_per_device == 1
    assert stage2.TRAIN.backbone_lr == 1e-5
    assert stage2.DATA.train.split == "train_stage2"


def test_strip_register_tokens_preserves_cls_and_patch_tokens():
    """DINOv3 register tokens must not be interpreted as patch tokens downstream."""
    hidden_state = torch.arange(2 * 201 * 4, dtype=torch.float32).reshape(2, 201, 4)

    stripped = strip_register_tokens(
        hidden_state,
        has_cls_token=True,
        num_register_tokens=4,
        expected_patch_token_count=196,
    )

    assert stripped.shape == (2, 197, 4)
    assert torch.equal(stripped[:, :1], hidden_state[:, :1])
    assert torch.equal(stripped[:, 1:], hidden_state[:, 5:])


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


def test_default_lr_schedule_is_linear_warmup_then_constant():
    """Default training LR should not cosine-anneal after warmup."""
    param = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([param], lr=1.0)
    scheduler = get_linear_warmup_constant_schedule(optimizer, num_warmup_steps=3)

    observed = []
    for _ in range(7):
        optimizer.step()
        scheduler.step()
        observed.append(scheduler.get_last_lr()[0])

    assert observed[0] < observed[1] <= observed[2]
    assert observed[2:] == [1.0, 1.0, 1.0, 1.0, 1.0]


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
