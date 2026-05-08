"""Regression tests for training-time bbox jitter preprocessing."""

from __future__ import annotations

from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from src.constant import HAND_JOINT_COUNT, MANO_JOINT_COUNT, MANO_SHAPE_DIM
from src.data.preprocess import preprocess_batch


def _make_batch(num_frames: int = 3):
    batch_size = 1
    image_size = 128
    joint_img = torch.zeros(batch_size, num_frames, HAND_JOINT_COUNT, 2, dtype=torch.float32)
    joint_img[..., 0] = torch.linspace(42.0, 82.0, HAND_JOINT_COUNT).view(1, 1, HAND_JOINT_COUNT)
    joint_img[..., 1] = torch.linspace(48.0, 88.0, HAND_JOINT_COUNT).view(1, 1, HAND_JOINT_COUNT)
    hand_bbox = torch.tensor([40.0, 46.0, 84.0, 90.0], dtype=torch.float32).view(1, 1, 4)
    hand_bbox = hand_bbox.repeat(batch_size, num_frames, 1)

    return {
        "__key__": ["sample"],
        "imgs_path": [["frame.png" for _ in range(num_frames)]],
        "handedness": ["right"],
        "imgs": [torch.zeros(num_frames, 3, image_size, image_size, dtype=torch.uint8)],
        "hand_bbox": hand_bbox,
        "joint_img": joint_img,
        "joint_cam": torch.zeros(batch_size, num_frames, HAND_JOINT_COUNT, 3),
        "joint_rel": torch.zeros(batch_size, num_frames, HAND_JOINT_COUNT, 3),
        "joint_valid": torch.ones(batch_size, num_frames, HAND_JOINT_COUNT),
        "mano_pose": torch.zeros(batch_size, num_frames, MANO_JOINT_COUNT * 3),
        "mano_shape": torch.zeros(batch_size, num_frames, MANO_SHAPE_DIM),
        "mano_valid": torch.zeros(batch_size, num_frames),
        "timestamp": torch.arange(num_frames, dtype=torch.float32).view(1, num_frames),
        "focal": torch.ones(batch_size, num_frames, 2) * 1000.0,
        "princpt": torch.ones(batch_size, num_frames, 2) * 64.0,
        "has_intr": torch.ones(batch_size, num_frames),
    }


def _preprocess(batch, augmentation_flag: bool, bbox_jitter=None):
    return preprocess_batch(
        batch_origin=batch,
        patch_size=(64, 64),
        patch_expanstion=2.0,
        scale_z_range=(1.0, 1.0),
        scale_f_range=(1.0, 1.0),
        persp_rot_max=0.0,
        joint_rep_type="3",
        augmentation_flag=augmentation_flag,
        device=torch.device("cpu"),
        pixel_aug=None,
        perspective_normalization=False,
        bbox_jitter=bbox_jitter,
    )[0]


def test_bbox_jitter_is_disabled_on_eval_path():
    batch = _make_batch(num_frames=2)
    bbox_jitter = {
        "enabled": True,
        "prob": 1.0,
        "temporal_mode": "frame",
        "center_shift": 0.0,
        "scale_range": [1.5, 1.5],
        "aspect_ratio_range": [1.0, 1.0],
        "min_edge_px": 8.0,
    }

    out = _preprocess(batch, augmentation_flag=False, bbox_jitter=bbox_jitter)

    assert torch.allclose(out["hand_bbox"], batch["hand_bbox"])


def test_bbox_jitter_scales_training_bbox_and_patch_geometry():
    batch = _make_batch(num_frames=2)
    no_jitter = {
        "enabled": False,
    }
    bbox_jitter = {
        "enabled": True,
        "prob": 1.0,
        "temporal_mode": "frame",
        "center_shift": 0.0,
        "scale_range": [1.5, 1.5],
        "aspect_ratio_range": [1.0, 1.0],
        "min_edge_px": 8.0,
    }

    torch.manual_seed(123)
    baseline = _preprocess(batch, augmentation_flag=True, bbox_jitter=no_jitter)
    torch.manual_seed(123)
    out = _preprocess(batch, augmentation_flag=True, bbox_jitter=bbox_jitter)

    original_size = baseline["hand_bbox"][..., 2:] - baseline["hand_bbox"][..., :2]
    jittered_size = out["hand_bbox"][..., 2:] - out["hand_bbox"][..., :2]
    patch_size = out["patch_bbox"][..., 2:] - out["patch_bbox"][..., :2]
    expected_patch_edge = torch.max(jittered_size, dim=-1, keepdim=True).values * 2.0
    assert torch.allclose(jittered_size, original_size * 1.5)
    assert torch.allclose(patch_size, expected_patch_edge.expand_as(patch_size))


def test_stage_configs_enable_expected_bbox_jitter_modes():
    config_dir = str((Path(__file__).resolve().parents[1] / "config").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        stage1 = compose(config_name="stage1")
        stage2 = compose(config_name="stage2")

    assert stage1.TRAIN.bbox_jitter.enabled is True
    assert stage1.TRAIN.bbox_jitter.temporal_mode == "frame"
    assert stage1.TRAIN.bbox_jitter.scale_range == [0.75, 1.35]
    assert stage2.TRAIN.bbox_jitter.enabled is True
    assert stage2.TRAIN.bbox_jitter.temporal_mode == "clip"
    assert stage2.TRAIN.bbox_jitter.frame_scale_range == [0.95, 1.05]
