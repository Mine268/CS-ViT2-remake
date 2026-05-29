from __future__ import annotations

import kornia.geometry.conversions as KC
import torch
from src.model.loss import RemakeLoss
from src.model.ti import TITokenTransform, invert_ti_camera, invert_ti_pose_root


def test_ti_transform_is_identity_at_init():
    module = TITokenTransform(dim=32, depth=2, heads=4, dropout=0.0)
    tokens = torch.randn(2, 8, 32)
    scale = torch.tensor([0.95, 1.05], dtype=tokens.dtype)
    angle_rad = torch.tensor([0.25, -0.4], dtype=tokens.dtype)

    out = module(tokens, scale=scale, angle_rad=angle_rad)

    assert torch.allclose(out, tokens, atol=1e-6)


def test_invert_ti_pose_root_only_rotates_global_orient():
    pose = torch.zeros(2, 48)
    pose[0, 2] = 0.8
    pose[1, 2] = -0.3
    pose[:, 3:] = torch.randn(2, 45)
    angle_rad = torch.tensor([0.2, -0.5], dtype=pose.dtype)

    pose_inv = invert_ti_pose_root(pose, angle_rad=angle_rad, joint_rep_type="3")

    expected_root = torch.tensor([0.6, 0.2], dtype=pose.dtype)
    assert torch.allclose(pose_inv[:, 2], expected_root, atol=1e-5)
    assert torch.allclose(pose_inv[:, :2], torch.zeros_like(pose_inv[:, :2]), atol=1e-6)
    assert torch.allclose(pose_inv[:, 3:], pose[:, 3:], atol=1e-6)


def test_invert_ti_pose_root_matches_left_rotation_for_generic_axis_angle():
    pose = torch.zeros(2, 48)
    pose[:, :3] = torch.tensor([[0.2, -0.1, 0.4], [-0.3, 0.5, 0.1]])
    angle_rad = torch.tensor([0.35, -0.2], dtype=pose.dtype)

    pose_inv = invert_ti_pose_root(pose, angle_rad=angle_rad, joint_rep_type="3")
    actual = KC.axis_angle_to_rotation_matrix(pose_inv[:, :3])

    cos = torch.cos(-angle_rad)
    sin = torch.sin(-angle_rad)
    zeros = torch.zeros_like(cos)
    ones = torch.ones_like(cos)
    root_rot = torch.stack(
        [
            torch.stack([cos, -sin, zeros], dim=-1),
            torch.stack([sin, cos, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=-2,
    )
    expected = torch.matmul(root_rot, KC.axis_angle_to_rotation_matrix(pose[:, :3]))

    assert torch.allclose(actual, expected, atol=1e-5)


def test_invert_ti_pose_root_axis_angle_backward_is_finite():
    pose = torch.randn(16, 48) * 0.2
    pose.requires_grad_(True)
    angle_rad = torch.linspace(-0.5, 0.5, 16)

    pose_inv = invert_ti_pose_root(pose, angle_rad=angle_rad, joint_rep_type="3")
    loss = pose_inv[:, :3].abs().mean()
    loss.backward()

    assert torch.isfinite(pose.grad).all()


def test_invert_ti_camera_uses_ray_and_rho():
    pred_ray_unit = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    pred_rho = torch.tensor([[100.0], [200.0]])
    scale = torch.tensor([2.0, 0.5])
    angle_rad = torch.tensor([torch.pi / 2, -torch.pi / 2])

    trans_inv, ray_inv, rho_inv = invert_ti_camera(
        pred_ray_unit=pred_ray_unit,
        pred_rho=pred_rho,
        scale=scale,
        angle_rad=angle_rad,
    )

    expected_ray = torch.tensor(
        [
            [0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )
    expected_rho = torch.tensor([[50.0], [400.0]])
    assert torch.allclose(ray_inv, expected_ray, atol=1e-5)
    assert torch.allclose(rho_inv, expected_rho, atol=1e-6)
    assert torch.allclose(trans_inv, expected_ray * expected_rho, atol=1e-4)


def test_remake_loss_enabled_terms_skip_camera_head_requirements():
    loss_fn = RemakeLoss(
        lambda_theta=1.0,
        lambda_shape=1.0,
        lambda_trans=1.0,
        lambda_rel=1.0,
        lambda_img=1.0,
        lambda_uv_patch=1.0,
        lambda_root_z_cls=1.0,
        lambda_root_z_res=1.0,
        hm_centers=(torch.linspace(0.0, 1.0, 2), torch.linspace(0.0, 1.0, 2), None),
        hm_sigma=1.0,
        pred_joint_z_min_mm=1.0,
        reproj_loss_type="l1",
        reproj_loss_delta=1.0,
        ego_datasets=["AssemblyHands"],
        aux_datasets=[],
    )
    pose_pred = torch.zeros(1, 1, 48)
    shape_pred = torch.zeros(1, 1, 10)
    trans_pred = torch.zeros(1, 1, 3)
    with torch.no_grad():
        joint_rel_gt, _ = loss_fn.rmano_layer(pose_pred, shape_pred)
    trans_gt = torch.tensor([[[10.0, 20.0, 500.0]]])
    joint_cam_gt = joint_rel_gt + trans_gt[:, :, None, :]
    batch = {
        "mano_pose": pose_pred.clone(),
        "mano_shape": shape_pred.clone(),
        "has_mano": torch.ones(1, 1),
        "joint_2d_valid": torch.ones(1, 1, joint_rel_gt.shape[2]),
        "joint_3d_valid": torch.ones(1, 1, joint_rel_gt.shape[2]),
        "has_intr": torch.ones(1, 1),
        "joint_cam": joint_cam_gt,
        "joint_rel": joint_rel_gt,
        "joint_patch_resized": torch.zeros(1, 1, joint_rel_gt.shape[2], 2),
        "hand_bbox": torch.tensor([[[0.0, 0.0, 128.0, 128.0]]]),
        "focal": torch.tensor([[[1000.0, 1000.0]]]),
        "princpt": torch.tensor([[[256.0, 256.0]]]),
        "data_source": ["AssemblyHands"],
    }

    loss, state, result = loss_fn(
        pose_pred,
        shape_pred,
        trans_pred,
        cam_aux={},
        batch=batch,
        enabled_terms={"theta", "shape", "trans", "joint_rel"},
    )

    expected_trans_loss = torch.tensor((10.0 + 20.0 + 500.0) / 3.0)
    assert torch.allclose(loss, expected_trans_loss, atol=1e-5)
    assert torch.allclose(state["loss_uv_patch"], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(state["loss_rho_cls"], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(state["loss_rho_res"], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(state["loss_joint_img"], torch.tensor(0.0), atol=1e-6)
    assert result["joint_cam_pred"].shape == joint_cam_gt.shape
