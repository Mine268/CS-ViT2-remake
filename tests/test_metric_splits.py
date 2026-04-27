"""Tests for split metric computation across ego and aux dataset groups."""

import torch

from src.utils.metric import MetricMeter


def test_metric_meter_splits_absolute_and_relative_metrics_by_group():
    meter = MetricMeter()

    joint_cam_gt = torch.zeros(3, 1, 1, 3)
    joint_rel_gt = torch.zeros(3, 1, 1, 3)
    verts_cam_gt = torch.zeros(3, 1, 2, 3)
    verts_rel_gt = torch.zeros(3, 1, 2, 3)

    joint_cam_pred = joint_cam_gt.clone()
    joint_rel_pred = joint_rel_gt.clone()
    verts_cam_pred = verts_cam_gt.clone()
    verts_rel_pred = verts_rel_gt.clone()

    # Sample 0 is ego with absolute error 1, relative error 10, vertex errors 2 and 20.
    joint_cam_pred[0, 0, 0, 0] = 1.0
    joint_rel_pred[0, 0, 0, 0] = 10.0
    verts_cam_pred[0, 0, :, 0] = 2.0
    verts_rel_pred[0, 0, :, 0] = 20.0

    # Sample 1 is aux with absolute error 3, relative error 30, vertex errors 4 and 40.
    joint_cam_pred[1, 0, 0, 0] = 3.0
    joint_rel_pred[1, 0, 0, 0] = 30.0
    verts_cam_pred[1, 0, :, 0] = 4.0
    verts_rel_pred[1, 0, :, 0] = 40.0

    # Sample 2 is aux but invalid and should be ignored by all masks.
    joint_cam_pred[2, 0, 0, 0] = 99.0
    joint_rel_pred[2, 0, 0, 0] = 99.0
    verts_cam_pred[2, 0, :, 0] = 99.0
    verts_rel_pred[2, 0, :, 0] = 99.0

    has_mano = torch.tensor([[1.0], [1.0], [0.0]])
    joint_3d_valid = torch.tensor([[[1.0]], [[1.0]], [[0.0]]])
    norm_valid = torch.ones_like(has_mano)
    ego_mask = torch.tensor([[1.0], [0.0], [0.0]])
    aux_mask = torch.tensor([[0.0], [1.0], [1.0]])

    metrics = meter(
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
    )

    assert torch.isclose(metrics["micro_mpjpe_all"], torch.tensor(2.0))
    assert torch.isclose(metrics["micro_mpjpe_ego"], torch.tensor(1.0))
    assert torch.isclose(metrics["micro_mpjpe_aux"], torch.tensor(3.0))

    assert torch.isclose(metrics["micro_mpjpe_rel_all"], torch.tensor(20.0))
    assert torch.isclose(metrics["micro_mpjpe_rel_ego"], torch.tensor(10.0))
    assert torch.isclose(metrics["micro_mpjpe_rel_aux"], torch.tensor(30.0))

    assert torch.isclose(metrics["micro_mpvpe_all"], torch.tensor(3.0))
    assert torch.isclose(metrics["micro_mpvpe_ego"], torch.tensor(2.0))
    assert torch.isclose(metrics["micro_mpvpe_aux"], torch.tensor(4.0))

    assert torch.isclose(metrics["micro_mpvpe_rel_all"], torch.tensor(30.0))
    assert torch.isclose(metrics["micro_mpvpe_rel_ego"], torch.tensor(20.0))
    assert torch.isclose(metrics["micro_mpvpe_rel_aux"], torch.tensor(40.0))

    assert torch.isclose(metrics["micro_rte_all"], torch.tensor(2.0))
    assert torch.isclose(metrics["micro_rte_ego"], torch.tensor(1.0))
    assert torch.isclose(metrics["micro_rte_aux"], torch.tensor(3.0))
