"""Small regression test for dataset-group mask construction."""

import torch

from src.model.loss import build_dataset_group_mask


def test_build_dataset_group_mask_marks_ego_samples_only():
    """Only configured ego samples should be marked as valid in the returned mask."""
    mask = build_dataset_group_mask(
        data_sources=["HOT3D", "InterHand2.6M", "AssemblyHands"],
        target_sources=["HOT3D", "AssemblyHands"],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert torch.equal(mask.squeeze(-1), torch.tensor([1.0, 0.0, 1.0]))
