from __future__ import annotations

"""Distributed evaluation entrypoint that exports predictions into `.pt` and `.h5` files."""

import json
import os
import os.path as osp
from typing import Dict, List

from accelerate import Accelerator
import h5py
import hydra
from omegaconf import DictConfig
import torch

from src.data.preprocess import preprocess_batch
from src.train.engine import build_eval_dataloader, create_accelerator, setup_model


def _to_numpy(value):
    """Convert tensors to numpy for serialization while leaving scalars/strings untouched."""
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return value


def _save_local_results(path: str, results: Dict[str, List]):
    """Save per-rank prediction buffers before the main process merges them."""
    serializable = {}
    for key, values in results.items():
        serializable[key] = [_to_numpy(value) for value in values]
    torch.save(serializable, path)


def _merge_rank_results(output_dir: str, world_size: int):
    """Merge the per-rank `.pt` prediction shards written during distributed evaluation."""
    merged: Dict[str, List] = {}
    for rank in range(world_size):
        path = osp.join(output_dir, f"predictions_rank{rank:02d}.pt")
        local = torch.load(path, weights_only=False)
        for key, values in local.items():
            merged.setdefault(key, []).extend(values)
    return merged


def _write_hdf5(path: str, merged: Dict[str, List]):
    """Persist merged predictions into a compact HDF5 file for downstream analysis."""
    with h5py.File(path, "w") as f:
        for key, values in merged.items():
            if len(values) == 0:
                continue
            first = values[0]
            if isinstance(first, str):
                dt = h5py.string_dtype(encoding="utf-8")
                f.create_dataset(key, data=values, dtype=dt)
            else:
                stacked = torch.stack([torch.as_tensor(v) for v in values]).cpu().numpy()
                f.create_dataset(key, data=stacked, compression="gzip")


@hydra.main(version_base=None, config_path="../config", config_name="stage1")
def main(cfg: DictConfig):
    """Run evaluation on `DATA.test.source` and export predictions."""
    accelerator = create_accelerator(cfg)
    net = setup_model(cfg)
    test_loader = build_eval_dataloader(
        source_patterns=cfg.DATA.test.source,
        cfg_split=cfg.DATA.test,
        num_frames=cfg.MODEL.num_frame,
        batch_size=cfg.TEST.batch_size,
        num_workers=1,
        prefetch_factor=1,
        seed=42,
        accelerator=accelerator,
        infinite=False,
    )
    if test_loader is None:
        raise ValueError("No test data found. Please set DATA.test.source")

    net = accelerator.prepare(net)
    accelerator.unwrap_model(net).load_pretrained(cfg.TEST.checkpoint_path)
    net.eval()

    output_dir = cfg.TEST.output_dir
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        with open(osp.join(output_dir, "test_config.json"), "w", encoding="utf-8") as f:
            json.dump({"checkpoint_path": cfg.TEST.checkpoint_path}, f, indent=2)
    accelerator.wait_for_everyone()

    # Keep the export schema explicit so downstream consumers can rely on stable field names.
    local_results: Dict[str, List] = {
        "__key__": [],
        "joint_cam_pred": [],
        "vert_cam_pred": [],
        "mano_pose_pred": [],
        "mano_shape_pred": [],
        "trans_pred": [],
        "joint_cam_gt": [],
        "focal": [],
        "princpt": [],
    }

    total_samples = 0
    for batch_origin in test_loader:
        if cfg.TEST.max_samples is not None and total_samples >= int(cfg.TEST.max_samples):
            break
        batch, _, _ = preprocess_batch(
            batch_origin=batch_origin,
            patch_size=[cfg.MODEL.img_size, cfg.MODEL.img_size],
            patch_expanstion=cfg.TRAIN.expansion_ratio,
            scale_z_range=[1.0, 1.0],
            scale_f_range=[1.0, 1.0],
            persp_rot_max=0.0,
            joint_rep_type=cfg.MODEL.joint_type,
            augmentation_flag=False,
            device=accelerator.device,
            pixel_aug=None,
            perspective_normalization=cfg.TRAIN.get("perspective_normalization", False),
        )
        result = accelerator.unwrap_model(net).predict_full(
            img=batch["patches"],
            bbox=batch["patch_bbox"],
            focal=batch["focal"],
            princpt=batch["princpt"],
            timestamp=batch["timestamp"],
            hand_bbox=batch["hand_bbox"],
        )
        batch_size = batch["patches"].shape[0]
        total_samples += batch_size
        for bx in range(batch_size):
            local_results["__key__"].append(batch["__key__"][bx])
            local_results["joint_cam_pred"].append(result["joint_cam_pred"][bx, 0].detach().cpu())
            local_results["vert_cam_pred"].append(result["vert_cam_pred"][bx, 0].detach().cpu())
            local_results["mano_pose_pred"].append(result["mano_pose_pred"][bx, 0].detach().cpu())
            local_results["mano_shape_pred"].append(result["mano_shape_pred"][bx, 0].detach().cpu())
            local_results["trans_pred"].append(result["trans_pred_denorm"][bx, 0].detach().cpu())
            local_results["joint_cam_gt"].append(batch["joint_cam"][bx, -1].detach().cpu())
            local_results["focal"].append(batch["focal"][bx, -1].detach().cpu())
            local_results["princpt"].append(batch["princpt"][bx, -1].detach().cpu())

    local_path = osp.join(output_dir, f"predictions_rank{accelerator.process_index:02d}.pt")
    _save_local_results(local_path, local_results)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        merged = _merge_rank_results(output_dir, accelerator.num_processes)
        _write_hdf5(osp.join(output_dir, "predictions.h5"), merged)


if __name__ == "__main__":
    main()
