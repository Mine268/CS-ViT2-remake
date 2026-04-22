from __future__ import annotations

"""Render a handful of qualitative samples for quick visual inspection."""

import os
import os.path as osp

import cv2
import hydra
from omegaconf import DictConfig
import torch

from src.data.preprocess import preprocess_batch
from src.train.engine import build_eval_dataloader, create_accelerator, setup_model
from src.utils.vis import vis


@hydra.main(version_base=None, config_path="../config", config_name="stage1")
def main(cfg: DictConfig):
    """Export a small set of projected qualitative examples as PNG files."""
    accelerator = create_accelerator(cfg)
    if accelerator.num_processes != 1:
        raise ValueError("export_sample.py only supports single-process execution")

    net = setup_model(cfg)
    loader = build_eval_dataloader(
        source_patterns=cfg.DATA.test.source or cfg.DATA.val.source,
        cfg_split=cfg.DATA.test if cfg.DATA.test.source else cfg.DATA.val,
        num_frames=cfg.MODEL.num_frame,
        batch_size=1,
        num_workers=1,
        prefetch_factor=1,
        seed=42,
        accelerator=accelerator,
        infinite=False,
    )
    if loader is None:
        raise ValueError("No export source found. Set DATA.test.source or DATA.val.source")

    if cfg.TEST.checkpoint_path:
        net.load_pretrained(cfg.TEST.checkpoint_path)
    net.to(accelerator.device)
    net.eval()

    output_dir = osp.join(cfg.TEST.output_dir, "export_sample")
    os.makedirs(output_dir, exist_ok=True)

    for idx, batch_origin in enumerate(loader):
        if idx >= 8:
            break
        batch, trans_2d_mat, _ = preprocess_batch(
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
        output_state = net(batch)
        image = vis(batch, trans_2d_mat, output_state["result"], tx=batch["patches"].shape[1] - 1, bx=0)
        out_path = osp.join(output_dir, f"sample_{idx:02d}.png")
        cv2.imwrite(out_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
