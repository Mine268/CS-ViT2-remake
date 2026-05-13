from __future__ import annotations

"""Grab random samples from WDS shards for demo."""

import argparse
import json
import os
import os.path as osp
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.wds import get_dataloader


def grab_random_samples(
    url: List[str],
    num_samples: int,
    num_frames: int,
    stride: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    loader = get_dataloader(
        url=url,
        num_frames=num_frames,
        stride=stride,
        batch_size=min(num_samples, 8),
        num_workers=2,
        prefetch_factor=2,
        infinite=False,
        seed=seed,
        clip_sampling_mode="random_clip",
        clips_per_sequence=1,
        shardshuffle=seed,
        post_clip_shuffle=200,
    )
    samples = []
    for batch in loader:
        batch_size = int(batch["hand_bbox"].shape[0])
        for bx in range(batch_size):
            if len(samples) >= num_samples:
                break
            tx = int(batch["imgs"][bx].shape[0] - 1)
            img = batch["imgs"][bx][tx]  # [C, H, W], uint8
            source_index = batch.get("source_index", [[{}]])[bx][tx]
            si = source_index if isinstance(source_index, dict) else {}
            samples.append({
                "image": img.permute(1, 2, 0).cpu().numpy().astype(np.uint8),
                "__key__": str(batch["__key__"][bx]),
                "data_source": str(batch.get("data_source", ["unknown"])[bx]),
                "handedness": str(batch["handedness"][bx]),
                "hand_bbox": batch["hand_bbox"][bx, tx].cpu().numpy().astype(np.float32).tolist(),
                "focal": batch["focal"][bx, tx].cpu().numpy().astype(np.float32).tolist(),
                "princpt": batch["princpt"][bx, tx].cpu().numpy().astype(np.float32).tolist(),
                "source_index": si,
            })
        if len(samples) >= num_samples:
            break
    return samples


def save_samples(
    samples: List[Dict[str, Any]],
    output_dir: str,
    dataset_label: str,
    split_name: str,
):
    """Save each sample as a PNG image with metadata JSON."""
    from PIL import Image

    for i, s in enumerate(samples):
        sample_dir = osp.join(output_dir, f"sample_{i:02d}")
        os.makedirs(sample_dir, exist_ok=True)

        # Save image
        frame_path = osp.join(sample_dir, "frame.png")
        Image.fromarray(s["image"]).save(frame_path)

        # Save metadata (compatible with demo_stage1 --detector gt expectations)
        meta = {
            "dataset": dataset_label,
            "split": split_name,
            "sample_key": s["__key__"],
            "data_source": s["data_source"],
            "handedness": s["handedness"],
            "hand_bbox": s["hand_bbox"],
            "focal": s["focal"],
            "princpt": s["princpt"],
            "source_index": s["source_index"],
        }
        with open(osp.join(sample_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(samples)} samples to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Grab random WDS samples for demo.")
    parser.add_argument("--dataset", required=True, choices=["AssemblyHands", "HOT3D"])
    parser.add_argument("--split", required=True)
    parser.add_argument("--wds-root", default="/data_0/renkaiwen/webdatasets2_remake")
    parser.add_argument("--output-root", default="example/random_sequences")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import glob
    shard_pattern = osp.join(args.wds_root, args.dataset, args.split, "*.tar")
    urls = sorted(glob.glob(shard_pattern))

    print(f"Grabbing {args.num_samples} random samples from {args.dataset}/{args.split}...")
    samples = grab_random_samples(
        url=urls,
        num_samples=args.num_samples,
        num_frames=1,
        stride=1,
        seed=args.seed + args.start_index,
    )
    print(f"  Got {len(samples)} samples")

    output_name = f"{args.dataset.lower()}_{args.split}"
    output_dir = osp.join(args.output_root, output_name)
    save_samples(samples, output_dir, args.dataset, args.split)


if __name__ == "__main__":
    main()
