from __future__ import annotations

"""Full-dataset statistics on constrained bbox jitter parameter distributions.

Samples frames from training data, applies constrained jitter, and records
the actual (α, δx, δy) values to verify the augmentation covers the desired range.
"""

import argparse, json, os, sys, time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.wds import get_dataloader
from src.data.preprocess import _jitter_hand_bbox, MIN_PATCH_EDGE_PIXELS


def _image_wh_from_imgs(imgs_list, device, dtype):
    wh = []
    for t in imgs_list:
        _, _, h, w = t.shape
        wh.append([float(w), float(h)])
    return torch.tensor(wh, device=device, dtype=dtype)[:, None, :]


def analyze_jitter_params(
    dataset_name: str,
    split_name: str,
    wds_root: str,
    max_samples: int,
    output_path: str,
    jitter_cfg: Dict,
):
    import glob
    urls = sorted(glob.glob(os.path.join(wds_root, dataset_name, split_name, "*.tar")))
    print(f"Dataset: {dataset_name}/{split_name}, {len(urls)} shards")

    loader = get_dataloader(
        url=urls, num_frames=1, stride=1, batch_size=16,
        num_workers=0, prefetch_factor=1, infinite=False,
        clip_sampling_mode="dense", shardshuffle=False, post_clip_shuffle=0,
    )

    # Collect per-frame jitter parameters
    records: List[Dict[str, float]] = []
    count = 0
    t0 = time.time()

    for batch in loader:
        bs = int(batch["hand_bbox"].shape[0])
        for bx in range(bs):
            if max_samples > 0 and count >= max_samples:
                break

            hand_bbox = batch["hand_bbox"][bx]  # [1, 4]
            imgs = batch["imgs"][bx]  # list of tensors

            image_wh = _image_wh_from_imgs([imgs], hand_bbox.device, hand_bbox.dtype)

            # Store baseline
            x1, y1, x2, y2 = hand_bbox[0, 0].tolist(), hand_bbox[0, 1].tolist(), hand_bbox[0, 2].tolist(), hand_bbox[0, 3].tolist()
            base_cx = (x1 + x2) / 2
            base_cy = (y1 + y2) / 2
            base_sz = max(x2 - x1, y2 - y1)

            if base_sz < MIN_PATCH_EDGE_PIXELS:
                count += 1
                continue

            # Apply jitter
            torch.manual_seed(42 + count)
            jittered = _jitter_hand_bbox(
                hand_bbox=hand_bbox.unsqueeze(0),  # [1, 1, 4]
                image_wh=image_wh,
                bbox_jitter=jitter_cfg,
            )  # [1, 1, 4]

            jx1, jy1, jx2, jy2 = [float(v) for v in jittered[0, 0]]
            jt_cx = (jx1 + jx2) / 2
            jt_cy = (jy1 + jy2) / 2
            jt_sz = max(jx2 - jx1, jy2 - jy1)

            alpha = jt_sz / base_sz
            delta_x = (jt_cx - base_cx) / base_sz
            delta_y = (jt_cy - base_cy) / base_sz
            max_allowed = (alpha - 1.0) / 2.0

            records.append({
                "alpha": alpha,
                "delta_x": delta_x,
                "delta_y": delta_y,
                "abs_delta": float(np.sqrt(delta_x**2 + delta_y**2)),
                "max_allowed": max_allowed,
                "in_constraint": abs(delta_x) <= max_allowed + 1e-5 and abs(delta_y) <= max_allowed + 1e-5,
            })
            count += 1

        if max_samples > 0 and count >= max_samples:
            break
        if count % 5000 == 0:
            elapsed = time.time() - t0
            print(f"  {count} samples ({count / max(elapsed, 1):.0f} fps)")

    elapsed = time.time() - t0
    print(f"  Done: {len(records)} samples in {elapsed:.0f}s ({len(records)/max(elapsed,1):.0f} fps)")

    # Compute statistics
    arr = {k: np.array([r[k] for r in records]) for k in records[0]}

    def pstats(a, name):
        q = np.percentile(a, [1, 5, 10, 25, 50, 75, 90, 95, 99])
        return {
            "name": name, "count": int(len(a)),
            "mean": float(np.mean(a)), "std": float(np.std(a)),
            "p1": float(q[0]), "p5": float(q[1]), "p10": float(q[2]),
            "p25": float(q[3]), "p50": float(q[4]), "p75": float(q[5]),
            "p90": float(q[6]), "p95": float(q[7]), "p99": float(q[8]),
            "min": float(np.min(a)), "max": float(np.max(a)),
        }

    summary = {
        "dataset": dataset_name, "split": split_name,
        "num_samples": len(records),
        "jitter_config": jitter_cfg,
        "alpha": pstats(arr["alpha"], "scale α"),
        "delta_x": pstats(arr["delta_x"], "center shift x"),
        "delta_y": pstats(arr["delta_y"], "center shift y"),
        "abs_delta": pstats(arr["abs_delta"], "center shift magnitude"),
        "in_constraint_rate": float(np.mean(arr["in_constraint"])),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print report
    print(f"\n{'='*60}")
    print(f"Jitter Parameter Distribution: {dataset_name}/{split_name}")
    print(f"n={len(records)}  constraint_ok={summary['in_constraint_rate']:.4f}")
    print(f"{'='*60}")
    for key, label in [("alpha", "α (scale)"), ("delta_x", "δx"), ("delta_y", "δy"), ("abs_delta", "|δ|")]:
        s = summary[key]
        print(f"  {label:<12s}: p50={s['p50']:.3f}  p5={s['p5']:.3f}  p95={s['p95']:.3f}  "
              f"mean={s['mean']:.3f}±{s['std']:.3f}")
    print(f"  Saved to {output_path}")

    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--wds-root", default="/data_0/renkaiwen/webdatasets2_remake")
    p.add_argument("--output-dir", default="example/bbox_analysis")
    p.add_argument("--max-samples", type=int, default=0, help="0 = all available")
    p.add_argument("--scale-log-mean", type=float, default=0.36)
    p.add_argument("--scale-log-std", type=float, default=0.15)
    p.add_argument("--center-shift", type=float, default=0.5)
    args = p.parse_args()

    jitter_cfg = {
        "enabled": True, "prob": 1.0, "temporal_mode": "frame",
        "constrained": True,
        "scale_log_mean": args.scale_log_mean,
        "scale_log_std": args.scale_log_std,
        "scale_range": [1.0, 2.0],
        "center_shift": args.center_shift,
        "aspect_ratio_range": [1.0, 1.0],
        "frame_center_shift": 0.0, "frame_scale_range": [1.0, 1.0],
        "min_edge_px": 8.0,
    }

    output_name = f"jitter_params_{args.dataset.lower()}_{args.split}.json"
    output_path = os.path.join(args.output_dir, output_name)

    analyze_jitter_params(
        args.dataset, args.split, args.wds_root,
        args.max_samples, output_path, jitter_cfg,
    )


if __name__ == "__main__":
    main()
