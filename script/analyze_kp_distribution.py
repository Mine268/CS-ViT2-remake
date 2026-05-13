from __future__ import annotations

"""Analyze per-joint KP conditional distribution within GT vs WiLoR bbox.

For each of the 21 hand joints, compute the 2D distribution (mean, covariance)
in normalized bbox coordinates [0,1]x[0,1], then compare GT vs WiLoR.
"""

import argparse
import json
import os
import os.path as osp
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.wds import get_dataloader

HAND_JOINT_NAMES = [
    "Wrist", "Thumb_1", "Thumb_2", "Thumb_3", "Thumb_4",
    "Index_1", "Index_2", "Index_3", "Index_4",
    "Middle_1", "Middle_2", "Middle_3", "Middle_4",
    "Ring_1", "Ring_2", "Ring_3", "Ring_4",
    "Pinky_1", "Pinky_2", "Pinky_3", "Pinky_4",
]


def sample_frames(url, num_samples):
    loader = get_dataloader(
        url=url, num_frames=1, stride=1, batch_size=8,
        num_workers=0, prefetch_factor=1, infinite=False, seed=42,
        clip_sampling_mode="dense", shardshuffle=False, post_clip_shuffle=0,
    )
    frames = []
    for batch in loader:
        for bx in range(int(batch["hand_bbox"].shape[0])):
            if len(frames) >= num_samples:
                break
            tx = 0
            frames.append({
                "gt_bbox": batch["hand_bbox"][bx, tx].cpu().numpy().astype(np.float32),
                "gt_kps": batch["joint_img"][bx, tx].cpu().numpy().astype(np.float32),
                "kps_valid": batch["joint_2d_valid"][bx, tx].cpu().numpy().astype(np.float32),
                "image": batch["imgs"][bx][tx].permute(1, 2, 0).cpu().numpy().astype(np.uint8),
                "handedness": str(batch["handedness"][bx]),
            })
        if len(frames) >= num_samples:
            break
    return frames


def kps_in_bbox(kps, bbox, valid):
    """Convert KPs from image coords to normalized bbox coords [0,1]."""
    x1, y1, x2, y2 = bbox.astype(np.float64)
    bw, bh = x2 - x1, y2 - y1
    if bw < 1e-6 or bh < 1e-6:
        return np.zeros((21, 2)), np.zeros(21, dtype=bool)
    u = (kps[:, 0] - x1) / bw
    v = (kps[:, 1] - y1) / bh
    return np.stack([u, v], axis=-1), (valid > 0.5)


def run_analysis(dataset_name, split_name, wds_root, num_samples, output_path, device):
    import glob, time, cv2
    from hand_bbox_module.hand_bbox_detector import HandBBoxDetector
    wl_detector = HandBBoxDetector(
        model_path=str(REPO_ROOT / "hand_bbox_module" / "weights" / "detector.pt"),
        device=device, verbose=False,
    )

    urls = sorted(glob.glob(osp.join(wds_root, dataset_name, split_name, "*.tar")))
    print(f"Sampling {num_samples} frames from {len(urls)} shards...")
    frames = sample_frames(urls, num_samples)
    print(f"  Got {len(frames)} frames")

    # Per-joint accumulators for GT and WiLoR bboxes
    gt_kps_all = [[] for _ in range(21)]   # list of [N_frames, 2]
    wl_kps_all = [[] for _ in range(21)]

    t0 = time.time()
    for idx, frame in enumerate(frames):
        if (idx + 1) % 200 == 0:
            print(f"  {idx+1}/{len(frames)} ({(idx+1)/(time.time()-t0):.1f} fps)")

        gt_bbox = frame["gt_bbox"]
        kps = frame["gt_kps"]
        valid = frame["kps_valid"]

        # GT bbox: per-joint positions
        kp_uv, kp_ok = kps_in_bbox(kps, gt_bbox, valid)
        for j in range(21):
            if kp_ok[j]:
                gt_kps_all[j].append(kp_uv[j])

        # WiLoR bbox
        img_bgr = cv2.cvtColor(frame["image"], cv2.COLOR_RGB2BGR)
        raw = wl_detector.detect(img_bgr, conf=0.3, iou=0.45)
        wl_dets = [{"bbox": np.asarray(d.bbox, dtype=np.float32),
                     "handedness": ("left" if str(d.handedness).lower() == "left" else "right"),
                     "score": float(d.confidence)}
                   for d in raw]
        if wl_dets:
            wl_match = wl_dets[0]
            for d in wl_dets:
                if d["handedness"] == frame["handedness"]:
                    wl_match = d
                    break
            wl_bbox = wl_match["bbox"]
            kp_uv_wl, _ = kps_in_bbox(kps, wl_bbox, valid)
            for j in range(21):
                if kp_ok[j]:
                    wl_kps_all[j].append(kp_uv_wl[j])

    print(f"  Done in {time.time()-t0:.0f}s")

    # Compute per-joint statistics
    gt_stats = {}
    wl_stats = {}
    for j in range(21):
        if len(gt_kps_all[j]) > 0:
            arr = np.array(gt_kps_all[j])  # [N, 2]
            mean = arr.mean(axis=0)
            cov = np.cov(arr.T) if arr.shape[0] > 1 else np.eye(2)
            gt_stats[f"joint_{j:02d}"] = {
                "name": HAND_JOINT_NAMES[j],
                "mean_u": float(mean[0]), "mean_v": float(mean[1]),
                "std_u": float(np.sqrt(cov[0, 0])), "std_v": float(np.sqrt(cov[1, 1])),
                "corr_uv": float(cov[0, 1] / (np.sqrt(cov[0, 0]) * np.sqrt(cov[1, 1]) + 1e-10)),
                "p5_u": float(np.percentile(arr[:, 0], 5)), "p95_u": float(np.percentile(arr[:, 0], 95)),
                "p5_v": float(np.percentile(arr[:, 1], 5)), "p95_v": float(np.percentile(arr[:, 1], 95)),
                "count": int(arr.shape[0]),
            }
        if len(wl_kps_all[j]) > 0:
            arr = np.array(wl_kps_all[j])
            mean = arr.mean(axis=0)
            cov = np.cov(arr.T) if arr.shape[0] > 1 else np.eye(2)
            wl_stats[f"joint_{j:02d}"] = {
                "name": HAND_JOINT_NAMES[j],
                "mean_u": float(mean[0]), "mean_v": float(mean[1]),
                "std_u": float(np.sqrt(cov[0, 0])), "std_v": float(np.sqrt(cov[1, 1])),
                "corr_uv": float(cov[0, 1] / (np.sqrt(cov[0, 0]) * np.sqrt(cov[1, 1]) + 1e-10)),
                "p5_u": float(np.percentile(arr[:, 0], 5)), "p95_u": float(np.percentile(arr[:, 0], 95)),
                "p5_v": float(np.percentile(arr[:, 1], 5)), "p95_v": float(np.percentile(arr[:, 1], 95)),
                "count": int(arr.shape[0]),
            }

    # Compute shift between GT and WiLoR per-joint means
    shift_stats = {}
    for j in range(21):
        jk = f"joint_{j:02d}"
        if jk in gt_stats and jk in wl_stats:
            shift_stats[jk] = {
                "name": HAND_JOINT_NAMES[j],
                "shift_u": float(wl_stats[jk]["mean_u"] - gt_stats[jk]["mean_u"]),
                "shift_v": float(wl_stats[jk]["mean_v"] - gt_stats[jk]["mean_v"]),
                "scale_u": float(wl_stats[jk]["std_u"] / max(gt_stats[jk]["std_u"], 1e-10)),
                "scale_v": float(wl_stats[jk]["std_v"] / max(gt_stats[jk]["std_v"], 1e-10)),
            }

    result = {
        "dataset": dataset_name, "split": split_name,
        "num_frames": len(frames),
        "gt_per_joint": gt_stats,
        "wilor_per_joint": wl_stats,
        "shift_per_joint": shift_stats,
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved to {output_path}")

    # Print summary
    print("\n=== Per-Joint KP Distribution Summary ===")
    print(f"{'Joint':<12} {'GT mean_uv':<18} {'WiLoR mean_uv':<18} {'shift_uv':<16} {'scale_u':<8} {'scale_v':<8}")
    print("-" * 80)
    for j in range(21):
        jk = f"joint_{j:02d}"
        if jk not in gt_stats or jk not in wl_stats:
            continue
        g, w, s = gt_stats[jk], wl_stats[jk], shift_stats.get(jk, {})
        print(f"{g['name']:<12} ({g['mean_u']:.2f},{g['mean_v']:.2f})      "
              f"({w['mean_u']:.2f},{w['mean_v']:.2f})      "
              f"({s.get('shift_u',0):+.2f},{s.get('shift_v',0):+.2f})   "
              f"{s.get('scale_u',1):.2f}    {s.get('scale_v',1):.2f}")

    # Model fit assessment
    print("\n=== Model Fit Assessment ===")
    # Check if WiLoR bbox KPs are a shifted+scaled version of GT bbox KPs
    shifts_u = [s["shift_u"] for s in shift_stats.values()]
    shifts_v = [s["shift_v"] for s in shift_stats.values()]
    scales_u = [s["scale_u"] for s in shift_stats.values()]
    scales_v = [s["scale_v"] for s in shift_stats.values()]

    print(f"Mean shift: ({np.mean(shifts_u):.3f}, {np.mean(shifts_v):.3f})")
    print(f"Std of shifts: ({np.std(shifts_u):.3f}, {np.std(shifts_v):.3f})")
    print(f"Mean scale: ({np.mean(scales_u):.3f}, {np.mean(scales_v):.3f})")
    print(f"Std of scales: ({np.std(scales_u):.3f}, {np.std(scales_v):.3f})")

    # If shifts are consistent across joints → simple translation model fits
    # If scales are consistent across joints → simple scaling model fits
    print(f"\nshift_u CV={np.std(shifts_u)/max(abs(np.mean(shifts_u)),1e-6):.2f} "
          f"(<0.3 = consistent translation in u)")
    print(f"shift_v CV={np.std(shifts_v)/max(abs(np.mean(shifts_v)),1e-6):.2f} "
          f"(<0.3 = consistent translation in v)")

    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["AssemblyHands", "HOT3D"])
    p.add_argument("--split", required=True)
    p.add_argument("--wds-root", default="/data_0/renkaiwen/webdatasets2_remake")
    p.add_argument("--output-dir", default="example/bbox_analysis")
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_name = f"kp_dist_{args.dataset.lower()}_{args.split}.json"
    output_path = osp.join(args.output_dir, output_name)

    run_analysis(args.dataset, args.split, args.wds_root,
                 args.num_samples, output_path, args.device)


if __name__ == "__main__":
    main()
