from __future__ import annotations

"""
Model P(bbox | kps) for GT and WiLoR detectors.

Goal: design augmentation such that augmented GT bbox follows P_wilor(bbox|kps).

For each frame we have:
  - kps: 2D keypoints in image space [21, 2]
  - gt_bbox: tight envelope of kps
  - wl_bbox: WiLoR detector output

We model bbox as (center_x, center_y, size, aspect) relative to kps statistics.
Then compare GT distribution vs WiLoR distribution.
"""

import argparse, json, os, sys, time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.data.wds import get_dataloader


def bbox_from_kps(kps, valid_mask):
    """Compute tight bbox envelope from valid keypoints."""
    ok = kps[valid_mask > 0.5]
    if len(ok) == 0:
        return np.array([0, 0, 1, 1], dtype=np.float32)
    x1, y1 = ok[:, 0].min(), ok[:, 1].min()
    x2, y2 = ok[:, 0].max(), ok[:, 1].max()
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def bbox_params(bbox):
    """Decompose bbox into (cx, cy, size, aspect)."""
    x1, y1, x2, y2 = bbox.astype(np.float64)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    size = max(w, h)
    aspect = w / h
    return np.array([cx, cy, size, aspect])


def kps_statistics(kps, valid_mask):
    """Compute kps center, span, and aspect ratio."""
    ok = kps[valid_mask > 0.5]
    if len(ok) == 0:
        return np.array([0, 0, 1, 1, 0])
    cx = ok[:, 0].mean()
    cy = ok[:, 1].mean()
    span_x = ok[:, 0].max() - ok[:, 0].min()
    span_y = ok[:, 1].max() - ok[:, 1].min()
    span = max(span_x, span_y)
    aspect = span_x / max(span_y, 1e-6)
    return np.array([cx, cy, span, aspect, len(ok)])


def analyze_dataset(dataset_name, split_name, wds_root, num_samples, output_path, device):
    import glob, cv2
    from hand_bbox_module.hand_bbox_detector import HandBBoxDetector

    wl_detector = HandBBoxDetector(
        model_path=str(REPO_ROOT / "hand_bbox_module" / "weights" / "detector.pt"),
        device=device, verbose=False,
    )

    urls = sorted(glob.glob(os.path.join(wds_root, dataset_name, split_name, "*.tar")))
    loader = get_dataloader(
        url=urls, num_frames=1, stride=1, batch_size=8,
        num_workers=0, prefetch_factor=1, infinite=False, seed=42,
        clip_sampling_mode="dense", shardshuffle=False, post_clip_shuffle=0,
    )

    # Collect: for each frame, kps → gt_bbox_params, wl_bbox_params
    records = []
    print(f"Processing {dataset_name}/{split_name}...")
    t0 = time.time()
    count = 0

    for batch in loader:
        for bx in range(int(batch["hand_bbox"].shape[0])):
            if count >= num_samples:
                break
            tx = 0
            kps = batch["joint_img"][bx, tx].cpu().numpy().astype(np.float32)
            kps_valid = batch["joint_2d_valid"][bx, tx].cpu().numpy().astype(np.float32)
            handedness = str(batch["handedness"][bx])
            img = batch["imgs"][bx][tx].permute(1, 2, 0).cpu().numpy().astype(np.uint8)

            # GT bbox = tight envelope of valid KPs
            gt_bbox = bbox_from_kps(kps, kps_valid)
            kps_stat = kps_statistics(kps, kps_valid)

            # WiLoR
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            raw = wl_detector.detect(img_bgr, conf=0.3, iou=0.45)
            wl_dets = [{"bbox": np.asarray(d.bbox, dtype=np.float32),
                         "handedness": ("left" if str(d.handedness).lower() == "left" else "right")}
                       for d in sorted(raw, key=lambda d: d.confidence, reverse=True)]

            wl_match = None
            for d in wl_dets:
                if d["handedness"] == handedness:
                    wl_match = d; break
            if wl_match is None and wl_dets:
                wl_match = wl_dets[0]

            if wl_match is not None:
                wl_bbox = wl_match["bbox"]
                gt_params = bbox_params(gt_bbox)
                wl_params = bbox_params(wl_bbox)
                records.append({
                    "kps_cx": float(kps_stat[0]), "kps_cy": float(kps_stat[1]),
                    "kps_span": float(kps_stat[2]), "kps_aspect": float(kps_stat[3]),
                    "kps_n_valid": int(kps_stat[4]),
                    "gt_cx": float(gt_params[0]), "gt_cy": float(gt_params[1]),
                    "gt_size": float(gt_params[2]), "gt_aspect": float(gt_params[3]),
                    "wl_cx": float(wl_params[0]), "wl_cy": float(wl_params[1]),
                    "wl_size": float(wl_params[2]), "wl_aspect": float(wl_params[3]),
                })
            count += 1
            if count % 500 == 0:
                print(f"  {count}/{num_samples} ({(time.time()-t0)/max(count,1):.3f}s/frame)")
        if count >= num_samples:
            break

    print(f"  Collected {len(records)} records in {time.time()-t0:.0f}s")

    # ===== ANALYSIS =====
    arr = {k: np.array([r[k] for r in records]) for k in records[0]}

    # Key metrics: how does WiLoR bbox relate to GT bbox and KPs?
    # 1. Size: wl_size / gt_size
    # 2. Center offset: (wl_cx - gt_cx)/gt_size, (wl_cy - gt_cy)/gt_size
    # 3. Can we model these as functions of kps position within gt_bbox?

    size_ratio = arr["wl_size"] / arr["gt_size"]
    cx_shift = (arr["wl_cx"] - arr["gt_cx"]) / arr["gt_size"]
    cy_shift = (arr["wl_cy"] - arr["gt_cy"]) / arr["gt_size"]

    # KPs position within GT bbox (how "centered" is the hand?)
    kps_cx_in_gt = (arr["kps_cx"] - arr["gt_cx"]) / arr["gt_size"]
    kps_cy_in_gt = (arr["kps_cy"] - arr["gt_cy"]) / arr["gt_size"]
    kps_span_ratio = arr["kps_span"] / arr["gt_size"]  # ~1.0 for tight bbox

    def stats(arr, name, precision=3):
        q = np.percentile(arr, [5, 25, 50, 75, 95])
        return {
            "name": name, "mean": round(float(np.mean(arr)), precision),
            "std": round(float(np.std(arr)), precision),
            "p5": round(float(q[0]), precision), "p25": round(float(q[1]), precision),
            "p50": round(float(q[2]), precision), "p75": round(float(q[3]), precision),
            "p95": round(float(q[4]), precision),
        }

    summary = {
        "dataset": dataset_name, "split": split_name, "n": len(records),
        "size_ratio": stats(size_ratio, "wl_size/gt_size"),
        "cx_shift": stats(cx_shift, "(wl_cx-gt_cx)/gt_size"),
        "cy_shift": stats(cy_shift, "(wl_cy-gt_cy)/gt_size"),
        "kps_cx_in_gt": stats(kps_cx_in_gt, "kps_cx relative to gt_bbox"),
        "kps_cy_in_gt": stats(kps_cy_in_gt, "kps_cy relative to gt_bbox"),
        "kps_span_ratio": stats(kps_span_ratio, "kps_span/gt_size"),
    }

    # Correlation: do KPs position predict WiLoR bbox parameters?
    corr_cx = np.corrcoef(kps_cx_in_gt, cx_shift)[0, 1]
    corr_cy = np.corrcoef(kps_cy_in_gt, cy_shift)[0, 1]
    corr_span_size = np.corrcoef(kps_span_ratio, size_ratio)[0, 1]
    summary["correlations"] = {
        "kps_cx_vs_cx_shift": round(float(corr_cx), 4),
        "kps_cy_vs_cy_shift": round(float(corr_cy), 4),
        "kps_span_vs_size_ratio": round(float(corr_span_size), 4),
    }

    # Model: WiLoR bbox ≈ GT bbox expanded with specific pattern
    # wl_cx = gt_cx + gt_size * (cx_bias + cx_noise)
    # wl_cy = gt_cy + gt_size * (cy_bias + cy_noise)
    # wl_size = gt_size * (size_scale + size_noise)
    # wl_aspect ≈ gt_aspect (hand aspect doesn't change)

    summary["augmentation_model"] = {
        "description": "To make GT bbox look like WiLoR bbox during training:",
        "center_x": f"gt_cx + gt_size * N({np.mean(cx_shift):.3f}, {np.std(cx_shift):.3f})",
        "center_y": f"gt_cy + gt_size * N({np.mean(cy_shift):.3f}, {np.std(cy_shift):.3f})",
        "size": f"gt_size * logN({np.mean(np.log(size_ratio)):.3f}, {np.std(np.log(size_ratio)):.3f})",
    }

    # Current jitter equivalent
    jc = {"center_shift": 0.3, "scale_range": [0.5, 1.5]}
    actual_cx = np.percentile(np.abs(cx_shift), 95)
    actual_cy = np.percentile(np.abs(cy_shift), 95)
    actual_scale_p5 = np.exp(np.mean(np.log(size_ratio)) - 2 * np.std(np.log(size_ratio)))
    actual_scale_p95 = np.exp(np.mean(np.log(size_ratio)) + 2 * np.std(np.log(size_ratio)))

    summary["recommendation"] = {
        "center_shift": round(float(max(actual_cx, actual_cy, 0.5)), 3),
        "center_bias_x": round(float(np.mean(cx_shift)), 3),
        "center_bias_y": round(float(np.mean(cy_shift)), 3),
        "center_std_x": round(float(np.std(cx_shift)), 3),
        "center_std_y": round(float(np.std(cy_shift)), 3),
        "scale_range": [round(float(actual_scale_p5), 3), round(float(actual_scale_p95), 3)],
        "scale_log_mean": round(float(np.mean(np.log(size_ratio))), 3),
        "scale_log_std": round(float(np.std(np.log(size_ratio))), 3),
        "current_jitter_covered": {
            "center_p95": [round(float(actual_cx), 3), round(float(actual_cy), 3)],
            "covered_by_0.3": bool(actual_cx <= 0.3 and actual_cy <= 0.3),
            "scale_p5_p95": [round(float(actual_scale_p5), 3), round(float(actual_scale_p95), 3)],
            "covered_by_0.5_1.5": bool(actual_scale_p5 >= 0.5 and actual_scale_p95 <= 1.5),
        },
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print report
    s = summary
    sr = s["size_ratio"]; scx = s["cx_shift"]; scy = s["cy_shift"]
    co = s["correlations"]; am = s["augmentation_model"]; rec = s["recommendation"]

    print(f"\n{'='*70}")
    print(f"P(bbox | kps) Analysis: {dataset_name}/{split_name}  (n={len(records)})")
    print(f"{'='*70}")
    print(f"\nWiLoR bbox relative to GT bbox (GT = tight kps envelope):")
    print(f"  size_ratio:  p50={sr['p50']}  p5={sr['p5']}  p95={sr['p95']}")
    print(f"  cx_shift:    p50={scx['p50']}  p5={scx['p5']}  p95={scx['p95']}")
    print(f"  cy_shift:    p50={scy['p50']}  p5={scy['p5']}  p95={scy['p95']}")
    print(f"\nCorrelations (KPs position vs WiLoR bbox shift):")
    print(f"  kps_cx → cx_shift: {co['kps_cx_vs_cx_shift']}")
    print(f"  kps_cy → cy_shift: {co['kps_cy_vs_cy_shift']}")
    print(f"  kps_span → size_ratio: {co['kps_span_vs_size_ratio']}")
    print(f"\nAugmentation model:")
    print(f"  {am['center_x']}")
    print(f"  {am['center_y']}")
    print(f"  {am['size']}")
    print(f"\nRecommended vs current jitter:")
    print(f"  center_shift: {rec['center_shift']} (current: 0.3)")
    print(f"  scale_range: {rec['scale_range']} (current: [0.5,1.5])")
    print(f"\nSaved to {output_path}")

    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["AssemblyHands", "HOT3D"])
    p.add_argument("--split", required=True)
    p.add_argument("--wds-root", default="/data_0/renkaiwen/webdatasets2_remake")
    p.add_argument("--output-dir", default="example/bbox_analysis")
    p.add_argument("--num-samples", type=int, default=2000)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_name = f"bbox_given_kps_{args.dataset.lower()}_{args.split}.json"
    output_path = os.path.join(args.output_dir, output_name)

    s = analyze_dataset(args.dataset, args.split, args.wds_root,
                        args.num_samples, output_path, args.device)


if __name__ == "__main__":
    main()
