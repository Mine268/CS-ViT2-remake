from __future__ import annotations

"""Analyze bbox distribution differences between GT, MediaPipe KP, and WiLoR.

Samples frames from WDS shards, runs multiple detectors, and computes statistics
to inform bbox jitter augmentation parameters.
"""

import argparse
import json
import os
import os.path as osp
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.wds import get_dataloader


# ---------------------------------------------------------------------------
# Cached detector singletons (created once, reused across frames)
# ---------------------------------------------------------------------------

_MP_DETECTOR = None
_WILOR_DETECTOR = None


def _normalize_handedness(value: str) -> str:
    return "left" if value.lower() == "left" else "right"


def _clip_bbox(bbox_xyxy: np.ndarray, width: int, height: int) -> np.ndarray:
    bbox = bbox_xyxy.astype(np.float32, copy=True)
    bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0.0, float(max(width - 1, 0)))
    bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0.0, float(max(height - 1, 0)))
    return bbox


def _find_mediapipe_model() -> Optional[str]:
    candidates = [
        REPO_ROOT / "model" / "mediapipe" / "hand_landmarker.task",
        Path.home() / ".cache" / "mediapipe" / "hand_landmarker.task",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


def _init_mp_detector():
    """Create and cache the MediaPipe hand detector (solutions or tasks API)."""
    global _MP_DETECTOR
    if _MP_DETECTOR is not None:
        return _MP_DETECTOR
    try:
        import mediapipe as mp
    except ModuleNotFoundError:
        print("[warn] mediapipe not installed")
        _MP_DETECTOR = False
        return False

    if hasattr(mp, "solutions"):
        _MP_DETECTOR = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return _MP_DETECTOR

    model_path = _find_mediapipe_model()
    if model_path is None:
        print("[warn] mediapipe tasks API found but no hand_landmarker.task")
        _MP_DETECTOR = False
        return False

    from mediapipe.tasks.python.core import base_options
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
    _MP_DETECTOR = HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=base_options.BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )
    return _MP_DETECTOR


def detect_mediapipe_kp(image_rgb: np.ndarray) -> List[Dict[str, Any]]:
    mp_obj = _init_mp_detector()
    if mp_obj is False or mp_obj is None:
        return []
    height, width = image_rgb.shape[:2]

    # Solutions API
    import mediapipe as mp
    if hasattr(mp, "solutions") and hasattr(mp_obj, "process"):
        result = mp_obj.process(image_rgb)
        if not result.multi_hand_landmarks:
            return []
        dets = []
        handedness_items = result.multi_handedness or []
        for idx, landmarks in enumerate(result.multi_hand_landmarks):
            xs = np.asarray([lm.x * width for lm in landmarks.landmark], dtype=np.float32)
            ys = np.asarray([lm.y * height for lm in landmarks.landmark], dtype=np.float32)
            kps = np.stack([xs, ys], axis=-1)
            bbox = _clip_bbox(
                np.asarray([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32),
                width=width, height=height,
            )
            label = "right"
            if idx < len(handedness_items) and handedness_items[idx].classification:
                label = str(handedness_items[idx].classification[0].label).lower()
            dets.append({"bbox": bbox, "kps": kps, "handedness": _normalize_handedness(label)})
        return dets

    # Tasks API
    from mediapipe.tasks.python.vision import HandLandmarker
    if isinstance(mp_obj, HandLandmarker):
        try:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=np.ascontiguousarray(image_rgb, dtype=np.uint8))
            result = mp_obj.detect(mp_img)
        except Exception:
            return []
        if result is None or not result.hand_landmarks:
            return []
        dets = []
        for idx, landmarks in enumerate(result.hand_landmarks):
            xs = np.asarray([lm.x * width for lm in landmarks], dtype=np.float32)
            ys = np.asarray([lm.y * height for lm in landmarks], dtype=np.float32)
            kps = np.stack([xs, ys], axis=-1)
            bbox = _clip_bbox(
                np.asarray([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32),
                width=width, height=height,
            )
            label = "right"
            if idx < len(result.handedness) and len(result.handedness[idx]) > 0:
                label = str(result.handedness[idx][0].category_name).lower()
            dets.append({"bbox": bbox, "kps": kps, "handedness": _normalize_handedness(label)})
        return dets

    return []


def _init_wilor_detector(device: str = "cuda:0"):
    global _WILOR_DETECTOR
    if _WILOR_DETECTOR is not None:
        return _WILOR_DETECTOR
    try:
        from hand_bbox_module.hand_bbox_detector import HandBBoxDetector
    except ModuleNotFoundError:
        print("[warn] hand_bbox_module not installed")
        _WILOR_DETECTOR = False
        return False
    _WILOR_DETECTOR = HandBBoxDetector(
        model_path=str(REPO_ROOT / "hand_bbox_module" / "weights" / "detector.pt"),
        device=device,
        verbose=False,
    )
    return _WILOR_DETECTOR


def detect_wilor(image_rgb: np.ndarray, device: str = "cuda:0") -> List[Dict[str, Any]]:
    detector = _init_wilor_detector(device)
    if detector is False or detector is None:
        return []
    import cv2
    height, width = image_rgb.shape[:2]
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    raw = detector.detect(image_bgr, conf=0.3, iou=0.45)
    dets = []
    for det in sorted(raw, key=lambda d: float(d.confidence), reverse=True):
        bbox = _clip_bbox(np.asarray(det.bbox, dtype=np.float32), width=width, height=height)
        dets.append({
            "bbox": bbox,
            "kps": None,
            "handedness": _normalize_handedness(str(det.handedness)),
            "score": float(det.confidence),
        })
    return dets


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------


def sample_frames(url: List[str], num_samples: int, seed: int = 42) -> List[Dict[str, Any]]:
    loader = get_dataloader(
        url=url, num_frames=1, stride=1, batch_size=8,
        num_workers=0, prefetch_factor=1, infinite=False, seed=seed,
        clip_sampling_mode="dense", clips_per_sequence=1,
        shardshuffle=False, post_clip_shuffle=0,
    )
    frames = []
    for batch in loader:
        batch_size = int(batch["hand_bbox"].shape[0])
        for bx in range(batch_size):
            if len(frames) >= num_samples:
                break
            tx = 0
            img = batch["imgs"][bx][tx]
            frames.append({
                "image": img.permute(1, 2, 0).cpu().numpy().astype(np.uint8),
                "height": int(img.shape[1]), "width": int(img.shape[2]),
                "gt_hand_bbox": batch["hand_bbox"][bx, tx].cpu().numpy().astype(np.float32),
                "gt_joint_img": batch["joint_img"][bx, tx].cpu().numpy().astype(np.float32),
                "gt_joint_2d_valid": batch["joint_2d_valid"][bx, tx].cpu().numpy().astype(np.float32),
                "handedness": str(batch["handedness"][bx]),
            })
        if len(frames) >= num_samples:
            break
    return frames


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def bbox_properties(bbox: np.ndarray, width: int, height: int) -> Dict[str, float]:
    x1, y1, x2, y2 = bbox.astype(np.float64)
    bw, bh = x2 - x1, y2 - y1
    return {
        "cx_norm": float(((x1 + x2) * 0.5) / width),
        "cy_norm": float(((y1 + y2) * 0.5) / height),
        "edge_norm": float(max(bw, bh) / max(width, height)),
        "aspect": float(bw / max(bh, 1e-6)),
    }


def bbox_relative(bbox: np.ndarray, ref_bbox: np.ndarray) -> Dict[str, float]:
    x1, y1, x2, y2 = bbox.astype(np.float64)
    rx1, ry1, rx2, ry2 = ref_bbox.astype(np.float64)
    bw, bh = x2 - x1, y2 - y1
    rbw, rbh = rx2 - rx1, ry2 - ry1
    ref_edge = max(rbw, rbh)
    if ref_edge < 1e-6:
        return {"scale_ratio": 1.0, "center_shift_norm": 0.0, "dx_norm": 0.0, "dy_norm": 0.0}

    # IoU
    ix1, iy1 = max(x1, rx1), max(y1, ry1)
    ix2, iy2 = min(x2, rx2), min(y2, ry2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, bw) * max(0.0, bh)
    area_b = max(0.0, rbw) * max(0.0, rbh)
    iou = inter / (area_a + area_b - inter) if (area_a + area_b - inter) > 1e-6 else 0.0

    scale_ratio = max(bw, bh) / ref_edge
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    rcx, rcy = (rx1 + rx2) * 0.5, (ry1 + ry2) * 0.5
    cs = np.sqrt((cx - rcx) ** 2 + (cy - rcy) ** 2) / ref_edge

    return {
        "scale_ratio": float(scale_ratio),
        "center_shift_norm": float(cs),
        "dx_norm": float((cx - rcx) / ref_edge),
        "dy_norm": float((cy - rcy) / ref_edge),
        "iou": float(iou),
    }


def kps_relative_to_bbox(kps: np.ndarray, bbox: np.ndarray,
                          valid: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    x1, y1, x2, y2 = bbox.astype(np.float64)
    bw, bh = x2 - x1, y2 - y1
    if bw < 1e-6 or bh < 1e-6:
        return {"kp_u": np.zeros(0, dtype=np.float32), "kp_v": np.zeros(0, dtype=np.float32)}
    u = (kps[:, 0] - x1) / bw
    v = (kps[:, 1] - y1) / bh
    if valid is not None:
        mask = valid > 0.5
        u, v = u[mask], v[mask]
    return {"kp_u": u.astype(np.float32), "kp_v": v.astype(np.float32)}


def match_detection(detections: List[Dict], gt_handedness: str) -> Optional[Dict]:
    for det in detections:
        if det["handedness"] == gt_handedness:
            return det
    return detections[0] if detections else None


def percentile_stats(arr: np.ndarray) -> Dict[str, float]:
    q = np.percentile(arr, [5, 10, 25, 50, 75, 90, 95])
    return {
        "mean": float(np.mean(arr)), "std": float(np.std(arr)),
        "p5": float(q[0]), "p10": float(q[1]), "p25": float(q[2]),
        "p50": float(q[3]), "p75": float(q[4]), "p90": float(q[5]), "p95": float(q[6]),
        "min": float(np.min(arr)), "max": float(np.max(arr)), "count": int(len(arr)),
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def run_analysis(dataset_name: str, split_name: str, wds_root: str,
                 num_samples: int, output_path: str, device: str = "cuda:0"):
    import glob
    urls = sorted(glob.glob(osp.join(wds_root, dataset_name, split_name, "*.tar")))
    print(f"Dataset: {dataset_name}/{split_name}, {len(urls)} shards")

    print(f"Sampling {num_samples} frames...")
    frames = sample_frames(urls, num_samples=num_samples)
    print(f"  Got {len(frames)} frames")

    stats: Dict[str, List[Dict]] = defaultdict(list)
    kp_stats: Dict[str, List[np.ndarray]] = defaultdict(list)

    t_start = time.time()
    for idx, frame in enumerate(frames):
        if (idx + 1) % 200 == 0:
            elapsed = time.time() - t_start
            print(f"  Processing {idx + 1}/{len(frames)} "
                  f"({(idx + 1) / elapsed:.1f} fps, ETA {elapsed / (idx + 1) * (len(frames) - idx - 1):.0f}s)")

        h, w = frame["height"], frame["width"]
        gt_bbox = frame["gt_hand_bbox"]
        gt_kps = frame["gt_joint_img"]
        gt_kps_valid = frame["gt_joint_2d_valid"]
        handedness = frame["handedness"]

        # GT bbox properties
        stats["gt_props"].append(bbox_properties(gt_bbox, w, h))
        gt_kp_rel = kps_relative_to_bbox(gt_kps, gt_bbox, gt_kps_valid)
        kp_stats["gt_kp_u"].append(gt_kp_rel["kp_u"])
        kp_stats["gt_kp_v"].append(gt_kp_rel["kp_v"])

        # MediaPipe KP
        mp_dets = detect_mediapipe_kp(frame["image"])
        mp_match = match_detection(mp_dets, handedness)
        if mp_match is not None:
            mb = mp_match["bbox"]
            stats["mp_props"].append(bbox_properties(mb, w, h))
            stats["mp_vs_gt"].append(bbox_relative(mb, gt_bbox))
            mp_kp_rel = kps_relative_to_bbox(gt_kps, mb, gt_kps_valid)
            kp_stats["mp_kp_u"].append(mp_kp_rel["kp_u"])
            kp_stats["mp_kp_v"].append(mp_kp_rel["kp_v"])

        # WiLoR (scale=1.0)
        wl_dets = detect_wilor(frame["image"], device=device)
        wl_match = match_detection(wl_dets, handedness)
        if wl_match is not None:
            wb = wl_match["bbox"]
            stats["wl_props"].append(bbox_properties(wb, w, h))
            stats["wl_vs_gt"].append(bbox_relative(wb, gt_bbox))
            wl_kp_rel = kps_relative_to_bbox(gt_kps, wb, gt_kps_valid)
            kp_stats["wl_kp_u"].append(wl_kp_rel["kp_u"])
            kp_stats["wl_kp_v"].append(wl_kp_rel["kp_v"])

        # MP vs WiLoR
        if mp_match is not None and wl_match is not None:
            stats["mp_vs_wl"].append(bbox_relative(mp_match["bbox"], wl_match["bbox"]))

    elapsed = time.time() - t_start
    print(f"  Done. {len(frames)} frames in {elapsed:.0f}s ({len(frames)/elapsed:.1f} fps)")

    # Compute summary
    summary = {}
    for key, values in stats.items():
        if len(values) == 0:
            continue
        cols = list(values[0].keys())
        col_stats = {}
        for col in cols:
            col_stats[col] = percentile_stats(np.array([v[col] for v in values]))
        summary[key] = col_stats

    kp_summary = {}
    for key, array_list in kp_stats.items():
        if len(array_list) == 0:
            continue
        kp_summary[key] = percentile_stats(np.concatenate(array_list))

    result = {
        "dataset": dataset_name, "split": split_name,
        "num_frames": len(frames), "summary": summary, "kp_summary": kp_summary,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved to {output_path}")
    return result


def analyze_jitter_coverage(result: Dict[str, Any]) -> str:
    lines = ["=" * 70, "JITTER COVERAGE ANALYSIS", "=" * 70,
             f"\nCurrent jitter: center_shift=0.3, scale=[0.5, 1.5], aspect=[0.6, 1.6]"]
    summary = result["summary"]
    kp = result.get("kp_summary", {})

    for det_label, rel_key, prop_key in [
        ("MediaPipe KP vs GT", "mp_vs_gt", "mp_props"),
        ("WiLoR vs GT", "wl_vs_gt", "wl_props"),
    ]:
        if rel_key not in summary:
            lines.append(f"\n{det_label}: no data")
            continue
        cols = summary[rel_key]
        lines.append(f"\n--- {det_label} ---")
        for col_name in ["scale_ratio", "center_shift_norm", "dx_norm", "dy_norm", "iou"]:
            if col_name not in cols:
                continue
            s = cols[col_name]
            lines.append(f"  {col_name:<20s}: p5={s['p5']:.3f} p50={s['p50']:.3f} p95={s['p95']:.3f}  "
                         f"mean={s['mean']:.3f}±{s['std']:.3f}")

    lines.append("\n--- Detector bbox properties ---")
    for key, label in [("gt_props", "GT"), ("mp_props", "MediaPipe"), ("wl_props", "WiLoR")]:
        if key not in summary:
            continue
        cols = summary[key]
        lines.append(f"  {label}: edge_norm p50={cols['edge_norm']['p50']:.3f} "
                     f"aspect p50={cols['aspect']['p50']:.3f}")

    lines.append("\n--- 2D Keypoint distribution within bbox (u=horizontal, v=vertical) ---")
    for key_base, label in [("gt", "GT"), ("mp", "MediaPipe"), ("wl", "WiLoR")]:
        u_key, v_key = f"{key_base}_kp_u", f"{key_base}_kp_v"
        if u_key not in kp or v_key not in kp:
            continue
        su, sv = kp[u_key], kp[v_key]
        lines.append(f"  {label} bbox: "
                     f"u=[{su['p5']:.3f}, {su['p95']:.3f}] p50={su['p50']:.3f}  "
                     f"v=[{sv['p5']:.3f}, {sv['p95']:.3f}] p50={sv['p50']:.3f}")

    lines.append("\n--- Coverage check ---")
    for det_label, rel_key in [("MediaPipe vs GT", "mp_vs_gt"), ("WiLoR vs GT", "wl_vs_gt")]:
        if rel_key not in summary:
            continue
        s = summary[rel_key]
        if "scale_ratio" in s:
            ok = s["scale_ratio"]["p5"] >= 0.5 and s["scale_ratio"]["p95"] <= 1.5
            lines.append(f"  {det_label} scale p5-p95=[{s['scale_ratio']['p5']:.3f},{s['scale_ratio']['p95']:.3f}] "
                         f"covered by [0.5,1.5]: {'OK' if ok else 'GAP'}")
        if "center_shift_norm" in s:
            ok = s["center_shift_norm"]["p95"] <= 0.3
            lines.append(f"  {det_label} center_shift p95={s['center_shift_norm']['p95']:.3f} "
                         f"covered by 0.3: {'OK' if ok else 'GAP'}")
        if "dx_norm" in s and "dy_norm" in s:
            cx = max(abs(s["dx_norm"]["p5"]), abs(s["dx_norm"]["p95"]))
            cy = max(abs(s["dy_norm"]["p5"]), abs(s["dy_norm"]["p95"]))
            lines.append(f"  {det_label} dx_max={cx:.3f} dy_max={cy:.3f} "
                         f"vs center_shift=0.3: {'OK' if max(cx,cy)<=0.3 else 'GAP'}")

    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Analyze bbox distribution differences")
    p.add_argument("--dataset", required=True, choices=["AssemblyHands", "HOT3D"])
    p.add_argument("--split", required=True)
    p.add_argument("--wds-root", default="/data_0/renkaiwen/webdatasets2_remake")
    p.add_argument("--output-dir", default="example/bbox_analysis")
    p.add_argument("--num-samples", type=int, default=5000)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_name = f"{args.dataset.lower()}_{args.split}"
    output_path = osp.join(args.output_dir, f"bbox_stats_{output_name}.json")

    result = run_analysis(args.dataset, args.split, args.wds_root,
                          args.num_samples, output_path, args.device)

    report = analyze_jitter_coverage(result)
    print("\n" + report)

    report_path = osp.join(args.output_dir, f"coverage_report_{output_name}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
