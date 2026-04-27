from __future__ import annotations

"""Visual and numeric verification for exported AssemblyHands val clip-native WebDataset shards."""

import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
import webdataset as wds

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.constant import MANO_JOINTS_CONNECTION
from src.data.preprocess import preprocess_batch
from src.data.schema import normalize_decoded_clip_sample
from src.data.wds import preprocess_frame


SOURCE_GLOB = "/data_0/renkaiwen/webdatasets2_remake/AssemblyHands/val_stage2/*.tar"
OUTPUT_DIR = REPO_ROOT / "temp" / "assemblyhands_val_wds_check"
MAX_VIS_FRAMES = 8
RANDOM_SEED = 42


def _draw_hand(
    img: np.ndarray,
    joints: np.ndarray,
    valid: np.ndarray,
    color: tuple[int, int, int],
    radius: int = 2,
) -> np.ndarray:
    out = img.copy()
    joints = np.asarray(joints, dtype=np.float64)
    valid = np.asarray(valid, dtype=np.float32)
    drawable = (
        np.isfinite(joints).all(axis=-1)
        & (np.abs(joints) < 1e6).all(axis=-1)
        & (valid > 0.5)
    )
    points = np.zeros_like(joints, dtype=np.int32)
    if np.any(drawable):
        points[drawable] = np.round(joints[drawable]).astype(np.int32, copy=False)
    for pre, nex in MANO_JOINTS_CONNECTION:
        if not (drawable[pre] and drawable[nex]):
            continue
        out = cv2.line(out, tuple(points[pre]), tuple(points[nex]), color=color, thickness=1)
    for idx, point in enumerate(points):
        if not drawable[idx]:
            continue
        out = cv2.circle(out, tuple(point), radius=radius, color=color, thickness=-1)
    return out


def _decode_bgr_image(img_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image bytes from exported WebDataset sample")
    return image


def _project_joint_cam(joint_cam: np.ndarray, focal: np.ndarray, princpt: np.ndarray) -> np.ndarray:
    z = np.clip(joint_cam[:, 2:3], a_min=1e-8, a_max=None)
    return joint_cam[:, :2] / z * focal[None, :] + princpt[None, :]


def _crop_bbox(image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox.astype(np.float32)
    x1 = max(0, min(int(np.floor(x1)), max(w - 1, 0)))
    y1 = max(0, min(int(np.floor(y1)), max(h - 1, 0)))
    x2 = max(x1 + 1, min(int(np.ceil(x2)), w))
    y2 = max(y1 + 1, min(int(np.ceil(y2)), h))
    return image[y1:y2, x1:x2].copy()


def _tensor_patch_to_bgr(patch: torch.Tensor) -> np.ndarray:
    patch = torch.clamp(patch * 255.0, 0.0, 255.0).to(torch.uint8)
    rgb = patch.permute(1, 2, 0).cpu().numpy()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _draw_text_block(img: np.ndarray, lines: List[str], x: int = 12, y0: int = 20) -> np.ndarray:
    out = img.copy()
    for idx, line in enumerate(lines):
        y = y0 + idx * 18
        cv2.putText(
            out,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def _make_frame_record(sample: Dict[str, Any], tx: int) -> Dict[str, Any]:
    joint_2d_valid = np.asarray(sample["joint_2d_valid"][tx], dtype=np.float32)
    joint_3d_valid = np.asarray(sample["joint_3d_valid"][tx], dtype=np.float32)
    valid_2d_mask = joint_2d_valid > 0.5
    valid_reproj_mask = (joint_2d_valid > 0.5) & (joint_3d_valid > 0.5)

    joint_img = np.asarray(sample["joint_img"][tx], dtype=np.float32)
    joint_cam = np.asarray(sample["joint_cam"][tx], dtype=np.float32)
    focal = np.asarray(sample["focal"][tx], dtype=np.float32)
    princpt = np.asarray(sample["princpt"][tx], dtype=np.float32)
    reproj = _project_joint_cam(joint_cam, focal=focal, princpt=princpt)
    reproj_error = np.linalg.norm(reproj - joint_img, axis=-1)
    negative_depth_mask = joint_cam[:, 2] <= 1e-6

    if np.any(valid_reproj_mask):
        mean_reproj_error = float(np.mean(reproj_error[valid_reproj_mask]))
        max_reproj_error = float(np.max(reproj_error[valid_reproj_mask]))
        negative_depth_rate = float(np.mean(negative_depth_mask[valid_reproj_mask]))
    else:
        mean_reproj_error = 0.0
        max_reproj_error = 0.0
        negative_depth_rate = 0.0

    hand_bbox = np.asarray(sample["hand_bbox"][tx], dtype=np.float32)
    in_bbox_mask = (
        (joint_img[:, 0] >= hand_bbox[0])
        & (joint_img[:, 0] <= hand_bbox[2])
        & (joint_img[:, 1] >= hand_bbox[1])
        & (joint_img[:, 1] <= hand_bbox[3])
    )
    bbox_valid_mask = valid_2d_mask
    bbox_inside_rate = float(np.mean(in_bbox_mask[bbox_valid_mask])) if np.any(bbox_valid_mask) else 1.0

    joint_hand_bbox = np.asarray(sample["joint_hand_bbox"][tx], dtype=np.float32)
    joint_hand_recomputed = joint_img - hand_bbox[None, :2]
    joint_hand_bbox_error = np.linalg.norm(joint_hand_bbox - joint_hand_recomputed, axis=-1)
    joint_hand_bbox_max_error = float(np.max(joint_hand_bbox_error)) if len(joint_hand_bbox_error) > 0 else 0.0

    additional_desc = sample["additional_desc"][tx]
    source_index = sample["source_index"][tx]
    bad_3d_frame_masked = bool(additional_desc.get("bad_3d_frame_masked", False))
    return {
        "sample_key": sample["__key__"],
        "frame_in_chunk": tx,
        "frame_idx": int(source_index.get("frame_idx", tx)),
        "handedness": sample["handedness"],
        "joint_img_source": str(additional_desc.get("joint_img_source", "unknown")),
        "camera": str(source_index.get("camera", "unknown")),
        "seq_name": str(source_index.get("seq_name", "unknown")),
        "bad_3d_frame_masked": bad_3d_frame_masked,
        "bad_3d_mask_reason": str(additional_desc.get("bad_3d_mask_reason", "none")),
        "bad_3d_negative_depth_rate_raw": float(additional_desc.get("bad_3d_negative_depth_rate", 0.0)),
        "bad_3d_mean_reproj_error_px_raw": float(additional_desc.get("bad_3d_mean_reproj_error_px", 0.0)),
        "bad_3d_max_reproj_error_px_raw": float(additional_desc.get("bad_3d_max_reproj_error_px", 0.0)),
        "mean_reproj_error": mean_reproj_error,
        "max_reproj_error": max_reproj_error,
        "bbox_inside_rate": bbox_inside_rate,
        "joint_hand_bbox_max_error": joint_hand_bbox_max_error,
        "negative_depth_rate": negative_depth_rate,
        "num_valid_2d": int(np.sum(valid_2d_mask)),
        "num_valid_reproj": int(np.sum(valid_reproj_mask)),
    }


def _select_visualization_frames(frame_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    seen_keys: set[tuple[str, int]] = set()

    normal_frames = [
        item for item in frame_records if item["mean_reproj_error"] <= 1.0 and item["negative_depth_rate"] == 0.0
    ]
    masked_frames = [item for item in frame_records if item["bad_3d_frame_masked"]]

    for pool in (sorted(normal_frames, key=lambda item: item["mean_reproj_error"]),):
        if len(pool) == 0:
            continue
        for idx in [0, len(pool) // 2, len(pool) - 1]:
            item = pool[idx]
            key = (item["sample_key"], item["frame_in_chunk"])
            if key in seen_keys:
                continue
            selected.append(item)
            seen_keys.add(key)

    for item in sorted(
        masked_frames,
        key=lambda item: item["bad_3d_mean_reproj_error_px_raw"],
        reverse=True,
    ):
        if len(selected) >= MAX_VIS_FRAMES:
            break
        key = (item["sample_key"], item["frame_in_chunk"])
        if key in seen_keys:
            continue
        selected.append(item)
        seen_keys.add(key)
    return selected[:MAX_VIS_FRAMES]


def _render_frame(sample: Dict[str, Any], frame_record: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    tx = int(frame_record["frame_in_chunk"])
    image = _decode_bgr_image(sample["imgs_bytes"][tx])
    joint_img = np.asarray(sample["joint_img"][tx], dtype=np.float32)
    joint_2d_valid = np.asarray(sample["joint_2d_valid"][tx], dtype=np.float32)
    joint_3d_valid = np.asarray(sample["joint_3d_valid"][tx], dtype=np.float32)
    focal = np.asarray(sample["focal"][tx], dtype=np.float32)
    princpt = np.asarray(sample["princpt"][tx], dtype=np.float32)
    joint_cam = np.asarray(sample["joint_cam"][tx], dtype=np.float32)
    hand_bbox = np.asarray(sample["hand_bbox"][tx], dtype=np.float32)
    joint_hand_bbox = np.asarray(sample["joint_hand_bbox"][tx], dtype=np.float32)
    reproj = _project_joint_cam(joint_cam, focal=focal, princpt=princpt)
    reproj_valid = (joint_2d_valid > 0.5) & (joint_3d_valid > 0.5)

    origin = image.copy()
    cv2.rectangle(
        origin,
        (int(round(hand_bbox[0])), int(round(hand_bbox[1]))),
        (int(round(hand_bbox[2])), int(round(hand_bbox[3]))),
        (255, 255, 0),
        1,
    )
    origin = _draw_hand(origin, joint_img, joint_2d_valid, color=(0, 255, 0))
    origin = _draw_hand(origin, reproj, reproj_valid.astype(np.float32), color=(0, 0, 255))
    if frame_record["bad_3d_frame_masked"]:
        origin = _draw_hand(origin, reproj, joint_2d_valid, color=(255, 0, 255))
    origin = _draw_text_block(
        origin,
        [
            f"{frame_record['seq_name']} | {frame_record['camera']} | {frame_record['handedness']}",
            f"frame_idx={frame_record['frame_idx']} source={frame_record['joint_img_source']}",
            f"reproj_mean={frame_record['mean_reproj_error']:.3f}px max={frame_record['max_reproj_error']:.3f}px",
            f"bbox_inside_rate={frame_record['bbox_inside_rate']:.3f} hand_bbox_err={frame_record['joint_hand_bbox_max_error']:.6f}",
            f"negative_depth_rate={frame_record['negative_depth_rate']:.3f}",
            f"bad_3d_masked={frame_record['bad_3d_frame_masked']} reason={frame_record['bad_3d_mask_reason']}",
            f"raw_bad3d_mean={frame_record['bad_3d_mean_reproj_error_px_raw']:.3f}px",
        ],
    )

    crop = _crop_bbox(image, hand_bbox)
    crop = _draw_hand(crop, joint_hand_bbox, joint_2d_valid, color=(255, 200, 0))
    crop = _draw_text_block(crop, ["joint_hand_bbox on hand crop"], x=8, y0=18)

    batch_origin = preprocess_frame(sample)
    single_batch = {key: [value] if isinstance(value, str) else value for key, value in batch_origin.items()}
    for key, value in batch_origin.items():
        if isinstance(value, torch.Tensor):
            single_batch[key] = value.unsqueeze(0)
        elif isinstance(value, list):
            single_batch[key] = [value]
        elif isinstance(value, str):
            single_batch[key] = [value]
        else:
            single_batch[key] = [value]
    batch_out, _, _ = preprocess_batch(
        batch_origin=single_batch,
        patch_size=(224, 224),
        patch_expanstion=2.0,
        scale_z_range=(1.0, 1.0),
        scale_f_range=(1.0, 1.0),
        persp_rot_max=0.0,
        joint_rep_type="3",
        augmentation_flag=False,
        device=torch.device("cpu"),
        pixel_aug=None,
        perspective_normalization=False,
    )
    patch = _tensor_patch_to_bgr(batch_out["patches"][0, tx])
    patch_joints = batch_out["joint_patch_resized"][0, tx].cpu().numpy()
    patch_valid = batch_out["joint_2d_valid"][0, tx].cpu().numpy()
    patch = _draw_hand(patch, patch_joints, patch_valid, color=(0, 255, 255))
    patch = _draw_text_block(patch, ["preprocess patch + joint_patch_resized"], x=8, y0=18)

    frame_stem = f"{frame_record['joint_img_source']}__{frame_record['sample_key']}__t{tx:03d}"
    origin_path = output_dir / f"{frame_stem}__origin.png"
    crop_path = output_dir / f"{frame_stem}__crop.png"
    patch_path = output_dir / f"{frame_stem}__patch.png"
    meta_path = output_dir / f"{frame_stem}__meta.json"
    cv2.imwrite(str(origin_path), origin)
    cv2.imwrite(str(crop_path), crop)
    cv2.imwrite(str(patch_path), patch)
    meta_path.write_text(json.dumps(frame_record, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "origin": str(origin_path),
        "crop": str(crop_path),
        "patch": str(patch_path),
        "meta": str(meta_path),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(RANDOM_SEED)

    urls = sorted(
        str(path)
        for path in Path("/data_0/renkaiwen/webdatasets2_remake/AssemblyHands/val_stage2").glob("*.tar")
    )
    if len(urls) == 0:
        raise FileNotFoundError(f"No exported AssemblyHands val shards matched {SOURCE_GLOB}")

    frame_records: List[Dict[str, Any]] = []
    source_stats: Dict[str, Dict[str, float]] = {}
    camera_masked_counts: Dict[str, int] = {}
    all_mean_errors: List[float] = []
    all_max_errors: List[float] = []
    all_bbox_rates: List[float] = []
    all_hand_bbox_errors: List[float] = []
    all_negative_depth_rates: List[float] = []
    masked_3d_frames = 0
    total_samples = 0
    total_frames = 0

    dataset = wds.WebDataset(
        urls,
        shardshuffle=False,
        nodesplitter=lambda src: src,
        workersplitter=lambda src: src,
    ).decode()
    for decoded_sample in dataset:
        sample = normalize_decoded_clip_sample(decoded_sample)
        total_samples += 1
        total_frames += int(sample["num_frames"])
        for tx in range(int(sample["num_frames"])):
            frame_record = _make_frame_record(sample, tx)
            frame_records.append(frame_record)

            source_name = frame_record["joint_img_source"]
            stats = source_stats.setdefault(
                source_name,
                {
                    "num_frames": 0.0,
                    "mean_reproj_error_sum": 0.0,
                    "max_reproj_error_max": 0.0,
                    "bbox_inside_rate_sum": 0.0,
                    "joint_hand_bbox_max_error_max": 0.0,
                    "masked_bad_3d_frames": 0.0,
                },
            )
            stats["num_frames"] += 1.0
            stats["mean_reproj_error_sum"] += frame_record["mean_reproj_error"]
            stats["max_reproj_error_max"] = max(stats["max_reproj_error_max"], frame_record["max_reproj_error"])
            stats["bbox_inside_rate_sum"] += frame_record["bbox_inside_rate"]
            stats["joint_hand_bbox_max_error_max"] = max(
                stats["joint_hand_bbox_max_error_max"],
                frame_record["joint_hand_bbox_max_error"],
            )
            if frame_record["bad_3d_frame_masked"]:
                stats["masked_bad_3d_frames"] += 1.0
                masked_3d_frames += 1
                camera_masked_counts[frame_record["camera"]] = camera_masked_counts.get(frame_record["camera"], 0) + 1

            all_mean_errors.append(frame_record["mean_reproj_error"])
            all_max_errors.append(frame_record["max_reproj_error"])
            all_bbox_rates.append(frame_record["bbox_inside_rate"])
            all_hand_bbox_errors.append(frame_record["joint_hand_bbox_max_error"])
            all_negative_depth_rates.append(frame_record["negative_depth_rate"])

    vis_frames = _select_visualization_frames(frame_records)
    vis_targets: Dict[str, List[Dict[str, Any]]] = {}
    for frame_record in vis_frames:
        vis_targets.setdefault(frame_record["sample_key"], []).append(frame_record)

    rendered: List[Dict[str, Any]] = []
    render_dataset = wds.WebDataset(
        urls,
        shardshuffle=False,
        nodesplitter=lambda src: src,
        workersplitter=lambda src: src,
    ).decode()
    for decoded_sample in render_dataset:
        sample = normalize_decoded_clip_sample(decoded_sample)
        target_records = vis_targets.get(sample["__key__"])
        if not target_records:
            continue
        for frame_record in target_records:
            rendered.append(
                {
                    "frame_record": frame_record,
                    "files": _render_frame(
                        sample,
                        frame_record=frame_record,
                        output_dir=OUTPUT_DIR,
                    ),
                }
            )

    summarized_source_stats = {}
    for source_name, stats in source_stats.items():
        num_frames = max(stats["num_frames"], 1.0)
        summarized_source_stats[source_name] = {
            "num_frames": int(stats["num_frames"]),
            "mean_reproj_error": stats["mean_reproj_error_sum"] / num_frames,
            "max_reproj_error": stats["max_reproj_error_max"],
            "mean_bbox_inside_rate": stats["bbox_inside_rate_sum"] / num_frames,
            "max_joint_hand_bbox_error": stats["joint_hand_bbox_max_error_max"],
            "masked_bad_3d_frames": int(stats["masked_bad_3d_frames"]),
        }

    robust_errors = np.asarray(all_mean_errors, dtype=np.float64)
    normal_mask = robust_errors <= 1.0
    summary = {
        "num_samples": total_samples,
        "num_frames": total_frames,
        "overall": {
            "mean_reproj_error": float(np.mean(all_mean_errors)),
            "median_mean_reproj_error": float(np.median(all_mean_errors)),
            "p95_mean_reproj_error": float(np.percentile(all_mean_errors, 95)),
            "max_reproj_error": float(np.max(all_max_errors)),
            "mean_bbox_inside_rate": float(np.mean(all_bbox_rates)),
            "min_bbox_inside_rate": float(np.min(all_bbox_rates)),
            "max_joint_hand_bbox_error": float(np.max(all_hand_bbox_errors)),
            "num_frames_mean_reproj_error_gt_1px": int(np.sum(robust_errors > 1.0)),
            "num_frames_mean_reproj_error_gt_10px": int(np.sum(robust_errors > 10.0)),
            "num_frames_with_negative_depth": int(np.sum(np.asarray(all_negative_depth_rates) > 0.0)),
            "mean_reproj_error_normal_frames_only": float(np.mean(robust_errors[normal_mask])) if np.any(normal_mask) else 0.0,
            "masked_bad_3d_frames": masked_3d_frames,
        },
        "by_joint_img_source": summarized_source_stats,
        "masked_bad_3d_cameras": dict(
            sorted(camera_masked_counts.items(), key=lambda item: item[1], reverse=True)
        ),
        "visualizations": rendered,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
