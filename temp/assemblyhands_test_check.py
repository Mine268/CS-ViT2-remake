from __future__ import annotations

"""Validate AssemblyHands test-eccv2024 annotations and render a few qualitative overlays."""

import json
from pathlib import Path
import random
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.constant import MANO_JOINTS_CONNECTION


DATASET_ROOT = Path("/mnt/qnap/data/datasets/AssemblyHands")
ANNOT_ROOT = DATASET_ROOT / "annotations" / "test-eccv2024"
OUTPUT_DIR = REPO_ROOT / "temp" / "assemblyhands_test_check"
PATH_PREFIX_OLD = "ego_images_rectified/test_HANDS2024/"
PATH_PREFIX_NEW = "ego_images_rectified/test-eccv2024/"
PROJECT_HAND_ORDER_FROM_AH = [20, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16]


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_image_path(relative_path: str) -> Path:
    if relative_path.startswith(PATH_PREFIX_OLD):
        relative_path = relative_path.replace(PATH_PREFIX_OLD, PATH_PREFIX_NEW, 1)
    return DATASET_ROOT / relative_path


def _resolve_camera_key(camera_name: str, camera_keys: List[str]) -> str | None:
    """Match the ego_data camera field against calibration keys."""
    if camera_name in camera_keys:
        return camera_name
    candidate = f"{camera_name}_mono10bit"
    if candidate in camera_keys:
        return candidate
    return None


def _project_points(world_coord: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    """Project world-space joints into the ego image using `K * [R|t]`."""
    points_h = np.concatenate([world_coord.astype(np.float64), np.ones((world_coord.shape[0], 1))], axis=1)
    cam = (extrinsic @ points_h.T).T
    xy = cam[:, :2] / np.clip(cam[:, 2:3], a_min=1e-8, a_max=None)
    uv = (intrinsic[:2, :2] @ xy.T).T + intrinsic[:2, 2]
    return uv


def _draw_hand(img: np.ndarray, points: np.ndarray, valid: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    points_int = points.astype(np.int32)
    for pre, nex in MANO_JOINTS_CONNECTION:
        if valid[pre] <= 0 or valid[nex] <= 0:
            continue
        img = cv2.line(img, tuple(points_int[pre]), tuple(points_int[nex]), color=color, thickness=1)
    for idx, point in enumerate(points_int):
        if valid[idx] <= 0:
            continue
        img = cv2.circle(img, tuple(point), radius=2, color=color, thickness=-1)
    return img


def _remap_assemblyhands_hand_order(values: np.ndarray) -> np.ndarray:
    """Reorder one 21-joint hand from AssemblyHands order into the project's joint order."""
    return values[PROJECT_HAND_ORDER_FROM_AH]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ego_data = _load_json(ANNOT_ROOT / "assemblyhands_test_ego_data_v1-1.json")
    ego_calib = _load_json(ANNOT_ROOT / "assemblyhands_test_ego_calib_v1-1.json")
    joint_3d = _load_json(ANNOT_ROOT / "assemblyhands_test_joint_3d_v1-1.json")

    images = ego_data["images"]
    annotations = ego_data["annotations"]
    annotations_by_image_id = {int(ann["image_id"]): ann for ann in annotations}
    calib = ego_calib["calibration"]
    joint_3d_ann = joint_3d["annotations"]

    stats = {
        "num_images": len(images),
        "num_annotations_2d": len(annotations),
        "num_sequences_calib": len(calib),
        "num_sequences_3d": len(joint_3d_ann),
        "path_prefix_mismatch_count": 0,
        "missing_image_count": 0,
        "missing_2d_annotation_count": 0,
        "missing_calib_seq_count": 0,
        "missing_calib_cam_count": 0,
        "missing_calib_frame_count": 0,
        "missing_joint3d_seq_count": 0,
        "missing_joint3d_frame_count": 0,
        "joint_count_mismatch_count": 0,
    }

    valid_visualization_candidates: List[Tuple[Dict, Dict, np.ndarray, np.ndarray, np.ndarray]] = []

    for image_info in images:
        image_id = int(image_info["id"])
        relative_path = str(image_info["file_name"])
        if relative_path.startswith(PATH_PREFIX_OLD):
            stats["path_prefix_mismatch_count"] += 1
        image_path = _resolve_image_path(relative_path)
        if not image_path.exists():
            stats["missing_image_count"] += 1
            continue

        ann2d = annotations_by_image_id.get(image_id)
        if ann2d is None:
            stats["missing_2d_annotation_count"] += 1
            continue

        seq_name = str(image_info["seq_name"])
        frame_key = str(ann2d["frame_id"]).zfill(6)
        camera_name = str(image_info["camera"])

        if seq_name not in calib:
            stats["missing_calib_seq_count"] += 1
            continue
        camera_key = _resolve_camera_key(camera_name, list(calib[seq_name]["intrinsics"].keys()))
        if camera_key is None:
            stats["missing_calib_cam_count"] += 1
            continue
        if frame_key not in calib[seq_name]["extrinsics"]:
            stats["missing_calib_frame_count"] += 1
            continue

        if seq_name not in joint_3d_ann:
            stats["missing_joint3d_seq_count"] += 1
            continue
        if frame_key not in joint_3d_ann[seq_name]:
            stats["missing_joint3d_frame_count"] += 1
            continue

        world_coord = np.asarray(joint_3d_ann[seq_name][frame_key]["world_coord"], dtype=np.float64)
        joint_valid = np.asarray(joint_3d_ann[seq_name][frame_key]["joint_valid"], dtype=np.float32)
        keypoints_2d = np.asarray(ann2d["keypoints"], dtype=np.float64)[:, :2]
        keypoints_valid = np.asarray(ann2d["joint_valid"], dtype=np.float32)
        if not (world_coord.shape == (42, 3) and joint_valid.shape == (42,) and keypoints_2d.shape == (42, 2)):
            stats["joint_count_mismatch_count"] += 1
            continue

        K = np.asarray(calib[seq_name]["intrinsics"][camera_key], dtype=np.float64)
        Rt = np.asarray(calib[seq_name]["extrinsics"][frame_key][camera_key], dtype=np.float64)
        proj = _project_points(world_coord, K, Rt)

        if len(valid_visualization_candidates) < 64:
            valid_visualization_candidates.append((image_info, ann2d, proj, keypoints_2d, keypoints_valid))

    random.seed(42)
    random.shuffle(valid_visualization_candidates)
    vis_count = min(6, len(valid_visualization_candidates))
    vis_outputs = []
    for vis_idx in range(vis_count):
        image_info, ann2d, proj, keypoints_2d, keypoints_valid = valid_visualization_candidates[vis_idx]
        image = cv2.imread(str(_resolve_image_path(image_info["file_name"])))
        if image is None:
            continue

        right_slice = slice(0, 21)
        left_slice = slice(21, 42)
        image = _draw_hand(
            image,
            _remap_assemblyhands_hand_order(keypoints_2d[right_slice]),
            _remap_assemblyhands_hand_order(keypoints_valid[right_slice]),
            (0, 255, 0),
        )
        image = _draw_hand(
            image,
            _remap_assemblyhands_hand_order(keypoints_2d[left_slice]),
            _remap_assemblyhands_hand_order(keypoints_valid[left_slice]),
            (0, 180, 0),
        )
        image = _draw_hand(
            image,
            _remap_assemblyhands_hand_order(proj[right_slice]),
            _remap_assemblyhands_hand_order(keypoints_valid[right_slice]),
            (0, 0, 255),
        )
        image = _draw_hand(
            image,
            _remap_assemblyhands_hand_order(proj[left_slice]),
            _remap_assemblyhands_hand_order(keypoints_valid[left_slice]),
            (180, 0, 255),
        )

        bbox = ann2d["bbox"]
        for hand_name, color in [("right", (255, 255, 0)), ("left", (255, 180, 0))]:
            if hand_name not in bbox:
                continue
            if bbox[hand_name] is None:
                continue
            x1, y1, x2, y2 = bbox[hand_name]
            image = cv2.rectangle(
                image,
                (int(round(x1)), int(round(y1))),
                (int(round(x2)), int(round(y2))),
                color,
                1,
            )

        out_name = f"vis_{vis_idx:02d}_{image_info['seq_name']}_{image_info['camera']}_{image_info['frame_idx']}.png"
        out_path = OUTPUT_DIR / out_name
        cv2.imwrite(str(out_path), image)
        vis_outputs.append(str(out_path))

    summary = {
        **stats,
        "visualizations_written": vis_outputs,
    }
    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
