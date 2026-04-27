from __future__ import annotations

"""Render qualitative overlays for the AssemblyHands validation split."""

import json
from pathlib import Path
import random
import sys
import tarfile
from typing import Dict, List, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.constant import MANO_JOINTS_CONNECTION


DATASET_ROOT = Path("/mnt/qnap/data/datasets/AssemblyHands")
ANNOT_ROOT = DATASET_ROOT / "annotations" / "val"
IMAGE_ROOT = DATASET_ROOT / "ego_images_rectified" / "val"
OUTPUT_DIR = REPO_ROOT / "temp" / "assemblyhands_val_check"

# AssemblyHands stores one hand as [thumb4, thumb3, thumb2, thumb1, index4, ..., pinky1, wrist].
# The project expects [wrist, thumb1, thumb2, thumb3, thumb4, index1, ..., pinky4].
PROJECT_HAND_ORDER_FROM_AH = [20, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16]


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_camera_key(camera_name: str, camera_keys: List[str]) -> str | None:
    if camera_name in camera_keys:
        return camera_name
    candidate = f"{camera_name}_mono10bit"
    if candidate in camera_keys:
        return candidate
    return None


def _build_tar_path(seq_name: str) -> Path:
    """Map a validation sequence name to the tar.gz archive that stores its ego images."""
    prefix = seq_name[: -6] if seq_name.endswith("_164345") else seq_name
    matches = sorted(IMAGE_ROOT.glob(f"{prefix}*.tar.gz"))
    if len(matches) == 0:
        raise FileNotFoundError(f"No val tar.gz found for sequence {seq_name}")
    return matches[0]


def _read_image_from_tar(seq_name: str, camera_key: str, frame_idx: int) -> np.ndarray | None:
    """Read one JPEG frame directly from the AssemblyHands validation tar.gz archive."""
    tar_path = _build_tar_path(seq_name)
    member_name = f"{seq_name}/{camera_key}/{frame_idx:06d}.jpg"
    with tarfile.open(tar_path, "r:*") as tar:
        extracted = tar.extractfile(member_name)
        if extracted is None:
            return None
        payload = extracted.read()
    arr = np.frombuffer(payload, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _project_points(world_coord: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    points_h = np.concatenate([world_coord.astype(np.float64), np.ones((world_coord.shape[0], 1))], axis=1)
    cam = (extrinsic @ points_h.T).T
    uv = (intrinsic[:2, :2] @ (cam[:, :2] / np.clip(cam[:, 2:3], a_min=1e-8, a_max=None)).T).T + intrinsic[:2, 2]
    return uv


def _draw_hand(
    img: np.ndarray,
    points: np.ndarray,
    valid: np.ndarray,
    color: Tuple[int, int, int],
    radius: int = 2,
) -> np.ndarray:
    points_int = np.round(points).astype(np.int32)
    for pre, nex in MANO_JOINTS_CONNECTION:
        if valid[pre] <= 0 or valid[nex] <= 0:
            continue
        img = cv2.line(img, tuple(points_int[pre]), tuple(points_int[nex]), color=color, thickness=1)
    for idx, point in enumerate(points_int):
        if valid[idx] <= 0:
            continue
        img = cv2.circle(img, tuple(point), radius=radius, color=color, thickness=-1)
    return img


def _remap_assemblyhands_hand_order(values: np.ndarray) -> np.ndarray:
    """Reorder one 21-joint hand from AssemblyHands order into the project's joint order."""
    return values[PROJECT_HAND_ORDER_FROM_AH]


def _put_legend(img: np.ndarray) -> np.ndarray:
    legend_items = [
        ("GT 2D Right", (0, 255, 0)),
        ("GT 2D Left", (0, 180, 0)),
        ("Proj 3D Right", (0, 0, 255)),
        ("Proj 3D Left", (180, 0, 255)),
        ("BBox Right", (255, 255, 0)),
        ("BBox Left", (255, 180, 0)),
    ]
    y = 20
    for label, color in legend_items:
        cv2.rectangle(img, (10, y - 8), (22, y + 4), color, thickness=-1)
        cv2.putText(img, label, (30, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        y += 18
    return img


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ego = _load_json(ANNOT_ROOT / "assemblyhands_val_ego_data_v1-1.json")
    calib_raw = _load_json(ANNOT_ROOT / "assemblyhands_val_ego_calib_v1-1.json")
    j3d_raw = _load_json(ANNOT_ROOT / "assemblyhands_val_joint_3d_v1-1.json")

    calib = calib_raw["calibration"] if "calibration" in calib_raw else {k: v for k, v in calib_raw.items() if k != "info"}
    joint_3d = j3d_raw["annotations"] if "annotations" in j3d_raw else {k: v for k, v in j3d_raw.items() if k != "info"}
    annotations_by_image_id = {int(ann["image_id"]): ann for ann in ego["annotations"]}

    valid_candidates = []
    for image_info in ego["images"]:
        ann2d = annotations_by_image_id[int(image_info["id"])]
        seq_name = str(image_info["seq_name"])
        frame_key = str(ann2d["frame_id"]).zfill(6)
        camera_name = str(image_info["camera"])

        if seq_name not in calib or seq_name not in joint_3d or frame_key not in joint_3d[seq_name]:
            continue
        camera_key = _resolve_camera_key(camera_name, list(calib[seq_name]["intrinsics"].keys()))
        if camera_key is None or frame_key not in calib[seq_name]["extrinsics"]:
            continue

        valid = np.asarray(ann2d["joint_valid"], dtype=np.float32)
        if valid.sum() <= 0:
            continue

        valid_candidates.append((image_info, ann2d, camera_key))

    random.seed(42)
    random.shuffle(valid_candidates)
    vis_count = min(6, len(valid_candidates))
    outputs = []
    for idx in range(vis_count):
        image_info, ann2d, camera_key = valid_candidates[idx]
        seq_name = str(image_info["seq_name"])
        frame_key = str(ann2d["frame_id"]).zfill(6)

        image = _read_image_from_tar(seq_name, camera_key, int(image_info["frame_idx"]))
        if image is None:
            continue

        keypoints_2d = np.asarray(ann2d["keypoints"], dtype=np.float64)[:, :2]
        keypoints_valid = np.asarray(ann2d["joint_valid"], dtype=np.float32)
        world_coord = np.asarray(joint_3d[seq_name][frame_key]["world_coord"], dtype=np.float64)
        K = np.asarray(calib[seq_name]["intrinsics"][camera_key], dtype=np.float64)
        Rt = np.asarray(calib[seq_name]["extrinsics"][frame_key][camera_key], dtype=np.float64)
        proj = _project_points(world_coord, K, Rt)

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

        for hand_name, color in [("right", (255, 255, 0)), ("left", (255, 180, 0))]:
            bbox = ann2d["bbox"].get(hand_name)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            image = cv2.rectangle(
                image,
                (int(round(x1)), int(round(y1))),
                (int(round(x2)), int(round(y2))),
                color,
                1,
            )

        image = _put_legend(image)
        out_path = OUTPUT_DIR / f"vis_{idx:02d}_{seq_name}_{camera_key}_{frame_key}.png"
        cv2.imwrite(str(out_path), image)
        outputs.append(str(out_path))

    summary = {
        "num_candidates": len(valid_candidates),
        "visualizations_written": outputs,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
