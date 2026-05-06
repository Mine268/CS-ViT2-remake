from __future__ import annotations

"""Stage-1 demo for image, image-folder, video, and local WDS debugging inputs."""

import argparse
from dataclasses import dataclass
import json
import os
import os.path as osp
from pathlib import Path
import sys
from typing import Iterable, Iterator, Literal, Optional

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.constant import MANO_JOINT_COUNT, MANO_JOINTS_CONNECTION
from src.data.preprocess import preprocess_batch
from src.data.wds import get_dataloader
from src.train.engine import setup_model
from src.utils.misc import expand_glob_patterns
from src.utils.proj import proj_points_3d


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
DEFAULT_WDS_SOURCE = "/data_0/renkaiwen/webdatasets2_remake/AssemblyHands/val_stage1/*.tar"


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics in original image pixels."""

    fx: float
    fy: float
    cx: float
    cy: float

    def focal_np(self) -> np.ndarray:
        return np.asarray([self.fx, self.fy], dtype=np.float32)

    def princpt_np(self) -> np.ndarray:
        return np.asarray([self.cx, self.cy], dtype=np.float32)


@dataclass(frozen=True)
class Detection:
    """One detected hand bbox in original image coordinates."""

    bbox_xyxy: np.ndarray
    handedness: Literal["left", "right"]
    score: float
    detector: str
    raw_bbox_xyxy: Optional[np.ndarray] = None


@dataclass(frozen=True)
class FrameItem:
    """Input frame plus optional dataset metadata."""

    frame_index: int
    frame_name: str
    image_rgb: np.ndarray
    intrinsics: Optional[CameraIntrinsics] = None
    gt_bbox_xyxy: Optional[np.ndarray] = None
    gt_handedness: Optional[Literal["left", "right"]] = None
    sample_key: Optional[str] = None


class MediaPipeHandDetector:
    """Thin optional wrapper around MediaPipe Hands."""

    def __init__(
        self,
        static_image_mode: bool,
        max_num_hands: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
        mediapipe_selfie: bool,
        mediapipe_model: Optional[str],
    ):
        try:
            import mediapipe as mp  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "MediaPipe is not installed. Install it in the current environment, or run "
                "with `--detector gt` for WDS debug / `--detector bbox --bbox x1 y1 x2 y2`."
            ) from exc

        self._api = "solutions" if hasattr(mp, "solutions") else "tasks"
        del mediapipe_selfie
        if self._api == "solutions":
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=static_image_mode,
                max_num_hands=max_num_hands,
                model_complexity=1,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            return

        model_path = _resolve_mediapipe_task_model(mediapipe_model)
        from mediapipe.tasks.python.core import base_options
        from mediapipe.tasks.python.vision import HandLandmarker
        from mediapipe.tasks.python.vision import HandLandmarkerOptions
        from mediapipe.tasks.python.vision import RunningMode
        from mediapipe.tasks.python.vision.core import image as mp_image

        self._mp_image = mp_image
        running_mode = RunningMode.IMAGE if static_image_mode else RunningMode.VIDEO
        self._landmarker = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=base_options.BaseOptions(model_asset_path=model_path),
                running_mode=running_mode,
                num_hands=max_num_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        )
        self._timestamp_ms = 0

    def close(self) -> None:
        if self._api == "solutions":
            self._hands.close()
        else:
            self._landmarker.close()

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        if self._api == "tasks":
            return self._detect_tasks(image_rgb)
        return self._detect_solutions(image_rgb)

    def _detect_solutions(self, image_rgb: np.ndarray) -> list[Detection]:
        height, width = image_rgb.shape[:2]
        result = self._hands.process(image_rgb)
        if not result.multi_hand_landmarks:
            return []

        detections: list[Detection] = []
        handedness_items = result.multi_handedness or []
        for idx, landmarks in enumerate(result.multi_hand_landmarks):
            xs = np.asarray([lm.x * width for lm in landmarks.landmark], dtype=np.float32)
            ys = np.asarray([lm.y * height for lm in landmarks.landmark], dtype=np.float32)
            bbox = np.asarray([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)
            bbox = _clip_bbox(bbox, width=width, height=height)

            label = "right"
            score = 1.0
            if idx < len(handedness_items) and handedness_items[idx].classification:
                cls = handedness_items[idx].classification[0]
                label = str(cls.label).lower()
                score = float(cls.score)
            if label not in {"left", "right"}:
                label = "right"

            detections.append(
                Detection(
                    bbox_xyxy=bbox,
                    handedness=_normalize_handedness(label),
                    score=score,
                    detector="mediapipe",
                )
            )
        return detections

    def _detect_tasks(self, image_rgb: np.ndarray) -> list[Detection]:
        height, width = image_rgb.shape[:2]
        mp_image = self._mp_image.Image(
            image_format=self._mp_image.ImageFormat.SRGB,
            data=np.ascontiguousarray(image_rgb, dtype=np.uint8),
        )
        if self._api == "tasks":
            try:
                result = self._landmarker.detect(mp_image)
            except ValueError:
                self._timestamp_ms += 33
                result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)
        if not result.hand_landmarks:
            return []

        detections: list[Detection] = []
        for idx, landmarks in enumerate(result.hand_landmarks):
            xs = np.asarray([lm.x * width for lm in landmarks], dtype=np.float32)
            ys = np.asarray([lm.y * height for lm in landmarks], dtype=np.float32)
            bbox = _clip_bbox(
                np.asarray([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32),
                width=width,
                height=height,
            )
            label = "right"
            score = 1.0
            if idx < len(result.handedness) and len(result.handedness[idx]) > 0:
                cls = result.handedness[idx][0]
                label = str(cls.category_name).lower()
                score = float(cls.score)
            if label not in {"left", "right"}:
                label = "right"
            detections.append(
                Detection(
                    bbox_xyxy=bbox,
                    handedness=_normalize_handedness(label),
                    score=score,
                    detector="mediapipe",
                )
            )
        return detections


class WilorHandBBoxDetector:
    """WiLoR-mini YOLO hand bbox detector with anatomical handedness labels."""

    def __init__(
        self,
        model_path: Optional[str],
        device: str,
        max_num_hands: int,
        conf: float,
        iou: float,
        verbose: bool,
    ) -> None:
        detector_cls = _load_wilor_detector_class()
        resolved_model = str(_default_wilor_model_path() if model_path is None else model_path)
        self._detector = detector_cls(
            model_path=resolved_model,
            device=device,
            verbose=verbose,
        )
        self._max_num_hands = int(max_num_hands)
        self._conf = float(conf)
        self._iou = float(iou)

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        height, width = image_rgb.shape[:2]
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        raw_detections = self._detector.detect(image_bgr, conf=self._conf, iou=self._iou)
        raw_detections = sorted(raw_detections, key=lambda det: float(det.confidence), reverse=True)
        if self._max_num_hands > 0:
            raw_detections = raw_detections[: self._max_num_hands]

        detections: list[Detection] = []
        for det in raw_detections:
            detections.append(
                Detection(
                    bbox_xyxy=_clip_bbox(
                        np.asarray(det.bbox, dtype=np.float32),
                        width=width,
                        height=height,
                    ),
                    handedness=_normalize_handedness(str(det.handedness)),
                    score=float(det.confidence),
                    detector="wilor",
                )
            )
        return detections


def _load_wilor_detector_class():
    try:
        from hand_bbox_module.hand_bbox_detector import HandBBoxDetector
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "WiLoR hand bbox detector dependency is missing. Install demo dependencies with "
            "`uv pip install -r hand_bbox_module/requirements.txt`, or use "
            "`--detector mediapipe` / `--detector bbox` / `--detector gt`."
        ) from exc
    return HandBBoxDetector


def _default_wilor_model_path() -> Path:
    return REPO_ROOT / "hand_bbox_module" / "weights" / "detector.pt"


class FixedBBoxDetector:
    """Use one command-line bbox for every frame."""

    def __init__(self, bbox_xyxy: np.ndarray, handedness: Literal["left", "right"]):
        self._bbox_xyxy = bbox_xyxy.astype(np.float32, copy=True)
        self._handedness = handedness

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        height, width = image_rgb.shape[:2]
        return [
            Detection(
                bbox_xyxy=_clip_bbox(self._bbox_xyxy, width=width, height=height),
                handedness=self._handedness,
                score=1.0,
                detector="bbox",
            )
        ]


class GtBBoxDetector:
    """Use the hand bbox stored in the WDS sample."""

    def detect(self, frame: FrameItem) -> list[Detection]:
        if frame.gt_bbox_xyxy is None:
            raise ValueError("--detector gt requires WDS samples with `hand_bbox`")
        height, width = frame.image_rgb.shape[:2]
        return [
            Detection(
                bbox_xyxy=_clip_bbox(frame.gt_bbox_xyxy, width=width, height=height),
                handedness=frame.gt_handedness or "right",
                score=1.0,
                detector="gt",
            )
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the stage1 model on images/folders/videos and export 3D hand results."
    )
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Input image, image folder, video, or WDS tar/glob. For quick AH-val debug, use "
            f"`--input '{DEFAULT_WDS_SOURCE}' --input-type wds --detector gt`."
        ),
    )
    parser.add_argument(
        "--input-type",
        choices=["auto", "image", "folder", "video", "wds"],
        default="auto",
        help="Input type. `auto` infers image/folder/video from the path.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Stage1 checkpoint directory or raw model.safetensors path.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for overlays and results.")
    parser.add_argument(
        "--intrinsics",
        nargs=4,
        type=float,
        metavar=("FX", "FY", "CX", "CY"),
        help="Camera intrinsics in original image pixels. WDS input uses shard intrinsics by default.",
    )
    parser.add_argument(
        "--default-focal-scale",
        type=float,
        default=1.2,
        help=(
            "Fallback focal length is this value times max(width, height) when --intrinsics is "
            "not set for image/folder/video."
        ),
    )
    parser.add_argument(
        "--detector-bbox-scale",
        type=float,
        default=None,
        help=(
            "Scale bbox edge length around its center for detector-produced boxes before "
            "stage1 preprocessing. Applies only to `wilor` and `mediapipe`; `gt` and fixed "
            "`bbox` remain unchanged. If omitted, uses detector-specific defaults based on "
            "AH/HOT3D GT-bbox statistics."
        ),
    )
    parser.add_argument(
        "--wilor-bbox-scale",
        type=float,
        default=0.75,
        help="Default preprocess bbox scale for WiLoR detections when --detector-bbox-scale is omitted.",
    )
    parser.add_argument(
        "--mediapipe-bbox-scale",
        type=float,
        default=1.05,
        help=(
            "Default preprocess bbox scale for MediaPipe detections when "
            "--detector-bbox-scale is omitted."
        ),
    )
    parser.add_argument(
        "--detector",
        choices=["wilor", "mediapipe", "bbox", "gt"],
        default="wilor",
        help=(
            "Hand detector. `wilor` uses hand_bbox_module by default; `gt` is intended only for "
            "WDS dataset debugging."
        ),
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Fixed bbox for `--detector bbox`, in original image pixels.",
    )
    parser.add_argument(
        "--handedness",
        choices=["left", "right"],
        default="right",
        help="Handedness for `--detector bbox`.",
    )
    parser.add_argument(
        "--mediapipe-selfie",
        action="store_true",
        help=(
            "Reserved for compatibility. MediaPipe handedness is used after normalizing the label "
            "to lower-case `left`/`right`."
        ),
    )
    parser.add_argument(
        "--mediapipe-model",
        default=None,
        help=(
            "Path to MediaPipe Tasks hand_landmarker.task. Required for MediaPipe builds that "
            "only expose `mediapipe.tasks` instead of the legacy `mediapipe.solutions` API. "
            "If omitted, the script checks model/mediapipe/hand_landmarker.task and "
            "example/demo_stage1/hand_landmarker.task."
        ),
    )
    parser.add_argument(
        "--wilor-model",
        default=None,
        help=(
            "Path to WiLoR hand bbox detector .pt. If omitted, uses "
            "hand_bbox_module/weights/detector.pt."
        ),
    )
    parser.add_argument(
        "--wilor-conf",
        type=float,
        default=0.3,
        help="WiLoR hand bbox detector confidence threshold.",
    )
    parser.add_argument(
        "--wilor-iou",
        type=float,
        default=0.45,
        help="WiLoR hand bbox detector NMS IoU threshold.",
    )
    parser.add_argument(
        "--wilor-verbose",
        action="store_true",
        help="Show Ultralytics detector logs for the WiLoR hand bbox detector.",
    )
    parser.add_argument("--max-num-hands", type=int, default=2)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--recursive", action="store_true", help="Recursively scan image folders.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--config-name", default="stage1")
    parser.add_argument(
        "--config-override",
        action="append",
        default=[],
        help="Extra Hydra override, can be passed multiple times.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="WDS loader batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="WDS loader workers.")
    parser.add_argument("--stride", type=int, default=1, help="WDS clip stride.")
    parser.add_argument("--draw-mesh", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--draw-joints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-npz", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-overlays", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _clip_bbox(bbox_xyxy: np.ndarray, width: int, height: int) -> np.ndarray:
    bbox = bbox_xyxy.astype(np.float32, copy=True)
    bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0.0, float(max(width - 1, 0)))
    bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0.0, float(max(height - 1, 0)))
    if bbox[2] < bbox[0]:
        bbox[0], bbox[2] = bbox[2], bbox[0]
    if bbox[3] < bbox[1]:
        bbox[1], bbox[3] = bbox[3], bbox[1]
    return bbox


def _ensure_min_bbox_edge(bbox_xyxy: np.ndarray, width: int, height: int, min_edge: float = 8.0):
    bbox = _clip_bbox(bbox_xyxy, width=width, height=height)
    cx = float((bbox[0] + bbox[2]) * 0.5)
    cy = float((bbox[1] + bbox[3]) * 0.5)
    half = max(float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]), min_edge) * 0.5
    return _clip_bbox(
        np.asarray([cx - half, cy - half, cx + half, cy + half], dtype=np.float32),
        width=width,
        height=height,
    )


def _scale_bbox_about_center(
    bbox_xyxy: np.ndarray,
    width: int,
    height: int,
    scale: float,
) -> np.ndarray:
    bbox = _clip_bbox(bbox_xyxy, width=width, height=height)
    scale = float(scale)
    if scale <= 0:
        raise ValueError(f"bbox scale must be positive, got {scale}")
    if abs(scale - 1.0) < 1e-6:
        return bbox
    center = (bbox[:2] + bbox[2:]) * 0.5
    half_size = (bbox[2:] - bbox[:2]) * 0.5 * scale
    return _clip_bbox(
        np.concatenate([center - half_size, center + half_size]).astype(np.float32),
        width=width,
        height=height,
    )


def _maybe_scale_detector_detections(
    detections: list[Detection],
    image_rgb: np.ndarray,
    scale: Optional[float],
    wilor_scale: float,
    mediapipe_scale: float,
) -> list[Detection]:
    height, width = image_rgb.shape[:2]
    scaled: list[Detection] = []
    for det in detections:
        if det.detector not in {"wilor", "mediapipe"}:
            scaled.append(det)
            continue
        det_scale = float(
            scale
            if scale is not None
            else (wilor_scale if det.detector == "wilor" else mediapipe_scale)
        )
        if abs(det_scale - 1.0) < 1e-6:
            scaled.append(det)
            continue
        raw_bbox = det.raw_bbox_xyxy if det.raw_bbox_xyxy is not None else det.bbox_xyxy
        scaled.append(
            Detection(
                bbox_xyxy=_scale_bbox_about_center(
                    det.bbox_xyxy,
                    width=width,
                    height=height,
                    scale=det_scale,
                ),
                handedness=det.handedness,
                score=det.score,
                detector=det.detector,
                raw_bbox_xyxy=raw_bbox.astype(np.float32, copy=True),
            )
        )
    return scaled


def _detector_bbox_scale_for_record(args: argparse.Namespace, detector_name: str) -> float:
    if args.detector_bbox_scale is not None:
        return float(args.detector_bbox_scale)
    if detector_name == "wilor":
        return float(args.wilor_bbox_scale)
    if detector_name == "mediapipe":
        return float(args.mediapipe_bbox_scale)
    return 1.0


def _infer_input_type(input_path: str, input_type: str) -> str:
    if input_type != "auto":
        return input_type
    path = Path(input_path)
    if path.is_dir():
        return "folder"
    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    if ext == ".tar" or any(ch in input_path for ch in "*?[]"):
        return "wds"
    raise ValueError(f"Cannot infer input type from {input_path!r}; pass --input-type explicitly.")


def _load_image_rgb(path: str) -> np.ndarray:
    image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _iter_image_paths(input_path: str, recursive: bool) -> list[str]:
    root = Path(input_path)
    if root.is_file():
        return [str(root)]
    pattern = "**/*" if recursive else "*"
    paths = [
        str(path)
        for path in sorted(root.glob(pattern))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    if len(paths) == 0:
        raise ValueError(f"No images found under {input_path}")
    return paths


def iter_image_frames(input_path: str, recursive: bool) -> Iterator[FrameItem]:
    for idx, path in enumerate(_iter_image_paths(input_path, recursive=recursive)):
        yield FrameItem(
            frame_index=idx,
            frame_name=Path(path).stem,
            image_rgb=_load_image_rgb(path),
        )


def iter_video_frames(input_path: str, max_frames: Optional[int]) -> Iterator[FrameItem]:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {input_path}")
    frame_idx = 0
    try:
        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break
            ok, image_bgr = cap.read()
            if not ok:
                break
            yield FrameItem(
                frame_index=frame_idx,
                frame_name=f"frame_{frame_idx:06d}",
                image_rgb=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
            )
            frame_idx += 1
    finally:
        cap.release()


def iter_wds_frames(args: argparse.Namespace, cfg: DictConfig) -> Iterator[FrameItem]:
    sources = expand_glob_patterns([args.input])
    if len(sources) == 0:
        raise ValueError(f"No WDS shards matched {args.input!r}")
    loader = get_dataloader(
        url=sources,
        num_frames=cfg.MODEL.num_frame,
        stride=args.stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=1,
        infinite=False,
        seed=42,
        clip_sampling_mode="dense",
        clips_per_sequence=None,
        shardshuffle=False,
        post_clip_shuffle=0,
    )
    emitted = 0
    for batch in loader:
        batch_size = int(batch["hand_bbox"].shape[0])
        for bx in range(batch_size):
            if args.max_frames is not None and emitted >= args.max_frames:
                return
            tx = int(batch["imgs"][bx].shape[0] - 1)
            image_rgb = batch["imgs"][bx][tx].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            focal = batch["focal"][bx, tx].cpu().numpy().astype(np.float32)
            princpt = batch["princpt"][bx, tx].cpu().numpy().astype(np.float32)
            sample_key = str(batch["__key__"][bx])
            source_index = batch.get("source_index", [[{}]])[bx][tx]
            frame_index = int(source_index.get("frame_idx_within_clip", emitted))
            frame_name = f"{sample_key}_{frame_index:06d}"
            yield FrameItem(
                frame_index=emitted,
                frame_name=frame_name,
                image_rgb=image_rgb,
                intrinsics=CameraIntrinsics(
                    fx=float(focal[0]),
                    fy=float(focal[1]),
                    cx=float(princpt[0]),
                    cy=float(princpt[1]),
                ),
                gt_bbox_xyxy=batch["hand_bbox"][bx, tx].cpu().numpy().astype(np.float32),
                gt_handedness=_normalize_handedness(str(batch["handedness"][bx])),
                sample_key=sample_key,
            )
            emitted += 1


def _normalize_handedness(value: str) -> Literal["left", "right"]:
    return "left" if value.lower() == "left" else "right"


def _resolve_mediapipe_task_model(model_path: Optional[str]) -> str:
    candidates = []
    if model_path:
        candidates.append(Path(model_path))
    candidates.extend(
        [
            REPO_ROOT / "model" / "mediapipe" / "hand_landmarker.task",
            REPO_ROOT / "example" / "demo_stage1" / "hand_landmarker.task",
            Path.home() / ".cache" / "mediapipe" / "hand_landmarker.task",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    searched = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(
        "This MediaPipe installation exposes the Tasks API only, so a hand landmarker model "
        "asset is required. Download/provide `hand_landmarker.task` and pass "
        "`--mediapipe-model /path/to/hand_landmarker.task`, or place it at "
        "`model/mediapipe/hand_landmarker.task`.\n"
        f"Searched:\n{searched}"
    )


def _frame_intrinsics(
    args: argparse.Namespace,
    frame: FrameItem,
) -> tuple[CameraIntrinsics, str]:
    if args.intrinsics is not None:
        fx, fy, cx, cy = [float(v) for v in args.intrinsics]
        return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy), "cli"
    if frame.intrinsics is not None:
        return frame.intrinsics, "wds"
    height, width = frame.image_rgb.shape[:2]
    focal = float(max(width, height)) * float(args.default_focal_scale)
    return CameraIntrinsics(
        fx=focal,
        fy=focal,
        cx=float(width) * 0.5,
        cy=float(height) * 0.5,
    ), "fallback"


def _build_single_frame_batch(
    image_rgb: np.ndarray,
    detection: Detection,
    intrinsics: CameraIntrinsics,
) -> dict:
    height, width = image_rgb.shape[:2]
    bbox = _ensure_min_bbox_edge(detection.bbox_xyxy, width=width, height=height)
    img_tensor = torch.from_numpy(np.ascontiguousarray(image_rgb)).permute(2, 0, 1)
    zeros_joint_2d = torch.zeros((1, 1, 21, 2), dtype=torch.float32)
    zeros_joint_3d = torch.zeros((1, 1, 21, 3), dtype=torch.float32)
    zeros_valid = torch.zeros((1, 1, 21), dtype=torch.float32)
    return {
        "__key__": ["demo_frame"],
        "imgs_path": [["demo_frame"]],
        "imgs": [img_tensor.unsqueeze(0)],
        "handedness": [detection.handedness],
        "data_source": ["demo"],
        "source_split": ["demo"],
        "source_index": [[{"frame_idx_within_clip": 0}]],
        "intr_type": ["cli"],
        "additional_desc": [[{}]],
        "hand_bbox": torch.from_numpy(bbox).view(1, 1, 4),
        "joint_img": zeros_joint_2d,
        "joint_hand_bbox": zeros_joint_2d.clone(),
        "joint_cam": zeros_joint_3d,
        "joint_rel": zeros_joint_3d.clone(),
        "joint_2d_valid": zeros_valid,
        "joint_3d_valid": zeros_valid.clone(),
        "joint_valid": zeros_valid.clone(),
        "mano_pose": torch.zeros((1, 1, MANO_JOINT_COUNT * 3), dtype=torch.float32),
        "mano_shape": torch.zeros((1, 1, 10), dtype=torch.float32),
        "has_mano": torch.zeros((1, 1), dtype=torch.float32),
        "mano_valid": torch.zeros((1, 1), dtype=torch.float32),
        "has_intr": torch.ones((1, 1), dtype=torch.float32),
        "timestamp": torch.zeros((1, 1), dtype=torch.float32),
        "focal": torch.from_numpy(intrinsics.focal_np()).view(1, 1, 2),
        "princpt": torch.from_numpy(intrinsics.princpt_np()).view(1, 1, 2),
    }


def _load_cfg(config_name: str, overrides: list[str]) -> DictConfig:
    with hydra.initialize_config_dir(config_dir=str(REPO_ROOT / "config"), version_base=None):
        cfg = hydra.compose(config_name=config_name, overrides=overrides)
    cfg.MODEL.stage = "stage1"
    cfg.MODEL.num_frame = 1
    return cfg


def _load_stage1_model(cfg: DictConfig, checkpoint: str, device: torch.device) -> torch.nn.Module:
    net = setup_model(cfg)
    state_dict = load_file(_resolve_model_path(checkpoint), device="cpu")
    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[demo] missing checkpoint keys: {missing[:10]}", flush=True)
    if unexpected:
        print(f"[demo] unexpected checkpoint keys: {unexpected[:10]}", flush=True)
    net.to(device)
    net.eval()
    return net


def _resolve_model_path(checkpoint: str) -> str:
    path = Path(checkpoint)
    model_path = path if path.suffix == ".safetensors" else path / "model.safetensors"
    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint model file not found: {model_path}")
    return str(model_path)


@torch.inference_mode()
def _predict_one_hand(
    net: torch.nn.Module,
    cfg: DictConfig,
    image_rgb: np.ndarray,
    detection: Detection,
    intrinsics: CameraIntrinsics,
    device: torch.device,
) -> dict[str, np.ndarray]:
    batch_origin = _build_single_frame_batch(image_rgb, detection, intrinsics)
    batch, _, _ = preprocess_batch(
        batch_origin=batch_origin,
        patch_size=[cfg.MODEL.img_size, cfg.MODEL.img_size],
        patch_expanstion=cfg.TRAIN.expansion_ratio,
        scale_z_range=[1.0, 1.0],
        scale_f_range=[1.0, 1.0],
        persp_rot_max=0.0,
        joint_rep_type=cfg.MODEL.joint_type,
        augmentation_flag=False,
        device=device,
        pixel_aug=None,
        perspective_normalization=cfg.TRAIN.get("perspective_normalization", False),
    )
    result = net.predict_full(
        img=batch["patches"],
        bbox=batch["patch_bbox"],
        focal=batch["focal"],
        princpt=batch["princpt"],
        timestamp=batch["timestamp"],
        hand_bbox=batch["hand_bbox"],
    )

    joint_cam = result["joint_cam_pred"][0, 0].detach().float().cpu()
    vert_cam = result["vert_cam_pred"][0, 0].detach().float().cpu()
    mano_pose = result["mano_pose_pred"][0, 0].detach().float().cpu()
    mano_shape = result["mano_shape_pred"][0, 0].detach().float().cpu()
    trans = result["trans_pred_denorm"][0, 0].detach().float().cpu()

    if detection.handedness == "left":
        joint_cam[:, 0] *= -1.0
        vert_cam[:, 0] *= -1.0
        trans[0] *= -1.0
        if str(cfg.MODEL.joint_type) == "3":
            mano_pose = mano_pose.view(MANO_JOINT_COUNT, 3)
            mano_pose[:, 1:] *= -1.0
            mano_pose = mano_pose.reshape(-1)

    focal = torch.from_numpy(intrinsics.focal_np())
    princpt = torch.from_numpy(intrinsics.princpt_np())
    joint_img = proj_points_3d(joint_cam, focal, princpt).numpy()
    vert_img = proj_points_3d(vert_cam, focal, princpt).numpy()
    return {
        "joint_cam": joint_cam.numpy(),
        "vert_cam": vert_cam.numpy(),
        "mano_pose": mano_pose.numpy(),
        "mano_shape": mano_shape.numpy(),
        "trans": trans.numpy(),
        "joint_img": joint_img,
        "vert_img": vert_img,
    }


def _mesh_edges(faces: np.ndarray) -> np.ndarray:
    edges = set()
    for a, b, c in faces.astype(np.int32):
        edges.add(tuple(sorted((int(a), int(b)))))
        edges.add(tuple(sorted((int(b), int(c)))))
        edges.add(tuple(sorted((int(c), int(a)))))
    return np.asarray(sorted(edges), dtype=np.int32)


def _draw_overlay(
    image_rgb: np.ndarray,
    detections: list[Detection],
    predictions: list[dict[str, np.ndarray]],
    mesh_edges: np.ndarray,
    draw_mesh: bool,
    draw_joints: bool,
) -> np.ndarray:
    canvas = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
    mesh_layer = canvas.copy()

    for det, pred in zip(detections, predictions):
        bbox = np.round(det.bbox_xyxy).astype(np.int32)
        color = (80, 220, 255) if det.handedness == "right" else (255, 160, 80)
        cv2.rectangle(canvas, tuple(bbox[:2]), tuple(bbox[2:]), color, 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{det.handedness}:{det.score:.2f}",
            (int(bbox[0]), max(16, int(bbox[1]) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

        if draw_mesh:
            verts = pred["vert_img"]
            for edge in mesh_edges:
                pts = verts[edge]
                p0 = _cv_point_or_none(pts[0], image_rgb.shape)
                p1 = _cv_point_or_none(pts[1], image_rgb.shape)
                if p0 is None or p1 is None:
                    continue
                cv2.line(mesh_layer, p0, p1, (40, 180, 255), 1, cv2.LINE_AA)

        if draw_joints:
            joints = pred["joint_img"]
            for pre, nex in MANO_JOINTS_CONNECTION:
                p0 = _cv_point_or_none(joints[pre], image_rgb.shape)
                p1 = _cv_point_or_none(joints[nex], image_rgb.shape)
                if p0 is None or p1 is None:
                    continue
                cv2.line(canvas, p0, p1, (30, 30, 255), 2, cv2.LINE_AA)
            for joint in joints:
                center = _cv_point_or_none(joint, image_rgb.shape)
                if center is None:
                    continue
                cv2.circle(canvas, center, 3, (0, 255, 0), -1, cv2.LINE_AA)

    if draw_mesh:
        canvas = cv2.addWeighted(mesh_layer, 0.35, canvas, 0.65, 0.0)
    return canvas


def _cv_point_or_none(point_xy: np.ndarray, image_shape: tuple[int, ...]) -> Optional[tuple[int, int]]:
    if not np.isfinite(point_xy).all():
        return None
    height, width = image_shape[:2]
    margin = float(max(width, height) * 4)
    x = float(point_xy[0])
    y = float(point_xy[1])
    if x < -margin or x > width + margin or y < -margin or y > height + margin:
        return None
    return int(round(x)), int(round(y))


def _prediction_npz_path(output_dir: str, frame_name: str, hand_idx: int) -> str:
    pred_dir = osp.join(output_dir, "predictions_npz")
    os.makedirs(pred_dir, exist_ok=True)
    return osp.join(pred_dir, f"{_safe_stem(frame_name)}_hand{hand_idx:02d}.npz")


def _safe_stem(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    return safe.strip("._") or "frame"


def _write_prediction_npz(
    path: str,
    prediction: dict[str, np.ndarray],
    detection: Detection,
    intrinsics: CameraIntrinsics,
    faces: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        joint_cam=prediction["joint_cam"],
        vert_cam=prediction["vert_cam"],
        mano_pose=prediction["mano_pose"],
        mano_shape=prediction["mano_shape"],
        trans=prediction["trans"],
        joint_img=prediction["joint_img"],
        vert_img=prediction["vert_img"],
        bbox_xyxy=detection.bbox_xyxy,
        handedness=np.asarray(detection.handedness),
        score=np.asarray(detection.score, dtype=np.float32),
        focal=intrinsics.focal_np(),
        princpt=intrinsics.princpt_np(),
        faces=faces.astype(np.int32),
    )


def _open_video_writer(
    args: argparse.Namespace,
    input_type: str,
    first_frame_rgb: np.ndarray,
) -> Optional[cv2.VideoWriter]:
    if input_type != "video" or not args.save_video:
        return None
    os.makedirs(args.output_dir, exist_ok=True)
    height, width = first_frame_rgb.shape[:2]
    cap = cv2.VideoCapture(args.input)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()
    writer = cv2.VideoWriter(
        osp.join(args.output_dir, "overlay.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("Failed to create output video writer")
    return writer


def _build_frame_iterator(
    args: argparse.Namespace,
    cfg: DictConfig,
    input_type: str,
) -> Iterable[FrameItem]:
    if input_type in {"image", "folder"}:
        return iter_image_frames(args.input, recursive=args.recursive)
    if input_type == "video":
        return iter_video_frames(args.input, max_frames=args.max_frames)
    if input_type == "wds":
        return iter_wds_frames(args, cfg)
    raise ValueError(f"Unsupported input type: {input_type}")


def _build_detector(args: argparse.Namespace, input_type: str):
    if args.detector == "wilor":
        return WilorHandBBoxDetector(
            model_path=args.wilor_model,
            device=args.device,
            max_num_hands=args.max_num_hands,
            conf=args.wilor_conf,
            iou=args.wilor_iou,
            verbose=args.wilor_verbose,
        )
    if args.detector == "mediapipe":
        return MediaPipeHandDetector(
            static_image_mode=input_type == "image",
            max_num_hands=args.max_num_hands,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            mediapipe_selfie=args.mediapipe_selfie,
            mediapipe_model=args.mediapipe_model,
        )
    if args.detector == "bbox":
        if args.bbox is None:
            raise ValueError("--detector bbox requires --bbox X1 Y1 X2 Y2")
        return FixedBBoxDetector(
            np.asarray(args.bbox, dtype=np.float32),
            handedness=_normalize_handedness(args.handedness),
        )
    if args.detector == "gt":
        if input_type != "wds":
            raise ValueError("--detector gt is only supported with --input-type wds")
        return GtBBoxDetector()
    raise ValueError(f"Unsupported detector: {args.detector}")


def main() -> None:
    args = parse_args()
    input_type = _infer_input_type(args.input, args.input_type)
    os.makedirs(args.output_dir, exist_ok=True)
    overlay_dir = osp.join(args.output_dir, "overlays")
    if args.save_overlays:
        os.makedirs(overlay_dir, exist_ok=True)

    cfg = _load_cfg(args.config_name, args.config_override)
    cfg_yaml_path = osp.join(args.output_dir, "resolved_config.yaml")
    with open(cfg_yaml_path, "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    device = torch.device(args.device)
    net = _load_stage1_model(cfg, args.checkpoint, device=device)
    faces = np.asarray(net.rmano_layer.faces, dtype=np.int32)
    mesh_edges = _mesh_edges(faces)

    detector = _build_detector(args, input_type)
    frame_iter = iter(_build_frame_iterator(args, cfg, input_type))
    video_writer: Optional[cv2.VideoWriter] = None
    jsonl_path = osp.join(args.output_dir, "predictions.jsonl")
    total_frames = 0
    total_hands = 0

    try:
        with open(jsonl_path, "w", encoding="utf-8") as jsonl:
            for frame in frame_iter:
                if input_type != "video" and args.max_frames is not None:
                    if total_frames >= args.max_frames:
                        break

                intrinsics, intr_source = _frame_intrinsics(args, frame)
                if isinstance(detector, GtBBoxDetector):
                    detections = detector.detect(frame)
                else:
                    detections = detector.detect(frame.image_rgb)
                    detections = _maybe_scale_detector_detections(
                        detections,
                        image_rgb=frame.image_rgb,
                        scale=args.detector_bbox_scale,
                        wilor_scale=args.wilor_bbox_scale,
                        mediapipe_scale=args.mediapipe_bbox_scale,
                    )

                predictions: list[dict[str, np.ndarray]] = []
                records: list[dict] = []
                for hand_idx, detection in enumerate(detections):
                    pred = _predict_one_hand(
                        net=net,
                        cfg=cfg,
                        image_rgb=frame.image_rgb,
                        detection=detection,
                        intrinsics=intrinsics,
                        device=device,
                    )
                    predictions.append(pred)
                    npz_path = _prediction_npz_path(args.output_dir, frame.frame_name, hand_idx)
                    if args.save_npz:
                        _write_prediction_npz(
                            npz_path,
                            prediction=pred,
                            detection=detection,
                            intrinsics=intrinsics,
                            faces=faces,
                        )
                    records.append(
                        {
                            "hand_index": hand_idx,
                            "handedness": detection.handedness,
                            "detector": detection.detector,
                            "score": detection.score,
                            "bbox_xyxy": detection.bbox_xyxy.tolist(),
                            "bbox_xyxy_raw": (
                                detection.raw_bbox_xyxy.tolist()
                                if detection.raw_bbox_xyxy is not None
                                else detection.bbox_xyxy.tolist()
                            ),
                            "bbox_scale": (
                                _detector_bbox_scale_for_record(args, detection.detector)
                                if detection.raw_bbox_xyxy is not None
                                else 1.0
                            ),
                            "intrinsics": {
                                "fx": intrinsics.fx,
                                "fy": intrinsics.fy,
                                "cx": intrinsics.cx,
                                "cy": intrinsics.cy,
                                "source": intr_source,
                            },
                            "npz_path": npz_path if args.save_npz else None,
                            "trans_cam_mm": pred["trans"].tolist(),
                            "mano_pose_axis_angle": pred["mano_pose"].tolist(),
                            "mano_shape": pred["mano_shape"].tolist(),
                        }
                    )

                overlay_bgr = _draw_overlay(
                    frame.image_rgb,
                    detections=detections,
                    predictions=predictions,
                    mesh_edges=mesh_edges,
                    draw_mesh=args.draw_mesh,
                    draw_joints=args.draw_joints,
                )
                if video_writer is None:
                    video_writer = _open_video_writer(args, input_type, frame.image_rgb)
                if video_writer is not None:
                    video_writer.write(overlay_bgr)
                overlay_path = None
                if args.save_overlays:
                    overlay_path = osp.join(overlay_dir, f"{_safe_stem(frame.frame_name)}.png")
                    cv2.imwrite(overlay_path, overlay_bgr)

                jsonl.write(
                    json.dumps(
                        {
                            "frame_index": frame.frame_index,
                            "frame_name": frame.frame_name,
                            "sample_key": frame.sample_key,
                            "num_hands": len(records),
                            "overlay_path": overlay_path,
                            "hands": records,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                total_frames += 1
                total_hands += len(records)
                print(
                    f"[demo] frame={frame.frame_index} hands={len(records)} "
                    f"name={frame.frame_name}",
                    flush=True,
                )
    finally:
        if video_writer is not None:
            video_writer.release()
        close = getattr(detector, "close", None)
        if callable(close):
            close()

    summary = {
        "input": args.input,
        "input_type": input_type,
        "checkpoint": args.checkpoint,
        "output_dir": args.output_dir,
        "num_frames": total_frames,
        "num_hands": total_hands,
        "predictions_jsonl": jsonl_path,
        "config": cfg_yaml_path,
    }
    with open(osp.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
