"""Standalone left/right hand bounding-box detector.

This module uses the YOLO detector from WiLoR-mini and only returns hand
bounding boxes with left/right labels. It does not load WiLoR 3D pose models.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence, Union

import cv2
import numpy as np
import torch
from ultralytics import YOLO


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "weights" / "detector.pt"


@dataclass(frozen=True)
class HandDetection:
    """One hand detection in original image coordinates."""

    bbox: List[float]
    confidence: float
    class_id: int
    handedness: str
    is_right: bool

    def to_dict(self) -> dict:
        return asdict(self)


class HandBBoxDetector:
    """Detect hand bounding boxes and handedness with WiLoR-mini's YOLO model."""

    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        device: Union[str, int, None] = None,
        verbose: bool = False,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO detector model not found: {self.model_path}")

        self.model = self._load_yolo_model(self.model_path)
        if device is not None:
            self.model.to(device)
        self.verbose = verbose

    def detect(
        self,
        image: Union[str, Path, np.ndarray],
        conf: float = 0.3,
        iou: float = 0.45,
    ) -> List[HandDetection]:
        """Detect hands from an image path or OpenCV BGR image.

        Args:
            image: Image path or OpenCV-style BGR numpy array with shape HxWx3.
            conf: Confidence threshold. WiLoR-mini uses 0.3 by default.
            iou: NMS IoU threshold.

        Returns:
            A list of detections. Each bbox is [x1, y1, x2, y2] in pixels.
            class_id 0 means left hand, and class_id 1 means right hand.
        """
        image_input = self._load_image(image)
        result = self.model(image_input, conf=conf, iou=iou, verbose=self.verbose)[0]

        detections: List[HandDetection] = []
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        for bbox, score, class_id in zip(boxes, confs, classes):
            is_right = bool(class_id == 1)
            detections.append(
                HandDetection(
                    bbox=[float(v) for v in bbox.tolist()],
                    confidence=float(score),
                    class_id=int(class_id),
                    handedness="right" if is_right else "left",
                    is_right=is_right,
                )
            )
        return detections

    @staticmethod
    def _load_image(image: Union[str, Path, np.ndarray]) -> np.ndarray:
        if isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("image array must have shape HxWx3")
            return image

        image_path = Path(image)
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"failed to read image: {image_path}")
        return img

    @staticmethod
    def _load_yolo_model(model_path: Path) -> YOLO:
        """Load legacy Ultralytics checkpoints under PyTorch 2.6+.

        Ultralytics 8.1.x calls ``torch.load`` without passing ``weights_only``.
        PyTorch 2.6 changed that default to ``True``, which rejects older YOLO
        checkpoint objects. This detector only loads the local trusted
        ``weights/detector.pt`` or an explicitly provided replacement path.
        """

        original_torch_load = torch.load

        def torch_load_compat(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_torch_load(*args, **kwargs)

        torch.load = torch_load_compat
        try:
            return YOLO(str(model_path))
        finally:
            torch.load = original_torch_load


def detections_to_dicts(detections: Sequence[HandDetection]) -> List[dict]:
    """Convert dataclass detections to JSON-serializable dictionaries."""
    return [det.to_dict() for det in detections]


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect left/right hand bboxes.")
    parser.add_argument("image", help="Input image path.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="YOLO detector .pt path.")
    parser.add_argument("--device", default=None, help="Device, for example: cpu, cuda, cuda:0.")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--verbose", action="store_true", help="Show Ultralytics logs.")
    args = parser.parse_args()

    detector = HandBBoxDetector(args.model, device=args.device, verbose=args.verbose)
    detections = detector.detect(args.image, conf=args.conf, iou=args.iou)
    print(json.dumps(detections_to_dicts(detections), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
