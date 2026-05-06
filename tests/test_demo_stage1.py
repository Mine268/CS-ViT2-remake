from __future__ import annotations

"""Lightweight tests for the stage1 demo helpers."""

import numpy as np

import script.demo_stage1 as demo_stage1
from script.demo_stage1 import (
    CameraIntrinsics,
    Detection,
    WilorHandBBoxDetector,
    _build_single_frame_batch,
    _draw_overlay,
    _ensure_min_bbox_edge,
    _infer_input_type,
    _maybe_scale_detector_detections,
    _safe_stem,
)


def test_demo_infers_input_types_and_sanitizes_output_names():
    assert _infer_input_type("frame.png", "auto") == "image"
    assert _infer_input_type("clip.mp4", "auto") == "video"
    assert _infer_input_type("/tmp/demo/*.tar", "auto") == "wds"
    assert _safe_stem("seq/a b:0001") == "seq_a_b_0001"


def test_demo_single_frame_batch_has_preprocess_schema():
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    detection = Detection(
        bbox_xyxy=np.asarray([10, 8, 30, 28], dtype=np.float32),
        handedness="right",
        score=1.0,
        detector="bbox",
    )
    intrinsics = CameraIntrinsics(fx=500.0, fy=510.0, cx=24.0, cy=16.0)

    batch = _build_single_frame_batch(image, detection, intrinsics)

    assert batch["imgs"][0].shape == (1, 3, 32, 48)
    assert batch["hand_bbox"].shape == (1, 1, 4)
    assert batch["joint_img"].shape == (1, 1, 21, 2)
    assert batch["mano_pose"].shape == (1, 1, 48)
    assert batch["focal"][0, 0].tolist() == [500.0, 510.0]
    assert batch["princpt"][0, 0].tolist() == [24.0, 16.0]


def test_demo_bbox_min_edge_and_overlay_drawing():
    bbox = _ensure_min_bbox_edge(
        np.asarray([5.0, 5.0, 6.0, 6.0], dtype=np.float32),
        width=32,
        height=32,
        min_edge=8.0,
    )
    assert bbox[2] - bbox[0] >= 8.0
    assert bbox[3] - bbox[1] >= 8.0

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    detection = Detection(
        bbox_xyxy=np.asarray([4, 4, 24, 24], dtype=np.float32),
        handedness="right",
        score=0.9,
        detector="bbox",
    )
    prediction = {
        "joint_img": np.tile(np.asarray([[16.0, 16.0]], dtype=np.float32), (21, 1)),
        "vert_img": np.tile(np.asarray([[16.0, 16.0]], dtype=np.float32), (3, 1)),
    }

    overlay = _draw_overlay(
        image,
        detections=[detection],
        predictions=[prediction],
        mesh_edges=np.asarray([[0, 1], [1, 2]], dtype=np.int32),
        draw_mesh=True,
        draw_joints=True,
    )

    assert overlay.shape == (32, 32, 3)
    assert overlay.dtype == np.uint8
    assert int(overlay.sum()) > 0


def test_demo_scales_only_detector_bboxes_and_preserves_raw_bbox():
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    detections = [
        Detection(
            bbox_xyxy=np.asarray([10.0, 12.0, 20.0, 22.0], dtype=np.float32),
            handedness="right",
            score=0.8,
            detector="wilor",
        ),
        Detection(
            bbox_xyxy=np.asarray([10.0, 12.0, 20.0, 22.0], dtype=np.float32),
            handedness="right",
            score=1.0,
            detector="gt",
        ),
    ]

    scaled = _maybe_scale_detector_detections(
        detections,
        image_rgb=image,
        scale=1.4,
        wilor_scale=0.75,
        mediapipe_scale=1.05,
    )

    assert scaled[0].bbox_xyxy.tolist() == [8.0, 10.0, 22.0, 24.0]
    assert scaled[0].raw_bbox_xyxy.tolist() == [10.0, 12.0, 20.0, 22.0]
    assert scaled[1].bbox_xyxy.tolist() == [10.0, 12.0, 20.0, 22.0]
    assert scaled[1].raw_bbox_xyxy is None


def test_demo_uses_detector_specific_default_bbox_scales():
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    detections = [
        Detection(
            bbox_xyxy=np.asarray([10.0, 10.0, 30.0, 30.0], dtype=np.float32),
            handedness="right",
            score=0.8,
            detector="wilor",
        ),
        Detection(
            bbox_xyxy=np.asarray([10.0, 10.0, 30.0, 30.0], dtype=np.float32),
            handedness="right",
            score=0.8,
            detector="mediapipe",
        ),
    ]

    scaled = _maybe_scale_detector_detections(
        detections,
        image_rgb=image,
        scale=None,
        wilor_scale=0.75,
        mediapipe_scale=1.05,
    )

    assert scaled[0].bbox_xyxy.tolist() == [12.5, 12.5, 27.5, 27.5]
    assert scaled[1].bbox_xyxy.tolist() == [9.5, 9.5, 30.5, 30.5]


def test_wilor_detector_wrapper_converts_rgb_and_normalizes_detections(monkeypatch):
    class FakeRawDetection:
        def __init__(self, bbox, handedness, confidence):
            self.bbox = bbox
            self.handedness = handedness
            self.confidence = confidence

    class FakeDetector:
        def __init__(self, model_path, device, verbose):
            self.model_path = model_path
            self.device = device
            self.verbose = verbose
            self.last_image = None
            self.last_conf = None
            self.last_iou = None

        def detect(self, image, conf, iou):
            self.last_image = image.copy()
            self.last_conf = conf
            self.last_iou = iou
            return [
                FakeRawDetection([-5.0, 1.0, 60.0, 20.0], "Right", 0.2),
                FakeRawDetection([2.0, 3.0, 9.0, 12.0], "Left", 0.9),
            ]

    fake_holder = {}

    def fake_hand_bbox_detector(model_path, device, verbose):
        fake = FakeDetector(model_path, device, verbose)
        fake_holder["detector"] = fake
        return fake

    monkeypatch.setattr(demo_stage1, "_load_wilor_detector_class", lambda: fake_hand_bbox_detector)
    image_rgb = np.zeros((16, 32, 3), dtype=np.uint8)
    image_rgb[..., 0] = 11
    image_rgb[..., 2] = 33

    detector = WilorHandBBoxDetector(
        model_path="/tmp/fake.pt",
        device="cpu",
        max_num_hands=1,
        conf=0.25,
        iou=0.55,
        verbose=True,
    )
    detections = detector.detect(image_rgb)

    assert len(detections) == 1
    assert detections[0].handedness == "left"
    assert detections[0].detector == "wilor"
    assert detections[0].score == 0.9
    assert detections[0].bbox_xyxy.tolist() == [2.0, 3.0, 9.0, 12.0]
    assert fake_holder["detector"].last_conf == 0.25
    assert fake_holder["detector"].last_iou == 0.55
    assert fake_holder["detector"].last_image[0, 0].tolist() == [33, 0, 11]
