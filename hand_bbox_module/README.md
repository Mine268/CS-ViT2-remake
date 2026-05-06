# Left/Right Hand BBox Detector

This folder is a standalone hand bounding-box detector extracted from
WiLoR-mini. It detects hands and returns each hand bbox plus left/right label.
It does not load WiLoR 3D pose, MANO, ONNX Runtime, or other large model files.

## Folder Contents

```text
hand_bbox_module/
├── README.md
├── requirements.txt
├── hand_bbox_detector.py
└── weights/
    └── detector.pt
```

`weights/detector.pt` is the YOLO hand detector used by WiLoR-mini. Keep this
relative path if you want the default API and CLI commands to work unchanged.

## Install in Another Project

Copy the whole `hand_bbox_module/` folder into the target project, then install
dependencies in that project's Python environment:

```bash
cd /path/to/target_project
pip install -r hand_bbox_module/requirements.txt
```

For GPU inference, install a CUDA-compatible PyTorch build first if your
environment does not already provide one. CPU inference also works.

This vendored copy also handles PyTorch 2.6+ checkpoint loading by explicitly
using the legacy trusted-checkpoint path for the local YOLO detector weights.

## Quick CLI Check

Run detection on any image:

```bash
python hand_bbox_module/hand_bbox_detector.py /path/to/image.jpg --device cpu
```

Use GPU when available:

```bash
python hand_bbox_module/hand_bbox_detector.py /path/to/image.jpg --device cuda:0
```

The command prints JSON:

```json
[
  {
    "bbox": [158.0, 242.0, 329.0, 455.0],
    "confidence": 0.9143,
    "class_id": 1,
    "handedness": "right",
    "is_right": true
  }
]
```

## Python API

```python
import cv2
from hand_bbox_module.hand_bbox_detector import HandBBoxDetector

detector = HandBBoxDetector(device="cuda:0")  # or device="cpu"

image = cv2.imread("/path/to/image.jpg")  # BGR ndarray
detections = detector.detect(image, conf=0.3, iou=0.45)

for det in detections:
    print(det.bbox, det.handedness, det.confidence)
```

You can also pass an image path directly:

```python
detections = detector.detect("/path/to/image.jpg")
```

If the weight file is stored elsewhere, pass it explicitly:

```python
detector = HandBBoxDetector(model_path="/path/to/detector.pt", device="cpu")
```

## Output Definition

Each detection is a `HandDetection` dataclass:

- `bbox`: `[x1, y1, x2, y2]` in original image pixel coordinates.
- `confidence`: detector confidence score.
- `class_id`: `0` for left hand, `1` for right hand.
- `handedness`: `"left"` or `"right"`.
- `is_right`: boolean version of `class_id == 1`.

Important: left/right means the person's anatomical left/right hand, not the
left/right side of the image.

## Thresholds

Defaults match the WiLoR-mini pipeline:

- `conf=0.3`: confidence threshold.
- `iou=0.45`: non-maximum suppression IoU threshold.

Increase `conf` to reduce false positives. Decrease it to keep weaker hands.

## Source in the Original Project

The extracted logic comes from:

```text
WiLoR-mini/wilor_mini/pipelines/wilor_hand_pose3d_estimation_pipeline.py
```

Specifically, the original pipeline loads `pretrained_models/detector.pt` with
Ultralytics `YOLO`, runs it at the beginning of `predict()`, and reads
`boxes.xyxy`, `boxes.conf`, and `boxes.cls`.

## Troubleshooting

- `YOLO detector model not found`: keep `weights/detector.pt` in this folder or
  pass `model_path` explicitly.
- `failed to read image`: confirm the image path exists and OpenCV supports the
  file format.
- CUDA errors: try `device="cpu"` first, or install a PyTorch build matching the
  local CUDA driver.
- No detections: lower `conf`, check that hands are visible, and verify the image
  is loaded correctly.
