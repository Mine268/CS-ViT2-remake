# 第 6 章：推理与 Demo 管线

> 本章覆盖推理流程：手部检测器（WiLoR/MediaPipe/GT/Fixed）、预处理、模型前向、输出格式和多输入类型支持。

---

## 6.1 Demo 脚本入口

`script/demo_stage1.py` (1341 行) — 支持 4 种输入类型的 Stage1 推理 demo。

### 6.1.1 命令行参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--input` | str | required | 图像/目录/视频/WDS tar/glob |
| `--input-type` | choice | `"auto"` | 自动推断或显式指定 |
| `--checkpoint` | str | required | Stage1 checkpoint 目录或 .safetensors |
| `--output-dir` | str | required | 输出目录 |
| `--intrinsics` | float×4 | None | FX FY CX CY |
| `--detector` | choice | `"wilor"` | 检测器类型 |
| `--detector-bbox-scale` | float | None | 统一检测器 bbox 缩放 |
| `--wilor-bbox-scale` | float | 0.75 | WiLoR 默认缩放 |
| `--mediapipe-bbox-scale` | float | 1.05 | MediaPipe 默认缩放 |
| `--max-frames` | int | None | 最大处理帧数 |
| `--batch-size` | int | 1 | WDS batch 大小 |
| `--device` | str | auto | CUDA/CPU |
| `--config-name` | str | `"stage1"` | Hydra 配置名 |
| `--draw-mesh` | bool | True | 绘制 mesh 叠图 |
| `--draw-joints` | bool | True | 绘制关节点 |
| `--save-npz` | bool | True | 保存 NPZ 预测 |
| `--save-overlays` | bool | True | 保存叠图 PNG |

### 6.1.2 输入类型自动推断

```python
def _infer_input_type(input_path, input_type):
    if input_type != "auto": return input_type
    if is_dir: return "folder"
    if suffix in IMAGE_EXTS: return "image"
    if suffix in VIDEO_EXTS: return "video"
    if ".tar" in path or glob_chars in path: return "wds"
    raise ValueError
```

---

## 6.2 手部检测器

### 6.2.1 WiLoR (默认推荐)

`script/demo_stage1.py:218-263` 中的 `WilorHandBBoxDetector`:

```python
class WilorHandBBoxDetector:
    def __init__(self, model_path, device, max_num_hands=2, conf=0.3, iou=0.45):
        self.detector = HandBBoxDetector(model_path, device)  # ultralytics YOLO-based

    def detect(self, image_rgb):
        dets = self.detector(image_bgr)  # 内部转 BGR
        dets = sorted(dets, key=lambda d: d["confidence"], reverse=True)
        return [Detection(bbox_xyxy=d["bbox"], handedness=d["handedness"],
                         score=d["confidence"], detector="wilor")]
```

- 权重路径: `hand_bbox_module/weights/detector.pt`
- 默认 confidence 阈值: 0.3
- 默认 IoU 阈值: 0.45
- **默认 bbox 缩放**: 0.75 (≤1.0: 向内收缩，减少背景，更匹配训练时的 tight bbox 分布)

### 6.2.2 MediaPipe

`script/demo_stage1.py:78-215` 中的 `MediaPipeHandDetector`:

支持两个 API 后端：
- `mediapipe.solutions` (legacy): `mp.solutions.hands.Hands.process()`
- `mediapipe.tasks` (new): `HandLandmarker.detect()` / `detect_for_video()`

从 hand landmarks 的 min/max 计算 bbox，读取 `multi_handedness` 获取左右手标签。

- **默认 bbox 缩放**: 1.05 (≥1.0: 向外扩展，因为 MediaPipe bbox 通常偏紧)

### 6.2.3 GT BBox (仅 WDS)

`script/demo_stage1.py:423-437` 中的 `GtBBoxDetector`:

直接从 WDS 样本的 `FrameItem.gt_bbox_xyxy` 读取真值 bbox，不需要外部检测器。仅用于调试和评估。

### 6.2.4 Fixed BBox

`script/demo_stage1.py:404-420` 中的 `FixedBBoxDetector`:

手动指定固定的 bbox 坐标 (`--bbox X1 Y1 X2 Y2`)，适用于已知手部位置的场景。

### 6.2.5 MediaPipe Keypoint BBox

`script/demo_stage1.py:282-401` 中的 `MediaPipeKeypointBBoxDetector`:

先用 MediaPipe 检测关键点，从关键点边界框构造 bbox，支持额外的 padding 比例。

### 6.2.6 BBox 缩放策略

```python
def _maybe_scale_detector_detections(detections, image_rgb, scale, wilor_scale, mediapipe_scale):
    for det in detections:
        if det.detector == "wilor":
            s = scale or wilor_scale  # default 0.75
        elif det.detector == "mediapipe":
            s = scale or mediapipe_scale  # default 1.05
        else:
            continue  # GT bbox 不缩放

        # 以 bbox 中心为锚点等比缩放
        bbox = _scale_bbox_about_center(det.bbox_xyxy, s)
        det.bbox_xyxy = bbox
        det.raw_bbox_xyxy = original  # 保留原始 bbox 用于记录
```

---

## 6.3 推理预处理

### 6.3.1 单帧 Batch 构造

```python
def _build_single_frame_batch(image_rgb, detection, intrinsics, device):
    # 构造完整的 batch 字典，模拟训练时的结构
    return {
        "imgs": [image_tensor],  # [1, 3, H, W]
        "hand_bbox": detection_bbox,  # [1, 1, 4]
        "joint_img": zeros(1, 1, 21, 2),
        "joint_cam": zeros(1, 1, 21, 3),
        "mano_pose": zeros(1, 1, 48),
        "mano_shape": zeros(1, 1, 10),
        "focal": intrinsics_focal,  # [1, 1, 2]
        "princpt": intrinsics_princpt,
        "timestamp": zeros(1, 1),
        "has_intr": ones(1, 1),
        ...
    }
```

### 6.3.2 预处理调用

```python
batch, _, _ = preprocess_batch(
    batch_origin, patch_size=[224, 224], patch_expanstion=2.0,
    scale_z_range=[1.0, 1.0], scale_f_range=[1.0, 1.0],
    persp_rot_max=0.0,
    augmentation_flag=False,  # 推理不增强！
    device=device, pixel_aug=None,
    perspective_normalization=False, bbox_jitter=None
)
```

推理预处理只做：square patch bbox → crop → resize → normalize。不使用任何数据增强。

---

## 6.4 模型前向

### 6.4.1 模型加载

```python
def _load_stage1_model(cfg, checkpoint, device):
    net = setup_model(cfg)  # PoseNet
    if checkpoint.endswith(".safetensors"):
        state_dict = load_file(checkpoint)
    else:
        state_dict = load_file(f"{checkpoint}/model.safetensors")
    net.load_state_dict(state_dict, strict=False)
    net.to(device)
    net.eval()  # 冻结所有 batch norm 和 dropout
```

### 6.4.2 预测

```python
def _predict_one_hand(net, cfg, image_rgb, detection, intrinsics, device):
    batch, _, _ = _build_and_preprocess(...)
    result = net.predict_full(
        img=batch["patches"],
        bbox=batch["patch_bbox"],
        focal=batch["focal"],
        princpt=batch["princpt"],
        hand_bbox=batch["hand_bbox"],
    )
    # result 包含 joint_cam_pred, vert_cam_pred, mano_pose_pred, mano_shape_pred, trans_pred
```

### 6.4.3 左手处理

```python
if detection.handedness == "left":
    joint_cam[..., 0] *= -1   # X 轴翻转
    vert_cam[..., 0] *= -1
    trans[0] *= -1
    mano_pose[..., 1:] *= -1  # 非根关节轴角取反
```

---

## 6.5 输出格式

### 6.5.1 输出目录结构

```
output_dir/
├── predictions.jsonl         # 每帧每手的预测记录
├── predictions_npz/          # 逐手 NPZ 文件
│   └── {frame_name}_hand{idx:02d}.npz
├── overlays/                 # 可视化叠图
│   └── {frame_name}.png
├── overlay.mp4               # 视频输入时的输出视频
├── summary.json              # 运行摘要
└── resolved_config.yaml      # 完整 Hydra 配置
```

### 6.5.2 predictions.jsonl 格式

每行一个手部预测：
```json
{
  "hand_index": 0,
  "handedness": "right",
  "detector": "wilor",
  "score": 0.95,
  "bbox_xyxy": [x1, y1, x2, y2],
  "bbox_xyxy_raw": [x1, y1, x2, y2],
  "bbox_scale": 0.75,
  "intrinsics_source": "cli",
  "intrinsics": {"fx": 900, "fy": 900, "cx": 640, "cy": 360},
  "npz_path": "predictions_npz/input_hand00.npz"
}
```

### 6.5.3 NPZ 文件内容

每个 `.npz` 包含：
- `joint_cam`: 相机空间 3D 关节 (21, 3)
- `vert_cam`: 相机空间 3D 顶点 (778, 3)
- `mano_pose`: 预测的 MANO 姿态 (48,) [axis-angle]
- `mano_shape`: 预测的 MANO 形状 (10,)
- `trans`: 相机平移 (3,)
- `joint_img`: 2D 关节点 (21, 2) [重投影]
- `vert_img`: 2D 顶点 (778, 2) [重投影]
- `bbox_xyxy`: 检测器输出的 bbox (4,)
- `handedness`: 左右手标签
- `score`: 检测置信度
- `focal`: 焦距 (2,)
- `princpt`: 主点 (2,)
- `faces`: MANO mesh 的面片索引 (1538, 3)

### 6.5.4 可视化叠图

```python
def vis(batch, trans_2d, result, tx, bx):
    # GT joints → 绿色圆圈 (通过 joint_2d_valid 过滤)
    # Pred joints → 红色圆圈
    # Bones → 灰色线 (按 MANO_JOINTS_CONNECTION 连接)
    # Mesh → 半透明 MANO mesh (如果 draw_mesh=True)
```

---

## 6.6 内参处理优先级

```python
def _frame_intrinsics(args, frame):
    # 1. CLI --intrinsics 参数 (最高优先级)
    if args.intrinsics: return CameraIntrinsics(fx, fy, cx, cy), "cli"

    # 2. WDS 样本中的 focal/princpt (仅 wds 输入)
    if frame.intrinsics: return frame.intrinsics, "wds"

    # 3. Fallback: 假设水平视角 ≈ 53° (focal = 1.2 × max(w,h))
    focal = default_focal_scale * max(w, h)  # 默认 1.2
    return CameraIntrinsics(focal, focal, w/2, h/2), "fallback"
```

---

## 6.7 Test/Inference 脚本

`script/test.py` (153 行) — 分布式批量推理：

### 6.7.1 输出格式

- 每个 rank 保存 `predictions_rank{rank:02d}.pt`
- 主进程合并为 `predictions.h5` (HDF5 格式, gzip 压缩)

### 6.7.2 HDF5 内容

```python
{
    "__key__": string array,       # 样本标识符
    "joint_cam_pred": float32,     # [N, 21, 3]
    "vert_cam_pred": float32,      # [N, 778, 3]
    "mano_pose_pred": float32,     # [N, 48]
    "mano_shape_pred": float32,    # [N, 10]
    "trans_pred": float32,         # [N, 3]
    "joint_cam_gt": float32,       # [N, 21, 3]  (如果存在)
    "focal": float32,              # [N, 2]
    "princpt": float32,            # [N, 2]
}
```

---

## 6.8 Export Sample 脚本

`script/export_sample.py` (72 行) — 定性导出前 8 个样本的可视化：

```python
for idx, batch in enumerate(val_loader):
    if idx >= 8: break
    batch, _, _ = preprocess_batch(batch, augmentation_flag=False, ...)
    output_state = net(batch)
    vis_img = vis(batch, ..., output_state["result"], tx=0, bx=0)
    cv2.imwrite(f"{output_dir}/sample_{idx:02d}.png", vis_img)
```
