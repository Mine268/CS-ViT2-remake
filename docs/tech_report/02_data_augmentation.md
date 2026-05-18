# 第 2 章：数据增强

> 本章覆盖所有训练期数据增强：bbox jitter（约束性/传统）、3D 几何增强、像素级增强、透视归一化。

---

## 2.1 增强概述

CS-ViT2-remake 的训练增强分为三个层次：

| 层次 | 增强类型 | 生效范围 | 配置路径 |
|------|---------|---------|----------|
| 几何增强 | BBox Jitter | Train only | `TRAIN.bbox_jitter` |
| 几何增强 | 3D 空间变换 | Train only | `TRAIN.scale_z_range`, `scale_f_range`, `persp_rot_max` |
| 像素增强 | ColorJitter 等 | Train only | `TRAIN.augmentation` |
| 透视归一化 | BBox center → 光轴 | Train only (可选) | `TRAIN.perspective_normalization` |

验证/测试/推理路径仅进行 patch crop + resize，不应用任何增强。

## 2.2 BBox Jitter

**设计目标**：模拟真实手部检测器（如 WiLoR、MediaPipe）输出的 bbox 噪声，使模型对检测器 bbox 的中心偏移和尺寸误差具有鲁棒性。

**代码位置**：`src/data/preprocess.py:466-579` 中的 `_jitter_hand_bbox`

### 2.2.1 公共参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `enabled` | bool | 开关 |
| `prob` | float (0.7) | 每个样本被 jitter 的概率 |
| `temporal_mode` | `"frame"` 或 `"clip"` | 时序采样策略 |
| `constrained` | bool (true) | 是否启用约束性采样 |
| `scale_range` | [min, max] (`[1.0, 2.0]`) | 膨胀系数范围 |
| `center_shift` | float (0.5) | 中心偏移上限（bbox 边长比例） |
| `aspect_ratio_range` | [min, max] (`[0.6, 1.6]`) | 宽高比扰动范围 |
| `min_edge_px` | float (8.0) | 最小 bbox 边长（像素） |

Stage2 额外参数：

| 参数 | Stage1 值 | Stage2 值 | 说明 |
|------|----------|----------|------|
| `temporal_mode` | `"frame"` | `"clip"` | clip 内共享 base jitter |
| `frame_center_shift` | 0.0 | 0.03 | 逐帧中心小扰动 |
| `frame_scale_range` | `[1.0, 1.0]` | `[0.95, 1.05]` | 逐帧尺度小扰动 |

### 2.2.2 时序模式

**`temporal_mode = "frame"`** (Stage1 默认)：每个帧独立采样 jitter 参数，`base_shape = (B, T)`。

**`temporal_mode = "clip"`** (Stage2 默认)：clip 内所有帧共享同一个 base jitter（`base_shape = (B, 1)`），然后叠加小幅逐帧噪声：
- `frame_center_shift`: 每帧独立采样的额外中心偏移上限
- `frame_scale_range`: 每帧独立采样的额外尺度因子范围

### 2.2.3 约束性模式 (`constrained=true`)

约束性 jitter 是**当前默认且推荐**的模式。核心思想：scale 膨胀因子 α 和 center shift δ 联合采样，满足几何包含约束 `|δ| ≤ (α−1)/2`。

**算法流程**：

1. 从 bbox 提取 `bbox_size`、`bbox_edge = max(bbox_w, bbox_h)`、`center`

2. 通过 apply_mask 决定哪些样本被 jitter：
   ```python
   apply_mask = rand(B, base_T) < prob  # 0.7 概率
   ```

3. **采样膨胀系数 α**：
   ```python
   alpha = exp(randn(B, base_T) * σ + μ)  # σ=0.15, μ=0.36
   alpha = clamp(alpha, 1.0, 2.0)         # scale_range
   base_scale = alpha.expand(B, T)
   ```
   α 服从对数正态分布，约 68% 的值在 `[exp(0.21), exp(0.51)] ≈ [1.23, 1.67]` 之间。

4. **计算最大允许偏移**：
   ```python
   max_shift = (alpha - 1.0) * 0.5  # 包含约束
   effective_cap = min(center_shift_cfg, max_shift)
   ```
   `center_shift_cfg=0.5` 是用户设定的上限。如果 α=1.2，则 `max_shift=0.1`，实际偏移上限为 `min(0.5, 0.1)=0.1`。

5. **采样中心偏移 δ**：
   ```python
   base_center_noise = (rand(B, T, 2) * 2 - 1) * effective_cap
   center_noisy = center + base_center_noise * bbox_edge
   ```
   δ 均匀分布在 `[-effective_cap, +effective_cap]` 区间（每个维度独立）。

6. **采样宽高比扰动**：
   ```python
   aspect = exp(log_uniform(base_shape, [0.6, 1.6]))
   size_noisy = bbox_size * base_scale
   size_noisy_x = size_noisy_x * sqrt(aspect)
   size_noisy_y = size_noisy_y / sqrt(aspect)
   ```

7. **裁剪到图像边界**：
   ```python
   _bbox_from_center_size_clipped(center_noisy, size_noisy, image_wh, min_edge_px=8.0)
   ```

8. 最后用 `torch.where(apply_mask, bbox_noisy, hand_bbox)` 混合：未中选样本保留原始 bbox。

### 2.2.4 传统模式 (`constrained=false`)

传统模式独立采样 scale 和 center shift，不施加包含约束：

```python
# 中心偏移
base_center_noise = (rand(B, T, 2) * 2 - 1) * center_shift
# Scale
base_scale = log_uniform(base_shape, scale_range)
```

此模式可能导致 bbox 中心漂移但 scale 不膨胀的不合理情况，**当前不推荐使用**。

### 2.2.5 效果验证

2026-05-08 demo 复盘：约束性 bbox jitter 显著提升了在 WiLoR detector 上的泛化能力。AssemblyHands val 上用 WiLoR detector 评测时，有 jitter 训练的模型漏检更少、姿态更准确。

---

## 2.3 3D 几何增强

**代码位置**：`src/data/preprocess.py:724-838` 中的 `preprocess_batch` 增强路径

### 2.3.1 随机旋转（绕 Z 轴）

```python
rad = rand(B, 1) * 2π  # [0, 2π)
rad = rad.expand(B, T)  # 所有帧共享
```

绕相机 Z 轴旋转相当于图像平面内旋转。

### 2.3.2 Z 方向缩放

```python
scale_z = rand(B, 1) * (max-min) + min
# stage1: scale_z_range=[1.0, 1.0] → 无缩放
```

模拟沿相机光轴的平移。`scale_z > 1` 使手部远离相机（缩小），`scale_z < 1` 使手部靠近相机（放大）。

### 2.3.3 内参增强

```python
scale_f = rand(B, 1) * (max-min) + min  # stage1: [1.0, 1.0] → 无变化
focal_new = focal * scale_f
princpt_noise = randn(B, 1, 2) * ||princpt|| * 0.1111
princpt_new = princpt + princpt_noise
```

模拟不同相机内参的变体。主点噪声幅度为原始主点范数的约 1/9。

### 2.3.4 透视旋转

```python
persp_dir_rad = rand(B, 1) * 2π          # 随机旋转轴方向
persp_rot_rad = rand(B, 1) * 0.0873      # 最大 5° (0.0873 rad)
persp_axis_angle = [cos(dir)*rot, sin(dir)*rot, 0]
```

模拟相机视角的微小变化。旋转轴在 XY 平面内（无 Z 分量），角度上限 5°。

### 2.3.5 3D 变换矩阵

`src/data/preprocess.py:137-174` 中的 `get_trans_3d_mat(rad, scale, axis_angle)` 构建 `[..., 3, 3]` 变换矩阵：

```python
mat[..., :2, :2] = [[cos, -sin], [sin, cos]]  # Z轴旋转
mat[..., 2, 2] = scale                          # Z轴缩放
if axis_angle is not None:
    mat = axis_angle_R @ mat                    # 全局旋转（后乘）
```

该矩阵应用于 `joint_cam` 和 `joint_rel`，也用于更新 MANO root rotation。

### 2.3.6 2D 透视变换矩阵

`src/data/preprocess.py:176-237` 中的 `get_trans_2d_mat` 构建图像空间透视变换：

```python
mat_2d = new_intr @ rot_scale @ old_intr_inv
```

其中：
- `old_intr_inv`：原始内参的逆矩阵
- `rot_scale`：Z 旋转 + 缩放（2D 比例因子为 `1/scale_z`，因为远距离 → 图像缩小）
- `new_intr`：新（增强后）内参矩阵

### 2.3.7 MANO Root Rotation 组合

`src/data/preprocess.py:818-838`：增强影响 MANO 姿态的根旋转：

```python
root_rot_mat = rot_z(rad)           # Z轴旋转矩阵
if correction_rot_mat:
    root_rot_mat = root_rot_mat @ correction_rot_mat
if persp_rot_max > 0:
    root_rot_mat = persp_rot_mat @ root_rot_mat
mano_root_new = root_rot_mat @ mano_root_old  # 组合旋转
```

---

## 2.4 像素级增强

**代码位置**：`src/data/preprocess.py:25-134` 中的 `PixelLevelAugmentation`

### 2.4.1 设计原则

- 通过配置文件控制各增强的开关和参数，便于消融实验
- 避免破坏 DINOv2 预训练分布的增强
- 避免与 heatmap 监督冲突的增强（如 RandomErasing）

### 2.4.2 可用增强列表

| 增强 | 类 | Stage1 状态 | Stage2 状态 |
|------|-----|-----------|-----------|
| ColorJitter | `KA.ColorJitter` | **enabled** | 继承 stage1 |
| GaussianNoise | `KA.RandomGaussianNoise` | **enabled** | 继承 stage1 |
| GaussianBlur | `KA.RandomGaussianBlur` | disabled | disabled |
| Sharpness | `KA.RandomSharpness` | disabled | disabled |
| Equalize | `KA.RandomEqualize` | disabled | disabled |
| MotionBlur | `KA.RandomMotionBlur` | disabled | disabled |
| RandomErasing | `KA.RandomErasing` | disabled (与 heatmap 冲突) | disabled |

### 2.4.3 当前配置

```yaml
TRAIN.augmentation:
  color_jitter:
    enabled: true
    brightness: 0.2
    contrast: 0.2
    saturation: 0.1
    hue: 0.0
    p: 0.5
  gaussian_noise:
    enabled: true
    mean: 0.0
    std: 0.03
    p: 0.5
```

### 2.4.4 Forward 实现

```python
def forward(self, input_tensor):  # [B, T, C, H, W]
    x = input_tensor.reshape(B*T, C, H, W)
    x = self.transforms(x)  # nn.Sequential of kornia augmentations
    x = x.reshape(B, T, C, H, W)
    return torch.clamp(x, 0.0, 1.0)
```

每帧独立应用（reshape 为 `B*T` 维），变换后 clamp 到 `[0, 1]`。

### 2.4.5 新增增强的方法

只需在配置文件的 `TRAIN.augmentation` 下添加对应键即可，模块会自动构建 pipeline。例如：

```yaml
TRAIN.augmentation:
  gaussian_blur:
    enabled: true
    kernel_size: [5, 5]
    sigma: [0.3, 1.0]
    p: 0.15
```

---

## 2.5 透视归一化 (Perspective Normalization)

**代码位置**：`src/data/preprocess.py:313-371` 中的 `compute_perspective_normalization_rotation`

### 2.5.1 目的

将 bbox 中心旋转到图像主点（光轴）方向。归一化后，手部在 crop patch 中始终位于中心，模型不需要学习 "手部在图像中的偏移" 这种几何关系。

### 2.5.2 数学推导

1. 计算 bbox 中心的归一化相机方向：
   ```
   rx = (bbox_cx - cx) / fx
   ry = (bbox_cy - cy) / fy
   rz = 1.0
   ray_norm = [rx, ry, 1] / ||[rx, ry, 1]||
   ```

2. 计算将 `ray_norm` 旋转到 `[0, 0, 1]`（光轴）的旋转：
   ```
   axis = cross(ray_norm, [0, 0, 1]) = [ry, -rx, 0]
   angle = atan2(||axis||, rz)
   ```

3. 构造轴角 → 旋转矩阵

4. 退化处理：当 `||axis|| < 1e-7`（bbox 已经在主点上），返回单位矩阵

### 2.5.3 当前状态

`TRAIN.perspective_normalization = false`（默认关闭）。在 `preprocess_batch` 中，若启用：
- 计算 `correction_rot_mat` 并融入 `trans_3d_mat` 和 `trans_2d_mat`
- 同时将 `persp_axis_angle` 清零（避免与透视归一化冲突）

---

## 2.6 左手翻转处理

`src/data/preprocess.py:898-930` — 模型只处理右手（MANO 右手模型），左手通过水平翻转转换为右手：

1. **图像翻转**：`patches[bx] = torch.flip(patches[bx], dims=[-1])`
2. **BBox 翻转**：`x1, x2 = width - x2, width - x1`
3. **2D 关节点翻转**：`joint_img[..., 0] = width - joint_img[..., 0]`
4. **3D X 轴取反**：`joint_cam[..., 0] *= -1`, `joint_rel[..., 0] *= -1`
5. **MANO 姿态取反**：非根关节的 axis-angle 全部取反 `mano_pose[:, :, 1:] *= -1`
6. **主点翻转**：`princpt_x = width - princpt_x`

---

## 2.7 增强完整数据流

```
batch_origin (原始 WDS 样本)
  │
  ├─ 提取 & clone 所有张量到 device
  │
  ├─ [augmentation_flag=True]:
  │   ├─ 可选 perspective_normalization → correction_rot_mat
  │   ├─ 采样随机参数: rad, scale_z, scale_f, princpt_noise, persp_axis_angle
  │   ├─ get_trans_3d_mat → 3D 变换矩阵
  │   ├─ get_trans_2d_mat → 2D 透视变换矩阵
  │   ├─ joint_cam/joint_rel ← einsum(..., trans_3d_mat)
  │   ├─ MANO root rotation 更新
  │   ├─ joint_img ← apply_perspective_to_points(trans_2d_mat, ...)
  │   ├─ 从 jittered joints 重算 hand_bbox
  │   ├─ _jitter_hand_bbox(hand_bbox, ...)  ← 约束性或传统 jitter
  │   ├─ _compute_square_patch_bbox(jittered_hand_bbox, expansion=2.0)
  │   ├─ warp_perspective 裁剪 → patches [B, T, 3, 224, 224]
  │   └─ pixel_aug(patches) → ColorJitter + GaussianNoise
  │
  ├─ [augmentation_flag=False]:
  │   └─ crop_and_resize 直接裁剪 → patches
  │
  ├─ 左手翻转处理 (patches, bboxes, joints, MANO, princpt)
  ├─ joint_patch_resized 计算
  ├─ MANO 姿态表示转换 (3 ←→ 6d ←→ quat)
  └─ 输出 batch_out 字典
```
