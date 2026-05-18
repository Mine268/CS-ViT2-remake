# 第 4 章：损失函数与监督路由

> 本章覆盖 RemakeLoss 的所有 8 个损失分量、ego/aux 监督分离、Robust L1 损失、heatmap 交叉熵、Rho multibin 监督和 dropout 渐进调度。

---

## 4.1 监督分离的核心设计

CS-ViT2-remake 的核心监督路由规则（`src/model/loss.py:84-91`）：

> - **ego 数据集** (HOT3D + AssemblyHands): 接收**绝对监督**（translation, rho, reprojection）
> - **ego + aux 数据集**: 接收**局部监督**（MANO pose/shape, relative joints）

这种分离的动机：只有 ego 数据集提供了精确的相机外参/内参，可以监督绝对深度。aux 数据集提供 MANO 标注但不一定有可靠的相机参数。

### 4.1.1 数据集分组

| 组 | Stage1 成员 | Stage2 成员 | 监督范围 |
|----|-----------|-----------|---------|
| ego | HOT3D, AssemblyHands | 同左 | Absolute: trans, rho, reproj |
| aux | InterHand2.6M, FreiHAND, MTC, DexYCB, HO3D_v3, RHD | InterHand2.6M, MTC, DexYCB, HO3D_v3 (无 FreiHAND/RHD) | Relative: pose, shape, rel joints |

### 4.1.2 掩码构建

`src/model/loss.py:51-63` 中的 `build_dataset_group_mask`:

```python
def build_dataset_group_mask(data_sources, target_sources, device, dtype):
    target_set = {str(x) for x in target_sources}
    mask = [str(source) in target_set for source in data_sources]
    return torch.tensor(mask, device=device, dtype=dtype).view(-1, 1)
    # [B, 1] float tensor: 1.0 if sample in group else 0.0
```

**训练时**：`data_source` 来自 `canonicalize_data_source_name` 规范化后的配置驱动名称。
**验证时**：同样通过 `net.loss_fn.ego_datasets`/`aux_datasets` 访问。

---

## 4.2 损失分量详解

`src/model/loss.py:192-338` 中的 `RemakeLoss.forward`:

### 4.2.1 `loss_theta` — MANO 姿态 (3.0)

```python
loss_theta = L1(pose_pred, pose_gt)  # 48-dim axis-angle
mask: has_mano * all_supervised_mask  # ego + aux
```

权重 `λ_theta = 3.0`，监督所有 16 个关节的 axis-angle 参数。

### 4.2.2 `loss_shape` — MANO 形状 (3.0)

```python
loss_shape = L1(shape_pred, shape_gt)  # 10-dim PCA
mask: has_mano * all_supervised_mask
```

权重 `λ_shape = 3.0`，将形状系数约束在零附近（MANO 先验）。

### 4.2.3 `loss_uv_patch` — Root UV Heatmap (1.0)

```python
loss_uv_patch = CE(log_hm_uv_patch, Gaussian_target(gt_uv))
mask: root_uv_valid * range_frame_filter
```

**Heatmap 目标构造** (`compute_hm_ce_2d`, `src/model/loss.py:170-190`):
```python
squared_diff = (gt_x - x_grid)² + (gt_y - y_grid)²
target_unnormalized = exp(-squared_diff / (2 * σ²))  # σ = 4.0
target_probs = target_unnormalized / sum(target_unnormalized)  # 归一化
loss = -sum(target_probs * log(pred_log_probs))  # Cross-entropy
```

Gaussian σ = 4.0 像素（在 heatmap 分辨率空间中）。

### 4.2.4 `loss_trans` — Root Translation (0.001)

```python
loss_trans = L1(trans_pred, trans_gt)  # 3-dim (X, Y, Z mm)
mask: ego_root_valid = ego_mask * root_valid * has_intr * (trans_z > 0) * frame_filter
```

权重很低 (`λ_trans = 0.001`)，因为 translation 的误差尺度（mm）远大于其他项。实际上 Rho 损失承担了主要的深度监督。

### 4.2.5 `loss_rho_cls` — Rho 分类 (1.0)

```python
loss_rho_cls = CrossEntropy(rho_cls_logits[valid], bin_idx[valid])
valid: ego_root_valid > 0.5
```

8 类分类（对应 8 个 Δlog ρ bin）。

### 4.2.6 `loss_rho_res` — Rho 残差 (1.0)

```python
pred_rho_res = rho_residuals.gather(dim=-1, index=bin_idx.unsqueeze(-1)).squeeze(-1)
loss_rho_res = SmoothL1Loss(pred_rho_res[valid], residual[valid], beta=0.1)
```

只监督 GT bin 对应的那一个残差值。Smooth L1 的 β=0.1，在残差小于 0.1 时退化为 L2。

### 4.2.7 `loss_joint_rel` — 相对关节 (0.012)

```python
joint_rel_pred = MANO_FK(pose_pred, shape_pred.detach())  # root-relative
loss_joint_rel = L1(joint_rel_pred, joint_rel_gt)  # [B, T, 21, 3] mm
mask: joint_3d_valid * all_supervised_mask
```

注意 `shape_pred.detach()` — 相对关节损失不反向传播到形状预测器。

### 4.2.8 `loss_joint_img` — 重投影 (0.002)

```python
joint_cam_pred = joint_rel_pred + trans_pred  # absolute
joint_img_pred = project_3d_to_2d(joint_cam_pred, focal, princpt)
joint_img_gt = project_3d_to_2d(joint_cam_gt, focal, princpt)
loss_joint_img = RobustL1Loss(joint_img_pred, joint_img_gt, delta=84.0)
mask: reproj_valid = joint_3d_valid * has_intr * (pred_z > 20mm) * ego_mask * frame_filter
```

RobustL1 的 δ=84 像素——在 84 像素以内使用 L1，以外使用对数软化。

### 4.2.9 损失汇总

```python
loss = (
    3.0 * loss_theta
    + 3.0 * loss_shape
    + 1.0 * loss_uv_patch
    + 0.001 * loss_trans
    + 1.0 * loss_rho_cls
    + 1.0 * loss_rho_res
    + 0.012 * loss_joint_rel
    + 0.002 * loss_joint_img
)
```

### 4.2.10 监控指标

除损失外，`loss_state` 还记录：
- `rho_bin_acc`: rho 分类准确率
- `rho_mae_mm`: rho 的 MAE (mm)
- `pred_joint_z_min`: 预测的 3D 关节最小 Z 值
- `reproj_valid_frac`: 重投影有效的样本比例
- `ego_root_valid_frac`: ego root 有效的样本比例

训练时的几何指标按 ego/aux/all 拆分：`micro_mpjpe_{all,ego,aux}`, `micro_mpvpe_{all,ego,aux}`, `micro_rte_{all,ego,aux}` 及其相对版本。

---

## 4.3 Root Frame Filter

`src/model/loss.py:155-168` 中的 `compute_root_frame_filter_mask` 过滤不适合绝对 root/rho 监督的帧：

```python
frame_mask = ones_like(has_intr)
if min_valid_joints_2d > 0:  # 16
    valid_count = sum(joint_2d_valid > 0.5, dim=-1)
    frame_mask *= (valid_count >= 16).float()
if min_hand_bbox_edge_px > 0:  # 8
    bbox_min_edge = min(bbox_w, bbox_h)
    frame_mask *= (bbox_min_edge >= 8).float()
```

---

## 4.4 Robust L1 Loss

`src/model/loss.py:16-35`:

```python
class RobustL1Loss(nn.Module):
    def __init__(self, delta=100.0):
        self.delta = delta

    def forward(self, pred, target):
        abs_diff = |pred - target|
        inside = abs_diff < delta
        loss_l1 = abs_diff                         # 小误差: L1
        loss_log = delta * (1 + log1p((abs_diff - delta) / delta))  # 大误差: 对数软化
        return where(inside, loss_l1, loss_log)
```

对于重投影损失，`delta = 84` 像素——重投影误差超过 84 像素时，损失增长从线性转为对数，减弱异常值的梯度影响。

---

## 4.5 训练期 Dropout 渐进调度

`src/utils/train_utils.py` 中的 `get_progressive_dropout`:

```python
def get_progressive_dropout(step, total_steps, warmup_steps=10000, target_dropout=0.1):
    if step < warmup_steps:
        return 0.0              # 前 10000 步: 无 dropout
    else:
        return target_dropout   # 10000 步后: 跳变到目标值 0.1
```

虽然名为 "progressive"，实际实现是阶跃函数（`total_steps` 被接受但忽略）。在前 10000 步中 dropout 为零，之后直接跳到 0.1。

**应用**：`PoseNet.set_dropout_rate` (`src/model/net.py:482-491`) 遍历 `handec` 和 `temporal_refiner` 中所有 `nn.Dropout` 模块，更新其 `p` 属性。

---

## 4.6 评估指标

`src/utils/metric.py`:

### 4.6.1 MetricMeter (训练用)

在每个训练步计算 15 个标量指标：
- `micro_mpjpe_{all,ego,aux}`: Mean Per-Joint Position Error (相机空间)
- `micro_mpjpe_rel_{all,ego,aux}`: 根相对 MPJPE
- `micro_mpvpe_{all,ego,aux}`: Mean Per-Vertex Position Error
- `micro_mpvpe_rel_{all,ego,aux}`: 根相对 MPVPE
- `micro_rte_{all,ego,aux}`: Root Translation Error

所有指标取最后一帧 `[:, -1:]`。

### 4.6.2 StreamingMetricMeter (验证用)

累加跨多个验证 batch 的误差和/计数，最后计算平均。支持通过 `accelerator.gather_for_metrics` 的分布式指标收集。

### 4.6.3 Best Model 选择标准

当前使用 `micro_rte_ego`（ego 数据集的 root translation error）作为选模指标。值越小越好。

---

## 4.7 损失路由完整覆盖矩阵

| 损失分量 | ego (HOT3D, AH) | aux (IH26M, MTC, ...) | 所需标注 |
|---------|-----------------|----------------------|---------|
| `loss_theta` | ✓ | ✓ | MANO pose |
| `loss_shape` | ✓ | ✓ | MANO shape |
| `loss_joint_rel` | ✓ | ✓ | 3D joints |
| `loss_uv_patch` | ✓ | ✗ | 2D joints + intrinsics |
| `loss_trans` | ✓ | ✗ | 3D root + intrinsics |
| `loss_rho_cls` | ✓ | ✗ | 3D root + intrinsics |
| `loss_rho_res` | ✓ | ✗ | 3D root + intrinsics |
| `loss_joint_img` | ✓ | ✗ | 3D joints + intrinsics |

---

## 4.8 完全损失计算流程

```
batch → predict_mano_param → pose_pred, shape_pred, trans_pred, cam_aux
  │
  ├─ MANO FK (detach shape): joint_rel_pred, vert_rel_pred
  │
  ├─ ego_mask = data_source ∈ {HOT3D, AssemblyHands}
  ├─ all_mask = data_source ∈ ego ∪ aux
  │
  ├─ loss_theta = L1(pose_pred, pose_gt) * all_mask * has_mano
  ├─ loss_shape = L1(shape_pred, shape_gt) * all_mask * has_mano
  ├─ loss_joint_rel = L1(joint_rel_pred, joint_rel_gt) * all_mask * joint_3d_valid
  │
  ├─ loss_uv_patch = CE(hm, gaussian_gt) * root_uv_valid * frame_filter
  │
  ├─ rho_gt = ||trans_gt||
  ├─ encoded_rho = encode_delta_log_rho(ρ_gt, log_ρ_prior, ...)
  ├─ loss_rho_cls = CE(rho_cls_logits, bin_idx) * ego_root_valid
  ├─ loss_rho_res = SmoothL1(residual_pred, residual_gt) * ego_root_valid
  ├─ loss_trans = L1(trans_pred, trans_gt) * ego_root_valid
  │
  ├─ joint_cam_pred = joint_rel_pred + trans_pred
  ├─ joint_img_pred = project(joint_cam_pred)
  ├─ loss_joint_img = RobustL1(joint_img_pred, joint_img_gt) * reproj_valid
  │
  └─ loss = weighted_sum(...)
```
