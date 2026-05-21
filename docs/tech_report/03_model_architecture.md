# 第 3 章：模型架构

> 本章覆盖完整的模型架构：ViT Backbone、透视信息嵌入器、MANO Transformer Decoder、Camera Head（Patch UV + Rho Multibin）、几何先验、时序编码器、MANO 手部模型。

---

## 3.1 总体结构

模型顶层是 `PoseNet` (`src/model/net.py`)，包含以下子模块。

### 3.1.0 完整 Pipeline 流程图

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                           INPUT PIPELINE                                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  WDS Tar ─→ decode ─→ clip_to_t_frames ─→ filter ─→ preprocess_frame       ║
║                                                                             ║
║  preprocess_batch(batch_origin):                                            ║
║    ├─ crop & resize (hand_bbox, expansion=2.0) → [B,T,3,224,224]           ║
║    ├─ pixel aug (ColorJitter + GaussianNoise, training only)                 ║
║    ├─ 3D geometric aug (rotation + scale + perspective, training only)       ║
║    └─ normalize (img_mean / img_std)                                        ║
║                                                                             ║
║  Output: patches[B,T,3,224,224], bbox[B,T,4], focal[B,T,2], princpt[B,T,2]  ║
║          joint_cam[B,T,21,3], hand_bbox[B,T,4], timestamp[B,T]              ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                      │
                                      ▼

╔══════════════════════════════════════════════════════════════════════════════╗
║ ① ViT BACKBONE  (src/model/backbone.py)                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Model: facebook/dinov2-large                                               ║
║  Params: 304M (patch=14, hidden=1024, 24 layers)                            ║
║                                                                             ║
║  patches [B×T, 3, 224, 224]                                                 ║
║       │                                                                     ║
║       ▼                                                                     ║
║  ┌──────────────────────────────────┐                                       ║
║  │  PatchEmbed (14×14) → 256 tokens │                                       ║
║  │  + CLS token                     │                                       ║
║  │  + Position Embedding            │                                       ║
║  └──────────────┬───────────────────┘                                       ║
║                 ▼                                                           ║
║  ┌──────────────────────────────────┐                                       ║
║  │  24× TransformerBlock            │                                       ║
║  │  - Multi-Head Self-Attention     │                                       ║
║  │  - MLP (GELU)                   │                                       ║
║  │  - LayerNorm                     │                                       ║
║  └──────────────┬───────────────────┘                                       ║
║                 ▼                                                           ║
║  Output: feats [B×T, 257, 1024]  (1 CLS + 256 patches)                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                      │
                                      ▼

╔══════════════════════════════════════════════════════════════════════════════╗
║ ② PERSP INFO EMBEDDER  (src/model/perspective.py)                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  PerspInfoEmbedderCrossAttn: 将相机几何注入视觉 tokens                        ║
║                                                                             ║
║  1. 8×8 规则网格 → 像素坐标 (grid_xy)                                        ║
║  2. 相机方向: directions = (grid_xy - princpt) / focal                      ║
║  3. context = concat(directions, grid_xy - bbox_center)  → [64, 4]          ║
║  4. 1-layer Cross-Attention: feats(query) ←→ context(key, value)             ║
║  5. Zero-init Linear residual: out = feats + zero_linear(attn_out)          ║
║                                                                             ║
║  Output: feats [B×T, 257, 1024]  (残差结构, 初始行为 identity)              ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                      │
                                      ▼

╔══════════════════════════════════════════════════════════════════════════════╗
║ ③ MANO TRANSFORMER DECODER  (src/model/heads.py)                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  MANOTransformerDecoderHead: 解码 MANO 参数 + 相机参数                        ║
║                                                                             ║
║  ┌──────────────────────────────────────────────────────────────┐           ║
║  │ 1 × Learnable Query Token  [pose_dim + shape_dim + trans_dim] │           ║
║  │    = 48 (16j×3) + 10 + 3 = 61 dim                            │           ║
║  └────────────────────┬─────────────────────────────────────────┘           ║
║                       ▼                                                     ║
║  ┌──────────────────────────────────────────────────────────────┐           ║
║  │  4-layer TransformerDecoder (16 heads, dim_head=64, dim=1024) │           ║
║  │  ┌─────────────┐  ┌─────────────────┐  ┌──────────────────┐  │           ║
║  │  │ Self-Attn   │→ │ Cross-Attn      │→ │  FFN (MLP 4096)  │  │           ║
║  │  │ (query×self)│  │ (query×feats)   │  │  GELU + Dropout  │  │           ║
║  │  └─────────────┘  └─────────────────┘  └──────────────────┘  │           ║
║  └────────────────────┬─────────────────────────────────────────┘           ║
║                       │                                                     ║
║          ┌────────────┼────────────┬──────────────┐                          ║
║          ▼            ▼            ▼              ▼                          ║
║     ┌────────┐  ┌─────────┐  ┌──────────┐  ┌──────────────┐                 ║
║     │decpose │  │decshape │  │deccam_uv │  │  decrho      │                 ║
║     │Linear  │  │Linear   │  │Softargmax│  │  RhoMultiBin │                 ║
║     │1024→48 │  │1024→10  │  │  Head    │  │  Head        │                 ║
║     └───┬────┘  └────┬────┘  └────┬─────┘  └──────┬───────┘                 ║
║         │            │            │               │                          ║
║         ▼            ▼            ▼               ▼                          ║
║    pose_aa[48]  shape[10]   root_uv[2]     rho_cls[8bins]+res                ║
║    (16j×3 axis-  (MANO      (normalized     → root_depth[1]                  ║
║     angle)       betas)     patch UV)       → camera_trans[3]                ║
║                                                                             ║
║  ┌──────────────────────────────────────────────────────────────┐           ║
║  │  Camera Head: patch_uv_rho_multibin                          │           ║
║  │                                                              │           ║
║  │  root_uv ────→ 从 patch UV 反算相机光线方向                  │           ║
║  │  root_depth ─→ ρ = ρ_prior × exp(Δρ_cls + Δρ_res)           │           ║
║  │  camera_trans = ray_direction(uv, focal, princpt) × ρ       │           ║
║  └──────────────────────────────────────────────────────────────┘           ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                      │
                    ┌─────────────────┴─────────────────┐
                    │  Stage 1            Stage 2       │
                    │  (逐帧独立)         (时序精炼)     │
                    ▼                                   ▼

╔══════════════════════════════════════════════════════════════════════════════╗
║ ④ STAGE2: TEMPORAL ENCODER  (src/model/temporal.py)   (仅 Stage2)            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  decoder tokens [B×T, D] → rearrange → [B, T, D]                            ║
║       │                                                                     ║
║       ▼                                                                     ║
║  ┌──────────────────────────────────────────────────────────────┐           ║
║  │  2-layer Causal Transformer Decoder                          │           ║
║  │  - RoPE position encoding (trope_scalar=20.0)                │           ║
║  │  - 16 heads, dim_head=64                                    │           ║
║  │  - Causal mask (future-blind)                                │           ║
║  │  - Zero-init linear residual                                │           ║
║  └────────────────────┬─────────────────────────────────────────┘           ║
║                       ▼                                                     ║
║  refined tokens [B, T, D] → rearrange → [B×T, D]                            ║
║       │                                                                     ║
║       ▼                                                                     ║
║  decode_token() → pose, shape, trans, cam_aux  (复用 stage1 的输出头)        ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                      │
                                      ▼

╔══════════════════════════════════════════════════════════════════════════════╗
║ ⑤ MANO FORWARD KINEMATICS  (smplx MANO model)                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  pose_aa[48] → global_orient[3] + hand_pose[45]                             ║
║  shape[10]   → betas                                                     ║
║  rmano_layer(betas, global_orient, hand_pose)                                ║
║       │                                                                     ║
║       ├─→ mesh vertices [778, 3]  (mm, root-relative)                       ║
║       └─→ 21 joints via J_regressor  (mm, root-relative)                    ║
║                                                                             ║
║  joint_cam_pred = joint_rel + camera_trans[None, :]                         ║
║  vert_cam_pred  = vert_rel  + camera_trans[None, :]                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                      │
                                      ▼

╔══════════════════════════════════════════════════════════════════════════════╗
║ ⑥ LOSS COMPUTATION  (src/model/loss.py)                                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                             ║
║  ┌───────────────────────┬────────┬──────────────────────────────────┐      ║
║  │       Loss            │ Weight │         Description               │      ║
║  ├───────────────────────┼────────┼──────────────────────────────────┤      ║
║  │ L_theta (MANO pose)   │  3.0   │ SmoothL1 | 仅 ego                  │      ║
║  │ L_shape (MANO shape)  │  3.0   │ L2 | 仅 ego                        │      ║
║  │ L_uv_patch (root UV)  │  1.0   │ Gaussian heatmap CE                │      ║
║  │ L_root_z_cls (ρ bin)  │  1.0   │ Cross-Entropy (8 bins)             │      ║
║  │ L_root_z_res (ρ res)  │  1.0   │ SmoothL1                          │      ║
║  │ L_trans (root trans)  │ 0.001  │ L1                                │      ║
║  │ L_rel (relative jts)  │ 0.012  │ L1 | ego: 3D joints, aux: 2D+Z    │      ║
║  │ L_img (reprojection)  │ 0.002  │ RobustL1(δ=84) | ego: 3D, aux: 2D  │      ║
║  └───────────────────────┴────────┴──────────────────────────────────┘      ║
║                                                                             ║
║  Supervision Routing:                                                       ║
║    ego (HOT3D, AssemblyHands): 全监督 (pose + shape + 3D joints)             ║
║    aux (其他 6 数据集): 仅 2D joints + root depth 监督，无 MANO 损失          ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 3.1.1 Stage 枚举

```python
class Stage(enum.Enum):
    STAGE1 = "stage1"  # 逐帧独立解码
    STAGE2 = "stage2"  # 时序精炼解码
```

### 3.1.2 PoseNet 构造函数参数 (42 个)

| 类别 | 参数 | 类型 | 说明 |
|------|------|------|------|
| Stage | `stage` | str | `"stage1"` / `"stage2"` |
| Stage | `stage1_weight_path` | Optional[str] | Stage2 的 stage1 权重路径 |
| Backbone | `backbone_str` | str | HuggingFace 模型 ID |
| Backbone | `img_size` | Optional[int] | 图像尺寸 (默认 224) |
| Backbone | `img_mean/std` | List[float] | 归一化参数 (ImageNet) |
| Backbone | `infusion_feats_lyr` | Optional[List[int]] | 中间层融合 (默认 None) |
| Backbone | `drop_cls` | bool | 丢弃 CLS token |
| Backbone | `backbone_kwargs` | Optional[Dict] | HF AutoModel 额外参数 |
| Backbone | `freeze_backbone` | bool | 冻结 backbone |
| Hand Decoder | `num_handec_layer` | int | Decoder 层数 (4) |
| Hand Decoder | `num_handec_head` | int | Attention 头数 (16) |
| Hand Decoder | `ndim_handec_mlp` | int | MLP 隐藏维度 (4096) |
| Hand Decoder | `ndim_handec_head` | int | 每头维度 (64) |
| Hand Decoder | `prob_handec_dropout` | float | Dropout 率 (0.1) |
| Hand Decoder | `handec_norm` | str | 归一化类型 (`"layer"`) |
| Hand Decoder | `ndim_handec_ctx` | Optional[int] | Cross-attn context 维度 (1024) |
| Hand Decoder | `handec_mean_init` | bool | MANO 均值初始化 query |
| Hand Decoder | `handec_cam_head_type` | str | 仅 `"patch_uv_rho_multibin"` |
| Persp Embed | `pie_type` | str | 仅 `"ca"` (cross-attention) |
| Persp Embed | `num_pie_sample` | int | Grid 采样密度 (8) |
| Temporal | `num_temporal_head` | int | 时序 attention 头数 (16) |
| Temporal | `num_temporal_layer` | int | 时序层数 (2) |
| Temporal | `trope_scalar` | float | RoPE 时间缩放 (20.0) |
| Temporal | `zero_linear` | bool | 零初始化时序残差 |
| Joint | `joint_rep_type` | str | `"3"` / `"6d"` / `"quat"` |
| Root-Z/Rho | `root_z_num_bins` | int | 分类 bin 数 (8) |
| Root-Z/Rho | `root_z_d_min/max` | float | Δlog 范围 `[-0.71, 0.75]` |
| Root-Z/Rho | `root_z_prior_k` | float | 先验常数 (121.0) |
| Root-Z/Rho | `root_z_geom_hidden_dim` | int | 几何特征 MLP 维度 (256) |
| Root-Z/Rho | `root_z_dropout` | float | Rho head dropout (0.1) |
| Loss | `lambda_*` (10 个) | float | 各损失分量权重 |
| Loss | `hm_sigma` | float | Heatmap Gaussian σ (4.0) |
| Loss | `pred_joint_z_min_mm` | float | 重投影最小 Z (20.0) |
| Loss | `reproj_loss_type` | str | `"robust_l1"` / `"l1"` |
| Loss | `reproj_loss_delta` | float | Robust L1 阈值 (84.0) |
| Loss | `ego/aux_datasets` | List[str] | 监督路由 |
| Filter | `root_min_valid_joints_2d` | int | 最少 2D 关节点 (16) |
| Filter | `root_min_hand_bbox_edge_px` | float | 最小 bbox 边长 (8.0) |

### 3.1.3 构造函数关键步骤

`src/model/net.py:113-226`：

1. **验证**：`norm_by_hand=False` (不支持)、`cam_head_type="patch_uv_rho_multibin"` (仅支持类型)
2. **创建 ViTBackbone**，禁用预训练 mask_token 的梯度
3. **注册归一化参数**：`img_mean`, `img_std` 作为 buffer
4. **提取 backbone 属性**：`has_cls_token`, `patch_size`, `hidden_size`, `img_size`, `num_patch`
5. **验证 `pie_type="ca"`**，删除 `pie_fusion`
6. **创建 PerspInfoEmbedderCrossAttn**：`num_token = num_patch² + int(has_cls & !drop_cls)` = 256 或 257
7. **加载 MANO 资产**：`J_regressor_mano` (16×778, buffer), `rmano_layer` (smplx, 冻结)
8. **创建 MANOTransformerDecoderHead**
9. **创建 TemporalEncoder**
10. **Stage1**: 冻结 `temporal_refiner`
11. **创建 RemakeLoss + MetricMeter**
12. **Stage2 + stage1_weight**: 调用 `load_pretrained`

---

## 3.2 ViT Backbone

**代码位置**：`src/model/backbone.py`

### 3.2.1 模型加载

```python
backbone_cfg = transformers.AutoConfig.from_pretrained(backbone_str)
self.model_type = backbone_cfg.model_type
self.has_cls_token = self.model_type not in {"swin", "swinv2"}
self.patch_size = backbone_cfg.patch_size  # 14 for DINOv2
self.hidden_size = backbone_cfg.hidden_size  # 1024 for DINOv2-large
self.feature_stride = getattr(backbone_cfg, "encoder_stride", None) or self.patch_size
```

**分辨率验证**：
```python
img_size % patch_size == 0          # 224 % 14 == 0 ✓
img_size % feature_stride == 0      # 224 % 14 == 0 ✓
num_patch = img_size // feature_stride  # 16
```

**加载预训练模型**：
```python
self.backbone = transformers.AutoModel.from_pretrained(
    backbone_str, output_hidden_states=True, **backbone_kwargs
)
```

### 3.2.2 Forward Pass

**无中间特征融合** (默认):
```python
return backbone_output.last_hidden_state  # [B, 257, 1024] for DINOv2
```

**有中间特征融合** (`infusion_feats_lyr` 非空):
1. 提取指定层的 `hidden_states`
2. 分离 CLS token / patch tokens
3. CLS → Linear→BN→ReLU (proj 到 64 维)
4. Patches → Conv2d→BN→ReLU (proj 到 64 维)
5. Concat 所有层特征 → 融合 MLP/Conv → 投影回 hidden_size
6. 重组为 `[CLS, patch_1, ..., patch_256]`

### 3.2.3 ViT vs Swin 差异

Swin/SwinV2 没有 CLS token：
- `has_cls_token = False`
- CLS 特征用 patch 均值代替
- 输出形状 `[B, 256, C]` 而非 `[B, 257, C]`

### 3.2.4 支持的 Backbone 列表

本地 `model/` 目录中已下载的预训练权重：
- `facebook/dinov2-large` (默认)
- `facebook/dinov2-base`, `facebook/dinov2-giant`
- `facebook/vit-mae-large`, `facebook/vit-mae-huge`
- `microsoft/swinv2-*` 系列
- `microsoft/swin-*` 系列
- `microsoft/resnet-50`

---

## 3.3 透视信息嵌入器

**代码位置**：`src/model/perspective.py`

### 3.3.1 动机

将 patch crop 的几何上下文（bbox 位置和大小、相机内参）显式注入视觉 tokens，使模型更好地理解 "patch 内的每个位置对应什么样的 3D 相机方向"。

### 3.3.2 结构

```python
PerspInfoEmbedderCrossAttn(hidden_size=1024, num_sample=8, num_token=257):
    self.net = TransformerDecoder(
        num_tokens=257, token_dim=1024, dim=1024,
        depth=1, heads=8, mlp_dim=4096, dim_head=64,
        context_dim=4  # (dx, dy, u, v)
    )
    self.zero_linear = Linear(1024, 1024, bias=False)
    nn.init.zeros_(self.zero_linear.weight)  # 初始为 identity
```

### 3.3.3 Forward 详细计算

输入: `feats [B, 257, 1024]`, `bbox [B, 4]`, `focal [B, 2]`, `princpt [B, 2]`

1. **构建图像空间采样网格** (8×8=64 个点):
   ```python
   grid_edge = linspace(1/16, 15/16, 8)  # 在 [0.0625, 0.9375] 均匀采样
   # 64 个点的 (x, y) 图像坐标
   x_grid = bbox_x1 + (bbox_x2 - bbox_x1) * grid_edge  # [B, 8]
   y_grid = bbox_y1 + (bbox_y2 - bbox_y1) * grid_edge  # [B, 8]
   grid_xy = stack([x_grid[:,:,None].expand(8,8),
                     y_grid[:,None,:].expand(8,8)])  # [B, 8, 8, 2]
   ```

2. **转换为归一化相机方向**:
   ```python
   directions = (grid_xy - princpt) / focal  # [B, 8, 8, 2]  归一化偏移
   directions = cat([directions, ones_like(directions[..., :1])], dim=-1)  # [B, 8, 8, 3]
   directions = directions / ||directions||  # 单位方向向量
   directions = directions[..., :2]  # 丢弃 Z (冗余)
   ```

3. **拼接归一化 UV 坐标** (提供空间位置线索):
   ```python
   grid_uv = [0,1]² 内的规则网格  # [8, 8, 2]
   directions = cat([directions, grid_uv.expand(B, -1, -1, -1)], dim=-1)
   # [B, 8, 8, 4]
   ```

4. **Flatten + Cross-Attention + Residual**:
   ```python
   directions = rearrange(directions, "b p q d -> b (p q) d")  # [B, 64, 4]
   out = self.net(feats, context=directions)  # Cross-attn: feats attend to 64 geometry tokens
   out = self.zero_linear(out) + feats  # Zero-init residual → starts as identity
   ```

**关键设计**：`zero_linear` 权重为零初始化，使模块初始行为为恒等映射，训练中逐渐学习注入几何信息。

---

## 3.4 MANO Transformer Decoder Head

**代码位置**：`src/model/heads.py:148-345`

### 3.4.1 结构

```python
MANOTransformerDecoderHead:
    self.transformer = TransformerDecoder(
        num_tokens=1,           # 单个 query token
        token_dim=npose+10+3,  # pose + shape + camera
        dim=1024, depth=4, heads=16, dim_head=64,
        mlp_dim=4096, context_dim=1024
    )
    self.decpose = Linear(1024, npose)      # MANO pose
    self.decshape = Linear(1024, 10)         # MANO shape
    self.deccam_uv = SoftargmaxHead2DJoint   # Root UV heatmap
    self.decrho = RhoMultiBinHead            # Camera distance ρ
```

### 3.4.2 Query Token 初始化

`src/model/heads.py:247-271` — 支持 MANO 均值初始化 (`use_mean_init=True`)：

1. 从 `MANO_MEAN_NPZ` 加载统计均值
2. Pose 加载为 6D 旋转表示 (16×6=96 维)
3. 根据 `joint_rep_type` 转换：
   - `"6d"`: 直接使用
   - `"3"`: 6d → rotation matrix → axis-angle → 48 维
   - `"quat"`: 6d → rotation matrix → quaternion → 64 维
4. Shape (10 维) 和 Camera (3 维) 直接加载
5. 拼接后作为 buffer `init_hand_pose`, `init_betas`, `init_cam`

**不使用均值初始化**时，全部从 `randn` 随机初始化。

### 3.4.3 encode_img — Decoder Forward

```python
def encode_img(self, x):  # x: [B, N, 1024] backbone tokens
    token = cat([init_pose, init_shape, init_cam], dim=1)[:, None, :]
    # token: [B, 1, npose+10+3]
    return self.transformer(token, context=x).squeeze(1)
    # → [B, 1024]  单个精炼后的 latent token
```

### 3.4.4 decode_token — 输出头

```python
def decode_token(self, token_out, patch_bbox, hand_bbox, focal, princpt):
    # token_out: [B, 1024]
    pred_pose = self.decpose(token_out)        # [B, 48]
    pred_shape = self.decshape(token_out)       # [B, 10]
    pred_uv_patch, log_hm = self.deccam_uv(token_out)  # [B, 2], [B, H, W]
    pred_rho, rho_aux = self.decrho(token_out, hand_bbox, focal, princpt)  # [B, 1], dict

    # Patch UV → Image UV
    pred_uv_img = patch_uv_to_image_uv(pred_uv_patch, patch_bbox, patch_size)

    # Image UV → Camera Ray
    pred_q, pred_ray_unit, pred_q_norm = image_uv_to_camera_ray(
        pred_uv_img, focal, princpt
    )

    # Camera Translation = Ray × Distance
    pred_cam = pred_ray_unit * pred_rho  # [B, 3]
```

**关键公式**: `camera_translation = ray_unit × ρ` — 从预测的 root 2D 位置和预测的相机距离恢复 3D 平移。

### 3.4.5 SoftargmaxHead2DJoint

`src/model/heads.py:30-57`

在 patch 空间内预测 root joint 的 2D 坐标：

```python
logits = Linear(token → H*W)
log_hm = log_softmax(logits).view(H, W)
hm = exp(log_hm)
hm_x = hm.sum(dim=-2)   # 边缘分布 (y 方向积分)
hm_y = hm.sum(dim=-1)   # 边缘分布 (x 方向积分)
pred_x = sum(hm_x * x_centers)
pred_y = sum(hm_y * y_centers)
```

UV 分辨率：`min(heatmap_resolution[0], max(8, patch_w//4))` × `min(heatmap_resolution[1], max(8, patch_h//4))`。对于 224×224 的 patch，约 56×56。

### 3.4.6 RhoMultiBinHead

`src/model/heads.py:60-145` — 详见第 3.6 节。

---

## 3.5 几何先验与 Multibin 预测

**代码位置**：`src/model/root_z.py`

### 3.5.1 常数

```python
ROOT_Z_GEOM_DIM = 6
RHO_GEOM_DIM = 6
ROOT_Z_MIN_PRIOR = 1e-6
```

### 3.5.2 Root-Z 先验

`src/model/root_z.py:16-73` 中的 `compute_root_z_prior_and_geom`:

```
z_prior = k × sqrt(fx × fy) / sqrt(bbox_w × bbox_h)
```

**物理直觉**：同样真实尺寸的手，图像中 bbox 越小通常越远；焦距越大，同样距离下表观尺寸越大。

**6 维几何特征**：
```
geom_feat = [
    log(z_prior),                        # 先验深度对数
    (bbox_cx - cx) / fx,                 # bbox 中心的归一化 x 偏移
    (bbox_cy - cy) / fy,                 # bbox 中心的归一化 y 偏移
    bbox_w / fx,                         # 归一化 bbox 宽度
    bbox_h / fy,                         # 归一化 bbox 高度
    log(bbox_w / bbox_h)                 # 对数宽高比
]
```

### 3.5.3 Rho 先验

`src/model/root_z.py:76-130` 中的 `compute_rho_prior_and_geom`:

```
ρ_prior = z_prior × ||q_ref||
q_ref = [(bbox_cx - cx)/fx, (bbox_cy - cy)/fy, 1]
```

Rho 先验在 root-z 先验基础上乘以 bbox 中心射线方向 `q_ref` 的长度。因为相机距离 `ρ = ||trans||` 不仅取决于深度 z，还取决于 root 相对主点的偏移。

**Rho 几何特征**（与 root-z 类似，但第一个分量是 `log_ρ_prior`）：
```
[log_ρ_prior, dx_ref, dy_ref, dw, dh, log_aspect_ratio]
```

### 3.5.4 Multibin 编码（监督端）

`src/model/root_z.py:193-224` 中的 `encode_delta_log_rho_targets`:

1. `delta = (d_max - d_min) / num_bins = (0.75 - (-0.71)) / 8 = 0.1825`
2. `Δlog ρ = log(ρ_gt) - log(ρ_prior)`
3. `Δlog ρ_clamped = clamp(Δlog ρ, -0.71, 0.75)` — 覆盖 `exp(-0.71)=0.49x` 到 `exp(0.75)=2.12x` 先验范围
4. `bin_idx = floor((Δlog ρ_clamped - d_min) / delta)` — 0 到 7
5. `bin_center = d_min + (idx + 0.5) * delta`
6. `residual = (Δlog ρ_clamped - bin_center) / delta` — 范围 `[-0.5, 0.5]`

返回 `{delta_log_rho, delta_log_rho_clamped, bin_idx, bin_center, residual, bin_size}`

### 3.5.5 Multibin 解码（推理端）

`src/model/root_z.py:297-340` 中的 `decode_delta_log_rho_predictions`:

1. `pred_bin = argmax(cls_logits)` — 选择最可能的 bin
2. `pred_res = gather(residuals, pred_bin).clamp(-0.5, 0.5)` — 获取 bin 内残差
3. `pred_Δlog = bin_center[pred_bin] + pred_res * delta`
4. `pred_log_ρ = log_ρ_prior + pred_Δlog`
5. `pred_ρ = exp(pred_log_ρ)` — 最终距离（mm）

### 3.5.6 RhoMultiBinHead 前向

`src/model/heads.py:107-145`:

```python
# 1. 计算几何先验
rho_prior, log_rho_prior, geom_feat = compute_rho_prior_and_geom(
    hand_bbox, focal, princpt, prior_k=121.0
)  # geom_feat: [B, 6]

# 2. 编码几何特征
geom_hidden = geom_proj(geom_feat)  # Linear(6→256)→GELU→Dropout→Linear(256→256)→GELU

# 3. 特征融合
fused = LayerNorm(cat([token, geom_hidden], dim=-1))  # [B, 1024+256]

# 4. 双头预测
rho_cls_logits = cls_head(fused)   # Linear(1280→8)  每个 bin 的分类分数
rho_residuals = res_head(fused)    # Linear(1280→8)  每个 bin 的残差

# 5. 解码
decoded = decode_delta_log_rho_predictions(rho_cls_logits, rho_residuals, log_rho_prior, ...)
pred_rho = decoded["pred_rho"]  # [B, 1]
```

---

## 3.6 Temporal Encoder (仅 Stage2)

**代码位置**：`src/model/temporal.py`

### 3.6.1 结构

```python
TemporalEncoder(dim=1024, num_head=16, num_layer=2, dropout=0.1,
                trope_scalar=20.0, zero_linear=True):
    # 2 层时序精炼块，每层：
    #   causal self-attention + feed-forward (PreNorm + residual)
    # 输出投影
    self.zero_linear = Linear(1024, 1024, bias=True)
    nn.init.zeros_(weight), nn.init.zeros_(bias)
```

### 3.6.2 CausalTRoPESelfAttention

带有 RoPE (Rotary Position Embedding) 的因果自注意力：

1. 时间戳归一化：`t = timestamp / trope_scalar`
2. Q, K, V 投影：`Linear(1024 → 3072)`, split 3 way
3. Multi-head reshape：`(B, T, 1024) → (B, 16, T, 64)`
4. **RoPE 应用于 Q 和 K**：
   ```python
   freqs = t * inv_freq  # inv_freq = 1/(10000^(2i/d))
   cos, sin = cos(freqs), sin(freqs)
   q_rot = rotate_pairs(q, cos, sin)
   k_rot = rotate_pairs(k, cos, sin)
   ```
5. 缩放点积 + **因果掩码**：`dots.masked_fill_(triu(diag=1), -inf)`
6. Softmax + Dropout + MatMul + Output Projection

### 3.6.3 因果掩码

```python
causal_mask = torch.ones(T, T).triu(diagonal=1)  # 上三角为 1（掩蔽）
dots.masked_fill_(causal_mask[None, None], float("-inf"))
```

位置 i 只能 attend 到位置 `≤ i`（包括自身）。这保证了时序推理的因果性——每帧只能看到当前及过去的帧。

### 3.6.4 零初始化残差

```python
def forward(self, token, timestamp):
    timestamp = timestamp / self.trope_scalar
    x = token
    for sa, ff in self.layers:
        x = sa(x, t=timestamp) + x
        x = ff(x) + x
    delta = self.zero_linear(x)
    return token + delta  # 初始 delta ≈ 0 → Stage2 热启动等价于逐帧推理
```

### 3.6.5 TRoPECrossAttention (通用交叉注意力)

`src/model/temporal.py:81-118` — 支持**独立时间戳**的交叉注意力：
- Query: 位置编码用 `tq`
- Key: 位置编码用 `tk`
- 用于 `TRoPETransformerCrossAttn` 中的 self-attn + cross-attn 堆叠

---

## 3.7 MANO 手部模型

`src/model/net.py:155-157` 和 `src/utils/mano.py`：

```python
self.rmano_layer = smplx.create(
    "model/smplx_models", "mano",
    is_rhand=True, use_pca=False
)
# 10 个 PCA 形状系数（非 PCA 模式但仍然是 10 维）
# 16 个关节（MANO 运动链）
# 778 个顶点
```

### 3.7.1 前向运动学

`src/model/net.py:331-359` 中的 `mano_to_pose`:

1. 输入 pose `[B, T, 48]` (axis-angle) 和 shape `[B, T, 10]`
2. Flatten 为 `[B*T, 48]` 和 `[B*T, 10]`
3. 若 `joint_rep_type` 非 `"3"`，先转换为 axis-angle
4. MANO 前向：`global_orient=pose[:,:3]`, `hand_pose=pose[:,3:]`, `transl=0`
5. 关节回归：`joints = einsum("nvd,jv→njd", vertices, J_regressor_mano)` — 16 个关节
6. 减根关节（detach）+ ×1000 → 毫米单位
7. 返回 `joint_rel [B, T, 21, 3]`, `vert_rel [B, T, 778, 3]`

### 3.7.2 关节对应关系

`src/constant.py` 中定义的 `HAND_JOINTS_ORDER` (21 个输出关节)：
```
Wrist, Thumb_1(CMC), Thumb_2(MCP), Thumb_3(IP), Thumb_4(Fingertip),
Index_1(MCP), Index_2(PIP), Index_3(DIP), Index_4(Fingertip),
Middle_1-4, Ring_1-4, Pinky_1-4
```

`MANO_JOINTS_ORDER` (16 个 MANO 原生关节)：
```
Wrist, Index_1-3, Middle_1-3, Pinky_1-3, Ring_1-3, Thumb_1-3
```

`MANO_JOINTS_CONNECTION` (20 条骨骼连接，用于可视化)。

---

## 3.8 关键推理流程

### 3.8.1 `predict_mano_param` — Stage1 vs Stage2 分支

`src/model/net.py:262-329`:

**Stage1** (逐帧独立):
```python
img = rearrange(img, "b t ... -> (b t) ...")  # [B*T, ...]
pose, shape, trans, cam_aux, _ = decode_hand_param(img, ...)
out_frames = 1
pose = rearrange(pose, "(b t) d -> b t d", t=1)  # [B, 1, 48]
```

**Stage2** (时序精炼):
```python
img = rearrange(img, "b t ... -> (b t) ...")  # [B*T, ...]
_, _, tokens_out = decode_hand_param(img, ...)  # 只用 tokens，不用初步预测
tokens_out = rearrange(tokens_out, "(b t) d -> b t d", t=T)  # [B, T, 1024]
tokens_out = temporal_refiner(tokens_out, timestamp)  # 因果时序精炼
(pose, shape, trans), cam_aux = handec.decode_token(
    rearrange(tokens_out, "b t d -> (b t) d"), ...
)
out_frames = T
```

### 3.8.2 `predict_full` — 推理辅助

```python
@torch.inference_mode()
def predict_full(self, img, bbox, focal, princpt, ...):
    pose, shape, trans, _ = predict_mano_param(...)
    # 只取最后一帧
    pose, shape, trans = pose[:, -1:], shape[:, -1:], trans[:, -1:]
    joint_rel, vert_rel = mano_to_pose(pose, shape)
    joint_cam = joint_rel + trans[:, :, None, :]
    vert_cam = vert_rel + trans[:, :, None, :]
    return {mano_pose_pred, mano_shape_pred, trans_pred,
            joint_cam_pred, vert_cam_pred, joint_rel_pred, vert_rel_pred,
            norm_scale, norm_valid}
```

### 3.8.3 Stage 依赖的 `train()` / `eval()` 行为

`src/model/net.py:493-506`:

| 模式 | Stage1 | Stage2 |
|------|--------|--------|
| `train(True)` | backbone 按 `freeze_backbone` 决定 | backbone/embedder/decoder 全部 frozen, 只训练 temporal_refiner |
| `train(False)` | 全部 eval | 全部 eval |

### 3.8.4 优化器参数组

`src/model/net.py:448-480` 中的 `get_optim_param_dict`:

- **Stage1**: Group 1: `persp_info_embedder + handec` @ `lr=1e-4`; Group 2 (可选): `backbone` @ `backbone_lr=1e-5`
- **Stage2**: 单组: `temporal_refiner` @ `lr=1e-4`
