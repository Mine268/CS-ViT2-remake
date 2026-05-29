# 第 8 章：配置参考

> 本章提供所有 Hydra 配置文件的完整参数表和 Stage1/Stage2 差异对照。

---

## 8.1 config/data.yaml — 数据配置

### DATA.datasets (数据集注册表)

| 数据集 | 别名 | 可用 Splits |
|--------|------|-------------|
| AssemblyHands | `assemblyhands` | train_stage1, train_stage2, val_stage1, val_stage2 |
| DexYCB | `dexycb` | train_stage1, train_stage2 |
| FreiHAND | `freihand` | train_stage1 |
| HO3D_v3 | `ho3d` | train_stage1, train_stage2 |
| HOT3D | `hot3d` | train_stage1, train_stage2 |
| InterHand2.6M | `ih26m` | train_stage1, train_stage2 |
| MTC | `mtc` | train_stage1, train_stage2 |
| RHD | `rhd` | train_stage1 |

所有路径格式：`/data_0/renkaiwen/webdatasets2_remake/{Dataset}/{split}/*.tar`

### DATA.train (训练)

| 参数 | 值 | 说明 |
|------|-----|------|
| `split` | `train_stage1` | Stage2 覆盖为 `train_stage2` |
| `stride` | 1 | Clip 采样步长 |
| `shardshuffle` | 64 | Shard 级洗牌缓冲区 |
| `post_clip_shuffle` | 64 | Clip 级洗牌缓冲区 |

**两级采样权重** (Stage1 默认)：

| 组 | 组权重 | 数据集权重 |
|----|--------|-----------|
| ego | 0.20 | HOT3D: 0.50, AssemblyHands: 0.50 |
| aux | 0.80 | InterHand2.6M: 0.25, FreiHAND: 0.1875, MTC: 0.1875, DexYCB: 0.125, HO3D_v3: 0.125, RHD: 0.125 |

**Stage2 aux 权重** (覆盖)：

| 数据集 | 权重 |
|--------|------|
| InterHand2.6M | 0.3636 |
| MTC | 0.2727 |
| DexYCB | 0.1818 |
| HO3D_v3 | 0.1818 |

**过滤器**：

| 参数 | 值 |
|------|-----|
| `filter.enabled` | true |
| `filter.min_valid_joints_2d` | 16 |
| `filter.min_hand_bbox_edge_px` | 8 |
| `filter.frame_policy` | `"all"` |

**采样**：

| 参数 | 值 |
|------|-----|
| `sampling.mode` | `"random_clip"` |
| `sampling.clips_per_sequence` | 1 |

### DATA.val / DATA.test

| 参数 | 值 |
|------|-----|
| `stride` | 1 |
| `filter.enabled` | false |
| `full_eval` | false |
| `max_val_step` | 1000 |
| `sampling.mode` | `"dense"` |
| `shardshuffle` | false |
| `post_clip_shuffle` | 0 |

---

## 8.2 config/model.yaml — 模型配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `img_size` | 224 | 输入 patch 尺寸 |
| `img_mean` | `[0.485, 0.456, 0.406]` | ImageNet 均值 |
| `img_std` | `[0.229, 0.224, 0.225]` | ImageNet 标准差 |
| `stage` | `stage1` | Stage2 覆盖为 `stage2` |
| `stage1_weight` | null | Stage2 必须指定 |
| `num_frame` | 1 | Stage2: 7 |
| `joint_type` | `"3"` | 旋转表示 (3/6d/quat) |
| `norm_by_hand` | false | 不支持 |

### MODEL.backbone

| 参数 | 值 |
|------|-----|
| `backbone_str` | `model/facebook/dinov2-large` |
| `infusion_layer` | null |
| `drop_cls` | false |
| `kwargs` | `{}` |

### MODEL.handec (Hand Decoder)

| 参数 | 值 |
|------|-----|
| `num_layer` | 4 |
| `num_head` | 16 |
| `dim_head` | 64 |
| `dim_mlp` | 4096 |
| `dropout` | 0.1 |
| `norm` | `"layer"` |
| `context_dim` | 1024 |
| `skip_token_embed` | false |
| `use_mean_init` | false |
| `denorm_output` | false |
| `cam_head_type` | `"patch_uv_rho_multibin"` |
| `heatmap_resolution` | `[512, 512, 1024]` |

### MODEL.handec.root_z (Rho Head)

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_bins` | 8 | Multibin 数量 |
| `d_min` | -0.71 | Δlog ρ 下界 |
| `d_max` | 0.75 | Δlog ρ 上界 |
| `prior_k` | 121.0 | 先验尺度常数 |
| `geom_hidden_dim` | 256 | 几何特征 MLP 维度 |
| `dropout` | 0.1 | Rho head dropout |
| `use_data_source_embed` | false | 未实现 |

### MODEL.persp_info_embed

| 参数 | 值 |
|------|-----|
| `type` | `"ca"` (仅支持) |
| `num_sample` | 8 |
| `pie_fusion` | `"all"` (保留，忽略) |

### MODEL.temporal_encoder

| 参数 | 值 |
|------|-----|
| `num_layer` | 2 |
| `num_head` | 16 |
| `trope_scalar` | 20.0 |
| `zero_linear` | true |

---

## 8.3 config/loss.yaml — 损失配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `lambda_theta` | 3.0 | MANO pose |
| `lambda_shape` | 3.0 | MANO shape |
| `lambda_uv_patch` | 1.0 | Root UV heatmap CE |
| `lambda_trans` | 0.001 | Root translation L1 |
| `lambda_root_z_cls` | 1.0 | Rho 分类 CE |
| `lambda_root_z_res` | 1.0 | Rho 残差 SmoothL1 |
| `lambda_rel` | 0.012 | Relative joints L1 |
| `lambda_img` | 0.002 | Reprojection RobustL1 |
| `supervise_heatmap` | true | 是否监督 heatmap |
| `heatmap_sigma` | 4.0 | Gaussian target σ |
| `pred_joint_z_min_mm` | 20.0 | 重投影最小 Z |
| `reproj_loss_type` | `"robust_l1"` | 重投影损失类型 |
| `reproj_loss_delta` | 84.0 | RobustL1 过渡点 |
| `root_filter.min_valid_joints_2d` | 16 | 最少 2D 关节点 |
| `root_filter.min_hand_bbox_edge_px` | 8 | 最小 bbox 边长 |

---

## 8.4 config/tracker.yaml — 追踪配置

| 参数 | 值 |
|------|-----|
| `enabled` | true |
| `type` | `"swanlab"` |
| `project` | `"cs-vit2-remake"` |
| `workspace` | `"mine268"` |
| `mode` | `"cloud"` |
| `print_to_console` | true |
| `log_images` | true |

---

## 8.5 config/stage1.yaml — Stage1 训练

### GENERAL

| 参数 | 值 |
|------|-----|
| `description` | `"stage1 no_norm patch_uv_rho_multibin"` |
| `total_samples` | 25,600,000 |
| `log_step` | 5 |
| `vis_step` | 100 |
| `checkpoint_step` | 5000 |
| `warmup_step` | 5000 |
| `dropout_warmup_step` | 10000 |
| `num_worker` | 2 |
| `prefetch_factor` | 1 |
| `seed` | 3229084 |
| `val_seed` | 42 |
| `resume_path` | null |

### TRAIN

| 参数 | 值 |
|------|-----|
| `lr` | 1e-4 |
| `backbone_lr` | 1e-5 |
| `grad_accum_step` | 1 |
| `max_grad` | 1.0 |
| `weight_decay` | 1e-4 |
| `sample_per_device` | 32 (make 默认覆盖为 42) |
| `mixed_precision` | `"bf16"` |
| `expansion_ratio` | 2.0 |
| `scale_z_range` | `[1.0, 1.0]` |
| `scale_f_range` | `[1.0, 1.0]` |
| `persp_rot_max` | 0.08727 (5°) |
| `perspective_normalization` | false |

### TRAIN.bbox_jitter

| 参数 | 值 |
|------|-----|
| `enabled` | true |
| `prob` | 0.7 |
| `temporal_mode` | `"frame"` |
| `constrained` | true |
| `scale_log_mean` | 0.36 |
| `scale_log_std` | 0.15 |
| `scale_range` | `[1.0, 2.0]` |
| `center_shift` | 0.5 |
| `aspect_ratio_range` | `[0.6, 1.6]` |
| `frame_center_shift` | 0.0 |
| `frame_scale_range` | `[1.0, 1.0]` |
| `min_edge_px` | 8.0 |

### TRAIN.augmentation (像素级增强)

**ColorJitter**: enabled, brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0, p=0.5
**GaussianNoise**: enabled, mean=0.0, std=0.03, p=0.5

### TEST

| 参数 | 值 |
|------|-----|
| `output_dir` | `output/test_results` |
| `batch_size` | 16 |
| `save_format` | `"hdf5"` |
| `compression` | `"gzip"` |
| `max_samples` | null |
| `vis_step` | 100 |
| `enable_vis` | true |
| `checkpoint_path` | null |

---

## 8.6 Stage1 vs Stage2 差异对照

| 参数 | Stage1 | Stage2 | 说明 |
|------|--------|--------|------|
| `GENERAL.description` | `"... stage1 ..."` | `"... stage2 ..."` | |
| `GENERAL.total_samples` | 25,600,000 | 3,360,000 | ~70k steps × 6 spd × 8 GPU |
| `GENERAL.checkpoint_step` | 5000 | 3000 | Stage2 更频繁验证 |
| `DATA.train.split` | `train_stage1` | `train_stage2` | T=1 vs T=7 数据 |
| `DATA.train.groups.aux` | 6 数据集 | 4 数据集 | 排除 FreiHAND, RHD |
| `DATA.val.batch_size` | 16 | 6 | 匹配 sample_per_device |
| `DATA.val.source` | `val_stage1` | `val_stage2` | |
| `MODEL.stage` | `stage1` | `stage2` | |
| `MODEL.num_frame` | 1 | 7 | |
| `TRAIN.sample_per_device` | 32 | 6 | 7 帧需要更多显存 |
| `TRAIN.bbox_jitter.temporal_mode` | `"frame"` | `"clip"` | Clip 内共享 base jitter |
| `TRAIN.bbox_jitter.frame_center_shift` | 0.0 | 0.03 | Stage2 逐帧微扰 |
| `TRAIN.bbox_jitter.frame_scale_range` | `[1.0, 1.0]` | `[0.95, 1.05]` | Stage2 逐帧微扰 |

## 8.7 Makefile 变量

| 变量 | 默认值 | 用途 |
|------|--------|------|
| `SESSION_PREFIX` | `csvit2` | tmux session 名前缀 |
| `GPU_IDS` | `0,1,2,3` | GPU 设备 ID |
| `NUM_PROCESSES` | `4` | Accelerate 进程数 |
| `MAIN_PROCESS_PORT` | `0` | 分布式端口 (auto) |
| `DRY_RUN` | `0` | 仅打印命令 |
| `STAGE1_WEIGHT` | (空) | Stage2 的 stage1 权重路径 |
| `RUN_NAME` | (空) | 显式运行名 (自动加日期前缀) |
| `OVERRIDES` | (空) | 额外 Hydra overrides |
| `CONFIG_NAME` | target 默认值 | Hydra 配置名 |
| `STAGE1_DEFAULT_OVERRIDES` | `TRAIN.sample_per_device=42 LOSS.heatmap_sigma=4.0` | |
| `STAGE2_DEFAULT_OVERRIDES` | `TRAIN.sample_per_device=6 LOSS.heatmap_sigma=4.0` | |
| `DINO_STAGE1_LARGE_DEFAULT_OVERRIDES` | `LOSS.heatmap_sigma=4.0` | DINOv3-L/16 stage1 专用 target 的默认覆盖 |
| `DINO_STAGE1_LARGE_TI_DEFAULT_OVERRIDES` | `LOSS.heatmap_sigma=4.0 MODEL.ti.enabled=true` | DINOv3-L/16 + TI stage1 专用 target 的默认覆盖 |
| `DINO_STAGE1_DEFAULT_OVERRIDES` | `LOSS.heatmap_sigma=4.0` | DINOv3-H+/16 stage1 专用 target 的默认覆盖 |
| `DINO_STAGE2_DEFAULT_OVERRIDES` | `LOSS.heatmap_sigma=4.0` | DINOv3 stage2 专用 target 的默认覆盖 |

## 8.8 常用命令行覆盖示例

```bash
# 进入 Python 训练 TUI
make
make shell

# 修改 batch size
make train-stage1 OVERRIDES="TRAIN.sample_per_device=32"

# 修改损失权重
make train-stage1 OVERRIDES="LOSS.heatmap_sigma=4.0 LOSS.lambda_theta=1.0"

# 切换 backbone
make train-stage1 OVERRIDES="MODEL.backbone.backbone_str=model/facebook/dinov2-base"

# DINOv3-L/16 与 DINOv3-H+/16
make train-stage1-dinov3-large
make train-stage1-dinov3-large-ti
make train-stage2-dinov3-large STAGE1_WEIGHT=/path/to/dinov3_large_stage1/best_model
make train-stage1-dinov3
make train-stage2-dinov3 STAGE1_WEIGHT=/path/to/dinov3_stage1/best_model

# 修改数据采样权重
make train-stage1 OVERRIDES="DATA.train.groups.ego.weight=0.3 DATA.train.groups.aux.weight=0.7"

# 修改学习率
make train-stage1 OVERRIDES="TRAIN.lr=5e-5 TRAIN.backbone_lr=5e-6"

# 禁用 bbox jitter (消融实验)
make train-stage1 RUN_NAME=stage1-no-jitter OVERRIDES="TRAIN.bbox_jitter.enabled=false"

# 续接训练
make train-stage1 OVERRIDES="GENERAL.resume_path=/path/to/checkpoint-50000"

# 调试跑 (禁用 SwanLab)
make train-stage1 GPU_IDS=5 NUM_PROCESSES=1 \
  OVERRIDES="GENERAL.total_samples=32 TRACKER.enabled=false"
```

## 8.9 Python 依赖

```
accelerate==1.12.0      # 分布式训练
einops==0.8.1           # 张量重塑
h5py==3.15.1            # HDF5 输出
hydra-core==1.3.2       # 配置管理
kornia==0.8.2           # 图像变换/增强
numpy==2.2.6            # 数值计算
omegaconf==2.3.0        # 配置对象
opencv-python==4.12.0   # 图像 I/O
rich==14.2.0            # 终端美化
safetensors==0.7.0      # 模型序列化
smplx==0.1.28           # MANO 手部模型
swanlab                 # 实验追踪 (未固定版本)
torch==2.9.1            # 深度学习
torchvision==0.24.1     # 图像处理
transformers==4.57.3    # ViT Backbone
webdataset==1.0.2       # 数据管线
```

可选依赖：
- `dev`: `pytest==9.0.2`
- `demo`: `dill==0.3.8`, `ultralytics==8.1.34` (WiLoR 检测器)
