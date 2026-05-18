# 第 7 章：工程问题与解决方案

> 本章记录项目开发中遇到的实际工程问题、解决方案和经验教训。

---

## 7.1 Clip-Native 数据迁移 (10.4x 加速)

### 问题

原始训练数据以**完整 sequence** 为单位存储（`/data_0/renkaiwen/webdatasets2_512/`）。Stage1 只需 `T=1` 单帧，但数据加载器仍需反序列化整条 sequence（可能数百帧），再从其中切片出目标帧。这导致 loader + preprocess 吞吐仅 **10.2 samples/s**，严重影响训练效率。

### 解决方案

编写 `script/export_train_clips.py` 预切分工具，将 sequence 格式数据重新导出为 clip-native 格式：

1. 读取原始 sequence shards（`WebDataset, shardshuffle=false`）
2. 按 stage-aware stride 构建滑窗 clip 序列
3. 序列化为 JSON/pickle/NPY 成员写入新的 tar
4. 每个输出 tar 控制在约 1GB（`max_tar_size_bytes = 1GB`）

**Stage-aware 参数**：
- `stage1`: `clip_len=1, stride=1`
- `stage2`: `clip_len=7, stride=4`

Stage2 的 `stride=4` 使相邻 clips 仍有重叠，但不会像 `stride=1` 那样导致数据量爆炸。

### 结果

| 指标 | 旧格式 (Sequence) | 新格式 (Clip-Native) | 加速比 |
|------|------------------|---------------------|--------|
| Loader-only | 14.8 samples/s | 100.6 samples/s | 6.8x |
| Loader + Preprocess | 10.2 samples/s | 106.1 samples/s | **10.4x** |

### 后续清理

- `config/data.yaml` 中只保留新数据路径（`/data_0/renkaiwen/webdatasets2_remake/`）
- 旧的 sequence 数据路径（`/data_0/renkaiwen/webdatasets2_512/`）仅在 `export_train_clips.py` 中作为源数据引用

---

## 7.2 BBox Jitter 的约束性改进

### 问题

传统 bbox jitter 独立采样 scale 和 center shift：`scale ~ LogU(min, max)` 和 `center_noise ~ U(-1, 1) * cap` 不关联。可能导致 bbox 中心漂移 ±50% 边长但 scale 仅 1.01，几何上不合理——bbox 偏离手部区域。

### 解决方案

引入约束性采样方案（`_jitter_hand_bbox` 的 `constrained=true` 分支）：

1. Scale 膨胀因子 α 采样为对数正态（`σ=0.15, μ=0.36`）
2. 最大允许偏移 `max_shift = (α−1)/2`——数学上保证 jitter 后的 bbox 仍然包含原始 tight bbox 的中心区域
3. 实际偏移上限取 `min(用户设定上限, max_shift)`

**验证**：2026-05-08 实时 demo 复盘确认，约束性 bbox jitter 显著提升了模型在 WiLoR detector 上的泛化能力。在 AssemblyHands val 上用 WiLoR 评测，有 jitter 训练的模型漏检更少、姿态更准。

---

## 7.3 OmegaConf Struct Mode

### 问题

`engine.py` 中需要动态添加 `GENERAL.total_step` 到 config，但 OmegaConf DictConfig 为 struct mode：

```python
cfg.GENERAL.total_step = value  # ConfigAttributeError: Key 'total_step' not in struct
```

### 解决方案

使用 `OmegaConf.update` 并传递 `force_add=True`：

```python
from omegaconf import OmegaConf
OmegaConf.update(cfg, "GENERAL.total_step", value, force_add=True)
```

---

## 7.4 分布式训练进程误杀

### 问题

调试 GPU 5 时发现有进程占用 20GB VRAM（PID 1200683），直接 `kill 1200683`。但该 PID 是用户 4-GPU 分布式训练的**主进程**（torchrun/accelerate），SIGTERM 级联杀死了 GPU 0,1,3,4 上所有子进程——导致一个运行至 step 50000 的 stage1 训练被终止。

### 教训

1. 杀进程前必须：`ps aux | grep <PID>` 查看完整命令行 + `nvidia-smi` 查看 GPU 占用
2. 分布式训练架构：1 个父进程 + N 个子进程——杀父进程会级联杀死所有子进程
3. 不确定时先问用户确认

---

## 7.5 SwanLab 调试污染

### 问题

调试/冒烟训练产生大量无意义的 SwanLab 记录，污染实验历史（如 `total_samples=32` 的 3 步 smoke test 也会产生一个完整的 SwanLab run）。

### 解决方案

1. **代码层面**：`Tracker` 类检查 `cfg.TRACKER.enabled`，为 false 时所有 `log_scalars`、`log_image`、`finish` 变为 no-op，SwanLab 甚至不被 import
2. **使用规范**：调试跑必须在 Hydra override 中加 `TRACKER.enabled=false`

```bash
make train-stage1 GPU_IDS=5 NUM_PROCESSES=1 OVERRIDES="GENERAL.total_samples=32 ... TRACKER.enabled=false"
```

---

## 7.6 torch.compile 不可用

### 问题

尝试对模型应用 `torch.compile` 时遇到 graph breaks 和 segfault。

### 状态

`torch.compile` 暂不启用，作为未来优化方向。可能原因：
- `kornia` 的某些操作不支持 dynamo
- 自定义 Transformer decoder 的复杂控制流
- smplx MANO 层的非标准操作

---

## 7.7 AssemblyHands Test Set 占位值

### 问题

`AssemblyHands test-eccv2024` 的公开 JSON 文件中：
- 2D keypoints 是占位值（非真实坐标）
- 3D joints 是占位值
- extrinsics 是占位值

这意味着该 split **不可作为本地 GT 测试集**使用。

### 状态

- `AssemblyHands val` 的所有标注真实可用，作为唯一的本地验证集
- `AssemblyHands test-eccv2024` 仅用于格式/提交模板参考
- `HOT3D val` 尚未接入——官方公开口径下更适合从训练集切分本地验证集

---

## 7.8 Stage2 数据集排除

### 问题

`FreiHAND` 和 `RHD` 的原始 sequence 长度不足以产生 `T=7` 的 clips。如果包含它们会导致 `clip_to_t_frames` 跳过所有样本（`total_frames < num_frames`）。

### 解决方案

在 `config/stage2.yaml` 中使用 `_delete_: true` 覆盖 stage1 的 aux 组：

```yaml
DATA:
  train:
    groups:
      aux:
        _delete_: true  # 完全替换，而非合并
        weight: 0.80
        datasets:
          InterHand2.6M: 0.3636363636
          MTC: 0.2727272727
          DexYCB: 0.1818181818
          HO3D_v3: 0.1818181818
```

数据管线中 `_resolve_dataset_split_sources_if_available` 函数支持静默跳过无对应 split 的数据集，配置验证时不会报错。

---

## 7.9 多卡验证的 Clip 均衡

### 问题

多卡验证时，不同的 tar shard 可能包含不同数量的 clips，导致不同 rank 分到的 clip 数量不同。当使用 `accelerator.gather_for_metrics` 时，不同长度的 tensor 收集会导致 NCCL 死锁或数据错误。

### 解决方案

1. `estimate_wds_shard_clip_counts`：不解码图像，仅读取 `imgs_path.json` 估算每个 shard 的 clip 数
2. `build_balanced_clip_segments`：将全局 clip 范围按 rank 数量均匀划分为连续段
3. `equalize_rank_clip_segments`：将所有 rank 修剪到最小 clip 数（`target_count = min(local_counts)`）
4. `get_shared_eval_max_steps`：确保所有 rank 运行相同步数的验证

---

## 7.10 WebDataset dispatch_batches=False

### 问题

默认的 Accelerate DataLoader 配置会在 rank 0 上 split 和 concatenate iterable-style dataset 的 batch 项，但 WebDataset 的 batch 包含字符串元数据（`__key__`、`data_source` 等）和 list 类型的图像张量，无法被简单地 split/concat。

### 解决方案

```python
DataLoaderConfiguration(dispatch_batches=False)
```

此设置下每个进程独立获取自己的 batch，Accelerate 不做任何 batch 分发处理。

---

## 7.11 MANO Mask Token 梯度泄漏

### 问题

DINOv2 等 MIM 预训练模型包含 `mask_token` 参数，在微调时该参数仍然可训练。但项目从不使用 mask token（不做 MIM），该参数会在每次迭代中接收梯度但不参与前向计算。

### 解决方案

在 `PoseNet.__init__` 中显式禁用：

```python
mask_token = getattr(getattr(self.backbone.backbone, "embeddings", None), "mask_token", None)
if isinstance(mask_token, nn.Parameter):
    mask_token.requires_grad_(False)
```

---

## 7.12 数据源别名歧义

### 问题

不同数据集的 shard 内部可能使用不同的 `data_source` 命名（例如 `interhand26m` vs `InterHand2.6M` vs `interhand_26m`）。如果仅靠 shard 内的原始字符串做 ego/aux 路由，会出现漏匹配。

### 解决方案

1. 配置驱动的别名系统：`build_data_source_alias_map` 从 `DATA.datasets` 构建标准化映射
2. `normalize_data_source_alias_key`：小写 + 去除非字母数字字符→ 统一的查找键
3. 训练时使用 `force_data_source=True`：数据源名称由配置直接指定，不依赖 shard 内部标记

---

## 7.13 环境变量驱动的运行目录

### 问题

tmux 训练脚本需要在启动tmux时确定运行目录、运行名和日志路径，但这些信息需要在训练进程内部也能访问（用于 checkpoint 和 SwanLab 命名）。

### 解决方案

使用环境变量在 tmux 脚本和训练进程之间传递：
- `CSVIT2_RUN_DIR`：强制运行目录（`build_run_dir` 中优先检查）
- `CSVIT2_RUN_NAME`：强制运行名
- `CSVIT2_LOG_FILE`：日志文件路径

tmux 脚本通过 `tmux set-environment` 将这些变量注入 session。

---

## 7.14 性能基准

### Stage1 训练结果

最新 best model (step 95000, 4 GPU, `sample_per_device=42`, `total_samples=16,800,000`):

| 验证指标 | Best (step 95000) | Final (step 100000) |
|----------|-------------------|---------------------|
| `val/micro_rte_ego` | **18.07 mm** | 25.65 mm |
| `val/micro_mpjpe_ego` | **26.06 mm** | 33.08 mm |
| `val/micro_mpjpe_rel_ego` | **20.34 mm** | 20.78 mm |

Step 100000 出现性能反弹，best model 在 95000 是合理的。

### Demo 评估 (best model)

| 数据集 | 检测器 | 帧数 | 检出率 | 备注 |
|--------|--------|------|--------|------|
| AssemblyHands val | GT bbox | 20 | 100% (20/20) | 上界性能 |
| AssemblyHands val | WiLoR | 20 | 70% (14/20) | 6 帧漏检 (遮挡/模糊) |
| HOT3D train | GT bbox | 10 | - (10/10) | GT 仅单手标注 |
| HOT3D train | WiLoR | 10 | 200% (20/10) | 每帧双手检出 (>GT) |

---

## 7.15 测试覆盖

12 个 pytest 测试文件覆盖核心功能：

| 测试文件 | 覆盖内容 |
|---------|---------|
| `test_bbox_jitter.py` | Bbox jitter 增强逻辑 |
| `test_config_smoke.py` | 配置加载冒烟测试 |
| `test_data_source_routing.py` | 数据集别名路由 |
| `test_demo_stage1.py` | Demo 脚本功能 |
| `test_eval_dataloader.py` | 评估数据加载器 |
| `test_export_clips.py` | Clip 导出管线 |
| `test_metric_splits.py` | 指标按数据集组计算 |
| `test_patch_uv_rho_multibin.py` | Camera head 编解码 |
| `test_run_name.py` | 运行名生成 |
| `test_supervision_matrix.py` | 损失路由验证 |
| `test_tracker.py` | Tracker 初始化 |
| `test_train_data_plan.py` | 训练数据计划构建 |
