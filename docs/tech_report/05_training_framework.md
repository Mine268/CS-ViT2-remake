# 第 5 章：训练框架

> 本章覆盖完整的训练基础设施：Hydra 配置系统、两阶段训练策略、total_samples→total_step 换算、LR 调度、混合精度、checkpoint 管理、non-finite guard、SwanLab 集成和 tmux 后台训练。

---

## 5.1 Hydra 配置系统

### 5.1.1 配置组合

```
stage1.yaml:
  defaults: [data, model, loss, tracker, _self_]
  # _self_ 使得 stage1.yaml 中的字段可以覆盖 data/model/loss/tracker 中的默认值

stage2.yaml:
  defaults: [stage1, _self_]
  # 继承 stage1 的全部配置，仅覆盖不同的字段
```

### 5.1.2 配置组

| 配置组 | 文件 | 说明 |
|--------|------|------|
| `GENERAL` | stage1.yaml / stage2.yaml | 训练超参数、seed、步数 |
| `DATA` | data.yaml | 数据集注册、采样策略 |
| `MODEL` | model.yaml | 模型结构超参数 |
| `TRAIN` | stage1.yaml / stage2.yaml | 优化器、增强、batch 等 |
| `LOSS` | loss.yaml | 损失权重和参数 |
| `TRACKER` | tracker.yaml | SwanLab 追踪设置 |
| `TEST` | stage1.yaml | 推理/导出设置 |

大写组对应 OmegaConf struct，嵌套键为小写。

---

## 5.2 两阶段训练策略

### 5.2.1 Stage1 — 逐帧预训练

```
T = 1 (单帧)
训练参数: persp_info_embedder + handec (lr=1e-4) + backbone (可选, lr=1e-5)
冻结参数: temporal_refiner
total_samples: 25,600,000 (≈ 100k steps × 32 spd × 8 GPU)
数据 split: train_stage1 (clip_len=1, stride=1)
bbox_jitter: temporal_mode=frame (每帧独立)
```

### 5.2.2 Stage2 — 时序精炼

```
T = 7 (7 帧 clip)
训练参数: temporal_refiner only (lr=1e-4)
冻结参数: backbone + persp_info_embedder + handec
初始化: 从 Stage1 best model 加载权重 (MODEL.stage1_weight)
total_samples: 3,360,000 (≈ 70k steps × 6 spd × 8 GPU)
数据 split: train_stage2 (clip_len=7, stride=4)
bbox_jitter: temporal_mode=clip (clip 内共享 base jitter + 逐帧小噪声)
```

### 5.2.3 Stage2 数据集差异

FreiHAND 和 RHD 的原始序列长度不足以产生 T=7 的 clips，因此 Stage2 的 aux 组中排除：

```yaml
# stage2.yaml
DATA.train.groups.aux:
  _delete_: true  # 覆盖 stage1 的 aux
  weight: 0.80
  datasets:
    InterHand2.6M: 0.3636363636
    MTC: 0.2727272727
    DexYCB: 0.1818181818
    HO3D_v3: 0.1818181818
```

Stage2 aux 权重重新归一化，总和仍为 1.0。

---

## 5.3 total_samples → total_step 换算

`src/train/engine.py:403-420`:

```python
total_samples = int(cfg.GENERAL.get("total_samples", 0))
if total_samples > 0:
    total_batch = (sample_per_device × num_gpus × grad_accum_step)
    total_step = ceil(total_samples / total_batch)
    # 更新 config
    OmegaConf.update(cfg, "GENERAL.total_step", total_step, force_add=True)
    print(f"total_samples={total_samples} → total_step={total_step} "
          f"(batch={total_batch} = {spd}×{ngpus}×{grad_accum})")
elif "total_step" not in cfg.GENERAL:
    raise ValueError("Either GENERAL.total_samples or GENERAL.total_step must be set")
```

**关键设计**：`total_samples` 使训练数据量与 GPU 数量解耦——同样的 `total_samples` 值在任何 GPU 数量下都保证模型看到相同数量的样本。

**数值对应**：

| Config | total_samples | spd | GPUs | total_batch | total_step |
|--------|-------------|-----|------|-------------|------------|
| stage1 (8卡) | 25,600,000 | 32 | 8 | 256 | 100,000 |
| stage2 (8卡) | 3,360,000 | 6 | 8 | 48 | 70,000 |
| stage1 (make默认, 4卡) | 16,800,000 | 42 | 4 | 168 | 100,000 |

在 4 卡场景下，`total_step` 翻倍（每个 GPU 处理更多 batch），但 `total_samples` 保持一致。

---

## 5.4 训练循环详解

`src/train/engine.py:392-629` 中的 `train(cfg)`:

### 5.4.1 初始化

```python
# 1. 创建 accelerator
accelerator = create_accelerator(cfg)
# mixed_precision=bf16, gradient_accumulation_steps=1
# dispatch_batches=False (WebDataset 兼容)
# find_unused_parameters=False

# 2. 计算 total_step
total_samples → total_step

# 3. 设置 seed
set_seed(cfg.GENERAL.seed)  # 3229084

# 4. 构建输出目录
output_dir = build_run_dir(description)
# → checkpoint/YYYY-MM-DD/YYYY-MM-DD-HH-MM-SS-<slug>/
# 可通过环境变量 CSVIT2_RUN_DIR 覆盖

# 5. 保存配置快照
save_config_snapshot(cfg, output_dir, config_name)

# 6. 初始化 tracker (SwanLab)
tracker = Tracker(cfg, accelerator, experiment_name=basename(output_dir))

# 7. 构建数据加载器
train_loader = build_train_dataloader(cfg)
val_loader = build_eval_dataloader(...)

# 8. 构建模型
net = setup_model(cfg)  # PoseNet

# 9. 优化器
optimizer = AdamW(net.get_optim_param_dict(lr, backbone_lr), weight_decay=1e-4)

# 10. 学习率调度器
scheduler = get_linear_warmup_constant_schedule(optimizer, warmup_steps=5000)
# 前 5000 步线性 warmup (0 → target_lr)，之后常数

# 11. 像素增强
pixel_aug = PixelLevelAugmentation(cfg.TRAIN.augmentation)

# 12. Accelerate 包装
net, optimizer, train_loader, scheduler, val_loader = accelerator.prepare(...)

# 13. 可选 resume
if cfg.GENERAL.resume_path:
    accelerator.load_state(resume_path)
```

### 5.4.2 主循环每一步

```python
while global_step < total_step:
    # 1. 获取 batch
    batch_origin = next(train_iter)  # 无限迭代器

    # 2. Dropout 调度
    dropout_rate = get_progressive_dropout(step=global_step, ...)
    net.set_dropout_rate(dropout_rate)

    # 3. 预处理 + 数据增强
    batch, trans_2d_mat, _ = preprocess_batch(
        batch_origin, patch_size=[224,224], patch_expanstion=2.0,
        scale_z_range=[1.0,1.0], scale_f_range=[1.0,1.0],
        persp_rot_max=0.0873, augmentation_flag=True, device=device,
        pixel_aug=pixel_aug, perspective_normalization=False,
        bbox_jitter=cfg.TRAIN.bbox_jitter,
    )

    # 4. 前向 + 反向 (梯度累积上下文)
    with accelerator.accumulate(net):
        output_state = net(batch)
        loss = output_state["loss"]

        # 4a. Forward NaN 检查
        if not isfinite(loss):
            save_nonfinite_step_artifacts(trigger_type="forward_loss")
            raise RuntimeError("Non-finite loss")

        # 4b. 反向传播
        accelerator.backward(loss)

        # 4c. 梯度 NaN 检查
        if any non-finite grad:
            save_nonfinite_step_artifacts(trigger_type="backward_grad")
            raise RuntimeError("Non-finite gradient")

        # 4d. 梯度裁剪
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(net.parameters(), max_grad=1.0)

        # 4e. 优化器步进
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        # 4f. 参数 NaN 检查
        if any non-finite param:
            save_nonfinite_step_artifacts(trigger_type="post_step_param")
            raise RuntimeError("Non-finite parameter")

    # 5. 步进
    global_step += 1

    # 6. 日志 (每 log_step=5 步)
    tracker.log_scalars({loss_total, loss_*, train/micro_*_*}, step, split="train")

    # 7. 可视化 (每 vis_step=100 步)
    if log_images:
        vis_img = vis(batch, trans_2d_mat, result, ...)
        tracker.log_image("vis", vis_img, step)

    # 8. Checkpoint + 验证 (每 checkpoint_step 步)
    if global_step % checkpoint_step == 0:
        # 保存 rolling checkpoint
        accelerator.save_state(f"checkpoint-{global_step}")
        manage_checkpoints(output_dir, keep_last_n=3)

        # 运行验证
        val_metrics = validate(...)

        # 更新 best model
        if val_metrics["micro_rte_ego"] < best_info["best_value"]:
            save_best_model_variant(...)  # 保存到 best_model/
```

### 5.4.3 验证流程

`src/train/engine.py` 中的 `validate()`:

```python
@torch.no_grad()
def validate(cfg, accelerator, net, val_loader, global_step, tracker, max_eval_steps):
    net.eval()
    meter = StreamingMetricMeter()
    meter.reset()

    for val_step, batch_origin in enumerate(val_loader):
        if val_step >= eval_step_cap:
            break

        batch, _, _ = preprocess_batch(batch_origin, augmentation_flag=False, ...)
        output_state = net(batch)

        # 构建 ego/aux masks
        ego_mask = build_dataset_group_mask(data_source, ego_datasets, ...)
        aux_mask = build_dataset_group_mask(data_source, aux_datasets, ...)

        # 跨 rank 收集
        tensors = accelerator.gather_for_metrics([
            joint_cam_gt[:, -1:], joint_cam_pred[:, -1:],
            verts_cam_gt[:, -1:], verts_cam_pred[:, -1:],
            has_mano[:, -1:], joint_3d_valid[:, -1:],
            ego_mask, aux_mask
        ])

        # 累加统计
        meter.update(...)

    # 计算最终指标
    metrics = meter.compute()  # {val/micro_mpjpe_all, val/micro_rte_ego, ...}
    tracker.log_scalars(metrics, step=global_step, split="val")
    net.train()
    return metrics
```

### 5.4.4 验证超时保护

`StepTimeoutTerminator` (`src/train/engine.py`):
- 基于线程的看门狗
- 在 `__enter__` 时启动 daemon 线程，sleep `timeout_seconds`
- 超时时调用 `os._exit(124)` 硬终止（防止 DDP 死锁）
- `__exit__` 时取消

### 5.4.5 多卡验证的 Clip 均衡

```python
# build_eval_dataloader 中:
clip_counts = estimate_wds_shard_clip_counts(urls, num_frames, stride)
segments = build_balanced_clip_segments(urls, clip_counts, num_ranks)
equalize_rank_clip_segments(segments)  # 修剪到相同步数
```

---

## 5.5 学习率调度

`src/train/engine.py:55-66` 中的 `get_linear_warmup_constant_schedule`:

```python
def lr_lambda(current_step):
    if warmup_steps <= 0:
        return 1.0
    if current_step < warmup_steps:
        return current_step / max(1, warmup_steps)  # 0 → 1
    return 1.0  # 常数
```

- `warmup_steps = 5000`
- Warmup 后保持常数学习率，不使用 cosine annealing

Stage1 LR: `lr=1e-4` (persp+decoder), `backbone_lr=1e-5` (backbone)
Stage2 LR: `lr=1e-4` (temporal_refiner only)

---

## 5.6 优化器配置

```python
optimizer = AdamW(
    net.get_optim_param_dict(lr, backbone_lr),  # 分参数组
    weight_decay=1e-4
)
```

- `grad_accum_step = 1` (无梯度累积)
- `max_grad = 1.0` (梯度裁剪)
- `mixed_precision = bf16`
- `set_to_none=True` (梯度清零优化)

---

## 5.7 Checkpoint 管理

`src/train/checkpoint.py`:

### 5.7.1 输出目录结构

```
checkpoint/YYYY-MM-DD/YYYY-MM-DD-<run_name>/
├── config_stage1.yaml            # 完整解析后的 Hydra 配置
├── tmux.log                       # 训练日志 (tee)
├── checkpoints/
│   ├── checkpoint-95000/          # Rolling checkpoint (保留最近 3 个)
│   ├── checkpoint-100000/
│   └── ...
├── best_model/                    # Best validation model
│   ├── model.safetensors
│   ├── optimizer.bin
│   ├── scheduler.bin
│   └── random_states_0.pkl
├── best_model.json                # Best metrics 元数据
└── nonfinite_stop/                # NaN 诊断输出 (仅触发时)
    └── step-XXXXXX/
```

### 5.7.2 best_model.json 格式

```json
{
  "step": 95000,
  "config_name": "stage1",
  "timestamp": "2026-05-17T17:13:52.961071",
  "best_dir_name": "best_model",
  "micro_mpjpe_all": 26.058,
  "micro_mpjpe_ego": 26.058,
  "micro_rte_ego": 18.067,
  ...
}
```

### 5.7.3 运行名生成

```python
def build_run_dir(description):
    env_dir = os.environ.get("CSVIT2_RUN_DIR")  # tmux 脚本设置
    if env_dir: return env_dir

    env_name = os.environ.get("CSVIT2_RUN_NAME")
    now = datetime.now()
    date_dir = now.strftime("%Y-%m-%d")
    if env_name:
        run_name = ensure_date_prefixed(env_name, now)
    else:
        run_name = ensure_date_prefixed(
            f"{now:%H-%M-%S}-{slugify(description)[:80]}", now
        )
    return f"checkpoint/{date_dir}/{run_name}"
```

---

## 5.8 Non-Finite Guard

`src/train/nan_guard.py` — 三阶段 NaN/Inf 检测：

### 5.8.1 检测阶段

| 阶段 | trigger_type | 检查对象 | 位置 |
|------|-------------|---------|------|
| 1 | `"forward_loss"` | 前向 loss | `engine.py` after forward |
| 2 | `"backward_grad"` | 梯度 | `engine.py` after backward |
| 3 | `"post_step_param"` | 参数更新后 | `engine.py` after optimizer.step() |

### 5.8.2 诊断保存

触发时 `save_nonfinite_step_artifacts` 保存：

```
nonfinite_stop/step-XXXXXX/
├── summary.json          # 触发类型、offending ranks、世界大小
├── rank00_batch.pt       # 每个 rank 的完整 batch + model state (detach→cpu)
├── rank01_batch.pt
├── ...
└── model_state/          # 完整 accelerator state
```

### 5.8.3 Rank 收集

```python
local_rank_code = tensor([rank if triggered else -1])
gathered = accelerator.gather(local_rank_code)
offending_ranks = [r for r in gathered if r >= 0]
```

---

## 5.9 SwanLab 集成

`src/train/tracker.py` 中的 `Tracker`:

```python
class Tracker:
    def __init__(self, cfg, accelerator, experiment_name=None):
        self.enabled = bool(cfg.TRACKER.enabled) and accelerator.is_main_process
        # enabled=false → 所有方法变为 no-op (调试跑必须)
        if not self.enabled: return

        swanlab.init(
            project="cs-vit2-remake",
            workspace="mine268",
            mode="cloud",
            config=OmegaConf.to_container(cfg, resolve=True),  # 完整配置
            experiment_name=experiment_name,
        )

    def log_scalars(self, metrics, step, split):
        payload = {f"{split}/{key}": to_scalar(val) for key, val in metrics.items()}
        swanlab.log(payload, step=step, print_to_console=True)

    def log_image(self, name, image, step, split):
        swanlab.log({f"{split}/{name}": swanlab.Image(np.asarray(image))}, step=step)

    def finish(self):
        self.run.finish()  # 优雅关闭
```

`TRACKER.enabled=false` 可在 Hydra override 中禁用。

---

## 5.10 Tmux 后台训练

`script/run_train_tmux.sh`:

```bash
# 创建 detached tmux session
tmux new-session -d -s "${SESSION_NAME}" "bash -lc '...'"

# 设置 tmux 环境变量
tmux set-environment -t "${SESSION_NAME}" CSVIT2_RUN_DIR "${RUN_DIR}"
tmux set-environment -t "${SESSION_NAME}" CSVIT2_RUN_NAME "${RUN_NAME}"
tmux set-environment -t "${SESSION_NAME}" CSVIT2_LOG_FILE "${LOG_FILE}"

# 训练命令
cd ROOT_DIR && source .venv/bin/activate && \
  CSVIT2_RUN_DIR=... CSVIT2_RUN_NAME=... \
  accelerate launch --gpu_ids "${GPU_IDS}" --num_processes "${NUM_PROCESSES}" \
    -m script.train --config-name="${CONFIG_NAME}" OVERRIDES \
  2>&1 | tee -a tmux.log
```

### 5.10.1 Makefile 变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `GPU_IDS` | `0,1,2,3` | GPU 设备 ID |
| `NUM_PROCESSES` | `4` | Accelerate 进程数 |
| `CONFIG_NAME` | stage 名 | Hydra 配置名 |
| `RUN_NAME` | 自动生成 | 显式运行名 |
| `OVERRIDES` | 空 | 额外 Hydra overrides |
| `STAGE1_DEFAULT_OVERRIDES` | `TRAIN.sample_per_device=42 LOSS.heatmap_sigma=4.0` | Stage1 默认覆盖 |
| `STAGE2_DEFAULT_OVERRIDES` | `TRAIN.sample_per_device=6 LOSS.heatmap_sigma=4.0` | Stage2 默认覆盖 |
| `DINO_STAGE1_LARGE_DEFAULT_OVERRIDES` | `LOSS.heatmap_sigma=4.0` | DINOv3-L/16 Stage1 默认覆盖 |
| `DINO_STAGE1_LARGE_TI_DEFAULT_OVERRIDES` | `LOSS.heatmap_sigma=4.0 MODEL.ti.enabled=true` | DINOv3-L/16 + TI Stage1 默认覆盖 |
| `DINO_STAGE1_DEFAULT_OVERRIDES` | `LOSS.heatmap_sigma=4.0` | DINOv3-H+/16 Stage1 默认覆盖 |
| `DINO_STAGE2_DEFAULT_OVERRIDES` | `LOSS.heatmap_sigma=4.0` | DINOv3 Stage2 默认覆盖 |

### 5.10.2 Make 目标

```bash
make                                                 # 进入 Python 训练 TUI
make shell                                           # 进入 Python 训练 TUI
make train-stage1                                    # 启动 stage1
make train-stage2 STAGE1_WEIGHT=/path/to/best_model  # 启动 stage2
make train-stage1-dinov3-large                       # 启动 DINOv3-L/16 stage1
make train-stage1-dinov3-large-ti                    # 启动 DINOv3-L/16 stage1 + TI
make train-stage2-dinov3-large STAGE1_WEIGHT=/path   # 启动 DINOv3-L/16 stage2
make train-stage1-dinov3                             # 启动 DINOv3-H+/16 stage1
make train-stage2-dinov3 STAGE1_WEIGHT=/path         # 启动 DINOv3-H+/16 stage2
make attach-stage1                                   # 接入 tmux
make logs-stage1                                     # tail 日志
make stop-stage1                                     # 停止
make train-stage1 DRY_RUN=1                          # 仅打印命令
```

Python TUI 内部仍然复用这些显式 Make target，因此已有脚本无需改写。
