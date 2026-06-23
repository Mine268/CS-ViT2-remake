# Checkpoint 实验结果盘点

最后更新：2026-06-23。

`checkpoint` 是指向 `/data_0/renkaiwen/CS-ViT2-remake-checkpoints` 的符号链接。下表指标来自各 run 的 `best_model.json`；除特别说明外，均为 AssemblyHands validation 的 ego 指标，单位为 mm。训练引擎当前使用 `micro_rte_ego` 选择 `best_model`。本次同步确认：2026-05-29 之后没有新的 `best_model.json` 写入。

## 有 Best Model 的 Run

| Run | Config | Best step | MPJPE ego | Rel MPJPE ego | RTE ego | 大小 | 备注 |
|-----|--------|-----------|-----------|---------------|---------|------|------|
| `checkpoint/2026-04-24/10-46-54-smoke-stage1-valfix` | `stage1` | 1 | 112.44 | 94.18 | 83.51 | 8.9G | validation 修复 smoke；不是正式结果。 |
| `checkpoint/2026-04-24/10-55-28-smoke-stage2-valfix` | `stage2` | 1 | 102.36 | 101.63 | 68.65 | 3.4G | validation 修复 smoke；不是正式结果。 |
| `checkpoint/2026-04-24/2026-04-24-stage1-evalsync-smoke-v3` | `stage1` | 1 | 78.84 | 81.72 | 114.19 | 8.9G | evaluation sync smoke；不是正式结果。 |
| `checkpoint/2026-04-24/2026-04-24-stage2-evalsync-smoke-v3` | `stage2` | 1 | 78.45 | 81.68 | 114.19 | 3.4G | evaluation sync smoke；不是正式结果。 |
| `checkpoint/2026-04-27/2026-04-27-11-10-34-csvit2-stage1` | `stage1` | 60000 | 28.39 | 23.51 | 17.78 | 18G | 较早的完整 Stage1 baseline；README demo 示例仍引用该权重。 |
| `checkpoint/2026-04-29/2026-04-29-10-52-47-csvit2-stage2` | `stage2` | 6000 | 27.95 | 22.07 | 17.79 | 6.7G | 使用 Stage1 权重启动的完整 Stage2 run。 |
| `checkpoint/2026-05-06/2026-05-06-19-09-07-csvit2-stage1` | `stage1` | 90000 | 31.01 | 23.24 | 22.28 | 18G | bbox jitter 前的 Stage1 参考结果，用于 demo 对比。 |
| `checkpoint/2026-05-08/2026-05-08-stage1-bbox-jitter-v2` | `stage1` | 100000 | 27.34 | 22.98 | 16.67 | 18G | 扫描到的已完成 Stage1 run 中 RTE 最好；bbox jitter 提升了 detector bbox 鲁棒性。 |
| `checkpoint/2026-05-14/2026-05-14-16-57-16-csvit2-stage1` | `stage1` | 30000 | 30.04 | 24.18 | 19.32 | 18G | 中断/部分完成的 Stage1 run。 |
| `checkpoint/2026-05-15/2026-05-15-resume-50000` | `stage1` | 95000 | 26.06 | 20.34 | 18.07 | 18G | 扫描到的 Stage1 DINOv2 run 中 MPJPE/Rel MPJPE 最好；最终 step 有回退，因此 best 在 step 95000。 |
| `checkpoint/2026-05-27/2026-05-27-17-06-35-csvit2-stage1-dinov3-large` | `stage1_dinov3_large` | 40000 | 34.66 | 29.56 | 20.65 | 18G | DINOv3-L/16 run。扫描时仍在运行；命令里带 `TRAIN.sample_per_device=42`，与当前 `stage1_dinov3_large` 默认 batch 一致。 |
| `checkpoint/2026-05-29/2026-05-29-19-36-56-csvit2-stage1-dinov3-large-ti` | `stage1_dinov3_large` + `MODEL.ti.enabled=true` | 275000 | 29.15 | 22.77 | 19.45 | 18G | DINOv3-L/16 + TI 完整 Stage1 run；最终训练到 step 400000，best 出现在 step 275000。 |

## 当前结论

- 按当前 best-model 选择指标 `micro_rte_ego`，已完成 Stage1 checkpoint 中最好的是 `2026-05-08-stage1-bbox-jitter-v2`，RTE 为 `16.67 mm`。
- 按 MPJPE 和相对 MPJPE，扫描到的 Stage1 DINOv2 run 中最好的是 `2026-05-15-resume-50000`，MPJPE 为 `26.06 mm`，Rel MPJPE 为 `20.34 mm`。
- 已完成的 Stage2 候选是 `2026-04-29-10-52-47-csvit2-stage2`，step 6000 的 RTE 为 `17.79 mm`，MPJPE 为 `27.95 mm`。
- DINOv3-L/16 非 TI run 在 step 40000 的 best 为 `MPJPE=34.66 / RTE=20.65`；DINOv3-L/16 + TI 完整 run 在 step 275000 的 best 为 `MPJPE=29.15 / RTE=19.45`。两者不是严格控制变量的完整消融，但说明 TI 路径已经能稳定训练到完整 `total_samples`。
- step 1 的 smoke run 只用于验证 checkpoint/eval 管线，不用于模型效果比较。

## 当前配置口径补充

- `stage1_dinov3_large` 当前默认 `TRAIN.sample_per_device=42`，因此之后用 `make train-stage1-dinov3-large` 启动的新实验，若没有额外 override，都会沿用这个 batch。
- Makefile 已新增 `make train-stage1-dinov3-large-ti`，它在 `stage1_dinov3_large` 的基础上追加 `MODEL.ti.enabled=true`，用于启动 DINOv3-L/16 + TI 的 Stage1 训练。
- 2026-05-29 的早期 DINOv3-L + TI 启动暴露了两个实现问题：`TRAIN.sample_per_device=42` 时 TI 分支重复运行 backbone 导致 OOM；`TRAIN.sample_per_device=32/16` 时 axis-angle 根姿态逆旋转的 `matrix -> axis-angle` 反向在第 0 步产生 non-finite gradients。当前代码已改为 Stage1+TI 共享主分支 tokens，并用稳定 quaternion compose 做 axis-angle 逆旋转。后续新 TI run 应与这些失败启动分开记录。

## 实验 TODO：Transformation Isomorphism 消融

目标：证明 `MODEL.ti` / transformation isomorphism 正则分支的作用，而不是只证明 DINOv3-L 大模型本身有效。

建议做一个二维消融矩阵：

- 模型规模：`stage1` DINOv2-L、`stage1_dinov3_large` DINOv3-L/16、条件允许时加入 `stage1_dinov3` DINOv3-H+/16。
- 数据集/训练源：完整 8-dataset stage1、ego-only（HOT3D + AssemblyHands）、AssemblyHands-only 或 HOT3D-only、本地 cross-domain 验证集可用后再加入 cross-camera/cross-dataset。
- TI 开关：`MODEL.ti.enabled=false` vs `MODEL.ti.enabled=true`。
- 评估：AssemblyHands val 的 MPJPE/RTE/Rel MPJPE，同时按 bbox/visibility/hand-object/camera-domain 等 challenge metadata 分组；如果后续接入 HOT3D val，必须报告跨数据集泛化。

最小启动版本：

```bash
make train-stage1-dinov3-large RUN_NAME=dinov3-large-no-ti
make train-stage1-dinov3-large-ti RUN_NAME=dinov3-large-ti
```

控制要求：

- 使用相同 `GENERAL.total_samples`、`TRAIN.sample_per_device`、`LOSS.heatmap_sigma`、bbox jitter 和数据采样配置。
- 每组至少保留 `best_model.json`、`tmux.log` 和 Hydra config snapshot。
- 若只做单 seed，结论应表述为 “TI improves/does not improve in this setting”；若要作为论文结论，建议至少对关键设置做 2-3 个 seed 或 bootstrap 置信区间。
- 重点关注 TI 是否在 camera/scale/rotation 变化更强的 regime 上收益更明显；如果只在 in-domain 平均指标上提升，科研解释力度不足。

## 没有 `best_model.json` 的 Run

下面这些目录没有记录 best model，主要是早期 smoke/debug、失败启动或只留下日志/普通 checkpoint shard 的部分 run。

| Run | 大小 | 观察到的内容 |
|-----|------|--------------|
| `checkpoint/2026-04-22/13-21-33-csvit2-stage1` | 20K | 仅有 `tmux.log`。 |
| `checkpoint/2026-04-22/13-43-43-csvit2-stage1` | 20K | 仅有 `tmux.log`。 |
| `checkpoint/2026-04-23/12-19-07-stage1-clip-smoke` | 12K | 空/极小 smoke 目录。 |
| `checkpoint/2026-04-23/12-27-15-stage1-clip-smoke-clean` | 12K | 空/极小 smoke 目录。 |
| `checkpoint/2026-04-23/12-58-52-csvit2-stage1` | 28K | 仅有 `tmux.log`。 |
| `checkpoint/2026-04-23/13-01-07-stage1-clip-smoke-fix` | 12K | 空/极小 smoke 目录。 |
| `checkpoint/2026-04-23/13-02-02-csvit2-stage1` | 28K | 仅有 `tmux.log`。 |
| `checkpoint/2026-04-23/13-02-56-csvit2-stage1` | 504K | 仅有 `tmux.log`。 |
| `checkpoint/2026-04-23/14-29-08-stage1-metric-split-smoke` | 12K | 空/极小 smoke 目录。 |
| `checkpoint/2026-04-23/17-14-53-csvit2-stage1` | 3.4G | 有 `checkpoint-5000`，没有 best model。 |
| `checkpoint/2026-04-24/10-59-10-csvit2-stage1` | 4.5G | 有 `checkpoint-5000`，没有 best model。 |
| `checkpoint/2026-04-24/2026-04-24-15-26-41-csvit2-stage1` | 2.6G | 有 `checkpoint-5000`，没有 best model。 |
| `checkpoint/2026-04-29/2026-04-29-10-44-42-csvit2-stage2` | 8.0K | 失败/早停的 Stage2 启动目录。 |
| `checkpoint/2026-04-29/2026-04-29-10-45-43-csvit2-stage2` | 28K | 仅有 `tmux.log`。 |
| `checkpoint/2026-05-14/2026-05-14-16-51-17-csvit2-stage1` | 28K | 仅有 `tmux.log`。 |
| `checkpoint/2026-05-15/2026-05-15-19-12-50-smoke-test-total-samples` | 12K | total_samples smoke 目录。 |
| `checkpoint/2026-05-15/2026-05-15-19-14-07-smoke-test-total-samples` | 12K | total_samples smoke 目录。 |
| `checkpoint/2026-05-15/2026-05-15-19-15-48-smoke-test-total-samples` | 12K | total_samples smoke 目录。 |
| `checkpoint/2026-05-15/2026-05-15-19-16-52-smoke-test-total-samples` | 12K | total_samples smoke 目录。 |
| `checkpoint/2026-05-27/2026-05-27-03-12-50-csvit2-stage1-dinov3-large` | 20K | 失败/早停的 DINOv3-L 启动目录。 |
| `checkpoint/2026-05-27/2026-05-27-16-50-34-csvit2-stage1-dinov3-large` | 28K | 失败/早停的 DINOv3-L 启动目录。 |
| `checkpoint/2026-05-29/2026-05-29-18-15-42-csvit2-stage1-dinov3-large-ti` | 52K | DINOv3-L + TI 早期启动，`sample_per_device=42`，TI 分支重复跑 backbone 导致 OOM。 |
| `checkpoint/2026-05-29/2026-05-29-18-19-10-csvit2-stage1-dinov3-large-ti` | 1.7G | DINOv3-L + TI 早期启动，`sample_per_device=32`，第 0 步 backward non-finite gradients。 |
| `checkpoint/2026-05-29/2026-05-29-18-22-35-csvit2-stage1-dinov3-large-ti` | 1.7G | DINOv3-L + TI 早期启动，`sample_per_device=16`，第 0 步 backward non-finite gradients；`nonfinite_stop/step-000000` 已用于复盘。 |
| `checkpoint/2026-05-29/2026-05-29-18-38-28-stage1-dinov3-large-no-norm-patch-uv-rho-multibin` | 12K | 修复后 DINOv3-L + TI 一步 smoke，`sample_per_device=16`、`TRACKER.enabled=false`、`GENERAL.total_samples=16`；用于确认 backward guard 不再触发。 |
| `checkpoint/2026-05-29/2026-05-29-18-40-43-stage1-dinov3-large-no-norm-patch-uv-rho-multibin` | 12K | 修复后 DINOv3-L + TI 一步 smoke，`sample_per_device=42`、`TRACKER.enabled=false`、`GENERAL.total_samples=42`；用于确认默认 TI batch 不再 OOM。 |

## 其他 Artifact

- `checkpoint/stage1_full8_loader_benchmark.json`：完整 8 数据集 Stage1 设置下的 loader-only benchmark。
- `checkpoint/stage1_full8_preprocess_benchmark.json`：完整 8 数据集 Stage1 设置下的 loader + preprocess benchmark。
- `checkpoint/clip_export.log`：clip-native WebDataset 导出日志。
