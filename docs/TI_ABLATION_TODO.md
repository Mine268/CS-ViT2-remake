# Transformation Isomorphism Ablation TODO

最后更新：2026-06-23

## 目标

验证 `MODEL.ti` / transformation isomorphism 模块是否真的提升模型对几何变换的鲁棒性，而不是只受益于更大的 backbone、更长训练或随机训练波动。

核心问题：

- TI 是否在相同模型、相同训练数据、相同训练预算下带来稳定收益？
- TI 的收益是否随 DINOv3 系列模型规模变化，例如 DINOv3-S/16、DINOv3-B/16、DINOv3-L/16 是否一致？
- TI 的收益是否随训练数据量变化，并且这种变化是否能通过 `GENERAL.total_samples` 形成可控的数据规模消融？
- TI 的收益是否随训练源变化，例如 full stage1、ego-heavy、单 ego 数据集是否一致？
- TI 是否主要提升 camera/scale/rotation/cross-domain regime，而不只是平均 MPJPE 略微变化？

## 当前实现口径

- 配置入口：`MODEL.ti.enabled=true`。
- 当前只支持 Stage1：`MODEL.ti.apply_stage1=true`，`MODEL.ti.apply_stage2=false`。
- TI 分支在 `persp_info_embedder` 后做 token-level FiLM 条件变换，不做图像空间增强。
- Stage1+TI 共享主分支 backbone + perspective tokens，避免重复跑 DINOv3 backbone 导致显存翻倍。
- TI 分支逆变换后只监督 `theta/shape/joint_rel/trans`，不监督 transformed branch 的 `uv_patch/rho/reprojection`。
- 当前默认 TI 超参：`angle_range_deg=30.0`、`scale_range=[0.95, 1.05]`、`loss_weight=0.01`。

## 实验原则

- 控制变量必须严格：同一对照组只改变 `MODEL.ti.enabled`。
- 训练数据量用 `GENERAL.total_samples` 对齐，而不是用 epoch 或 wall-clock 对齐。
- batch size、GPU 数、LR、bbox jitter、heatmap sigma、seed、val split 必须一致。
- 统一使用同一个 validation protocol，当前首选 AssemblyHands `val_stage1`。
- 不能只看 best `micro_rte_ego`，还要看 `micro_mpjpe_ego`、`micro_mpjpe_rel_ego`、`micro_mpvpe_ego`、收敛速度和训练稳定性。
- 如果只做单 seed，结论只能写成 “在该设置下 TI 有收益/无收益”；如果作为论文结论，关键设置需要多 seed 或 bootstrap 置信区间。

## 第一轮核心矩阵

第一轮目标是用最小成本判断 TI 是否值得继续扩大实验。模型 scale 统一选用 DINOv3 系列 backbone，因为该系列提供同一预训练体系下的不同 scale 初始化权重，比混用 DINOv2/DINOv3 更适合作为 model scaling 消融。第一轮不使用 H+/G 这类超大模型，最高到 Large，避免算力成本和 batch size 差异掩盖 TI 本身的作用。

| 维度 | 设置 |
| --- | --- |
| 模型规模 | DINOv3-S/16、DINOv3-B/16、DINOv3-L/16 |
| 训练源 | full stage1、ego-heavy |
| TI 开关 | off、on |
| 数据规模 | 通过 `GENERAL.total_samples` 控制，例如 25%、50%、100% Stage1 数据量 |
| 验证集 | AssemblyHands `val_stage1` |

训练源定义：

- `full stage1`：使用当前默认 Stage1 训练分布，即 `DATA.train.groups.ego.weight=0.2`、`DATA.train.groups.aux.weight=0.8`。ego 组包含 HOT3D 和 AssemblyHands；aux 组包含 InterHand2.6M、FreiHAND、MTC、DexYCB、HO3D_v3、RHD。这是当前大数据 baseline 口径。
- `ego-heavy`：仍然使用同一套 Stage1 数据集和同一套监督逻辑，但把采样分布改成更偏第一人称 ego 数据，例如 `DATA.train.groups.ego.weight=0.8`、`DATA.train.groups.aux.weight=0.2`。它不是严格 ego-only，只是提高 HOT3D/AssemblyHands 的采样概率，用于观察 TI 在 ego/camera-space 监督更密集时是否更有效。
- `ego-only`：只使用 HOT3D/AssemblyHands，不采 aux。当前数据配置要求 ego/aux 两组都非空且权重为正，所以严格 ego-only 需要新增专用配置或放宽数据计划校验，不放在第一轮核心矩阵。

因此，`full stage1` 和 `ego-heavy` 的区别只是训练采样分布不同；底层仍然读取 clip-native stage1 WebDataset，验证集仍然使用同一个 AssemblyHands `val_stage1`，模型结构和 loss routing 不变。

第一轮建议先做 full stage1 的模型 scale × TI 对照，再决定是否加入 ego-heavy：

| ID | Model | Data | TI | 目的 |
| --- | --- | --- | --- | --- |
| A1 | DINOv3-S/16 | full stage1 | off | 小模型对照，观察低容量下 TI 是否提供几何正则 |
| A2 | DINOv3-S/16 | full stage1 | on | 与 A1 对照 |
| B1 | DINOv3-B/16 | full stage1 | off | 中等模型对照 |
| B2 | DINOv3-B/16 | full stage1 | on | 与 B1 对照 |
| C1 | DINOv3-L/16 | full stage1 | off | Large 模型对照 |
| C2 | DINOv3-L/16 | full stage1 | on | 已有完整 run，可作为第一版 TI 结果 |
| D1 | DINOv3-S/16 | ego-heavy | off | 检查 ego-heavy 下小模型是否更依赖 TI |
| D2 | DINOv3-S/16 | ego-heavy | on | 与 D1 对照 |
| E1 | DINOv3-B/16 | ego-heavy | off | 检查 ego-heavy 下中等模型的 TI 收益 |
| E2 | DINOv3-B/16 | ego-heavy | on | 与 E1 对照 |
| F1 | DINOv3-L/16 | ego-heavy | off | 检查 ego-heavy 下 Large 模型上限 |
| F2 | DINOv3-L/16 | ego-heavy | on | 与 F1 对照 |

前置状态：当前仓库已配置 DINOv3-L/16 与 DINOv3-H+/16；若执行上述第一轮矩阵，需要先补齐 DINOv3-S/16、DINOv3-B/16 的本地权重目录、Hydra config、Make/TUI 启动入口和 smoke。

## 数据规模消融

训练数据量统一由 `GENERAL.total_samples` 控制。这样即使 GPU 数、batch size 或 grad accumulation 不同，训练引擎也会根据全局 batch 自动换算 `total_step`，保证每个 run 看到的总样本数可比。

建议数据规模：

| Scale | `GENERAL.total_samples` | 说明 |
| --- | --- | --- |
| 25% | `6400000` | 低成本趋势点 |
| 50% | `12800000` | 中等数据量 |
| 100% | `25600000` | 当前 Stage1 默认训练量 |

第一轮最小数据规模矩阵：

- 对 DINOv3-S/16、DINOv3-B/16、DINOv3-L/16 先做 `100%` × `TI off/on`，验证 model scale 方向是否存在一致趋势。
- 对 DINOv3-L/16 做 `25% / 50% / 100%` × `TI off/on`，验证 TI 是否改变 data scaling 曲线。
- 如果算力允许，再对 DINOv3-S/16 和 DINOv3-B/16 补 `25% / 50%`，形成完整 model scale × data scale × TI 矩阵。

解释口径：

- 如果 TI 在小数据量更有效，可能说明它提供了几何正则化，缓解数据不足。
- 如果 TI 只在大数据量有效，可能说明模型需要足够多样性才能学习 TI 分支约束。
- 如果 TI 收益随 S -> B -> L 变小，可能说明更大模型已经部分吸收几何变换鲁棒性。
- 如果 TI 收益随 S -> B -> L 变大，可能说明 TI 对更大容量模型提供了更强的可利用训练信号。

## 第二轮扩展矩阵

第一轮若观察到 TI 的稳定收益，再扩展：

- 在 DINOv3-H+/16 上做扩展验证。注意它当前默认 `TRAIN.sample_per_device=2`，训练成本和吞吐与 L 不可直接比较，需要单独记录 effective samples/sec 和总 compute。
- 数据源加入严格 ego-only、AssemblyHands-only 与 HOT3D-only，用于区分 TI 对不同 ego 数据域的收益。
- 加入 2-3 个 seed，优先对最关键的 DINOv3-L full stage1 和 ego-heavy 设置做重复。
- 如果有 HOT3D validation 或 cross-camera/cross-dataset validation，可加入 cross-domain 指标。

## 数据配置前置项

当前 `DATA.train.groups` 要求 ego/aux 两组都存在，并且组权重、组内数据集权重都必须为正。因此：

- `full stage1` 可以直接使用默认配置。
- `ego-heavy` 可以直接通过 override 实现，例如 `DATA.train.groups.ego.weight=0.8 DATA.train.groups.aux.weight=0.2`。
- 严格 `ego-only`、`AssemblyHands-only`、`HOT3D-only` 不能简单把 aux 权重设为 0，否则会违反当前数据计划校验。
- 若需要严格单数据源实验，应新增专用 config 或扩展 `src/data/config.py`，允许实验配置显式禁用某个 group，同时保持 supervision mask 语义清晰。

建议第一轮先使用 `ego-heavy=0.8/0.2`，避免先改数据管线。若 ego-heavy 中观察到明显趋势，再补严格 ego-only 配置。

## 推荐启动命令

DINOv3-L/16 full stage1，无 TI：

```bash
make train-stage1-dinov3-large RUN_NAME=ti-ablate-dinov3l-full-off
```

DINOv3-L/16 full stage1，启用 TI：

```bash
make train-stage1-dinov3-large-ti RUN_NAME=ti-ablate-dinov3l-full-on
```

DINOv3-L/16 ego-heavy，无 TI：

```bash
make train-stage1-dinov3-large RUN_NAME=ti-ablate-dinov3l-egoheavy-off OVERRIDES="DATA.train.groups.ego.weight=0.8 DATA.train.groups.aux.weight=0.2"
```

DINOv3-L/16 ego-heavy，启用 TI：

```bash
make train-stage1-dinov3-large-ti RUN_NAME=ti-ablate-dinov3l-egoheavy-on OVERRIDES="DATA.train.groups.ego.weight=0.8 DATA.train.groups.aux.weight=0.2"
```

DINOv3-L/16 25% 数据量，启用 TI：

```bash
make train-stage1-dinov3-large-ti RUN_NAME=ti-ablate-dinov3l-full-on-25p OVERRIDES="GENERAL.total_samples=6400000"
```

DINOv3-L/16 50% 数据量，启用 TI：

```bash
make train-stage1-dinov3-large-ti RUN_NAME=ti-ablate-dinov3l-full-on-50p OVERRIDES="GENERAL.total_samples=12800000"
```

DINOv3-L/16 100% 数据量，启用 TI：

```bash
make train-stage1-dinov3-large-ti RUN_NAME=ti-ablate-dinov3l-full-on-100p OVERRIDES="GENERAL.total_samples=25600000"
```

DINOv3-S/16、DINOv3-B/16 的命令需要在补齐本地权重、Hydra config 和 Make/TUI target 后再固化。不要用 DINOv3-H+/16 代替第一轮的小模型 scale 点。

## Smoke 要求

每种新组合正式训练前先跑 smoke，关闭 SwanLab：

```bash
make train-stage1 DRY_RUN=1 OVERRIDES="MODEL.ti.enabled=true TRACKER.enabled=false GENERAL.total_samples=64"
```

```bash
make train-stage1-dinov3-large-ti DRY_RUN=1 OVERRIDES="TRACKER.enabled=false GENERAL.total_samples=84"
```

正式 smoke 应至少确认：

- dataloader 能取到 batch；
- forward/backward 不触发 non-finite guard；
- TI on/off 的 loss key 与日志字段正常；
- checkpoint 目录、`tmux.log`、Hydra config 正常写入；
- `TRACKER.enabled=false`，不污染 SwanLab。

## 需要记录的指标

训练元信息：

- run name、git commit、config name、完整 overrides；
- GPU 数、`TRAIN.sample_per_device`、global batch、`GENERAL.total_samples`；
- 由 `GENERAL.total_samples` 推导出的 `total_step`；
- 是否启用 bbox jitter；
- 是否启用 TI 以及 TI 超参；
- best checkpoint step 与最终 checkpoint step。

验证指标：

- `val/micro_mpjpe_ego`
- `val/micro_mpjpe_rel_ego`
- `val/micro_mpvpe_ego`
- `val/micro_mpvpe_rel_ego`
- `val/micro_rte_ego`
- best checkpoint 选择指标当前为 `micro_rte_ego`，但结论中必须同时报告 MPJPE 和 RTE。

稳定性指标：

- 是否发生 OOM；
- 是否发生 non-finite guard；
- samples/sec 或 step time；
- TI 额外训练开销。

## Challenge-stratified 分析 TODO

为了证明 TI 的作用机制，不能只报告 aggregate 指标。后续需要按几何挑战分组：

- scale/bbox regime：按 bbox 边长、bbox jitter 前后尺度、patch 中手部占比划分；
- rotation regime：按手部 2D 主方向或图像内旋转角估计划分；
- camera regime：按手部中心到图像中心距离、fx/fy、鱼眼边缘程度划分；
- occlusion/inter-hand regime：按 2D keypoint valid ratio、左右手 bbox overlap、手物 overlap 划分；
- domain regime：按 dataset、subject、camera、scene、object/action 划分。

第一版可以先做 `dataset + bbox scale + image center distance` 三个分组，因为它们最容易从现有样本元信息或几何量中提取。

## 判定标准

TI 可以被认为有效的最低标准：

- 在至少一个主要设置中，TI on 相比 TI off 同时改善 `micro_mpjpe_ego` 或 `micro_rte_ego`，且没有显著牺牲另一个指标；
- 训练稳定，不引入 OOM/non-finite；
- 收益不只出现在单个偶然 checkpoint，而是在 best 与 late-stage validation 上趋势一致；
- 在 scale/rotation/camera 相关分组中收益更明显，符合 TI 的设计动机。

TI 结论不充分的情况：

- 只在 DINOv3-L 单 run 上提升，但 DINOv3-S/B、不同数据量或 ego-heavy 中不复现；
- 只改善 RTE 但显著恶化 MPJPE，或反过来；
- 平均指标提升但 challenge 分组没有任何几何相关规律；
- TI 收益小于 run-to-run 随机波动且没有重复 seed 支撑。

## 输出物

第一轮完成后需要同步：

- 更新 `docs/EXPERIMENT_RESULTS.md` 的 TI 消融表；
- 保存每个 run 的 `best_model.json`、`tmux.log` 和 resolved config；
- 生成一张 TI on/off 对比表，按 model/data 分组；
- 如果完成 challenge-stratified 分析，生成分组指标表和可视化图；
- 明确下一步是扩展到 DINOv3-H+、严格 ego-only、单数据集，还是停止该方向。
