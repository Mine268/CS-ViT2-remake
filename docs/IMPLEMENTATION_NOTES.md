# 实现说明

本仓库是对旧版 CS-ViT2 的单主线重构，关键约定如下：

1. 只保留 `patch_uv_rho_multibin` camera head。
2. 只保留 `WebDataset V2 + dataset registry + ego/aux two-level random mix + clip filter` 数据主线。
3. 所有训练数据集必须先登记在 `config/data.yaml -> DATA.datasets`，包括 canonical 名字、别名和 split 路径。
4. `DATA.train.groups.ego` 与 `DATA.train.groups.aux` 同时定义：
   - 哪些数据集属于 ego / aux
   - 每个组内部的数据集随机采样权重
   - ego / aux 两个组之间的随机采样权重
5. 训练时样本的 `data_source` 以 dataset registry 中的 canonical 名字为准，ego/aux 路由不依赖 shard 内部的 `data_source.json`。
6. 当前默认监督口径仍然是：`HOT3D` 与 `AssemblyHands` 作为 ego，承担绝对 root/rho 监督。`stage1` 的 aux 为 `InterHand2.6M`、`DexYCB`、`HO3D_v3`、`FreiHAND`、`RHD`、`MTC`；`stage2` 使用 clip-native `train_stage2` 数据，因此当前 aux 为 `InterHand2.6M`、`DexYCB`、`HO3D_v3`、`MTC`。
7. 实验记录通过 `Tracker` 薄封装接入 SwanLab，仅主进程记录。
8. 训练/验证日志中的核心几何指标已按 `ego / aux / all` 拆分：
   - 绝对指标：`micro_mpjpe_*`、`micro_mpvpe_*`、`micro_rte_*`
   - 相对指标：`micro_mpjpe_rel_*`、`micro_mpvpe_rel_*`
   其中绝对验证选模当前使用 `micro_rte_ego`。
9. 训练保留 `forward_loss / backward_grad / post_step_param` 三阶段 non-finite stop。
10. `AssemblyHands val` 已确认可以用于本地验证；clip-native 的 `val_stage1(T=1,stride=1)` 与 `val_stage2(T=7,stride=1)` 数据已在项目外预处理完成，当前默认 `stage1/stage2` 配置已经分别把它们接成验证集，并使用独立 `DATA.val.batch_size`。多卡 finite validation 会先均衡各 rank 的 clip 段，并把实际验证步数通过 `max_eval_steps` 显式传给 `validate()`。
11. 旧的 CS-ViT2 在 `/data_1/renkaiwen/CS-ViT2` 下。

目录说明：

- `src/data/`: schema、采样与预处理
- `src/model/`: backbone、perspective、head、temporal、loss、net
- `src/train/`: 训练引擎、tracker、checkpoint、nan guard
- `script/`: train/test/export_sample 入口
