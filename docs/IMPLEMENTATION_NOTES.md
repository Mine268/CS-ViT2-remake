# 实现说明

本仓库是对旧版 CS-ViT2 的单主线重构，关键约定如下：

1. 只保留 `patch_uv_rho_multibin` camera head。
2. 只保留 `WebDataset V2 + dataset reweight + clip filter` 数据主线。
3. `HOT3D` 与 `AssemblyHands` 作为 `ego_abs` 数据集，承担绝对 root/rho 监督。
4. `InterHand2.6M`、`DexYCB`、`HO3D_v3`、`FreiHAND`、`RHD`、`MTC` 作为 `aux_local`，只承担局部监督。
5. 实验记录通过 `Tracker` 薄封装接入 SwanLab，仅主进程记录。
6. 训练保留 `forward_loss / backward_grad / post_step_param` 三阶段 non-finite stop。
7. 旧的 CS-ViT2 在 `/data_1/renkaiwen/CS-ViT2` 下。

目录说明：

- `src/data/`: schema、采样与预处理
- `src/model/`: backbone、perspective、head、temporal、loss、net
- `src/train/`: 训练引擎、tracker、checkpoint、nan guard
- `script/`: train/test/export_sample 入口
