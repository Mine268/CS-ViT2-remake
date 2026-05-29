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
7.1. SwanLab 的 `step` 轴当前使用累计训练 sample 数，而不是优化器 `global_step`。训练循环内部的 checkpoint 命名、`best_model.json` 和日志触发频率仍然使用 `global_step`，只是在上报给 SwanLab 时转换为 `samples_seen = min(global_step * total_batch, total_samples)`。
8. 训练/验证日志中的核心几何指标已按 `ego / aux / all` 拆分：
   - 绝对指标：`micro_mpjpe_*`、`micro_mpvpe_*`、`micro_rte_*`
   - 相对指标：`micro_mpjpe_rel_*`、`micro_mpvpe_rel_*`
   其中绝对验证选模当前使用 `micro_rte_ego`。
9. 训练保留 `forward_loss / backward_grad / post_step_param` 三阶段 non-finite stop。
10. `AssemblyHands val` 已确认可以用于本地验证；clip-native 的 `val_stage1(T=1,stride=1)` 与 `val_stage2(T=7,stride=1)` 数据已在项目外预处理完成，当前默认 `stage1/stage2` 配置已经分别把它们接成验证集，并使用独立 `DATA.val.batch_size`。多卡 finite validation 会先均衡各 rank 的 clip 段，并把实际验证步数通过 `max_eval_steps` 显式传给 `validate()`。
11. `stage2` 默认训练步数设为 `70000`。依据是导出统计中 `stage1=3978109` clips、`stage2=647039` clips，结合默认 per-device batch `42 -> 6`，按 stage1 最佳验证点约 `60000` step 的样本曝光量等比换算得到约 `68314` step，并向上取整。
12. 默认 LR scheduler 只做 `GENERAL.warmup_step` 的 linear warmup，warmup 后保持常数学习率；不再使用 cosine annealing，也不再保留 `GENERAL.cosine_cycle` 默认配置。
13. 默认启用 `TRAIN.bbox_jitter`。该增强只在训练预处理路径生效，从 tight bbox 出发扰动中心、边长和宽高比，并用扰动后的 bbox 统一驱动 crop、patch bbox、perspective info 与 root-depth 几何输入；validation/test 不启用。2026-05-08 demo 复盘确认 bbox jitter 提升 realtime inference detector bbox 鲁棒性，是正确方向。
14. `src/model/backbone.py` 统一处理 ViT register tokens。DINOv3 会输出 `cls + 4 register + patch` tokens，wrapper 在进入 perspective embedder 和 hand decoder 前丢弃 register tokens，只保留 `cls + patch`。DINOv3-L/16 对应 `stage1_dinov3_large` / `stage2_dinov3_large`，使用 `model/facebook/dinov3-vitl16-pretrain-lvd1689m`、`MODEL.handec.context_dim=1024`；其中 `stage1_dinov3_large` 当前默认 `TRAIN.sample_per_device=42`。DINOv3-H+/16 对应 `stage1_dinov3` / `stage2_dinov3`，使用 `model/facebook/dinov3-vith16plus`、`MODEL.handec.context_dim=1280`。两者默认 `TRAIN.backbone_lr=1e-5` 即 full fine-tune。
15. 训练入口现在有两层：`make` / `make shell` 默认启动 Python 训练 TUI。启动训练时先选 `stage`，再选 `backbone`，再选是否启用 `TI`，最后在常用参数菜单里设置 `GPU_IDS`、`NUM_PROCESSES`、`TRAIN.sample_per_device`、`RUN_NAME`、`MAIN_PROCESS_PORT`、`OVERRIDES` 等并确认启动；同一个 TUI 里也可以 attach tmux、tail log、停止训练和按 session 名过滤多个实验。底层仍然复用显式 Make target。`script/run_train_tmux.sh` 支持 `CONFIG_NAME` 环境变量，Makefile 提供 `train-stage1-dinov3-large`、`train-stage1-dinov3-large-ti`、`train-stage2-dinov3-large`、`train-stage1-dinov3`、`train-stage2-dinov3` 专用入口。其中 `train-stage1-dinov3-large-ti` 会在 `stage1_dinov3_large` 的基础上追加 `MODEL.ti.enabled=true`。DINOv3 实验应优先使用这些入口，避免普通 `train-stage1/train-stage2` 的 DINOv2 默认 batch 覆盖 DINOv3 配置。
16. 当前支持一个 Stage1-only 的 TI v1 特征正则分支，配置入口是 `MODEL.ti`。它在 `perspective embedder` 之后对 token 做 FiLM 条件变换，再复用同一个 `handec` 解码；训练时额外采样 `(scale, angle)`，对 transformed branch 的 `global_orient` 做逆旋转，对 `pred_ray_unit/pred_rho` 做逆变换并重组 `trans`，然后只在 `theta/shape/joint_rel/trans` 上与原 GT 比较。该分支不做图像增强，不做 feature consistency loss，也不对 `uv_patch/rho/reprojection` 头施加 transformed supervision。Stage1 训练时主分支和 TI 分支共享同一次 backbone + perspective token 编码，避免 DINOv3-L 上重复跑 backbone 导致显存翻倍；axis-angle `global_orient` 的逆旋转使用稳定的 quaternion compose，避免 `matrix -> axis-angle` 反向在 bf16 下产生 non-finite 梯度。`MODEL.ti.apply_stage2` 预留但尚未实现。
17. 旧的 CS-ViT2 在 `/data_1/renkaiwen/CS-ViT2` 下。

目录说明：

- `src/data/`: schema、采样与预处理
- `src/model/`: backbone、perspective、head、temporal、loss、net
- `src/train/`: 训练引擎、tracker、checkpoint、nan guard
- `script/`: train/test/export_sample 入口
