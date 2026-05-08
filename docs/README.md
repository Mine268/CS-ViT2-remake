# 文档总览

本目录记录 remake 的关键约定，优先查看：

- `IMPLEMENTATION_NOTES.md`
- `CLIP_DATA_REORG.md`
- `LEGACY_REFERENCE_POLICY.md`
- 仓库根目录的 `HANDOFF.md`
- 项目 Codex skill：`.agents/skills/dataset-visual-check/SKILL.md`

Stage 1 demo 入口见仓库根目录 `README.md` 的 “Stage 1 Demo” 小节。该脚本只消费 clip-native/WDS 或普通媒体输入，不引入 sequence chunk 数据路径。普通媒体输入默认使用仓库内 `hand_bbox_module/` 的 WiLoR-mini 手部 bbox 检测器，MediaPipe 仅作为显式回退检测器保留。外部 detector bbox 可通过 `--detector-bbox-scale` 统一覆盖，或通过 detector-specific 默认值在 preprocess 前等比缩放；GT/WDS debug bbox 不受该参数影响。

训练期 bbox 鲁棒性增强由 `TRAIN.bbox_jitter` 控制，只在 `augmentation_flag=True` 的训练预处理路径生效。它从数据集 tight bbox 出发随机扰动中心、边长和宽高比，用扰动后的 bbox 统一驱动 crop/patch bbox、perspective info 与 root-depth 几何输入；2D/3D 标注和相机内参本身不被该增强直接改写。Stage 1 默认逐帧采样扰动，Stage 2 默认 clip-level 主扰动加小幅逐帧噪声；validation/test 路径默认不传入该配置，保持 GT-bbox 指标可比。

2026-05-08 复盘结论：加入 bbox jitter 后，stage1 demo 对 realtime inference detector bbox 的鲁棒性更强，说明 bbox jitter 是正确方向，后续 bbox/detector 相关训练实验应优先保留该增强。

数据集导出、bbox/patch、投影或左右手翻转相关可视化验证流程已经固化为项目 skill：`.agents/skills/dataset-visual-check/SKILL.md`。之后遇到类似验证任务，优先按该 skill 的坐标系规则生成 `temp/<check_name>/summary.json` 和 `vis_*.png`，避免把 preprocess 坐标直接画回 raw image 的错误。

默认 LR scheduler 只做训练开始阶段的 linear warmup，warmup 后保持常数学习率；不再使用 cosine annealing。

如果代码结构或训练口径发生变化，先更新这里的索引，再同步更新对应专题文档。
