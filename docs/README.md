# 文档总览

本目录记录 remake 的关键约定，优先查看：

- `IMPLEMENTATION_NOTES.md`
- `CLIP_DATA_REORG.md`
- `LEGACY_REFERENCE_POLICY.md`
- 仓库根目录的 `HANDOFF.md`

Stage 1 demo 入口见仓库根目录 `README.md` 的 “Stage 1 Demo” 小节。该脚本只消费 clip-native/WDS 或普通媒体输入，不引入 sequence chunk 数据路径。普通媒体输入默认使用仓库内 `hand_bbox_module/` 的 WiLoR-mini 手部 bbox 检测器，MediaPipe 仅作为显式回退检测器保留。外部 detector bbox 可通过 `--detector-bbox-scale` 统一覆盖，或通过 detector-specific 默认值在 preprocess 前等比缩放；GT/WDS debug bbox 不受该参数影响。

如果代码结构或训练口径发生变化，先更新这里的索引，再同步更新对应专题文档。
