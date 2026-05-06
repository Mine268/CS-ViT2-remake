# CS-ViT2-remake

精简版 CS-ViT2 重构仓库，只保留当前主线：

- `stage1` / `stage2`
- `no_norm`
- `WebDataset V2`
- `dataset reweight`
- `patch_uv_rho_multibin`
- `SwanLab`
- non-finite stop

## 环境

```bash
uv venv .venv --python=3.12
source .venv/bin/activate
uv sync
accelerate config
```

## 格式化

```bash
uvx ruff format .
uvx ruff check .
```

## 训练

推荐直接使用 `make`，训练会自动托管到 `tmux` 后台 session。每次运行都会生成一个显式 `run_name`，并统一用于：

- `checkpoint/YYYY-MM-DD/<run_name>/`
- `checkpoint/YYYY-MM-DD/<run_name>/tmux.log`
- SwanLab 云端实验名

`run_name` 本身现在也会包含 `YYYY-MM-DD` 日期前缀；如果显式传 `RUN_NAME=stage1-ablation-a`，最终会被规范成类似 `2026-04-24-stage1-ablation-a`。

可先运行 `make help` 查看完整帮助，包括每个 make 变量的默认值、用途，以及常见 `OVERRIDES` 示例。

当前 tracker 默认会把标量日志同时发给 SwanLab，并通过 `print_to_console` 镜像到本地终端 / `tmux.log`。如需关闭本地镜像，可在配置里将 `TRACKER.print_to_console=false`。

### Stage 1

```bash
make train-stage1
make attach-stage1
make logs-stage1
```

常见覆盖参数：

```bash
make train-stage1 OVERRIDES="TRAIN.sample_per_device=32 LOSS.heatmap_sigma=4.0"
make train-stage1 RUN_NAME=stage1-ablation-a
```

### Stage 2

```bash
make train-stage2 STAGE1_WEIGHT=/path/to/stage1/best_model
make attach-stage2
make logs-stage2
```

自定义 GPU 或进程数：

```bash
make train-stage1 GPU_IDS=0,1 NUM_PROCESSES=2
make train-stage2 STAGE1_WEIGHT=/path/to/stage1/best_model GPU_IDS=4,5,6,7 NUM_PROCESSES=4
```

查看 / 停止 tmux session：

```bash
make tmux-ls
make stop-stage1
make stop-stage2
```

如需先检查最终命令而不启动训练：

```bash
make train-stage1 DRY_RUN=1
```

## 数据配置

训练数据现在完全由 [config/data.yaml](/data_1/renkaiwen/CS-ViT2-remake/config/data.yaml) 驱动：

- `DATA.datasets`: 统一登记每个数据集的 canonical 名字、别名和各 split 路径。
- `DATA.train.groups.ego/aux.datasets`: 用数据集名字声明监督分组，并配置组内随机采样权重。
- `DATA.train.groups.ego/aux.weight`: 配置 ego 和 aux 两个采样组之间的随机采样权重。

训练时会直接把样本的 `data_source` 写成 registry 里的 canonical 名字，不依赖 shard 里自带的 `data_source.json` 来决定监督分流；因此 ego/aux mask 以配置为准。

当前主训练已经切到 clip-native 数据：

- `stage1` 使用 `DATA.train.split=train_stage1`
- `stage2` 使用 `DATA.train.split=train_stage2`
- 训练配置中不再保留旧的 sequence 训练路径，旧目录仅通过导出脚本作为原始源数据读取

## Clip 导出

为了消除“先读取整条 sequence，再切训练 clip”的读取瓶颈，项目新增了 clip-native 导出脚本：

```bash
.venv/bin/python script/export_train_clips.py --help
```

导出目标目录固定为 `/data_0/renkaiwen/webdatasets2_remake/`，详细设计、运行方式和验证结果见：

- [docs/CLIP_DATA_REORG.md](/data_1/renkaiwen/CS-ViT2-remake/docs/CLIP_DATA_REORG.md)

当前默认导出参数：

- `stage1`: `clip_len=1`, `stride=1`
- `stage2`: `clip_len=7`, `stride=4`

## AssemblyHands Val

`AssemblyHands val` 的 clip-native 验证数据已在项目外预处理完成，当前直接使用：

- `DATA.datasets.AssemblyHands.splits.val_stage1`
- `DATA.datasets.AssemblyHands.splits.val_stage2`
- `stage1` 默认验证源为 `DATA.datasets.AssemblyHands.splits.val_stage1`
- `stage2` 默认验证源为 `DATA.datasets.AssemblyHands.splits.val_stage2`
- validation 使用独立 `DATA.val.batch_size`，不再默认复用 `TRAIN.sample_per_device`

## 验证 / 测试数据现状

- `AssemblyHands val`: 已确认可用，`2D + 3D + calibration` 数值自洽，可作为本地验证集使用。
- `AssemblyHands test-eccv2024`: 目录和文件齐全，但公开 JSON 中的 `2D keypoints / 3D joints / extrinsics` 是占位值，不可作为本地真实 GT 测试集直接使用。
- `HOT3D`: 当前项目尚未接入正式 `val/test`；官方公开口径下更适合从训练集切一个本地 `val`。

临时检查脚本：

- [assemblyhands_val_check.py](/data_1/renkaiwen/CS-ViT2-remake/temp/assemblyhands_val_check.py)
- [assemblyhands_val_wds_check.py](/data_1/renkaiwen/CS-ViT2-remake/temp/assemblyhands_val_wds_check.py)
- [assemblyhands_test_check.py](/data_1/renkaiwen/CS-ViT2-remake/temp/assemblyhands_test_check.py)

## 测试

```bash
source .venv/bin/activate
uv sync --extra dev
pytest -q

source .venv/bin/activate
python -m script.test \
  --config-name=stage1 \
  TEST.checkpoint_path=/path/to/checkpoint \
  DATA.test.source='[/path/to/test/*.tar]'
```

## Stage 1 Demo

`script/demo_stage1.py` 支持对图像、图像目录、视频和本地 WDS 样本运行第一阶段模型推理，输出原相机空间中的 3D 手部姿态估计结果。普通图像 / 视频输入默认使用仓库内的 WiLoR-mini 手部 bbox 检测器 `hand_bbox_module/`，返回 bbox 和解剖学左右手标签；如需安装 demo 检测依赖可运行 `uv sync --extra demo` 或 `uv pip install -r hand_bbox_module/requirements.txt`。MediaPipe 仍保留为回退选项，可显式传 `--detector mediapipe`。本地数据集 debug 可使用 `--detector gt` 直接读取 WDS 中的手框，不依赖外部检测器。

示例：

```bash
source .venv/bin/activate
python script/demo_stage1.py \
  --input /path/to/image_or_video \
  --checkpoint /data_0/renkaiwen/CS-ViT2-remake-checkpoints/2026-04-27/2026-04-27-11-10-34-csvit2-stage1/best_model \
  --output-dir output/demo_stage1 \
  --intrinsics 900 900 640 360
```

WiLoR bbox 检测器可通过 `--wilor-conf`、`--wilor-iou`、`--wilor-model` 调整阈值和权重路径；默认权重为 `hand_bbox_module/weights/detector.pt`。外部检测器产生的 bbox 会在进入 stage1 preprocess 前按中心等比缩放，该参数只作用于 `wilor` 和 `mediapipe` 输出，不改变 `--detector gt` 或手工 `--detector bbox`。基于 AssemblyHands/HOT3D 各 200 帧与 WDS 真 bbox 的统计，默认 `--wilor-bbox-scale 0.75`、`--mediapipe-bbox-scale 1.05`；如需统一覆盖或完全使用原始 detector bbox，传 `--detector-bbox-scale 1.0`。

AssemblyHands val 样本 debug：

```bash
python script/demo_stage1.py \
  --input '/data_0/renkaiwen/webdatasets2_remake/AssemblyHands/val_stage1/*.tar' \
  --input-type wds \
  --detector gt \
  --checkpoint /data_0/renkaiwen/CS-ViT2-remake-checkpoints/2026-04-27/2026-04-27-11-10-34-csvit2-stage1/best_model \
  --output-dir output/demo_stage1_ah_val \
  --max-frames 8
```

输出包括：

- `predictions.jsonl`: 每帧的 bbox、handedness、MANO pose/shape、相机空间平移和 `.npz` 路径。
- `predictions_npz/*.npz`: 每只手的 `joint_cam`, `vert_cam`, `mano_pose`, `mano_shape`, `trans`, `joint_img`, `vert_img`, `faces`。
- `overlays/*.png`: MANO mesh 和 21 关节重投影叠图；视频输入还会输出 `overlay.mp4`。

## 文档

- [docs/README.md](/data_1/renkaiwen/CS-ViT2-remake/docs/README.md)
- [docs/IMPLEMENTATION_NOTES.md](/data_1/renkaiwen/CS-ViT2-remake/docs/IMPLEMENTATION_NOTES.md)
- [HANDOFF.md](/data_1/renkaiwen/CS-ViT2-remake/HANDOFF.md)
