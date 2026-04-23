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

## 文档

- [docs/README.md](/data_1/renkaiwen/CS-ViT2-remake/docs/README.md)
- [docs/IMPLEMENTATION_NOTES.md](/data_1/renkaiwen/CS-ViT2-remake/docs/IMPLEMENTATION_NOTES.md)
