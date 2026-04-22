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

## 训练

### Stage 1

```bash
source .venv/bin/activate
accelerate launch --main_process_port 0 --gpu_ids 0,1,2,3 --num_processes 4 -m script.train \
  --config-name=stage1 \
  TRAIN.sample_per_device=42 \
  LOSS.heatmap_sigma=4.0
```

### Stage 2

```bash
source .venv/bin/activate
accelerate launch --main_process_port 0 --gpu_ids 0,1,2,3 --num_processes 4 -m script.train \
  --config-name=stage2 \
  MODEL.stage1_weight=/path/to/stage1/best_model \
  TRAIN.sample_per_device=6 \
  LOSS.heatmap_sigma=4.0
```

## 测试

```bash
source .venv/bin/activate
python -m script.test \
  --config-name=stage1 \
  TEST.checkpoint_path=/path/to/checkpoint \
  DATA.test.source='[/path/to/test/*.tar]'
```

## 文档

- [docs/README.md](/data_1/renkaiwen/CS-ViT2-remake/docs/README.md)
- [docs/IMPLEMENTATION_NOTES.md](/data_1/renkaiwen/CS-ViT2-remake/docs/IMPLEMENTATION_NOTES.md)
