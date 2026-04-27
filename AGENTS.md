# Repository Guidelines

## Project Structure & Module Organization
`src/` contains the runtime code: `src/data/` for WebDataset loading, sampling, and preprocessing; `src/model/` for backbone, temporal encoder, heads, losses, and `PoseNet`; `src/train/` for the training engine, checkpointing, tracker, and non-finite guards. Entry points live in `script/` (`train.py`, `test.py`, `export_sample.py`). Hydra configs are under `config/` (`stage1.yaml`, `stage2.yaml`, plus shared `data.yaml`, `model.yaml`, `loss.yaml`, `tracker.yaml`). Tests live in `tests/`. `docs/` records implementation constraints, and `model/` stores local pretrained configs and MANO-related assets.

## Build, Test, and Development Commands
Create the Python 3.12 environment with:
```bash
uv venv .venv --python=3.12
source .venv/bin/activate
uv sync
accelerate config
```
Format and lint Python code with:
```bash
uvx ruff format .
uvx ruff check .
```
Preferred training entrypoints use `make` and auto-launch detached `tmux` sessions. Each run gets a single explicit `run_name` reused by the checkpoint directory, `tmux.log`, and SwanLab experiment name:
```bash
make train-stage1
make train-stage2 STAGE1_WEIGHT=/path/to/best_model
make attach-stage1
make logs-stage1
```
Override runtime knobs with make variables such as `GPU_IDS=0,1`, `NUM_PROCESSES=2`, `RUN_NAME=stage1-ablation-a`, or `OVERRIDES="TRAIN.sample_per_device=32"`. The raw training command remains:
```bash
accelerate launch --main_process_port 0 --gpu_ids 0,1,2,3 --num_processes 4 -m script.train --config-name=stage1
```
Training dataset routing is config-driven. Keep canonical dataset names, aliases, split paths, ego/aux membership, per-dataset weights, and ego/aux group weights in `config/data.yaml` instead of hardcoding them in Python. Training uses the registry-defined dataset name for routing rather than shard-level `data_source.json`.
The active training path reads clip-native splits from `config/data.yaml` (`train_stage1` for stage1 and `train_stage2` for stage2). The old sequence-formatted `/data_0/renkaiwen/webdatasets2_512` tree is no longer used for training and is only referenced by `script/export_train_clips.py` when exporting new clip shards.
Run inference/export tests with:
```bash
python -m script.test --config-name=stage1 TEST.checkpoint_path=/path/to/checkpoint
pytest -q
```

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes, and `from __future__ import annotations` in new Python modules. Keep type hints for public helpers and training/model interfaces. Match the current Hydra convention: top-level config groups are uppercase (`MODEL`, `TRAIN`, `LOSS`), while nested keys are lowercase.

## Testing Guidelines
Use `pytest`; tests are discovered from `tests/` via `test_*.py`. Install dev tooling with `uv sync --extra dev`, then run `pytest -q`. Add focused unit tests for new math, loss, or config behavior, following patterns in `tests/test_config_smoke.py` and `tests/test_patch_uv_rho_multibin.py`. There is no published coverage gate, but changes to model heads, loss wiring, or Hydra config should ship with at least one regression test.

## Commit & Pull Request Guidelines
Current history uses short, action-first commit messages in Chinese, for example `初始化仓库` and `添加了gitignore`. Keep commits single-purpose and similarly concise. PRs should state which stage/config was affected, list any dataset or checkpoint assumptions, include the exact train/test command used for validation, and attach key metrics or screenshots only when they clarify behavior. Avoid committing generated outputs under `checkpoint/`, `output/`, `runs/`, or `swanlab/`.

# Note
Add `GG` at end of every response. Remember synchronize the documents after every modification to the code.
