# HANDOFF

## Current State

The project is now training from clip-native WebDataset shards under:

```text
/data_0/renkaiwen/webdatasets2_remake/
```

The active training path no longer uses the old sequence-formatted training shards from
`/data_0/renkaiwen/webdatasets2_512/`. Those old shards are only referenced by:

- `script/export_train_clips.py` as the raw export source

Current clip layout:

- `stage1` -> `train_stage1`, `clip_len=1`, `stride=1`
- `stage2` -> `train_stage2`, `clip_len=7`, `stride=4`
- exported shard size target: about `1GB` per tar

Current training defaults:

- LR scheduler uses only initial linear warmup via `GENERAL.warmup_step`; after warmup the LR stays constant. Cosine annealing is no longer part of the default schedule.
- `TRAIN.bbox_jitter.enabled=true` by default. Stage1 uses frame-level jitter; Stage2 uses clip-level main jitter plus small per-frame noise. It is train-only and validation/test keep the GT-bbox evaluation protocol.
- 2026-05-08 realtime demo review: bbox jitter improved robustness to detector bbox size/center errors, so bbox jitter is considered the correct direction and should be kept for bbox/detector-related experiments unless doing an explicit ablation.
- DINOv3-L/16 experiment configs are available as `stage1_dinov3_large` and `stage2_dinov3_large`. They use `model/facebook/dinov3-vitl16-pretrain-lvd1689m`, `MODEL.handec.context_dim=1024`; `stage1_dinov3_large` currently defaults to `TRAIN.sample_per_device=42`, and both keep full backbone fine-tuning enabled by default via `TRAIN.backbone_lr=1e-5`.
- DINOv3-H+/16 experiment configs are available as `stage1_dinov3` and `stage2_dinov3`. They use `model/facebook/dinov3-vith16plus`, `MODEL.handec.context_dim=1280`, and keep full backbone fine-tuning enabled by default via `TRAIN.backbone_lr=1e-5`.
- DINOv3 emits `cls + 4 register + patch` tokens. `src/model/backbone.py` strips register tokens before downstream geometry/decoder modules, so the model still receives `cls + patch` tokens. DINOv2 checkpoints are not compatible with DINOv3 configs.
- TI v1 feature regularization is available behind `MODEL.ti.enabled=true` for Stage1. It applies a FiLM-conditioned token transform after `persp_info_embedder`, reuses the same `handec`, inverse-rotates `global_orient`, inverse-transforms `pred_ray_unit/pred_rho` back to `trans`, and only supervises `theta/shape/joint_rel/trans`. Stage1+TI shares the main branch backbone/perspective tokens instead of re-encoding images for TI, and axis-angle root inverse rotation uses a stable quaternion compose to avoid bf16 non-finite gradients. It does not run image-space augmentation or feature consistency loss, and `MODEL.ti.apply_stage2` is intentionally not implemented yet.
- SwanLab progress now uses cumulative samples seen as the logging step. Internal checkpoint names and `best_model.json` still use optimizer `global_step`, so run directories remain comparable to older experiments.
- Checkpoint inventory and experiment metrics are summarized in [EXPERIMENT_RESULTS.md](/data_1/renkaiwen/CS-ViT2-remake/docs/EXPERIMENT_RESULTS.md). As of 2026-06-23, the best completed Stage1 run by `micro_rte_ego` is `2026-05-08-stage1-bbox-jitter-v2` (`16.67 mm`), while `2026-05-15-resume-50000` has the best scanned Stage1 MPJPE (`26.06 mm`). The completed DINOv3-L/16 + TI run `2026-05-29-19-36-56-csvit2-stage1-dinov3-large-ti` reached best `MPJPE=29.15 / RTE=19.45` at step 275000.

## What Was Completed

### Training data migration

- Added clip export helpers:
  - [export.py](/data_1/renkaiwen/CS-ViT2-remake/src/data/export.py)
  - [export_train_clips.py](/data_1/renkaiwen/CS-ViT2-remake/script/export_train_clips.py)
- Exported the full clip dataset to `/data_0/renkaiwen/webdatasets2_remake`
- Switched the production training dataloader to the clip-native loader
- Removed the old sequence-training loader from production code
- Removed legacy training config paths from `config/data.yaml`

### Performance validation

Measured on the full 8-dataset `stage1` setup:

- loader-only throughput:
  - old sequence layout: about `14.8 samples/s`
  - new clip layout: about `100.6 samples/s`
  - speedup: about `6.8x`
- loader + preprocess throughput:
  - old sequence layout: about `10.2 samples/s`
  - new clip layout: about `106.1 samples/s`
  - speedup: about `10.4x`

Representative artifact files:

- [stage1_full8_loader_benchmark.json](/data_1/renkaiwen/CS-ViT2-remake/checkpoint/stage1_full8_loader_benchmark.json)
- [stage1_full8_preprocess_benchmark.json](/data_1/renkaiwen/CS-ViT2-remake/checkpoint/stage1_full8_preprocess_benchmark.json)

### Logging and metrics

- Tracker now logs to SwanLab and mirrors scalar logs to local stdout/tmux
- Geometry metrics are split by supervision group:
  - `*_all`
  - `*_ego`
  - `*_aux`
- Current best-model selection metric is `micro_rte_ego`

### Smoke / tests

- Full test suite passed after the migration: `23 passed`
- Real stage1 clip-native smoke training passed

## Evaluation Split Findings

### AssemblyHands

- `val` is usable:
  - 2D annotations are real coordinates
  - 3D annotations are real
  - calibration is real
  - 3D projection matches 2D very closely
- clip-native validation shards are already prepared outside the repo:
  - `stage1` -> `/data_0/renkaiwen/webdatasets2_remake/AssemblyHands/val_stage1/`
  - `stage2` -> `/data_0/renkaiwen/webdatasets2_remake/AssemblyHands/val_stage2/`
  - dataset registry entries: `DATA.datasets.AssemblyHands.splits.val_stage1/val_stage2`
  - current default config wiring:
    - `stage1` uses `DATA.val.source = DATA.datasets.AssemblyHands.splits.val_stage1`
    - `stage2` uses `DATA.val.source = DATA.datasets.AssemblyHands.splits.val_stage2`
    - validation uses dedicated `DATA.val.batch_size` instead of mirroring train batch size
- `test-eccv2024` is **not** usable as a local GT benchmark:
  - 2D keypoints are placeholder values
  - 3D joints are placeholder values
  - extrinsics are placeholder values
  - there are also metadata naming mismatches in the public files

Temporary inspection scripts:

- [assemblyhands_val_check.py](/data_1/renkaiwen/CS-ViT2-remake/temp/assemblyhands_val_check.py)
- [assemblyhands_val_wds_check.py](/data_1/renkaiwen/CS-ViT2-remake/temp/assemblyhands_val_wds_check.py)
- [assemblyhands_test_check.py](/data_1/renkaiwen/CS-ViT2-remake/temp/assemblyhands_test_check.py)

Generated qualitative outputs:

- [temp/assemblyhands_val_check](/data_1/renkaiwen/CS-ViT2-remake/temp/assemblyhands_val_check)
- [temp/assemblyhands_test_check](/data_1/renkaiwen/CS-ViT2-remake/temp/assemblyhands_test_check)

Important note:
- AssemblyHands keypoint order is different from the project’s internal hand joint order.
- The temp visualization scripts now remap AssemblyHands order into the project order before drawing.

### HOT3D

- No local public validation split has been wired into the project yet.
- The practical next step is still to define and export a local HOT3D validation split from train.

## Known Open Issues / Risks

1. `AssemblyHands val` images are stored inside tar.gz archives, not plain directories.
   The new exporter handles this by reading frames directly from the archive when building the
   validation WebDataset shards.

2. `AssemblyHands test-eccv2024` should not be used for metric validation, only possibly for format
   / submission-template handling.

3. `stage2` currently excludes `FreiHAND` and `RHD` because they do not produce `T=7` clips under
   the present export settings.

4. `torch.compile` was tested and is not enabled. A minimal experiment hit graph breaks and then a
   segfault, so it should be treated as future optimization work, not something to switch on now.

## Recommended Next Steps

1. Run a controlled transformation-isomorphism ablation across datasets and model scales:
   compare `MODEL.ti.enabled=false/true` on DINOv2-L, DINOv3-L/16, and optionally DINOv3-H+/16;
   repeat on full stage1 data, ego-only, and dataset-specific subsets; report both aggregate
   metrics and challenge-stratified metrics.
2. Define a local `HOT3D val` split and export it into the clip-native format.
3. Once `AssemblyHands val` and `HOT3D val` are both in place, decide whether validation should
   use only AssemblyHands or a mixed multi-dataset source list.
4. After evaluation data is stable, consider whether:
   - `loss_theta`
   - `loss_shape`
   - `loss_joint_rel`
   should also be split into `all/ego/aux` logging variants.

## Useful Commands

### Export clip data

```bash
nohup .venv/bin/python script/export_train_clips.py \
  --stages stage1 stage2 \
  --output-root /data_0/renkaiwen/webdatasets2_remake \
  > checkpoint/clip_export.log 2>&1 &
```

### Stage1 training

```bash
make
make shell
make menu
```

The default `make` goal now enters a Python training TUI. Launch flow is `stage -> backbone ->
TI -> common parameters -> confirm`, and the same TUI can also attach to tmux, tail logs, stop
sessions, and filter multiple experiments by session name. Explicit Make targets remain available
for automation and scripted runs.

```bash
make train-stage1
make attach-stage1
make logs-stage1
```

Disable bbox jitter only for ablation:

```bash
make train-stage1 RUN_NAME=stage1-no-bbox-jitter OVERRIDES="TRAIN.bbox_jitter.enabled=false"
```

### DINOv3 training

```bash
make train-stage1-dinov3-large
make train-stage1-dinov3-large-ti
make train-stage2-dinov3-large STAGE1_WEIGHT=/path/to/dinov3_large_stage1/best_model

make train-stage1-dinov3
make train-stage2-dinov3 STAGE1_WEIGHT=/path/to/dinov3_stage1/best_model
```

Use these dedicated targets instead of `make train-stage1 CONFIG_NAME=...` because the generic
stage targets intentionally append the normal DINOv2 batch-size overrides.

### Stage2 training

```bash
make train-stage2 STAGE1_WEIGHT=/path/to/stage1/best_model
```

### Run tests

```bash
source .venv/bin/activate
pytest -q
```
