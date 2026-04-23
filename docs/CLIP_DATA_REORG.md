# Clip Data Reorganization

## Goal

The original training data under `/data_0/renkaiwen/webdatasets2_512/` stores one **sequence** per sample.
The current stage1 pipeline often trains on only `T=1` frame, but still needs to deserialize a full
sequence sample before slicing out the requested clip. This is the main data-loading bottleneck.

This reorganization exports a new clip-native dataset under:

```text
/data_0/renkaiwen/webdatasets2_remake/
```

The new layout stores one fixed-length clip per sample:

- `train_stage1`: clip length `1`
- `train_stage2`: clip length `7`

The default export stride is stage-aware:

- `stage1`: stride `1`
- `stage2`: stride `4`

For stage2 this means neighboring clips still overlap, but the dataset does not explode as much as
with stride `1`.

## Current Scope

This change adds:

- a clip export pipeline
- a clip-native dataloader implementation for benchmarking and later integration
- validation scripts and temporary profiling tools

The main training path now reads the exported clip-native dataset directly:

- `stage1` -> `train_stage1`
- `stage2` -> `train_stage2`

Because `FreiHAND` and `RHD` do not produce `T=7` clips, they are excluded from the current
stage2 aux group.

The production training config no longer keeps the old sequence-format training paths. The original
`/data_0/renkaiwen/webdatasets2_512/` tree is now only used by `script/export_train_clips.py` as a
raw export source.

## Export Script

Use:

```bash
.venv/bin/python script/export_train_clips.py --help
```

Example dry validation on a small subset:

```bash
.venv/bin/python script/export_train_clips.py \
  --datasets HOT3D InterHand2.6M \
  --stages stage1 stage2 \
  --max-input-shards 1 \
  --output-root /data_0/renkaiwen/webdatasets2_remake
```

Full background export example:

```bash
nohup .venv/bin/python script/export_train_clips.py \
  --stages stage1 stage2 \
  --output-root /data_0/renkaiwen/webdatasets2_remake \
  > checkpoint/clip_export.log 2>&1 &
```

By default, the exporter rotates shards at about **1GB per tar**.
By default, the exporter also uses `clip_stride=1` for stage1 and `clip_stride=4` for stage2.

## Output Layout

For each dataset:

```text
/data_0/renkaiwen/webdatasets2_remake/<DATASET>/train_stage1/*.tar
/data_0/renkaiwen/webdatasets2_remake/<DATASET>/train_stage2/*.tar
```

Each sample key is:

```text
<original_sequence_key>__start_<clip_start>__len_<clip_len>
```

With the default stage2 stride, neighboring `T=7` clips overlap by `3` frames.

The member layout matches the normalized clip schema already used by the remake codebase, so each
exported sample stores only the frames and labels for that clip.

## Validation Summary

Small-scale real export validation was completed for:

- `HOT3D`
- `InterHand2.6M`

Correctness checks confirmed:

- stage1 exported samples have `num_frames == 1`
- stage2 exported samples have `num_frames == 7`
- `data_source` is preserved as the canonical dataset name
- per-frame arrays and image lists have the expected clip length

## Measured Layout Improvement

Representative measurements:

- `HOT3D` stage1 old-sequence layout vs new clip layout: about **27.3x** faster
- `HOT3D` stage2 old-sequence layout vs new clip layout: about **5.5x** faster
- `HOT3D` stage1 batch-level `next(iter(loader))` comparison (`batch_size=32`, `num_workers=0`):
  about **2.0x** faster

Important nuance:

- datasets that are already mostly single-frame or very short-sequence in the old layout will not
  see the same level of improvement
- the biggest gains come from long-sequence sources such as `HOT3D`

## Related Temp Tools

Temporary profiling utilities live under:

```text
temp/dataloader_profile/
```

They were used to verify that the original bottleneck came from sequence-level sample deserialization
before runtime clip slicing.
