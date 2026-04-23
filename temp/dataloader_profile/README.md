# Dataloader Profile Temp Experiment

This directory holds one-off profiling utilities used to diagnose the current clip-native training input pipeline.

The main entrypoint is:

```bash
.venv/bin/python temp/dataloader_profile/bench_dataloader.py --help
```

Typical examples:

```bash
# Current training setup, loader-only timing
.venv/bin/python temp/dataloader_profile/bench_dataloader.py --mode loader

# Current setup, loader + preprocess timing on one GPU
.venv/bin/python temp/dataloader_profile/bench_dataloader.py --mode preprocess --device cuda:0

# Compare a reduced four-dataset mix against the current eight-dataset mix
.venv/bin/python temp/dataloader_profile/bench_dataloader.py \
  --mode loader \
  --dataset-names HOT3D AssemblyHands InterHand2.6M DexYCB
```
