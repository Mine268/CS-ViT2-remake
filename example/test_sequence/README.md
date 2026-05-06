# AssemblyHands Test Sequence Demo

This folder contains a 7-frame continuous clip sampled from AssemblyHands `val_stage2`.
It is placed directly under `example/` so it is independent from the single-image
`example/demo_stage1/` samples.

Source metadata is stored in `sequence_meta.json`.

Run:

```bash
python script/demo_stage1.py \
  --input example/test_sequence \
  --input-type folder \
  --detector mediapipe \
  --min-detection-confidence 0.1 \
  --min-tracking-confidence 0.1 \
  --intrinsics 133.741608 133.863464 321.243256 237.782974 \
  --checkpoint /path/to/stage1/best_model \
  --output-dir example/test_sequence/results
```

Saved result:

- `results/predictions.jsonl`
- `results/predictions_npz/frame_*.npz`
- `results/overlays/frame_*.png`

The saved run detected one hand in all 7 frames.
