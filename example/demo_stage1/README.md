# Stage 1 Demo Example

This folder contains one small AssemblyHands val frame for checking the stage1 demo without
MediaPipe. The bbox and camera intrinsics are stored in `sample.json`.

Run the fixed-bbox example:

```bash
python script/demo_stage1.py \
  --input example/demo_stage1/input.png \
  --input-type image \
  --detector bbox \
  --bbox 339.101257 355.587280 414.045929 444.258453 \
  --handedness left \
  --intrinsics 133.741608 133.863464 321.243256 237.782974 \
  --checkpoint /path/to/stage1/best_model \
  --output-dir output/demo_stage1_example
```

Expected outputs:

- `output/demo_stage1_example/predictions.jsonl`
- `output/demo_stage1_example/predictions_npz/*.npz`
- `output/demo_stage1_example/overlays/input.png`

Run the MediaPipe example:

```bash
python script/demo_stage1.py \
  --input example/demo_stage1/mediapipe_input.png \
  --input-type image \
  --detector mediapipe \
  --min-detection-confidence 0.1 \
  --min-tracking-confidence 0.1 \
  --intrinsics 133.741608 133.863464 321.243256 237.782974 \
  --checkpoint /path/to/stage1/best_model \
  --output-dir output/demo_stage1_mediapipe_example
```

The repository checks `model/mediapipe/hand_landmarker.task` by default for MediaPipe Tasks.
The detected bbox and score for this image are stored in `mediapipe_sample.json`.

Continuous sequence example is stored separately at `example/test_sequence`.
