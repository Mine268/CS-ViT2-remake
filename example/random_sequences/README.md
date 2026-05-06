# Random Sequence Demo Results

This folder contains four randomly selected continuous clips:

- `seq_00_assemblyhands_val_stage2`: AssemblyHands val-stage2, 7 frames
- `seq_01_assemblyhands_val_stage2`: AssemblyHands val-stage2, 7 frames
- `seq_02_hot3d_train_stage2`: HOT3D train-stage2, 7 frames
- `seq_03_hot3d_train_stage2`: HOT3D train-stage2, 7 frames

Each sequence folder contains:

- `frame_*.png`: sampled continuous frames
- `sequence_meta.json`: source WDS sample, intrinsics, and MediaPipe probe results
- `results/predictions.jsonl`: per-frame predictions
- `results/predictions_npz/*.npz`: camera-space joints/vertices, MANO pose/shape/trans, projections
- `results/overlays/*.png`: reprojection overlays

`selection_summary.json` summarizes the dataset, sample key, number of frames, number of detected
hands, overlay count, npz count, and handedness per frame for all selected sequences.
