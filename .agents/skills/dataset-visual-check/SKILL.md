---
name: dataset-visual-check
description: Use when validating hand dataset exports, WebDataset samples, preprocess geometry, bbox jitter, 2D/3D projection, crop/patch alignment, or left/right hand flip handling in this repository. This skill standardizes visual verification outputs and prevents coordinate-system mistakes.
---

# Dataset Visual Check

## Purpose

Use this skill before trusting any dataset conversion, bbox preprocessing change, camera/perspective transform change, or demo detector bbox adjustment. The output should make coordinate-system mistakes visually obvious and should include numeric summary evidence.

## Required Outputs

Write verification artifacts under `temp/<check_name>/` unless the user asks for another location. Match the existing style used by `temp/assemblyhands_val_check`:

- `summary.json` with candidate counts, visualization paths, config used, and per-frame numeric records.
- `vis_XX_<sample-id>.png` images with text legends.
- Optional per-frame `meta.json` files only when the image would become too cluttered.

Do not write generated visualization outputs outside `temp/`, `example/`, or an explicitly requested directory. Do not delete `/data_0` data.

## Coordinate-System Rules

Always draw boxes, joints, and images in the same coordinate system.

- Raw dataset coordinates: draw only raw image, raw `joint_img`, raw `hand_bbox`, raw reprojection.
- Preprocess image-plane coordinates: first apply `trans_2d_mat` to the raw image with `cv2.warpPerspective`; if `batch_out["flip"][bx]` is true, horizontally flip the warped image before drawing `batch_out["joint_img"]`, `batch_out["hand_bbox"]`, or `batch_out["patch_bbox"]`.
- Patch coordinates: draw on `batch_out["patches"][bx, tx]` using `batch_out["joint_patch_resized"][bx, tx]`.
- Never draw preprocess-output bbox directly on the raw image for left-hand samples or augmented samples. This was the exact failure mode caught in the bbox jitter check.

## Standard Panels

For preprocess or bbox validation, prefer one horizontal PNG with these panels:

1. Raw origin: raw image, raw 2D joints, raw tight bbox.
2. Preprocess image plane: warped/flipped image, preprocess joints, baseline bbox, jitter/detector bbox, patch bbox.
3. Baseline patch: patch image without the tested change and patch-space joints.
4. Changed patch: patch image with the tested change and patch-space joints.

Use distinct colors and write a legend into the image:

- Green: joints in the current panel coordinate system.
- Cyan: baseline/tight hand bbox.
- Orange: changed/noisy/detector hand bbox.
- Magenta: patch bbox.
- Red or purple: reprojection or invalid/masked diagnostic overlays, if relevant.

## Numeric Checks

Include numeric records in `summary.json` that match the operation being validated:

- Bbox jitter: baseline bbox, changed bbox, baseline patch bbox, changed patch bbox, `edge_ratio`, `center_offset_over_edge`, and whether `trans_2d_mat` stayed the same when it should.
- Projection validation: mean/max reprojection error, negative-depth rate, valid joint counts, bbox-inside rate.
- Crop/patch validation: patch bbox edge, expansion ratio, number of valid joints inside patch, and any clipped bbox flags.
- Detector comparison: raw detector bbox, scaled bbox, GT bbox, IoU, edge ratio, area ratio, center offset over GT edge.

For deterministic visual checks, set seeds explicitly with `random.seed(...)`, `np.random.seed(...)` if used, and `torch.manual_seed(...)` before each paired preprocess call.

## Repository Helpers

Prefer existing helpers before writing new logic:

- Decode clip-native WDS samples with `src.data.schema.normalize_decoded_clip_sample`.
- Convert a normalized frame to batch input with `src.data.wds.preprocess_frame`.
- Run project preprocessing with `src.data.preprocess.preprocess_batch`.
- Draw hand skeletons using `src.constant.MANO_JOINTS_CONNECTION`.
- Use existing reference scripts in `temp/` as patterns:
  - `temp/assemblyhands_val_check.py` for raw AssemblyHands annotation overlay format.
  - `temp/assemblyhands_val_wds_check.py` for WDS decode, reprojection metrics, crop/patch visual checks.
  - `temp/bbox_jitter_check.py` for bbox jitter visual checks.

## Validation Procedure

1. Inspect the relevant existing reference script and the target data format.
2. Build a small deterministic candidate set, usually 6-8 frames for visualization and up to a few hundred frames for summary statistics.
3. Render visual panels and write `summary.json`.
4. Read back `summary.json` and list the generated files.
5. If a panel looks wrong, first audit coordinate systems, left/right flip, and whether augmentation changed image-plane coordinates.
6. Report both the output directory and the numeric evidence. If any visualization is invalid, say so and regenerate rather than claiming success.

## Smoke Commands

Typical checks after generating a temporary visualization script:

```bash
.venv/bin/python temp/<check_script>.py
find temp/<check_name> -maxdepth 1 -type f | sort
sed -n '1,120p' temp/<check_name>/summary.json
```

For code that should remain in the project, add focused tests. For throwaway visualization scripts under `temp/`, do not add them to git unless the user explicitly asks.
