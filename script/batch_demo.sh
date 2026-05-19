#!/bin/bash
# Batch demo: jitter v2 + pre-jitter, on AssemblyHands + HOT3D, GT + WiLoR
set -euo pipefail

REPO=/data_1/renkaiwen/CS-ViT2-remake
VENV=$REPO/.venv
JITTER_CKPT=$REPO/checkpoint/2026-05-15/2026-05-15-resume-50000/best_model
PREJITTER_CKPT=$REPO/checkpoint/2026-05-06/2026-05-06-19-09-07-csvit2-stage1/best_model

AH_SRC="/data_0/renkaiwen/webdatasets2_remake/AssemblyHands/val_stage1/*.tar"
HOT3D_SRC="/data_0/renkaiwen/webdatasets2_remake/HOT3D/train_stage1/*.tar"

MAX_FRAMES=16

run_one() {
    local label="$1" ckpt="$2" src="$3" detector="$4" out_dir="$5" gpu="$6"
    echo "=== [$label] detector=$detector gpu=$gpu ==="
    mkdir -p "$out_dir"
    CUDA_VISIBLE_DEVICES=$gpu $VENV/bin/python -u $REPO/script/demo_stage1.py \
        --input "$src" --input-type wds \
        --checkpoint "$ckpt" \
        --output-dir "$out_dir" \
        --detector "$detector" \
        --max-frames $MAX_FRAMES \
        --device cuda:0 \
        --batch-size 4 \
        --no-save-video \
        --save-overlays --save-npz
    echo "=== DONE [$label] ==="
}

# Round 1: Jitter v2 (4 parallel)
echo "========== ROUND 1: JITTER V2 =========="
run_one "jv2_AH_gt"     "$JITTER_CKPT"    "$AH_SRC"     "gt"    "$REPO/example/demo_jitter_v2/assemblyhands_val/gt"    1 &
run_one "jv2_AH_wilor"  "$JITTER_CKPT"    "$AH_SRC"     "wilor" "$REPO/example/demo_jitter_v2/assemblyhands_val/wilor" 3 &
run_one "jv2_HOT3D_gt"  "$JITTER_CKPT"    "$HOT3D_SRC"  "gt"    "$REPO/example/demo_jitter_v2/hot3d_train/gt"          4 &
run_one "jv2_HOT3D_wilor" "$JITTER_CKPT"  "$HOT3D_SRC"  "wilor" "$REPO/example/demo_jitter_v2/hot3d_train/wilor"       5 &
wait

# Round 2: Pre-jitter (4 parallel)
echo "========== ROUND 2: PRE-JITTER =========="
run_one "pj_AH_gt"      "$PREJITTER_CKPT" "$AH_SRC"     "gt"    "$REPO/example/demo_pre_jitter/assemblyhands_val/gt"    1 &
run_one "pj_AH_wilor"   "$PREJITTER_CKPT" "$AH_SRC"     "wilor" "$REPO/example/demo_pre_jitter/assemblyhands_val/wilor" 3 &
run_one "pj_HOT3D_gt"   "$PREJITTER_CKPT" "$HOT3D_SRC"  "gt"    "$REPO/example/demo_pre_jitter/hot3d_train/gt"          4 &
run_one "pj_HOT3D_wilor" "$PREJITTER_CKPT" "$HOT3D_SRC" "wilor" "$REPO/example/demo_pre_jitter/hot3d_train/wilor"       5 &
wait

echo "========== ALL DONE =========="
