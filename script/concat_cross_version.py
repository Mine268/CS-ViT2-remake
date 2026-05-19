#!/usr/bin/env python3
"""横向拼接 jitter v2 和 pre-jitter 的 overlay 图片，用于比较训练效果。"""
import cv2
import numpy as np
import os
import os.path as osp

REPO = "/data_1/renkaiwen/CS-ViT2-remake"
OUT_BASE = osp.join(REPO, "example/demo_cross_version")
JITTER_BASE = osp.join(REPO, "example/demo_jitter_v2")
PREJITTER_BASE = osp.join(REPO, "example/demo_pre_jitter")

DATASETS = ["assemblyhands_val", "hot3d_train"]
DETECTORS = ["gt", "wilor"]

os.makedirs(OUT_BASE, exist_ok=True)

for ds in DATASETS:
    for det in DETECTORS:
        jitter_dir = osp.join(JITTER_BASE, ds, det, "overlays")
        prejitter_dir = osp.join(PREJITTER_BASE, ds, det, "overlays")
        if not osp.isdir(jitter_dir) or not osp.isdir(prejitter_dir):
            print(f"SKIP {ds}/{det}: missing overlays")
            continue

        out_dir = osp.join(OUT_BASE, ds, det)
        os.makedirs(out_dir, exist_ok=True)

        jitter_files = set(f for f in os.listdir(jitter_dir) if f.endswith(".png"))
        prejitter_files = set(f for f in os.listdir(prejitter_dir) if f.endswith(".png"))
        common = sorted(jitter_files & prejitter_files)

        matched = 0
        for name in common:
            jitter_img = cv2.imread(osp.join(jitter_dir, name))
            prejitter_img = cv2.imread(osp.join(prejitter_dir, name))
            if jitter_img is None or prejitter_img is None:
                continue

            h, w = jitter_img.shape[:2]
            label_h = 36

            jitter_labeled = np.zeros((h + label_h, w, 3), dtype=np.uint8)
            jitter_labeled[:h] = jitter_img
            cv2.putText(jitter_labeled, "Jitter v2", (10, h + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            prejitter_labeled = np.zeros((h + label_h, w, 3), dtype=np.uint8)
            prejitter_labeled[:h] = prejitter_img
            cv2.putText(prejitter_labeled, "Pre-Jitter", (10, h + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            side_by_side = np.hstack([jitter_labeled, prejitter_labeled])
            cv2.imwrite(osp.join(out_dir, name), side_by_side)
            matched += 1

        print(f"[{ds}/{det}] matched {matched} pairs -> {out_dir}/")

print("Done.")
