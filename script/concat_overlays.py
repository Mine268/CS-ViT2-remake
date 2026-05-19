#!/usr/bin/env python3
"""横向拼接 GT bbox 和 WiLoR bbox 的 overlay 图片，同时比较 jitter v2 vs pre-jitter。"""
import cv2
import numpy as np
import os
import os.path as osp

REPO = "/data_1/renkaiwen/CS-ViT2-remake"
BASE_DIRS = {
    "jitter_v2": osp.join(REPO, "example/demo_jitter_v2"),
    "pre_jitter": osp.join(REPO, "example/demo_pre_jitter"),
}
DATASETS = ["assemblyhands_val", "hot3d_train"]

for version, base in BASE_DIRS.items():
    out_dir = osp.join(base, "side_by_side")
    os.makedirs(out_dir, exist_ok=True)

    for ds in DATASETS:
        gt_dir = osp.join(base, ds, "gt", "overlays")
        wl_dir = osp.join(base, ds, "wilor", "overlays")
        if not osp.isdir(gt_dir) or not osp.isdir(wl_dir):
            print(f"SKIP {version}/{ds}: missing overlays")
            continue

        gt_files = set(f for f in os.listdir(gt_dir) if f.endswith(".png"))
        wl_files = set(f for f in os.listdir(wl_dir) if f.endswith(".png"))
        common = sorted(gt_files & wl_files)

        matched = 0
        for name in common:
            gt_img = cv2.imread(osp.join(gt_dir, name))
            wl_img = cv2.imread(osp.join(wl_dir, name))
            if gt_img is None or wl_img is None:
                continue

            # Add labels at bottom
            h, w = gt_img.shape[:2]
            label_h = 36
            gt_labeled = np.zeros((h + label_h, w, 3), dtype=np.uint8)
            gt_labeled[:h] = gt_img
            cv2.putText(gt_labeled, "GT BBox", (10, h + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            wl_labeled = np.zeros((h + label_h, w, 3), dtype=np.uint8)
            wl_labeled[:h] = wl_img
            cv2.putText(wl_labeled, "WiLoR", (10, h + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # GT | WiLoR
            side_by_side = np.hstack([gt_labeled, wl_labeled])
            cv2.imwrite(osp.join(out_dir, f"{ds}_{name}"), side_by_side)
            matched += 1

        print(f"[{version}/{ds}] matched {matched} pairs -> {out_dir}/")

print("Done.")
