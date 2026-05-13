from __future__ import annotations
"""Statistics on constrained bbox jitter parameter distributions."""
import sys, json, os, time, glob, numpy as np, torch
sys.path.insert(0, ".")
from src.data.preprocess import _jitter_hand_bbox, MIN_PATCH_EDGE_PIXELS
from src.data.wds import get_dataloader

DATASET = sys.argv[1]   # AssemblyHands or HOT3D
SPLIT = sys.argv[2]     # train_stage1
MAX_SAMPLES = int(sys.argv[3])  # 100000

urls = sorted(glob.glob(f"/data_0/renkaiwen/webdatasets2_remake/{DATASET}/{SPLIT}/*.tar"))
print(f"{DATASET}/{SPLIT}: {len(urls)} shards, target {MAX_SAMPLES}")

cfg = {
    "enabled": True, "prob": 1.0, "temporal_mode": "frame", "constrained": True,
    "scale_log_mean": 0.36, "scale_log_std": 0.15,
    "scale_range": [1.0, 2.0], "center_shift": 0.5,
    "aspect_ratio_range": [1.0, 1.0],
    "frame_center_shift": 0.0, "frame_scale_range": [1.0, 1.0], "min_edge_px": 8.0,
}

records = []; count = 0; t0 = time.time()
loader = get_dataloader(url=urls, num_frames=1, stride=1, batch_size=16,
    num_workers=0, prefetch_factor=1, infinite=False,
    clip_sampling_mode="dense", shardshuffle=False, post_clip_shuffle=0)

for batch in loader:
    bs = int(batch["hand_bbox"].shape[0])
    for bx in range(bs):
        if count >= MAX_SAMPLES: break
        _, _, H, W = batch["imgs"][bx].shape
        hb = batch["hand_bbox"][bx, 0]
        x1, y1, x2, y2 = [float(v) for v in hb]
        bs2 = max(x2 - x1, y2 - y1)
        if bs2 < MIN_PATCH_EDGE_PIXELS:
            count += 1; continue

        hb_t = hb.unsqueeze(0).unsqueeze(0)
        torch.manual_seed(42 + count)
        jt = _jitter_hand_bbox(hb_t, torch.tensor([[[float(W), float(H)]]]), cfg)
        jx1, jy1, jx2, jy2 = [float(v) for v in jt[0, 0]]
        js = max(jx2 - jx1, jy2 - jy1)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        jcx, jcy = (jx1 + jx2) / 2, (jy1 + jy2) / 2
        a = js / bs2; dx = (jcx - cx) / bs2; dy = (jcy - cy) / bs2; M = (a - 1) / 2
        records.append({
            "alpha": float(a), "dx": float(dx), "dy": float(dy),
            "abs_d": float(np.sqrt(dx**2 + dy**2)), "M": float(M),
            "ok": bool(abs(dx) <= M + 1e-5 and abs(dy) <= M + 1e-5),
        })
        count += 1
    if count >= MAX_SAMPLES: break
    if count % 20000 == 0:
        print(f"  {count}/{MAX_SAMPLES} ({count / (time.time() - t0):.0f} fps)")

elapsed = time.time() - t0
print(f"Done: {len(records)} in {elapsed:.0f}s ({len(records) / max(elapsed, 1):.0f} fps)")
arr = {k: np.array([r[k] for r in records]) for k in records[0]}
for k, lbl in [("alpha", "a"), ("dx", "dx"), ("dy", "dy"), ("abs_d", "|d|")]:
    a = arr[k]; q = np.percentile(a, [5, 25, 50, 75, 95])
    print(f"{lbl}: p50={q[2]:.3f} p5={q[0]:.3f} p95={q[4]:.3f} mean={a.mean():.3f}")
print(f"ok_rate={arr['ok'].mean():.4f}")

os.makedirs("example/bbox_analysis", exist_ok=True)
out = f"example/bbox_analysis/jitter_params_{DATASET.lower()}_{SPLIT}.json"
s = {k: {"p50": float(np.percentile(arr[k], 50)), "p5": float(np.percentile(arr[k], 5)),
          "p95": float(np.percentile(arr[k], 95)), "mean": float(arr[k].mean()),
          "std": float(arr[k].std())} for k in ["alpha", "dx", "dy", "abs_d"]}
s["dataset"] = DATASET; s["split"] = SPLIT
s["num_samples"] = len(records); s["ok_rate"] = float(arr["ok"].mean())
with open(out, "w") as f: json.dump(s, f, indent=2)
print(f"Saved to {out}")
