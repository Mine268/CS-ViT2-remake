from __future__ import annotations

import datetime
import json
import os
import os.path as osp
import re
import shutil
from typing import Dict

from accelerate import Accelerator
from omegaconf import OmegaConf


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "run"


def build_run_dir(description: str) -> str:
    now = datetime.datetime.now()
    date_dir = now.strftime("%Y-%m-%d")
    run_name = now.strftime("%H-%M-%S") + "-" + slugify(description)[:80]
    return osp.join("checkpoint", date_dir, run_name)


def save_config_snapshot(cfg, output_dir: str, config_name: str):
    os.makedirs(output_dir, exist_ok=True)
    with open(osp.join(output_dir, f"config_{config_name}.yaml"), "w", encoding="utf-8") as f:
        OmegaConf.save(cfg, f)


def save_best_model_variant(
    accelerator: Accelerator,
    output_dir: str,
    global_step: int,
    val_metrics: Dict[str, float],
    config_name: str,
    best_dir_name: str,
    metadata_filename: str,
):
    if not accelerator.is_main_process:
        return
    best_model_dir = osp.join(output_dir, best_dir_name)
    accelerator.save_state(best_model_dir)
    metadata = {
        "step": int(global_step),
        "config_name": config_name,
        "timestamp": datetime.datetime.now().isoformat(),
        "best_dir_name": best_dir_name,
        **val_metrics,
    }
    with open(osp.join(output_dir, metadata_filename), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_best_metric_info(output_dir: str, metric_key: str, metadata_filename: str) -> Dict:
    metadata_path = osp.join(output_dir, metadata_filename)
    if not osp.exists(metadata_path):
        return {"best_value": float("inf"), "step": 0}
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "best_value": data.get(metric_key, float("inf")),
            "step": data.get("step", 0),
        }
    except (json.JSONDecodeError, OSError):
        return {"best_value": float("inf"), "step": 0}


def manage_checkpoints(output_dir: str, keep_last_n: int = 3):
    ckpt_parent_dir = os.path.join(output_dir, "checkpoints")
    if not os.path.exists(ckpt_parent_dir):
        return
    ckpts = [d for d in os.listdir(ckpt_parent_dir) if d.startswith("checkpoint-")]
    try:
        ckpts.sort(key=lambda x: int(x.split("-")[-1]))
    except ValueError:
        return
    if len(ckpts) > keep_last_n:
        for ckpt_to_del in ckpts[:-keep_last_n]:
            shutil.rmtree(os.path.join(ckpt_parent_dir, ckpt_to_del), ignore_errors=True)
