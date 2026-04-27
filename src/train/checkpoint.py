from __future__ import annotations

"""Checkpoint/output directory helpers shared by training and tmux launch scripts."""

import datetime
import json
import os
import os.path as osp
import re
import shutil
from typing import Dict


RUN_DIR_ENV = "CSVIT2_RUN_DIR"
RUN_NAME_ENV = "CSVIT2_RUN_NAME"
RUN_NAME_DATE_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:-|$)")

from accelerate import Accelerator
from omegaconf import OmegaConf


def slugify(text: str) -> str:
    """Convert free-form run descriptions into filesystem-friendly slug fragments."""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "run"


def ensure_date_prefixed_run_name(run_name: str, now: datetime.datetime) -> str:
    """Prefix `YYYY-MM-DD-` unless the provided run name already starts with that date form."""
    run_name = run_name.strip()
    if RUN_NAME_DATE_PREFIX_RE.match(run_name):
        return run_name
    return f"{now.strftime('%Y-%m-%d')}-{run_name}"


def build_run_dir(description: str) -> str:
    """
    Build the output directory for one run.

    Environment overrides are honored first so tmux launch scripts can force the exact run name
    and checkpoint location shared by logs, local artifacts, and SwanLab.
    """
    env_run_dir = os.environ.get(RUN_DIR_ENV)
    if env_run_dir:
        return env_run_dir

    env_run_name = os.environ.get(RUN_NAME_ENV)
    now = datetime.datetime.now()
    date_dir = now.strftime("%Y-%m-%d")
    if env_run_name:
        run_name = ensure_date_prefixed_run_name(env_run_name, now=now)
    else:
        run_name = ensure_date_prefixed_run_name(
            now.strftime("%H-%M-%S") + "-" + slugify(description)[:80],
            now=now,
        )
    return osp.join("checkpoint", date_dir, run_name)


def save_config_snapshot(cfg, output_dir: str, config_name: str):
    """Persist the fully resolved Hydra config next to the run artifacts for reproducibility."""
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
    """Save the current best model variant and a small JSON summary of the validation metrics."""
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
    """Load the currently tracked best validation metric, tolerating missing/corrupt metadata."""
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
    """Prune older rolling checkpoints while keeping the most recent `keep_last_n` versions."""
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
