from __future__ import annotations

"""Temporary benchmark script for isolating dataloader and preprocessing bottlenecks."""

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.config import build_train_data_plan
from src.data.preprocess import preprocess_batch
from src.train.engine import build_train_dataloader


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="stage1")
    parser.add_argument("--mode", choices=["loader", "preprocess"], default="loader")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--sample-per-device", type=int, default=None)
    parser.add_argument("--dataset-names", nargs="*", default=None)
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional path to write the benchmark summary as JSON.",
    )
    return parser.parse_args()


def _load_cfg(config_name: str) -> DictConfig:
    config_dir = str((Path(__file__).resolve().parents[2] / "config").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name)
    return cfg

def _filter_to_dataset_subset(cfg: DictConfig, dataset_names: Iterable[str]) -> None:
    dataset_names = {str(name) for name in dataset_names}
    if len(dataset_names) == 0:
        return

    group_cfg = cfg.DATA.train.groups
    active_group_weights: Dict[str, float] = {}
    for group_name in ("ego", "aux"):
        original_group_weight = float(group_cfg[group_name].weight)
        original_dataset_weights = {
            str(name): float(weight) for name, weight in group_cfg[group_name].datasets.items()
        }
        filtered = {name: weight for name, weight in original_dataset_weights.items() if name in dataset_names}
        if len(filtered) == 0:
            raise ValueError(
                f"Selected dataset subset leaves group '{group_name}' empty. "
                f"Subset must keep at least one dataset in each of ego and aux."
            )
        total = sum(filtered.values())
        group_cfg[group_name].datasets = {name: weight / total for name, weight in filtered.items()}
        active_group_weights[group_name] = original_group_weight

    total_group = sum(active_group_weights.values())
    for group_name in ("ego", "aux"):
        group_cfg[group_name].weight = active_group_weights[group_name] / total_group


def _apply_runtime_overrides(cfg: DictConfig, args: argparse.Namespace) -> None:
    if args.num_workers is not None:
        cfg.GENERAL.num_worker = int(args.num_workers)
    if args.prefetch_factor is not None:
        cfg.GENERAL.prefetch_factor = int(args.prefetch_factor)
    if args.sample_per_device is not None:
        cfg.TRAIN.sample_per_device = int(args.sample_per_device)
    if args.dataset_names:
        _filter_to_dataset_subset(cfg, args.dataset_names)


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summarize_timing(values: List[float]) -> Dict[str, float]:
    return {
        "mean_s": statistics.mean(values),
        "median_s": statistics.median(values),
        "p95_s": sorted(values)[max(0, int(len(values) * 0.95) - 1)],
        "min_s": min(values),
        "max_s": max(values),
    }


def main() -> None:
    args = _parse_args()
    cfg = _load_cfg(args.config_name)
    _apply_runtime_overrides(cfg, args)
    plan = build_train_data_plan(cfg.DATA)
    loader = build_train_dataloader(cfg)
    iterator = iter(loader)
    device = torch.device(args.device)

    loader_times: List[float] = []
    preprocess_times: List[float] = []
    total_times: List[float] = []
    source_counter: Counter[str] = Counter()
    unique_sources_per_batch: List[int] = []

    total_iters = args.warmup + args.batches
    for step_idx in range(total_iters):
        iter_start = time.perf_counter()
        loader_start = time.perf_counter()
        batch_origin = next(iterator)
        loader_end = time.perf_counter()

        batch_sources = list(batch_origin.get("data_source", []))
        source_counter.update(batch_sources)
        unique_sources_per_batch.append(len(set(batch_sources)))

        preprocess_elapsed = 0.0
        if args.mode == "preprocess":
            _sync_if_cuda(device)
            preprocess_start = time.perf_counter()
            preprocess_batch(
                batch_origin=batch_origin,
                patch_size=[cfg.MODEL.img_size, cfg.MODEL.img_size],
                patch_expanstion=cfg.TRAIN.expansion_ratio,
                scale_z_range=cfg.TRAIN.scale_z_range,
                scale_f_range=cfg.TRAIN.scale_f_range,
                persp_rot_max=cfg.TRAIN.persp_rot_max,
                joint_rep_type=cfg.MODEL.joint_type,
                augmentation_flag=True,
                device=device,
                pixel_aug=None,
                perspective_normalization=cfg.TRAIN.get("perspective_normalization", False),
            )
            _sync_if_cuda(device)
            preprocess_elapsed = time.perf_counter() - preprocess_start

        total_elapsed = time.perf_counter() - iter_start

        if step_idx >= args.warmup:
            loader_times.append(loader_end - loader_start)
            if args.mode == "preprocess":
                preprocess_times.append(preprocess_elapsed)
            total_times.append(total_elapsed)

    batch_size = int(cfg.TRAIN.sample_per_device)
    summary = {
        "config_name": args.config_name,
        "mode": args.mode,
        "device": str(device),
        "num_workers": int(cfg.GENERAL.num_worker),
        "prefetch_factor": int(cfg.GENERAL.prefetch_factor),
        "sample_per_device": batch_size,
        "batches_measured": args.batches,
        "group_weights": dict(plan.group_weights),
        "group_dataset_weights": {
            group_name: dict(weights) for group_name, weights in plan.group_dataset_weights.items()
        },
        "loader_timing": _summarize_timing(loader_times),
        "unique_sources_per_batch_mean": statistics.mean(unique_sources_per_batch[args.warmup:]),
        "source_counts": dict(source_counter),
        "loader_samples_per_second": batch_size / statistics.mean(loader_times),
        "end_to_end_samples_per_second": batch_size / statistics.mean(total_times),
    }
    if preprocess_times:
        summary["preprocess_timing"] = _summarize_timing(preprocess_times)

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
