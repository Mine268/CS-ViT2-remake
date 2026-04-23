from __future__ import annotations

"""Export fixed-length training clip shards from the original sequence WebDataset shards."""

import argparse
import json
import os.path as osp
from pathlib import Path
import sys
from typing import Dict, Iterable, List

from hydra import compose, initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.export import export_sequence_shards_to_clip_shards
from src.utils.misc import expand_glob_patterns


DEFAULT_OUTPUT_ROOT = "/data_0/renkaiwen/webdatasets2_remake"
DEFAULT_SOURCE_ROOT = "/data_0/renkaiwen/webdatasets2_512"
RAW_SEQUENCE_SOURCE_SUBDIRS = {
    "AssemblyHands": "AssemblyHands/train",
    "DexYCB": "DexYCB/s1/train",
    "FreiHAND": "FreiHAND/train",
    "HO3D_v3": "HO3D_v3/train",
    "HOT3D": "HOT3D/train",
    "InterHand2.6M": "InterHand2.6M/train",
    "MTC": "MTC/train",
    "RHD": "RHD/train",
}
STAGE_TO_CLIP_LENGTH = {
    "stage1": 1,
    "stage2": 7,
}
STAGE_TO_CLIP_STRIDE = {
    "stage1": 1,
    "stage2": 4,
}
STAGE_TO_OUTPUT_SPLIT = {
    "stage1": "train_stage1",
    "stage2": "train_stage2",
}
DEFAULT_MAX_TAR_SIZE_GB = 1.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=sorted(STAGE_TO_CLIP_LENGTH.keys()),
        default=["stage1", "stage2"],
        help="Which stage-specific clip datasets to export.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset subset. Defaults to all datasets from config/data.yaml.",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory where the exported remake clip shards will be written.",
    )
    parser.add_argument(
        "--source-root",
        default=DEFAULT_SOURCE_ROOT,
        help="Root directory containing the original sequence-formatted source shards.",
    )
    parser.add_argument(
        "--max-input-shards",
        type=int,
        default=None,
        help="Optional cap on how many source shards to process per dataset. Useful for smoke tests.",
    )
    parser.add_argument(
        "--clip-stride",
        type=int,
        default=None,
        help="Optional sliding-window stride override. Defaults to stage1=1 and stage2=4.",
    )
    parser.add_argument(
        "--max-tar-size-gb",
        type=float,
        default=DEFAULT_MAX_TAR_SIZE_GB,
        help="Maximum on-disk size of each exported tar shard in GB.",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional path to write a summary JSON for this export job.",
    )
    return parser.parse_args()


def _load_data_cfg():
    config_dir = str((Path(__file__).resolve().parents[1] / "config").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="stage1")
    return cfg.DATA


def _resolve_dataset_names(data_cfg, dataset_names: Iterable[str] | None) -> List[str]:
    if dataset_names is None or len(list(dataset_names)) == 0:
        return list(data_cfg.datasets.keys())
    resolved = [str(name) for name in dataset_names]
    unknown = [name for name in resolved if name not in data_cfg.datasets]
    if unknown:
        raise ValueError(f"Unknown dataset names: {unknown}")
    return resolved


def _resolve_source_shards(dataset_name: str, source_root: str) -> List[str]:
    if dataset_name not in RAW_SEQUENCE_SOURCE_SUBDIRS:
        raise ValueError(f"Dataset {dataset_name} has no raw sequence source mapping")
    pattern = str(Path(source_root) / RAW_SEQUENCE_SOURCE_SUBDIRS[dataset_name] / "*.tar")
    shard_paths = expand_glob_patterns([pattern])
    if len(shard_paths) == 0:
        raise ValueError(
            f"Dataset {dataset_name} matched no source shards from pattern {pattern}"
        )
    return shard_paths


def main() -> None:
    args = _parse_args()
    data_cfg = _load_data_cfg()
    dataset_names = _resolve_dataset_names(data_cfg, args.datasets)

    summary: Dict[str, Dict[str, Dict]] = {}
    for dataset_name in dataset_names:
        summary[dataset_name] = {}
        source_shards = _resolve_source_shards(dataset_name, source_root=args.source_root)
        if args.max_input_shards is not None:
            source_shards = source_shards[: int(args.max_input_shards)]

        for stage_name in args.stages:
            clip_num_frames = STAGE_TO_CLIP_LENGTH[stage_name]
            clip_stride = (
                int(args.clip_stride)
                if args.clip_stride is not None
                else STAGE_TO_CLIP_STRIDE[stage_name]
            )
            output_split = STAGE_TO_OUTPUT_SPLIT[stage_name]
            output_dir = osp.join(args.output_root, dataset_name, output_split)
            max_tar_size_bytes = int(float(args.max_tar_size_gb) * 1024 * 1024 * 1024)

            stats = export_sequence_shards_to_clip_shards(
                shard_paths=source_shards,
                dataset_name=dataset_name,
                split_name="train",
                clip_num_frames=clip_num_frames,
                clip_stride=clip_stride,
                output_dir=output_dir,
                max_tar_size_bytes=max_tar_size_bytes,
            )
            summary[dataset_name][stage_name] = stats.__dict__
            print(
                json.dumps(
                    {
                        "dataset": dataset_name,
                        "stage": stage_name,
                        "clip_stride": clip_stride,
                        **stats.__dict__,
                        "output_dir": output_dir,
                    },
                    ensure_ascii=False,
                )
            )

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
