from __future__ import annotations

"""Helpers for exporting fixed-length clip WebDataset shards from sequence-formatted inputs."""

from dataclasses import dataclass
import io
import json
import os
import os.path as osp
import pickle
import tarfile
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import webdataset as wds

from .schema import PER_FRAME_ARRAY_SPECS, normalize_decoded_clip_sample, slice_normalized_clip_sample


JSON_LIST_FIELDS = {
    "imgs_path": "imgs_path.json",
    "additional_desc": "additional_desc.json",
    "source_index": "source_index.json",
}
JSON_SCALAR_FIELDS = {
    "handedness": "handedness.json",
    "data_source": "data_source.json",
    "source_split": "source_split.json",
    "intr_type": "intr_type.json",
}
PICKLE_FIELDS = {
    "imgs_bytes": "img_bytes.pickle",
}
NPY_FIELDS = {field_name: f"{field_name}.npy" for field_name in PER_FRAME_ARRAY_SPECS.keys()}


@dataclass(frozen=True)
class ExportStats:
    """Basic export statistics emitted for one dataset/stage run."""

    dataset_name: str
    split_name: str
    clip_num_frames: int
    input_shards: int
    input_sequences: int
    output_clips: int
    output_shards: int
    max_tar_size_bytes: int


def build_clip_sample_key(sample_key: str, clip_start: int, clip_length: int) -> str:
    """Construct a deterministic key for one exported clip."""
    return f"{sample_key}__start_{clip_start:06d}__len_{clip_length:04d}"


def iter_sequence_clip_samples(
    shard_paths: Sequence[str],
    dataset_name: str,
    split_name: str,
    clip_num_frames: int,
    clip_stride: int,
) -> Iterable[Dict]:
    """
    Yield fixed-length clip samples from sequence-formatted source shards.

    The source shard is expected to follow the current sequence schema from
    `/data_0/renkaiwen/webdatasets2_512`. The yielded sample already matches the normalized clip
    schema expected by the remake codebase and therefore can be written directly as a clip shard.
    """
    dataset = wds.WebDataset(
        list(shard_paths),
        shardshuffle=False,
        nodesplitter=lambda src: src,
        workersplitter=lambda src: src,
    ).decode()

    for decoded_sample in dataset:
        sequence_sample = normalize_decoded_clip_sample(
            decoded_sample,
            default_data_source=dataset_name,
            default_source_split=split_name,
            force_data_source=True,
        )
        total_frames = int(sequence_sample["num_frames"])
        if total_frames < clip_num_frames:
            continue

        num_clips = (total_frames - clip_num_frames) // clip_stride + 1
        for clip_idx in range(num_clips):
            clip_start = clip_idx * clip_stride
            clip_end = clip_start + clip_num_frames
            clip_sample = slice_normalized_clip_sample(
                sequence_sample,
                start=clip_start,
                end=clip_end,
                slice_index=None,
            )
            clip_sample["__key__"] = build_clip_sample_key(
                sample_key=str(sequence_sample["__key__"]),
                clip_start=clip_start,
                clip_length=clip_num_frames,
            )
            yield clip_sample


def _add_bytes_member(tar: tarfile.TarFile, member_name: str, payload: bytes) -> None:
    """Write one bytes payload as a tar member."""
    info = tarfile.TarInfo(name=member_name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


def _encode_npy(value: np.ndarray) -> bytes:
    """Serialize one numpy array using the standard `.npy` format."""
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    return buffer.getvalue()


def _estimate_tar_member_size(payload: bytes) -> int:
    """
    Estimate how many bytes one tar member occupies on disk.

    Tar stores a 512-byte header plus payload padded up to the next 512-byte block.
    """
    payload_blocks = (len(payload) + 511) // 512
    return 512 + payload_blocks * 512


def serialize_clip_sample_members(clip_sample: Mapping[str, object]) -> List[tuple[str, bytes]]:
    """Serialize one normalized clip sample into individual WebDataset members."""
    sample_key = str(clip_sample["__key__"])
    members: List[tuple[str, bytes]] = []

    for field_name, member_suffix in JSON_LIST_FIELDS.items():
        payload = json.dumps(clip_sample[field_name], ensure_ascii=False).encode("utf-8")
        members.append((f"{sample_key}.{member_suffix}", payload))

    for field_name, member_suffix in JSON_SCALAR_FIELDS.items():
        payload = json.dumps(clip_sample[field_name], ensure_ascii=False).encode("utf-8")
        members.append((f"{sample_key}.{member_suffix}", payload))

    for field_name, member_suffix in PICKLE_FIELDS.items():
        payload = pickle.dumps(clip_sample[field_name], protocol=4)
        members.append((f"{sample_key}.{member_suffix}", payload))

    for field_name, member_suffix in NPY_FIELDS.items():
        payload = _encode_npy(np.asarray(clip_sample[field_name], dtype=np.float32))
        members.append((f"{sample_key}.{member_suffix}", payload))

    return members


def estimate_clip_sample_tar_size(clip_sample: Mapping[str, object]) -> int:
    """Estimate how large one exported clip sample will be once written into a tar shard."""
    members = serialize_clip_sample_members(clip_sample)
    return sum(_estimate_tar_member_size(payload) for _, payload in members)


def write_clip_sample_to_tar(tar: tarfile.TarFile, clip_sample: Mapping[str, object]) -> int:
    """Serialize one normalized clip sample into the on-disk WebDataset member layout."""
    total_bytes = 0
    for member_name, payload in serialize_clip_sample_members(clip_sample):
        _add_bytes_member(tar, member_name, payload)
        total_bytes += _estimate_tar_member_size(payload)
    return total_bytes


def export_sequence_shards_to_clip_shards(
    shard_paths: Sequence[str],
    dataset_name: str,
    split_name: str,
    clip_num_frames: int,
    clip_stride: int,
    output_dir: str,
    max_tar_size_bytes: int,
) -> ExportStats:
    """
    Convert a list of sequence shards into fixed-length clip shards.

    Args:
        shard_paths: Input sequence shard paths.
        dataset_name: Canonical dataset name used in exported metadata.
        split_name: Split label stored in `source_split.json`.
        clip_num_frames: Clip length, e.g. 1 for stage1 and 7 for stage2.
        clip_stride: Sliding-window stride when generating clips from a source sequence.
        output_dir: Directory that will receive numbered output tar files plus metadata.
        max_tar_size_bytes: Hard cap on how large each output tar is allowed to grow.
    """
    os.makedirs(output_dir, exist_ok=True)

    output_tar: Optional[tarfile.TarFile] = None
    output_tar_index = 0
    current_tar_bytes = 0
    output_clips = 0
    output_shards = 0
    input_sequences = 0

    def _open_next_tar() -> tarfile.TarFile:
        nonlocal output_tar_index, current_tar_bytes, output_shards
        tar_path = osp.join(output_dir, f"{output_tar_index:06d}.tar")
        output_tar_index += 1
        current_tar_bytes = 0
        output_shards += 1
        return tarfile.open(tar_path, mode="w")

    try:
        for clip_sample in iter_sequence_clip_samples(
            shard_paths=shard_paths,
            dataset_name=dataset_name,
            split_name=split_name,
            clip_num_frames=clip_num_frames,
            clip_stride=clip_stride,
        ):
            input_sequences += int(clip_sample["source_index"][0]["frame_idx_within_clip"] == 0)
            clip_tar_bytes = estimate_clip_sample_tar_size(clip_sample)
            # Reserve 1024 bytes for the tar EOF markers when deciding whether to rotate.
            if (
                output_tar is None
                or (current_tar_bytes > 0 and current_tar_bytes + clip_tar_bytes + 1024 > max_tar_size_bytes)
            ):
                if output_tar is not None:
                    output_tar.close()
                output_tar = _open_next_tar()
            current_tar_bytes += write_clip_sample_to_tar(output_tar, clip_sample)
            output_clips += 1
    finally:
        if output_tar is not None:
            output_tar.close()

    stats = ExportStats(
        dataset_name=dataset_name,
        split_name=split_name,
        clip_num_frames=clip_num_frames,
        input_shards=len(shard_paths),
        input_sequences=input_sequences,
        output_clips=output_clips,
        output_shards=output_shards,
        max_tar_size_bytes=max_tar_size_bytes,
    )
    with open(osp.join(output_dir, "_export_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats.__dict__, f, indent=2, ensure_ascii=False)
    return stats
