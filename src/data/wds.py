"""WebDataset readers, mixers, and evaluation helpers for the remake training pipeline."""

from __future__ import annotations

import json
import tarfile
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch
import torchvision
import webdataset as wds
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .schema import normalize_decoded_clip_sample, slice_normalized_clip_sample

COLLATE_LIST_KEYS = {"imgs"}


@dataclass(frozen=True)
class ClipSegment:
    """Half-open clip index range assigned to one shard during balanced full evaluation."""

    tar_path: str
    start_clip: int
    end_clip: int


def count_sample_clips(total_frames: int, num_frames: int, stride: int) -> int:
    """Return how many fixed-length clips can be extracted from one sequence."""
    total_clips = (total_frames - num_frames) // stride + 1
    return max(0, total_clips)


def _select_clip_indices(
    total_frames: int,
    num_frames: int,
    stride: int,
    sampling_mode: str,
    clips_per_sequence: Optional[int],
    rng: np.random.Generator,
) -> List[int]:
    """Choose clip indices from one sequence according to the configured sampling mode."""
    total_clips = count_sample_clips(total_frames, num_frames, stride)
    if total_clips <= 0:
        return []
    if sampling_mode == "dense":
        return list(range(total_clips))
    if sampling_mode != "random_clip":
        raise ValueError(f"Unsupported clip sampling mode: {sampling_mode}")
    if clips_per_sequence is None:
        clips_per_sequence = 1
    num_select = min(int(clips_per_sequence), total_clips)
    return rng.choice(total_clips, size=num_select, replace=False).tolist()


def clip_to_t_frames(
    num_frames,
    stride,
    source,
    sampling_mode: str = "dense",
    clips_per_sequence: Optional[int] = None,
    seed: Optional[int] = None,
    default_data_source: Optional[str] = None,
    default_source_split: str = "unknown",
    data_source_alias_map: Optional[Mapping[str, str]] = None,
    force_data_source: bool = False,
    sample_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
):
    """
    Turn decoded sequence samples into fixed-length training/eval clips.

    Args:
        default_data_source: Canonical dataset name coming from the registry for the current
            stream. When `force_data_source=True`, this becomes the authoritative dataset name
            stored in the emitted clip metadata.
    """
    worker_info = get_worker_info()
    worker_id = worker_info.id if worker_info is not None else 0
    # Offset the RNG per worker so random_clip mode does not duplicate slices across workers.
    rng_seed = None if seed is None else seed + worker_id
    rng = np.random.default_rng(rng_seed)

    for decoded_sample in source:
        clip_sample = normalize_decoded_clip_sample(
            decoded_sample,
            default_data_source=default_data_source,
            default_source_split=default_source_split,
            data_source_alias_map=data_source_alias_map,
            force_data_source=force_data_source,
        )
        total_frames = clip_sample["num_frames"]
        if total_frames < num_frames:
            continue

        clip_indices = _select_clip_indices(
            total_frames=total_frames,
            num_frames=num_frames,
            stride=stride,
            sampling_mode=sampling_mode,
            clips_per_sequence=clips_per_sequence,
            rng=rng,
        )
        for clip_idx in clip_indices:
            start = int(clip_idx) * stride
            end = start + num_frames
            clip = slice_normalized_clip_sample(clip_sample, start, end, slice_index=int(clip_idx))
            if sample_filter is not None and not sample_filter(clip):
                continue
            yield clip


def estimate_wds_shard_clip_counts(
    urls: Sequence[str],
    num_frames: int,
    stride: int,
) -> List[int]:
    """Estimate how many train/eval clips each tar shard contains without decoding images."""
    clip_counts: List[int] = []
    for url in urls:
        shard_clip_count = 0
        with tarfile.open(url, "r") as tf:
            for member in tf:
                if not member.isfile() or not member.name.endswith("imgs_path.json"):
                    continue
                file_obj = tf.extractfile(member)
                if file_obj is None:
                    continue
                imgs_path = json.load(file_obj)
                shard_clip_count += count_sample_clips(len(imgs_path), num_frames, stride)
        clip_counts.append(shard_clip_count)
    return clip_counts


def build_balanced_clip_segments(
    urls: Sequence[str],
    clip_counts: Sequence[int],
    num_parts: int,
) -> List[List[ClipSegment]]:
    """Split shard clip ranges into contiguous per-rank segments for balanced full_eval."""
    if len(urls) != len(clip_counts):
        raise ValueError(f"urls and clip_counts length mismatch: {len(urls)} vs {len(clip_counts)}")
    if num_parts <= 0:
        raise ValueError(f"num_parts must be positive, got {num_parts}")

    total_clips = int(sum(clip_counts))
    assignments: List[List[ClipSegment]] = [[] for _ in range(num_parts)]
    if total_clips <= 0:
        return assignments

    part_ranges = [
        (total_clips * rank // num_parts, total_clips * (rank + 1) // num_parts)
        for rank in range(num_parts)
    ]

    # Convert global clip ranges back into shard-local ranges so each rank reads only its slice.
    clip_cursor = 0
    for url, shard_clip_count in zip(urls, clip_counts):
        shard_start = clip_cursor
        shard_end = clip_cursor + int(shard_clip_count)
        clip_cursor = shard_end
        if shard_clip_count <= 0:
            continue
        for rank, (part_start, part_end) in enumerate(part_ranges):
            overlap_start = max(shard_start, part_start)
            overlap_end = min(shard_end, part_end)
            if overlap_start >= overlap_end:
                continue
            assignments[rank].append(
                ClipSegment(
                    tar_path=url,
                    start_clip=overlap_start - shard_start,
                    end_clip=overlap_end - shard_start,
                )
            )
    return assignments


def count_segment_clips(segments: Sequence[ClipSegment]) -> int:
    """Return how many clip samples are covered by a rank's segment list."""
    return int(sum(max(0, int(segment.end_clip) - int(segment.start_clip)) for segment in segments))


def trim_clip_segments_to_count(
    segments: Sequence[ClipSegment],
    target_num_clips: int,
) -> List[ClipSegment]:
    """Keep the earliest clips from `segments` until exactly `target_num_clips` remain."""
    if target_num_clips <= 0:
        return []

    kept: List[ClipSegment] = []
    remaining = int(target_num_clips)
    for segment in segments:
        segment_count = max(0, int(segment.end_clip) - int(segment.start_clip))
        if segment_count <= 0:
            continue
        if segment_count <= remaining:
            kept.append(segment)
            remaining -= segment_count
        else:
            kept.append(
                ClipSegment(
                    tar_path=segment.tar_path,
                    start_clip=int(segment.start_clip),
                    end_clip=int(segment.start_clip) + remaining,
                )
            )
            remaining = 0
        if remaining <= 0:
            break
    return kept


def equalize_rank_clip_segments(
    rank_segments: Sequence[Sequence[ClipSegment]],
) -> List[List[ClipSegment]]:
    """
    Trim rank-local segments so every rank iterates the same number of clips.

    `accelerator.gather_for_metrics` requires every process to execute the same number of collective
    calls. When rank-local iterable datasets have slightly different lengths, later ranks can stall
    inside NCCL collectives after earlier ranks finish validation. We avoid that by trimming every
    rank to the smallest local clip count.
    """
    if len(rank_segments) == 0:
        return []
    local_counts = [count_segment_clips(segments) for segments in rank_segments]
    target_count = min(local_counts)
    return [
        trim_clip_segments_to_count(segments, target_num_clips=target_count)
        for segments in rank_segments
    ]


def _decode_webp_bytes(img_bytes: bytes) -> torch.Tensor:
    """Decode one image payload, with a generic fallback for non-WEBP encoded frames."""
    buffer_np = np.frombuffer(img_bytes, dtype=np.uint8).copy()
    buffer = torch.from_numpy(buffer_np)
    try:
        return torchvision.io.decode_webp(buffer)
    except RuntimeError:
        return torchvision.io.decode_image(buffer, mode=torchvision.io.ImageReadMode.RGB)


def preprocess_frame(sample):
    """Decode image bytes and convert numpy-backed fields into torch tensors."""
    imgs_tensor = [_decode_webp_bytes(img_bytes) for img_bytes in sample["imgs_bytes"]]
    imgs_tensor = torch.stack(imgs_tensor)

    result = {
        "__key__": sample["__key__"],
        "imgs_path": sample["imgs_path"],
        "imgs": imgs_tensor,
        "handedness": sample["handedness"],
        "data_source": sample["data_source"],
        "source_split": sample["source_split"],
        "source_index": sample["source_index"],
        "intr_type": sample["intr_type"],
        "additional_desc": sample["additional_desc"],
    }
    for key, value in sample.items():
        if key in result or key in {"num_frames", "imgs_bytes"}:
            continue
        if isinstance(value, np.ndarray):
            result[key] = torch.from_numpy(value).float()
        else:
            result[key] = value
    return result


def collate_fn(batch_wds):
    """Custom collate that keeps list-like metadata intact while stacking tensors."""
    batch_filter = [b for b in batch_wds if b is not None]
    if len(batch_filter) == 0:
        return {}

    collated = {}
    for key in batch_filter[0].keys():
        if key in COLLATE_LIST_KEYS:
            collated[key] = [sample[key] for sample in batch_filter]
        else:
            values = [sample[key] for sample in batch_filter]
            if isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values)
            else:
                collated[key] = values
    return collated


def _build_clip_webdataset(
    url,
    num_frames: int,
    stride: int,
    infinite: bool = True,
    seed: Optional[int] = None,
    clip_sampling_mode: str = "dense",
    clips_per_sequence: Optional[int] = None,
    shardshuffle: Union[bool, int] = False,
    post_clip_shuffle: int = 200,
    shuffle_buffer: int = 20,
    default_data_source: Optional[str] = None,
    default_source_split: str = "unknown",
    data_source_alias_map: Optional[Mapping[str, str]] = None,
    force_data_source: bool = False,
    sample_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
):
    """Build one iterable WebDataset pipeline before batch collation and DataLoader wrapping."""
    dataset = (
        wds.WebDataset(
            url,
            resampled=infinite,
            shardshuffle=shardshuffle,
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
            seed=seed,
        )
        .shuffle(shuffle_buffer, initial=seed if seed is not None else 0, seed=seed)
        .decode()
    )

    dataset = dataset.compose(
        partial(
            clip_to_t_frames,
            num_frames,
            stride,
            sampling_mode=clip_sampling_mode,
            clips_per_sequence=clips_per_sequence,
            seed=seed,
            default_data_source=default_data_source,
            default_source_split=default_source_split,
            data_source_alias_map=data_source_alias_map,
            force_data_source=force_data_source,
            sample_filter=sample_filter,
        )
    )
    if post_clip_shuffle > 0:
        dataset = dataset.shuffle(post_clip_shuffle, initial=seed if seed is not None else 0, seed=seed)
    return dataset.map(preprocess_frame)


def _build_precut_clip_webdataset(
    url,
    infinite: bool = True,
    seed: Optional[int] = None,
    shardshuffle: Union[bool, int] = False,
    post_clip_shuffle: int = 200,
    default_data_source: Optional[str] = None,
    default_source_split: str = "unknown",
    data_source_alias_map: Optional[Mapping[str, str]] = None,
    force_data_source: bool = False,
    sample_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
):
    """
    Build a WebDataset pipeline for already-precut clip shards.

    Unlike `_build_clip_webdataset`, this path assumes every sample stored in the shard is already
    a fixed-length clip. It therefore skips sequence-to-clip slicing and only performs schema
    normalization, optional filtering, and frame decoding.
    """
    dataset = (
        wds.WebDataset(
            url,
            resampled=infinite,
            shardshuffle=shardshuffle,
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
        )
        .shuffle(20, initial=seed if seed is not None else 0, seed=seed)
        .decode()
    )

    def _normalize_and_filter(decoded_sample: Dict[str, Any]):
        clip_sample = normalize_decoded_clip_sample(
            decoded_sample,
            default_data_source=default_data_source,
            default_source_split=default_source_split,
            data_source_alias_map=data_source_alias_map,
            force_data_source=force_data_source,
        )
        if sample_filter is not None and not sample_filter(clip_sample):
            return None
        return preprocess_frame(clip_sample)

    dataset = dataset.map(_normalize_and_filter)
    if post_clip_shuffle > 0:
        dataset = dataset.shuffle(post_clip_shuffle, initial=seed if seed is not None else 0, seed=seed)
    return dataset


def get_dataloader(
    url,
    num_frames: int,
    stride: int,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    infinite: bool = True,
    seed: Optional[int] = None,
    clip_sampling_mode: str = "dense",
    clips_per_sequence: Optional[int] = None,
    shardshuffle: Union[bool, int] = False,
    post_clip_shuffle: int = 200,
    shuffle_buffer: int = 20,
    default_data_source: Optional[str] = None,
    default_source_split: str = "unknown",
    data_source_alias_map: Optional[Mapping[str, str]] = None,
    force_data_source: bool = False,
    sample_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> DataLoader:
    """
    Create a batched DataLoader from a single WebDataset source or source list.

    Evaluation and test loaders intentionally batch at the outer PyTorch `DataLoader` level instead
    of inside the WebDataset pipeline. This keeps `DataLoader.batch_size` explicit so downstream
    wrappers such as Accelerate can shard the iterable correctly in multi-process validation.
    """
    dataset = _build_clip_webdataset(
        url=url,
        num_frames=num_frames,
        stride=stride,
        infinite=infinite,
        seed=seed,
        clip_sampling_mode=clip_sampling_mode,
        clips_per_sequence=clips_per_sequence,
        shardshuffle=shardshuffle,
        post_clip_shuffle=post_clip_shuffle,
        shuffle_buffer=shuffle_buffer,
        default_data_source=default_data_source,
        default_source_split=default_source_split,
        data_source_alias_map=data_source_alias_map,
        force_data_source=force_data_source,
        sample_filter=sample_filter,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=False,
    )


def get_group_reweight_precut_clip_dataloader(
    group_dataset_sources: Mapping[str, Mapping[str, Sequence[str]]],
    group_dataset_weights: Mapping[str, Mapping[str, float]],
    group_weights: Mapping[str, float],
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    infinite: bool = True,
    seed: Optional[int] = None,
    shardshuffle: Union[bool, int] = False,
    post_clip_shuffle: int = 200,
    default_source_split: str = "unknown",
    data_source_alias_map: Optional[Mapping[str, str]] = None,
    sample_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
):
    """
    Two-level `RandomMix` dataloader for clip-native shards.

    This is the production training loader for the remake clip-native dataset. Each source shard
    already stores fixed-length clips, so the loader no longer needs to slice a long sequence at
    runtime.
    """
    group_streams = []
    group_probs = []
    for group_idx, group_name in enumerate(group_weights.keys()):
        dataset_streams = []
        dataset_probs = []
        for dataset_idx, dataset_name in enumerate(group_dataset_weights[group_name].keys()):
            dataset_seed = None if seed is None else seed + group_idx * 100003 + dataset_idx * 1009
            dataset_streams.append(
                _build_precut_clip_webdataset(
                    url=list(group_dataset_sources[group_name][dataset_name]),
                    infinite=infinite,
                    seed=dataset_seed,
                    shardshuffle=shardshuffle,
                    post_clip_shuffle=post_clip_shuffle,
                    default_data_source=dataset_name,
                    default_source_split=default_source_split,
                    data_source_alias_map=data_source_alias_map,
                    force_data_source=True,
                    sample_filter=sample_filter,
                )
            )
            dataset_probs.append(float(group_dataset_weights[group_name][dataset_name]))

        if len(dataset_streams) == 1:
            mixed_group_stream = dataset_streams[0]
        else:
            mixed_group_stream = wds.RandomMix(
                dataset_streams,
                probs=dataset_probs,
                longest=not infinite,
            )
        group_streams.append(mixed_group_stream)
        group_probs.append(float(group_weights[group_name]))

    if len(group_streams) == 1:
        mixed_dataset = group_streams[0]
    else:
        mixed_dataset = wds.RandomMix(group_streams, probs=group_probs, longest=not infinite)
    return DataLoader(
        mixed_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=collate_fn,
        pin_memory=False,
    )


class WDSClipSegmentDataset(IterableDataset):
    """Sequential reader used by full_eval to materialize only the assigned clip windows."""

    def __init__(
        self,
        segments: Sequence[ClipSegment],
        num_frames: int,
        stride: int,
        default_data_source: Optional[str] = None,
        default_source_split: str = "unknown",
        data_source_alias_map: Optional[Mapping[str, str]] = None,
        force_data_source: bool = False,
    ):
        super().__init__()
        self.segments = list(segments)
        self.num_frames = num_frames
        self.stride = stride
        self.default_data_source = default_data_source
        self.default_source_split = default_source_split
        self.data_source_alias_map = data_source_alias_map
        self.force_data_source = force_data_source

    def __iter__(self):
        """Iterate only the clip windows assigned to this rank for segmented evaluation."""
        for segment in self.segments:
            dataset = wds.WebDataset(
                [segment.tar_path],
                shardshuffle=False,
                nodesplitter=lambda src: src,
                workersplitter=lambda src: src,
            ).decode()
            clip_cursor = 0
            for decoded_sample in dataset:
                clip_sample = normalize_decoded_clip_sample(
                    decoded_sample,
                    default_data_source=self.default_data_source,
                    default_source_split=self.default_source_split,
                    data_source_alias_map=self.data_source_alias_map,
                    force_data_source=self.force_data_source,
                )
                total_frames = clip_sample["num_frames"]
                total_clips = count_sample_clips(total_frames, self.num_frames, self.stride)
                local_start = max(0, segment.start_clip - clip_cursor)
                local_end = min(total_clips, segment.end_clip - clip_cursor)
                if local_start < local_end:
                    # `segment` is expressed in shard-global clip indices, so convert it back to
                    # per-sample offsets before slicing the current decoded sequence.
                    for clip_idx in range(local_start, local_end):
                        start = clip_idx * self.stride
                        end = start + self.num_frames
                        yield preprocess_frame(
                            slice_normalized_clip_sample(
                                clip_sample,
                                start,
                                end,
                                slice_index=int(clip_idx),
                            )
                        )
                clip_cursor += total_clips
                if clip_cursor >= segment.end_clip:
                    break

    def __len__(self) -> int:
        """Expose the exact local clip count so `DataLoader` can compute a stable batch length."""
        return count_segment_clips(self.segments)


def get_segmented_wds_dataloader(
    segments: Sequence[ClipSegment],
    num_frames: int,
    stride: int,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    default_data_source: Optional[str] = None,
    default_source_split: str = "unknown",
    data_source_alias_map: Optional[Mapping[str, str]] = None,
    force_data_source: bool = False,
):
    """Wrap `WDSClipSegmentDataset` in a standard DataLoader for full-eval usage."""
    dataset = WDSClipSegmentDataset(
        segments=segments,
        num_frames=num_frames,
        stride=stride,
        default_data_source=default_data_source,
        default_source_split=default_source_split,
        data_source_alias_map=data_source_alias_map,
        force_data_source=force_data_source,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=collate_fn,
        pin_memory=False,
    )
