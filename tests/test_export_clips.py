"""Tests for fixed-length clip export helpers."""

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import webdataset as wds

from src.data.export import build_clip_sample_key
from src.data.export import export_sequence_shards_to_clip_shards
from src.data.schema import normalize_decoded_clip_sample


def _add_bytes_member(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


def _encode_npy(value: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    return buffer.getvalue()


def _write_minimal_sequence_tar(path: Path, sample_key: str = "demo_seq") -> None:
    with tarfile.open(path, "w") as tar:
        num_frames = 3
        _add_bytes_member(tar, f"{sample_key}.imgs_path.json", json.dumps(["a", "b", "c"]).encode("utf-8"))
        _add_bytes_member(
            tar,
            f"{sample_key}.img_bytes.pickle",
            __import__("pickle").dumps([b"0", b"1", b"2"], protocol=4),
        )
        _add_bytes_member(tar, f"{sample_key}.handedness.json", json.dumps("right").encode("utf-8"))
        _add_bytes_member(tar, f"{sample_key}.data_source.json", json.dumps("demo").encode("utf-8"))
        _add_bytes_member(tar, f"{sample_key}.source_split.json", json.dumps("train").encode("utf-8"))
        _add_bytes_member(
            tar,
            f"{sample_key}.additional_desc.json",
            json.dumps([{}, {}, {}]).encode("utf-8"),
        )
        _add_bytes_member(
            tar,
            f"{sample_key}.source_index.json",
            json.dumps([{"frame": 0}, {"frame": 1}, {"frame": 2}]).encode("utf-8"),
        )
        _add_bytes_member(tar, f"{sample_key}.intr_type.json", json.dumps("real").encode("utf-8"))

        zeros = {
            "hand_bbox.npy": np.zeros((num_frames, 4), dtype=np.float32),
            "joint_img.npy": np.zeros((num_frames, 21, 2), dtype=np.float32),
            "joint_hand_bbox.npy": np.zeros((num_frames, 21, 2), dtype=np.float32),
            "joint_cam.npy": np.zeros((num_frames, 21, 3), dtype=np.float32),
            "joint_rel.npy": np.zeros((num_frames, 21, 3), dtype=np.float32),
            "joint_2d_valid.npy": np.ones((num_frames, 21), dtype=np.float32),
            "joint_3d_valid.npy": np.ones((num_frames, 21), dtype=np.float32),
            "joint_valid.npy": np.ones((num_frames, 21), dtype=np.float32),
            "mano_pose.npy": np.zeros((num_frames, 48), dtype=np.float32),
            "mano_shape.npy": np.zeros((num_frames, 10), dtype=np.float32),
            "has_mano.npy": np.ones((num_frames,), dtype=np.float32),
            "mano_valid.npy": np.ones((num_frames,), dtype=np.float32),
            "has_intr.npy": np.ones((num_frames,), dtype=np.float32),
            "timestamp.npy": np.arange(num_frames, dtype=np.float32),
            "focal.npy": np.ones((num_frames, 2), dtype=np.float32),
            "princpt.npy": np.zeros((num_frames, 2), dtype=np.float32),
        }
        for member_name, value in zeros.items():
            _add_bytes_member(tar, f"{sample_key}.{member_name}", _encode_npy(value))


def test_build_clip_sample_key_includes_start_and_length():
    assert build_clip_sample_key("demo", 12, 7) == "demo__start_000012__len_0007"


def test_export_sequence_shards_to_clip_shards_roundtrips_one_sequence(tmp_path: Path):
    source_tar = tmp_path / "source.tar"
    output_dir = tmp_path / "clips"
    _write_minimal_sequence_tar(source_tar)

    stats = export_sequence_shards_to_clip_shards(
        shard_paths=[str(source_tar)],
        dataset_name="DemoSet",
        split_name="train",
        clip_num_frames=2,
        clip_stride=1,
        output_dir=str(output_dir),
        max_tar_size_bytes=1024 * 1024,
    )

    assert stats.output_clips == 2
    assert stats.max_tar_size_bytes == 1024 * 1024
    clip_tar = output_dir / "000000.tar"
    decoded = next(
        iter(
            wds.WebDataset([str(clip_tar)], shardshuffle=False, nodesplitter=lambda src: src, workersplitter=lambda src: src).decode()
        )
    )
    clip_sample = normalize_decoded_clip_sample(decoded)
    assert clip_sample["num_frames"] == 2
    assert clip_sample["data_source"] == "DemoSet"
