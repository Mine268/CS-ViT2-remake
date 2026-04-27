from __future__ import annotations

"""Regression tests for evaluation dataloader batching and Accelerate compatibility."""

import inspect
import io
import json
from pathlib import Path
import tarfile

import cv2
from hydra import compose, initialize_config_dir
import numpy as np

from src.data.wds import ClipSegment, equalize_rank_clip_segments, get_dataloader
from src.train.engine import create_accelerator, validate


def _add_bytes_member(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


def _encode_npy(value: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    return buffer.getvalue()


def _write_valid_sequence_tar(path: Path, sample_key: str = "demo_seq") -> None:
    num_frames = 3
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[..., 1] = 255
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Failed to encode synthetic PNG test image")
    png_bytes = encoded.tobytes()
    with tarfile.open(path, "w") as tar:
        _add_bytes_member(
            tar,
            f"{sample_key}.imgs_path.json",
            json.dumps(["a.png", "b.png", "c.png"]).encode("utf-8"),
        )
        _add_bytes_member(
            tar,
            f"{sample_key}.img_bytes.pickle",
            __import__("pickle").dumps([png_bytes, png_bytes, png_bytes], protocol=4),
        )
        _add_bytes_member(
            tar,
            f"{sample_key}.handedness.json",
            json.dumps("right").encode("utf-8"),
        )
        _add_bytes_member(
            tar,
            f"{sample_key}.data_source.json",
            json.dumps("DemoSet").encode("utf-8"),
        )
        _add_bytes_member(
            tar,
            f"{sample_key}.source_split.json",
            json.dumps("val").encode("utf-8"),
        )
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
        _add_bytes_member(
            tar,
            f"{sample_key}.intr_type.json",
            json.dumps("real").encode("utf-8"),
        )

        arrays = {
            "hand_bbox.npy": np.asarray([[0, 0, 1, 1]] * num_frames, dtype=np.float32),
            "joint_img.npy": np.zeros((num_frames, 21, 2), dtype=np.float32),
            "joint_hand_bbox.npy": np.zeros((num_frames, 21, 2), dtype=np.float32),
            "joint_cam.npy": np.ones((num_frames, 21, 3), dtype=np.float32),
            "joint_rel.npy": np.zeros((num_frames, 21, 3), dtype=np.float32),
            "joint_2d_valid.npy": np.ones((num_frames, 21), dtype=np.float32),
            "joint_3d_valid.npy": np.ones((num_frames, 21), dtype=np.float32),
            "joint_valid.npy": np.ones((num_frames, 21), dtype=np.float32),
            "mano_pose.npy": np.zeros((num_frames, 48), dtype=np.float32),
            "mano_shape.npy": np.zeros((num_frames, 10), dtype=np.float32),
            "has_mano.npy": np.zeros((num_frames,), dtype=np.float32),
            "mano_valid.npy": np.zeros((num_frames,), dtype=np.float32),
            "has_intr.npy": np.ones((num_frames,), dtype=np.float32),
            "timestamp.npy": np.arange(num_frames, dtype=np.float32),
            "focal.npy": np.ones((num_frames, 2), dtype=np.float32),
            "princpt.npy": np.zeros((num_frames, 2), dtype=np.float32),
        }
        for member_name, value in arrays.items():
            _add_bytes_member(tar, f"{sample_key}.{member_name}", _encode_npy(value))


def test_eval_dataloader_batches_outside_webdataset_and_survives_accelerate_prepare(tmp_path: Path):
    source_tar = tmp_path / "source.tar"
    _write_valid_sequence_tar(source_tar)

    loader = get_dataloader(
        url=[str(source_tar)],
        num_frames=1,
        stride=1,
        batch_size=2,
        num_workers=0,
        prefetch_factor=1,
        infinite=False,
        clip_sampling_mode="dense",
        shardshuffle=False,
        post_clip_shuffle=0,
    )
    assert loader.batch_size == 2

    config_dir = str((Path(__file__).resolve().parents[1] / "config").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="stage1")
    accelerator = create_accelerator(cfg)
    prepared_loader = accelerator.prepare(loader)

    batch = next(iter(prepared_loader))
    assert len(batch["imgs"]) == 2
    assert batch["imgs"][0].shape[0] == 1
    assert batch["joint_img"].shape[:2] == (2, 1)


def test_equalize_rank_clip_segments_trims_all_ranks_to_smallest_clip_count():
    rank_segments = [
        [ClipSegment("a.tar", 0, 5)],
        [ClipSegment("a.tar", 5, 8)],
        [ClipSegment("a.tar", 8, 12)],
    ]

    equalized = equalize_rank_clip_segments(rank_segments)

    assert [(seg.start_clip, seg.end_clip) for seg in equalized[0]] == [(0, 3)]
    assert [(seg.start_clip, seg.end_clip) for seg in equalized[1]] == [(5, 8)]
    assert [(seg.start_clip, seg.end_clip) for seg in equalized[2]] == [(8, 11)]


def test_validate_accepts_train_loop_eval_step_cap_keyword():
    assert "max_eval_steps" in inspect.signature(validate).parameters
