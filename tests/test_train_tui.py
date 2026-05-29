from __future__ import annotations

import pytest
from script.train_tui import (
    LaunchSettings,
    build_make_command,
    build_override_tokens,
    infer_num_processes,
    resolve_preset,
)


def test_resolve_preset_for_dinov3_large_ti_stage1():
    preset = resolve_preset("stage1", "dinov3_large", True)

    assert preset.target == "train-stage1-dinov3-large-ti"
    assert preset.config_name == "stage1_dinov3_large"
    assert preset.default_batch == 42
    assert preset.ti_via_target is True


def test_resolve_preset_rejects_stage2_ti():
    with pytest.raises(ValueError, match="stage1"):
        resolve_preset("stage2", "baseline", True)


def test_build_override_tokens_adds_ti_for_baseline_stage1():
    preset = resolve_preset("stage1", "baseline", True)
    settings = LaunchSettings(
        stage="stage1",
        backbone="baseline",
        ti_enabled=True,
        batch_size=42,
        heatmap_sigma=4.0,
    )

    assert build_override_tokens(settings, preset) == ["MODEL.ti.enabled=true"]


def test_build_make_command_for_stage2_includes_weight_and_overrides():
    preset = resolve_preset("stage2", "dinov3_large", False)
    settings = LaunchSettings(
        stage="stage2",
        backbone="dinov3_large",
        ti_enabled=False,
        session_prefix="csvit2",
        gpu_ids="0,1",
        num_processes=2,
        main_process_port="29501",
        run_name="stage2-dinov3-large",
        batch_size=2,
        heatmap_sigma=5.0,
        stage1_weight="/tmp/stage1-best",
        extra_overrides="GENERAL.total_samples=1000",
        dry_run=True,
    )

    command = build_make_command(settings, preset)

    assert command[2] == "train-stage2-dinov3-large"
    assert "NUM_PROCESSES=2" in command
    assert "MAIN_PROCESS_PORT=29501" in command
    assert "RUN_NAME=stage2-dinov3-large" in command
    assert "STAGE1_WEIGHT=/tmp/stage1-best" in command
    assert (
        "OVERRIDES=TRAIN.sample_per_device=2 LOSS.heatmap_sigma=5 GENERAL.total_samples=1000"
        in command
    )


def test_infer_num_processes_counts_gpu_list():
    assert infer_num_processes("0, 3, 4") == 3
    assert infer_num_processes("7") == 1
