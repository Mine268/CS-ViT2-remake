from __future__ import annotations

"""Tests for run-name normalization used by checkpoint/tmux launch helpers."""

import datetime

from src.train.checkpoint import ensure_date_prefixed_run_name


def test_ensure_date_prefixed_run_name_adds_current_date_when_missing():
    now = datetime.datetime(2026, 4, 24, 10, 0, 0)
    assert (
        ensure_date_prefixed_run_name("10-46-54-csvit2-stage1", now=now)
        == "2026-04-24-10-46-54-csvit2-stage1"
    )


def test_ensure_date_prefixed_run_name_keeps_existing_date_prefix():
    now = datetime.datetime(2026, 4, 24, 10, 0, 0)
    assert (
        ensure_date_prefixed_run_name("2026-04-24-stage1-ablation-a", now=now)
        == "2026-04-24-stage1-ablation-a"
    )
