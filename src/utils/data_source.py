from __future__ import annotations

"""Helpers for normalizing dataset names carried through the training pipeline."""

import re
from typing import Any, Mapping, Optional

import numpy as np


def normalize_data_source_alias_key(value: Any) -> str:
    """Convert a dataset alias into a comparison-friendly lookup key."""
    raw = stringify_data_source_value(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "", raw)


def stringify_data_source_value(value: Any) -> str:
    """Convert scalar-like dataset metadata into a plain Python string."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return stringify_data_source_value(value.item())
        raise ValueError(f"Expected scalar data_source entry, got shape={value.shape}")
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return ""
        if len(value) == 1:
            return stringify_data_source_value(value[0])
    return str(value)


def canonicalize_data_source_name(
    value: Any,
    alias_map: Optional[Mapping[str, str]] = None,
) -> str:
    """
    Convert a raw sample `data_source` value into the canonical dataset name defined by config.

    Args:
        value: Raw dataset identifier from metadata or config.
        alias_map: Mapping from normalized alias key to canonical dataset name.

    Returns:
        Canonical dataset name when a matching alias exists, otherwise the original stringified
        value. This keeps the helper generic and avoids hiding project-specific dataset names
        in code paths that should remain config-driven.
    """
    raw = stringify_data_source_value(value).strip()
    if alias_map is None:
        return raw
    return alias_map.get(normalize_data_source_alias_key(raw), raw)
