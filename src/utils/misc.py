from __future__ import annotations

"""Small generic utilities shared by config parsing and data loading."""

import glob
from collections.abc import Sequence as SequenceABC
from typing import Iterable, List, Sequence


def as_list(value) -> List:
    """Return `value` as a list without splitting strings into characters."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]


def expand_glob_patterns(patterns: Sequence[str]) -> List[str]:
    """Resolve and concatenate multiple filesystem glob patterns in a stable order."""
    files: List[str] = []
    for pattern in patterns:
        files.extend(sorted(glob.glob(str(pattern))))
    return files


def flatten(items: Iterable[Iterable]) -> List:
    """Flatten a one-level nested iterable into a plain Python list."""
    return [x for group in items for x in group]
