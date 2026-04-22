from __future__ import annotations

import glob
from typing import Iterable, List, Sequence


def as_list(value) -> List:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def expand_glob_patterns(patterns: Sequence[str]) -> List[str]:
    files: List[str] = []
    for pattern in patterns:
        files.extend(sorted(glob.glob(str(pattern))))
    return files


def flatten(items: Iterable[Iterable]) -> List:
    return [x for group in items for x in group]
