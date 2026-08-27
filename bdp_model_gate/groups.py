"""Iterating protected groups, including intersections.

Checking each protected attribute on its own is the common approach and it
misses the thing that matters most. A model can show acceptable disparity on
gender and acceptable disparity on region while failing badly for women in
one region — harm concentrates at intersections, and marginal checks are
blind to it by construction.

`iter_protected` yields the marginal attributes and, when asked, their
pairwise intersections, so every group-based check gets intersectional
coverage from one place rather than each reimplementing it.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import combinations

import pandas as pd

from ._logging import get_logger

logger = get_logger("groups")

#: Separator in a synthesised intersection name, e.g. "gender × region".
JOIN = " × "


def iter_protected(
    protected_df: pd.DataFrame,
    intersectional: bool = False,
    min_group_size: int = 30,
) -> Iterator[tuple[str, pd.Series]]:
    """Yields (label, group series) for each protected attribute.

    With `intersectional=True`, also yields the pairwise combinations. Only
    pairs are generated: three-way intersections fragment a validation set
    faster than any realistic `min_group_size` tolerates, and reporting a
    disparity computed over four rows is worse than not reporting one.

    An intersection whose every group falls below `min_group_size` is skipped
    with a log line rather than yielded, since the downstream check would
    only discard it anyway.
    """
    columns = list(protected_df.columns)
    for column in columns:
        yield column, protected_df[column]

    if not intersectional or len(columns) < 2:
        return

    for left, right in combinations(columns, 2):
        combined = protected_df[left].astype(str) + JOIN + protected_df[right].astype(str)
        usable = (combined.value_counts() >= min_group_size).sum()
        if usable < 2:
            logger.debug(
                "intersection %s%s%s has fewer than two groups of at least %d rows — "
                "skipping. Marginal checks on each attribute still apply.",
                left,
                JOIN,
                right,
                min_group_size,
            )
            continue
        yield f"{left}{JOIN}{right}", combined


__all__ = ["JOIN", "iter_protected"]
