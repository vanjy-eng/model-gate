"""Small statistics the checks share, implemented in numpy.

Everything here works on a **core install**. That is the point: leakage
detection and proxy correlation are the two checks you least want to be
unavailable because scikit-learn is missing — a validation set that is
secretly the training set invalidates every other number in the report,
and a gate that silently skips that check is worse than one that never
had it.

`rank_auc` deliberately does not go into `bdp_model_gate.metrics`. The
registry's `roc_auc` is documented as needing scikit-learn, and quietly
making it numpy-native would change a published contract as a side effect
of an unrelated release. This is an internal statistic, not a gateable
metric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks 1..n, with ties sharing their mean rank.

    Tie handling is not cosmetic here. A leaked feature is often a coarse
    copy of the target with many repeated values, and ordinal ranks would
    score it differently depending on the order the rows happened to arrive
    in — the same class of order-dependence `stable_sample` exists to
    prevent elsewhere.
    """
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)

    ordered = values[order]
    # Boundaries of runs of equal values, then flatten each run to its mean.
    change = np.flatnonzero(np.concatenate(([True], ordered[1:] != ordered[:-1], [True])))
    for start, stop in zip(change[:-1], change[1:]):
        if stop - start > 1:
            ranks[order[start:stop]] = (start + stop + 1) / 2
    return ranks


def rank_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve, via the Mann–Whitney U statistic.

    Returns NaN when one class is absent, since AUC is undefined there —
    the caller decides whether that is a skip or a finding.
    """
    positive = np.asarray(y_true).astype(bool)
    n_pos = int(positive.sum())
    n_neg = int(positive.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = average_ranks(np.asarray(scores, dtype=float))
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def correlation_ratio(values: pd.Series, groups: pd.Series) -> float:
    """eta^2: the share of a feature's variance explained by group membership.

    0 means the feature's distribution is identical across groups; 1 means
    knowing the group tells you the feature exactly. Unlike a Pearson
    correlation it needs no ordering on the groups, which is what makes it
    the right statistic against a categorical attribute.
    """
    overall_mean = values.mean()
    ss_between = sum(
        len(values[groups == g]) * (values[groups == g].mean() - overall_mean) ** 2
        for g in groups.unique()
    )
    ss_total = ((values - overall_mean) ** 2).sum()
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation between two vectors, or 0.0 where either is constant.

    A constant column correlates with nothing, and `np.corrcoef` returns NaN
    there with a runtime warning. 0.0 is the honest answer and keeps the
    caller from having to special-case it.
    """
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    if left.std() == 0 or right.std() == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


__all__ = ["average_ranks", "correlation_ratio", "pearson_r", "rank_auc"]
