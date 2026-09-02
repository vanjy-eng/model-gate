"""Exposure weighting, and the measures a pricing actuary actually reads.

Three ideas live here, all numpy-only so they work on a **core install**.

**Exposure.** A general-purpose regression suite treats every row as one
observation. An insurance book does not work that way: a policy written for
one month and a policy written for twelve months are not equal evidence about
a claims *rate*, and an unweighted RMSE says they are. `context.exposure` is
the per-row weight that fixes it, and every function here accepts it.

The convention, stated once because getting it wrong silently changes the
answer: **`y_true` and `y_pred` must be on the same basis as each other, and
`exposure` is how much weight the row's observation deserves.** Supply it
when the target is a *rate* (claims per vehicle-year, loss cost per
sum-insured-year); leave it out when the target is a per-policy *total*,
where the exposure is already baked into the value. Under either convention
the actual-over-expected ratio below is `sum(w*actual) / sum(w*expected)`,
which is why one formula serves both.

**Actual against expected, by band.** The standard pricing validation, and
more informative than an RMSE: "wrong by 25,000 naira" does not say whether
the book is under-priced overall or only in its top decile, and those need
different remedies.

**The Lorenz Gini.** Whether the model *orders* risk correctly, which is a
separate question from whether its level is right. A model can be perfectly
calibrated on average and rank no better than a coin flip.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

import numpy as np
import pandas as pd

from ._logging import get_logger
from ._sampling import stable_sample
from .exceptions import GateConfigurationError

logger = get_logger("actuarial")

#: Guard for ratio denominators.
_EPSILON = 1e-12

INCREASING = "increasing"
DECREASING = "decreasing"
DIRECTIONS = (INCREASING, DECREASING)


def exposure_array(context: Any) -> np.ndarray | None:
    """`context.exposure` as a float array, or None when it was not supplied.

    Shape and sign are checked eagerly by `core.validation`, so this only
    casts. Kept as one function rather than five inline `getattr` calls so
    that every check reads exposure the same way.
    """
    exposure = getattr(context, "exposure", None)
    if exposure is None:
        return None
    return np.asarray(exposure, dtype=float)


def weights_or_ones(exposure: np.ndarray | None, n: int) -> np.ndarray:
    """Exposure weights, or a vector of ones of the right length.

    Every weighted statistic below reduces to its unweighted form on ones,
    so a book with no exposure column takes exactly the same code path. That
    is deliberate: two paths would eventually disagree.
    """
    if exposure is None:
        return np.ones(n, dtype=float)
    return np.asarray(exposure, dtype=float)


def weighted_mean(values: Any, weights: Any = None) -> float:
    """Exposure-weighted mean, or NaN when the weights sum to nothing.

    NaN rather than 0.0: a segment carrying no exposure has no mean, and
    reporting zero would put a fabricated number in a governance report.
    """
    v = np.asarray(values, dtype=float)
    w = weights_or_ones(None if weights is None else np.asarray(weights, dtype=float), v.size)
    total = float(w.sum())
    if total <= 0:
        return float("nan")
    return float(np.dot(v, w) / total)


def weighted_quantile(values: Any, quantiles: Any, weights: Any = None) -> np.ndarray:
    """Quantiles of `values` where each row counts for `weights`.

    Used to cut prediction bands holding roughly equal *exposure* rather
    than equal row counts. On a motor book the two differ substantially:
    equal-count deciles put most of the year's risk in whichever decile
    happens to hold the annual policies.
    """
    v = np.asarray(values, dtype=float)
    q = np.asarray(quantiles, dtype=float)
    if weights is None:
        return np.quantile(v, q)

    w = np.asarray(weights, dtype=float)
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]
    total = float(w.sum())
    if total <= 0:
        return np.quantile(v, q)
    # Mid-point of each row's weight interval, which is the weighted
    # generalisation of numpy's default 'linear' quantile.
    positions = (np.cumsum(w) - 0.5 * w) / total
    return np.interp(q, positions, v)


def band_edges(values: Any, n_bands: int, weights: Any = None) -> np.ndarray:
    """Unique band boundaries over `values`, by weighted quantile.

    Duplicates are collapsed, so a prediction that is constant over half the
    book yields fewer bands rather than several empty ones. The caller
    checks the length: fewer than three edges means there are not two bands
    to compare.
    """
    quantiles = np.linspace(0.0, 1.0, int(n_bands) + 1)
    return np.unique(weighted_quantile(values, quantiles, weights))


def assign_bands(values: Any, edges: np.ndarray) -> np.ndarray:
    """Band index (0..len(edges)-2) for every row."""
    return np.clip(np.digitize(np.asarray(values, dtype=float), edges[1:-1]), 0, len(edges) - 2)


def actual_over_expected(actual: Any, expected: Any, weights: Any = None) -> float:
    """`sum(w*actual) / sum(w*expected)` — the A/E ratio.

    A ratio of totals, not a mean of per-row ratios. Those are different
    numbers, and the totals version is the one a pricing report means: it
    answers "did the book collect what it needed to?", where the per-row
    mean is dominated by small policies.

    Returns NaN when the expected total is zero, since there is nothing to
    be a ratio of.
    """
    a = np.asarray(actual, dtype=float)
    e = np.asarray(expected, dtype=float)
    w = weights_or_ones(None if weights is None else np.asarray(weights, dtype=float), a.size)
    expected_total = float(np.dot(e, w))
    if abs(expected_total) <= _EPSILON:
        return float("nan")
    return float(np.dot(a, w) / expected_total)


def lorenz_gini(y_true: Any, y_pred: Any, exposure: Any = None) -> float:
    """The exposure-weighted Lorenz Gini index: does the model *order* risk?

    Policies are sorted from cheapest to dearest prediction; the curve plots
    the cumulative share of **exposure** against the cumulative share of
    realised **loss**. A model with signal loads the losses into its
    dearest policies, so the curve sags below the diagonal, and the Gini is
    twice the area between them:

        0.0   the ordering carries nothing — cheap and dear policies cost
              the same per unit of exposure
        > 0   the usual case; higher is better discrimination
        < 0   the ordering is *inverted*, which is a finding rather than a
              bad score, and is invisible to any error metric

    The ceiling is not 1.0 and depends on the book: no rating structure can
    predict which individual policy has the accident. Compare against
    `lorenz_gini(y_true, y_true, exposure)`, which sorts by the realised
    outcome and so gives the highest attainable value for this data — the
    same function, called with the actuals as the score, which is why there
    is no second implementation to disagree with this one.

    Ties in `y_pred` are aggregated into a single point on the curve, so the
    result cannot depend on the order rows happened to arrive in. That is
    the same order-invariance `average_ranks` and `stable_sample` exist to
    provide elsewhere.

    Raises GateConfigurationError for inputs the index is not defined on: a
    negative realised outcome (the shares would not be shares) or a total
    outcome of zero (no loss to concentrate).
    """
    actual = np.asarray(y_true, dtype=float)
    score = np.asarray(y_pred, dtype=float)
    weights = weights_or_ones(
        None if exposure is None else np.asarray(exposure, dtype=float), actual.size
    )
    if actual.size != score.size:
        raise GateConfigurationError(
            f"lorenz_gini needs y_true and y_pred of equal length, got {actual.size} and "
            f"{score.size}"
        )
    if np.any(actual < 0):
        raise GateConfigurationError(
            "lorenz_gini needs a non-negative y_true — the Lorenz curve accumulates "
            "shares of a total, and a negative outcome makes those shares meaningless. "
            "Use 'r2' or 'mae' for a target that can go below zero."
        )

    shares_x, shares_y = lorenz_curve(actual, score, weights)
    area = float(np.sum(np.diff(shares_x) * (shares_y[1:] + shares_y[:-1]) / 2.0))
    return 1.0 - 2.0 * area


def lorenz_curve(y_true: Any, y_pred: Any, exposure: Any = None) -> tuple[np.ndarray, np.ndarray]:
    """The curve `lorenz_gini` measures the area of: `(exposure share, loss share)`.

    Separate from the index because the shape carries what the scalar cannot.
    A Gini of 0.30 earned by isolating one dreadful decile and a Gini of 0.30
    spread evenly across the book are different rating structures with
    different remedies, and the curve shows which you have. `plot()` draws
    this, so the chart and the number come from one computation.

    Both arrays start at `(0, 0)` and end at `(1, 1)`, with one interior
    point per distinct predicted value.
    """
    actual = np.asarray(y_true, dtype=float)
    score = np.asarray(y_pred, dtype=float)
    weights = weights_or_ones(
        None if exposure is None else np.asarray(exposure, dtype=float), actual.size
    )

    usable = np.isfinite(actual) & np.isfinite(score) & (weights > 0)
    if usable.sum() < 2:
        raise GateConfigurationError(
            "the Lorenz curve needs at least two rows with a positive exposure and "
            "finite values; the ordering of one policy says nothing"
        )
    actual, score, weights = actual[usable], score[usable], weights[usable]

    order = np.argsort(score, kind="stable")
    score, weights = score[order], weights[order]
    losses = actual[order] * weights

    total_weight = float(weights.sum())
    total_loss = float(losses.sum())
    if abs(total_loss) <= _EPSILON:
        raise GateConfigurationError(
            "the Lorenz curve needs a non-zero total realised outcome — with no loss to "
            "concentrate there is no concentration to measure"
        )

    # One point per distinct predicted value, not per row: within a block of
    # equal predictions the curve is a straight line however the rows are
    # permuted, and aggregating makes that exact rather than approximate.
    block_ends = np.flatnonzero(np.concatenate((score[1:] != score[:-1], [True])))
    shares_x = np.concatenate(([0.0], np.cumsum(weights)[block_ends] / total_weight))
    shares_y = np.concatenate(([0.0], np.cumsum(losses)[block_ends] / total_loss))
    return shares_x, shares_y


def partial_dependence(
    predict: Callable[[pd.DataFrame], Any],
    X: pd.DataFrame,
    feature: str,
    grid: Sequence[float] | np.ndarray,
    max_rows: int = 200,
    random_state: int = 42,
) -> np.ndarray:
    """Mean prediction as `feature` is swept across `grid`, everything else held.

    The empirical answer to "what does this model do with this rating
    factor?" — every other column keeps its real joint distribution, so the
    curve is not an extrapolation into feature space nobody occupies.

    Scored on a `stable_sample` of at most `max_rows` rows: this costs
    `len(grid) * max_rows` predictions, and a governance gate is not the
    place to spend a million of them. Content-addressed sampling means
    sorting the input cannot change the curve.
    """
    sample = stable_sample(X, max_rows, random_state)
    logger.debug(
        "partial dependence on %r: %d grid point(s) x %d row(s) = %d predictions",
        feature,
        len(grid),
        len(sample),
        len(grid) * len(sample),
    )
    means = []
    for value in grid:
        counterfactual = sample.copy()
        counterfactual[feature] = value
        means.append(float(np.mean(np.asarray(predict(counterfactual), dtype=float))))
    return np.asarray(means, dtype=float)


def monotonicity_breaks(
    curve: Any, direction: str, tolerance: float = 0.02
) -> list[tuple[int, float]]:
    """Steps of `curve` that move against `direction`, worst first.

    Each entry is `(index, drop)`, where `index` is the left end of the
    offending step and `drop` is how far the curve moved the wrong way as a
    fraction of the curve's own range. Expressing it relatively is what lets
    one tolerance work for a premium in naira and a probability alike; a
    fixed absolute tolerance would flag every naira model and no probability
    model.

    A flat curve has no range, so nothing can break it: an empty list.
    """
    values = np.asarray(curve, dtype=float)
    if direction not in DIRECTIONS:
        raise GateConfigurationError(
            f"monotonicity direction must be one of {', '.join(DIRECTIONS)} — got {direction!r}"
        )
    if values.size < 2:
        return []
    span = float(np.nanmax(values) - np.nanmin(values))
    if span <= _EPSILON:
        return []

    steps = np.diff(values)
    wrong_way = -steps if direction == INCREASING else steps
    breaks = [
        (int(i), float(drop / span)) for i, drop in enumerate(wrong_way) if drop / span > tolerance
    ]
    return sorted(breaks, key=lambda pair: -pair[1])


def relative_change(new: Any, old: Any) -> tuple[np.ndarray, np.ndarray]:
    """`(new - old) / old` per row, and the mask of rows it is defined on.

    Rows whose baseline is zero or negative are excluded rather than given a
    fabricated percentage: a premium moving from 0 to 500 is not an increase
    of any percent, and dividing by it would put an infinity in the report.
    """
    numerator = np.asarray(new, dtype=float)
    denominator = np.asarray(old, dtype=float)
    defined = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0)
    change = np.full(numerator.shape, np.nan)
    change[defined] = (numerator[defined] - denominator[defined]) / denominator[defined]
    return change, defined


__all__ = [
    "DECREASING",
    "DIRECTIONS",
    "INCREASING",
    "actual_over_expected",
    "assign_bands",
    "band_edges",
    "exposure_array",
    "lorenz_curve",
    "lorenz_gini",
    "monotonicity_breaks",
    "partial_dependence",
    "relative_change",
    "weighted_mean",
    "weighted_quantile",
    "weights_or_ones",
]
