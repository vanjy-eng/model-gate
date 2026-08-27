"""Calibration measurement.

A model is **calibrated** when its stated probabilities match observed
frequencies: of the cases it scores 0.7, about 70% should be positive.
Discrimination and calibration are independent properties — a model can rank
perfectly (AUC 1.0) while every probability it emits is twice too high.

That distinction is why this module exists. For credit scoring and insurance
pricing, calibration is often the property that matters: a well-ranked but
badly calibrated model misprices every policy while scoring well on every
metric the gate measured before 0.5.0.

Everything here is numpy-native, so it works on a core install.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._logging import get_logger
from .exceptions import GateConfigurationError

logger = get_logger("calibration")


@dataclass(frozen=True)
class CalibrationCurve:
    """Observed frequency against predicted probability, per bin.

    `count` is included because a bin holding four observations says almost
    nothing, and any renderer or reader needs to weight accordingly.
    """

    bin_edges: np.ndarray
    predicted: np.ndarray  # mean predicted probability within the bin
    observed: np.ndarray  # observed positive rate within the bin
    count: np.ndarray  # rows falling in the bin

    @property
    def populated(self) -> np.ndarray:
        """Mask of bins that actually contain observations."""
        return self.count > 0


def _validate_probabilities(y_prob: np.ndarray) -> np.ndarray:
    values = np.asarray(y_prob, dtype=float)
    if values.ndim != 1:
        raise GateConfigurationError(
            f"calibration needs a 1-D vector of probabilities, got shape {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise GateConfigurationError("calibration inputs contain NaN or infinite values")
    if values.min() < 0.0 or values.max() > 1.0:
        raise GateConfigurationError(
            "calibration needs probabilities in [0, 1]; got values in "
            f"[{values.min():.4g}, {values.max():.4g}]. Pass predicted probabilities "
            "rather than scores or hard labels."
        )
    return values


def calibration_curve(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10, strategy: str = "uniform"
) -> CalibrationCurve:
    """Bins predictions and compares mean prediction to observed frequency.

    `strategy="uniform"` splits [0, 1] into equal-width bins — the classic
    reliability diagram. `strategy="quantile"` splits by equal count, which
    matters for a skewed score distribution where uniform bins leave the
    interesting region nearly empty. Fraud and default scores are usually
    skewed, so quantile is often the honest choice.
    """
    probabilities = _validate_probabilities(y_prob)
    actuals = np.asarray(y_true, dtype=float)
    if actuals.shape != probabilities.shape:
        raise GateConfigurationError(
            f"y_true has shape {actuals.shape} but y_prob has {probabilities.shape}"
        )
    if n_bins < 2:
        raise GateConfigurationError(f"n_bins must be at least 2, got {n_bins}")

    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        edges = np.unique(np.quantile(probabilities, np.linspace(0.0, 1.0, n_bins + 1)))
        if len(edges) < 3:
            logger.debug(
                "quantile binning collapsed to %d edge(s) — the predictions are nearly "
                "constant; falling back to uniform bins",
                len(edges),
            )
            edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise GateConfigurationError(
            f"unknown binning strategy {strategy!r} — use 'uniform' or 'quantile'"
        )

    # np.digitize puts values equal to an interior edge in the upper bin; clip
    # so the final edge (1.0) lands in the last bin rather than one past it.
    index = np.clip(np.digitize(probabilities, edges[1:-1], right=False), 0, len(edges) - 2)
    n_actual_bins = len(edges) - 1

    count = np.bincount(index, minlength=n_actual_bins).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        predicted = np.bincount(index, weights=probabilities, minlength=n_actual_bins) / count
        observed = np.bincount(index, weights=actuals, minlength=n_actual_bins) / count
    predicted = np.nan_to_num(predicted, nan=0.0)
    observed = np.nan_to_num(observed, nan=0.0)

    return CalibrationCurve(bin_edges=edges, predicted=predicted, observed=observed, count=count)


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10, strategy: str = "uniform"
) -> float:
    """Mean gap between predicted and observed frequency, weighted by bin size.

    0.0 is perfect. A value of 0.05 means predictions are off by five
    percentage points on average.

    ECE is a summary and hides shape: two models with the same ECE can be
    miscalibrated in opposite directions, one over-confident only at the top
    and another wrong throughout. Read it alongside the curve, which is why
    `calibration_curve` is public.
    """
    curve = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)
    total = curve.count.sum()
    if total == 0:
        return 0.0
    gaps = np.abs(curve.observed - curve.predicted)
    return float(np.sum(curve.count * gaps) / total)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error of the probabilities. Lower is better; 0.0 perfect."""
    probabilities = _validate_probabilities(y_prob)
    actuals = np.asarray(y_true, dtype=float)
    return float(np.mean((probabilities - actuals) ** 2))


def brier_decomposition(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> dict[str, float]:
    """Murphy's decomposition: Brier = reliability - resolution + uncertainty.

    Worth the extra numbers because they separate two different failures:

    - **reliability** — how far predictions sit from observed frequency.
      Lower is better, and this is the part recalibration can fix.
    - **resolution** — how much predictions vary from the base rate. Higher is
      better; a model predicting the base rate for everyone is perfectly
      reliable and completely useless.
    - **uncertainty** — the base rate's own variance. A property of the
      problem, not the model, and a floor nothing can improve.

    A model with excellent reliability and near-zero resolution has learned
    nothing, and neither the Brier score nor ECE says so on its own.

    The identity holds exactly for the **binned** forecast, so both are
    returned: `binned_brier` is what `reliability - resolution + uncertainty`
    reconstructs, and `brier` is the score on the raw probabilities. They
    differ by the information binning discards, which is small but not zero —
    reporting only the raw score would leave an identity that almost adds up,
    and a number that almost adds up is worse than two that are labelled.
    """
    probabilities = _validate_probabilities(y_prob)
    actuals = np.asarray(y_true, dtype=float)
    curve = calibration_curve(actuals, probabilities, n_bins=n_bins)
    total = curve.count.sum()
    base_rate = float(np.mean(actuals)) if len(actuals) else 0.0

    if total == 0:
        return {
            "brier": 0.0,
            "binned_brier": 0.0,
            "reliability": 0.0,
            "resolution": 0.0,
            "uncertainty": 0.0,
            "base_rate": base_rate,
        }

    weights = curve.count / total
    reliability = float(np.sum(weights * (curve.predicted - curve.observed) ** 2))
    resolution = float(np.sum(weights * (curve.observed - base_rate) ** 2))
    uncertainty = float(base_rate * (1.0 - base_rate))

    return {
        "brier": brier_score(actuals, probabilities),
        "binned_brier": reliability - resolution + uncertainty,
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "base_rate": base_rate,
    }


__all__ = [
    "CalibrationCurve",
    "brier_decomposition",
    "brier_score",
    "calibration_curve",
    "expected_calibration_error",
]
