"""Sampling error, and what a gate should do when it cannot tell.

## The problem this exists for

Every check in this library compares a **point estimate to a fixed threshold
with no notion of sampling error**. `FairnessConfig.min_group_size = 30` is a
crude stand-in for that missing notion, and it is not enough: for a proportion
near 0.5, n=30 carries a standard error of about 0.09, so the *difference* of
two such proportions carries one near 0.13. Against a `disparity_threshold` of
0.10, the verdict is noise.

A gate that flips on resampling teaches people to route around it, which
protects nobody.

## What replaces it

An interval, and a three-way reading of where it sits:

    entirely above the threshold   the finding is real       -> flag
    straddling the threshold       *you do not know*         -> NEEDS_REVIEW
    entirely below                 clean                     -> OK

The middle row is the point. "The disparity might be 0.06 and might be 0.14"
is a governance conversation, not a build failure — and this library already
has the machinery to say so, because a non-blocking result routes to a human
instead of stopping a deploy.

`UncertaintyConfig.on_uncertain` lets a team overrule that. `"point"` decides
exactly as before, on the point estimate, while **still reporting the
interval** — accepting a risk and not being told about it are different
things, and only the first is a decision.

## Why the resampling is content-addressed

`tests/test_invariants.py::test_row_order_does_not_change_the_verdict` asserts
that permuting the validation set cannot move a verdict, and it is one of the
tests that found a real bug. A textbook bootstrap breaks it: `rng.integers`
under a fixed seed picks the same *positions* every run, so re-sorting the
input hands the statistic a different resample and, near a threshold, a
different verdict.

So the rows are put into a canonical order first, derived from their own
contents, exactly as `bdp_model_gate._sampling.stable_sample` does. Sorting a
CSV must not change whether a model ships.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from ._logging import get_logger

logger = get_logger("uncertainty")

#: What to do when the interval straddles the threshold.
REVIEW = "review"  #: route to a human — the default
BLOCK = "block"  #: precautionary: if it might breach, stop
POINT = "point"  #: decide on the point estimate, but still report the interval
ON_UNCERTAIN = (REVIEW, BLOCK, POINT)

#: The flag a straddling interval produces. One name across the whole suite
#: rather than a per-check variant: `CheckResult.check_name` already says which
#: check it came from, and a reviewer scanning a report should be able to learn
#: one flag that means "the data cannot decide this".
UNCERTAIN_FLAG = "UNCERTAIN"

#: Below this many rows a bootstrap interval is theatre rather than evidence:
#: resampling 12 rows tells you about those 12 rows. The check reports the
#: point estimate and says the interval was not computed.
MIN_ROWS_FOR_INTERVAL = 30


@dataclass(frozen=True)
class Interval:
    """A point estimate and the range the data can support.

    Attributes:
        point: The statistic as computed on the data itself — always the
            number the check would have reported before intervals existed.
        low: Lower percentile of the bootstrap distribution.
        high: Upper percentile.
        level: Confidence level the percentiles were taken at.
        samples: Bootstrap resamples that produced a finite statistic. Fewer
            than requested means some resamples were degenerate — a group
            missing from a resample, say — and the caller should know how much
            evidence is behind the range.
    """

    point: float
    low: float
    high: float
    level: float
    samples: int

    @property
    def width(self) -> float:
        return self.high - self.low

    def straddles(self, threshold: float) -> bool:
        """Whether the threshold falls inside the interval — the case where
        the data cannot say which side of it the truth is on."""
        return self.low <= threshold <= self.high

    def exceeds(self, threshold: float) -> bool:
        """Whether the *whole* interval sits above the threshold."""
        return self.low > threshold

    def as_metadata(self) -> dict[str, Any]:
        return {
            "point": round(float(self.point), 4),
            "ci_low": round(float(self.low), 4),
            "ci_high": round(float(self.high), 4),
            "ci_level": self.level,
            "bootstrap_samples": self.samples,
        }

    def describe(self) -> str:
        """The interval as it appears in a detail string."""
        return f"{self.point:.3f} [{self.low:.3f}, {self.high:.3f}] at {self.level:.0%}"


def canonical_order(frame: pd.DataFrame) -> np.ndarray:
    """Row positions sorted by row *content*.

    The whole reason this module can promise order-invariance. Two frames
    holding the same rows in different orders produce resamples over the same
    canonical sequence, so the bootstrap distribution is identical.

    Ties — genuinely duplicate rows — are harmless: they are interchangeable
    by definition, so which of them a resample draws cannot change the
    statistic.
    """
    digests = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype="uint64")
    return np.argsort(digests, kind="stable")


def bootstrap(
    statistic: Callable[[np.ndarray], float],
    frame: pd.DataFrame,
    *,
    samples: int = 1000,
    level: float = 0.95,
    random_state: int = 42,
    min_rows: int = MIN_ROWS_FOR_INTERVAL,
) -> Interval | None:
    """A percentile bootstrap interval for `statistic`, or None.

    `statistic` receives an array of **row positions into `frame`** and
    returns one number. Passing positions rather than a resampled frame is
    what keeps this affordable: a caller can index into arrays it already
    holds, and nothing re-runs a check a thousand times.

    Returns None when there is not enough data to resample (`min_rows`), when
    the statistic cannot be computed on the data itself, or when too few
    resamples produced a finite value to form percentiles from. **None means
    "no interval", never "an interval of zero width"** — the caller reports
    the point estimate and says the interval was unavailable, rather than
    presenting a fabricated certainty.

    Deterministic, and invariant to the order of `frame`'s rows: resampling
    happens over `canonical_order(frame)`.
    """
    n = len(frame)
    if n < max(2, int(min_rows)):
        logger.debug("bootstrap skipped: %d row(s) is below min_rows=%d", n, min_rows)
        return None

    positions = canonical_order(frame)
    try:
        point = float(statistic(positions))
    except Exception as exc:
        logger.debug("bootstrap skipped: the statistic raised on the full data (%r)", exc)
        return None
    if not np.isfinite(point):
        return None

    rng = np.random.default_rng(random_state)
    draws = []
    for _ in range(max(1, int(samples))):
        resample = positions[rng.integers(0, n, size=n)]
        try:
            value = float(statistic(resample))
        except Exception:
            # A resample can be degenerate — a protected group missing
            # entirely, a constant column. Skip it and count what remains,
            # so the caller can see how much evidence the range rests on.
            continue
        if np.isfinite(value):
            draws.append(value)

    # Percentiles from a handful of draws are not an interval. The floor is
    # deliberately generous: it is better to report no interval than one built
    # on twenty resamples that happened to succeed.
    if len(draws) < max(20, int(samples) // 10):
        logger.debug(
            "bootstrap skipped: only %d of %d resample(s) produced a finite statistic",
            len(draws),
            samples,
        )
        return None

    tail = (1.0 - level) / 2.0
    low, high = np.percentile(draws, [100 * tail, 100 * (1 - tail)])
    return Interval(
        point=point,
        low=float(low),
        high=float(high),
        level=level,
        samples=len(draws),
    )


def resolve(
    interval: Interval | None,
    threshold: float,
    *,
    point: float,
    risk_flag: str,
    on_uncertain: str = REVIEW,
    blocking: bool = True,
) -> tuple[str, bool, str]:
    """Turn an interval and a threshold into `(flag, blocking, note)`.

    The one place the three-way reading lives, so no check can implement it
    slightly differently:

    - no interval, or `on_uncertain="point"` -> compare the point estimate,
      exactly as before intervals existed
    - the interval sits entirely above the threshold -> `risk_flag`, blocking
      as the check declares
    - it straddles the threshold -> `on_uncertain` decides
    - entirely below -> OK

    `note` is the sentence appended to the detail string. It is never empty
    when an interval exists, because the reader has to be able to tell a
    verdict backed by evidence from one the data cannot support.
    """
    if interval is None:
        flag = risk_flag if point > threshold else "OK"
        return flag, blocking, ""

    if interval.exceeds(threshold):
        return (
            risk_flag,
            blocking,
            f" — the whole {interval.level:.0%} interval sits above {threshold:.3f}",
        )

    if not interval.straddles(threshold):
        return "OK", blocking, f" — the whole interval sits below {threshold:.3f}"

    # Straddling. Which *direction* the doubt runs in matters to the reader:
    # "looks clean but could breach" and "looks bad but might not" are
    # different conversations, and a single "straddles the threshold" sentence
    # collapses them. What it means for the pipeline is the caller's policy,
    # not this function's.
    if point <= threshold:
        straddle_note = (
            f" — the estimate is under {threshold:.3f} but the interval reaches "
            f"{interval.high:.3f}, so this sample cannot rule out a breach"
        )
    else:
        straddle_note = (
            f" — the estimate is over {threshold:.3f} but the interval reaches down "
            f"to {interval.low:.3f}, so this sample cannot confirm a breach"
        )
    if on_uncertain == BLOCK:
        return risk_flag, blocking, straddle_note + " (on_uncertain='block')"
    if on_uncertain == POINT:
        flag = risk_flag if point > threshold else "OK"
        return flag, blocking, straddle_note + " (accepted: on_uncertain='point')"
    return UNCERTAIN_FLAG, False, straddle_note


__all__ = [
    "BLOCK",
    "MIN_ROWS_FOR_INTERVAL",
    "ON_UNCERTAIN",
    "UNCERTAIN_FLAG",
    "POINT",
    "REVIEW",
    "Interval",
    "bootstrap",
    "canonical_order",
    "resolve",
]
