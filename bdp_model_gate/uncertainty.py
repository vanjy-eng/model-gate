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

import warnings
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from ._logging import get_logger
from .exceptions import GateConfigurationError

logger = get_logger("uncertainty")

#: Which side of the threshold is the *bad* side.
ABOVE = "above"  #: a gap, an error, a disparity — flag when it exceeds
BELOW = "below"  #: a score floor such as `min_score` — flag when it falls short
DIRECTIONS = (ABOVE, BELOW)

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

    def decide(self, threshold: float, flag_when: str) -> str:
        """`"bad"`, `"fine"` or `"uncertain"` for this threshold.

        The boundary handling is the fiddly part, and it is fixed by a rule
        that was already documented elsewhere: **a value exactly equal to the
        threshold passes.** `PerformanceThresholdCheck` has always said a
        score equal to `min_score` clears it, and a gap equal to
        `disparity_threshold` is not a finding.

        So the passing side is inclusive and the failing side is strict, and a
        zero-width interval sitting exactly on the threshold reads `"fine"`
        rather than `"uncertain"`. A unanimous sample is not an uncertain one
        — reporting doubt there would be as wrong as reporting a fabricated
        certainty anywhere else.
        """
        if flag_when == BELOW:
            if self.low >= threshold:
                return "fine"
            if self.high < threshold:
                return "bad"
        else:
            if self.high <= threshold:
                return "fine"
            if self.low > threshold:
                return "bad"
        return "uncertain"

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

    Numeric columns are **rank-transformed** before hashing, and that is not a
    detail. Hashing raw floats makes the ordering sensitive to arithmetic that
    ought not to matter: `exposure` and `exposure * 12` are the same book, and
    `x / mean(x)` against `12x / mean(12x)` differs in the last bits, so the
    digests differ, so the resample sequence differs, so a verdict could move
    on a change of *units*. Dense ranks are exactly invariant under any
    positive monotone rescaling, which is the property the library's
    scale-free claims already promise elsewhere.

    Ties — genuinely duplicate rows — are harmless: interchangeable rows
    cannot change the statistic whichever one a resample draws.
    """
    ranked = frame.copy()
    for column in ranked.columns:
        if ranked[column].dtype.kind in "iuf":
            ranked[column] = ranked[column].rank(method="dense")
    digests = pd.util.hash_pandas_object(ranked, index=False).to_numpy(dtype="uint64")
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
            # Warnings are errors *inside a draw*, and that is a correctness
            # rule rather than tidiness. A resample can lose the information a
            # statistic needs — a protected group, a whole class — and the
            # libraries do not agree on how to say so.
            # `roc_auc_score` returns NaN, which `isfinite` catches.
            # `average_precision_score` warns and returns **0.0**, which is
            # finite, plausible, and completely fabricated: left in, it drags
            # the lower bound of every interval to zero and manufactures
            # uncertainty that is not in the data.
            #
            # A warning is the only signal both cases share, so it is treated
            # as what it is: this draw is not evidence. If every draw warns
            # there is no interval, and the check says so rather than
            # inventing one.
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                value = float(statistic(resample))
        except Exception:
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


@dataclass(frozen=True)
class Verdict:
    """What a check should report, once the interval has been read."""

    flag: str
    blocking: bool
    note: str
    metadata: dict[str, Any]

    @property
    def uncertain(self) -> bool:
        return self.flag == UNCERTAIN_FLAG


class Uncertainty:
    """Config plus the two operations a check needs, bound together.

    Every check that compares a number to a threshold needs the same three
    lines — build an interval, read it against the threshold, put the result in
    the detail string and the metadata. Nine copies of that is nine chances to
    get the three-way reading subtly different, so it lives here and each check
    holds one of these.

    Validation happens at construction, so a typo'd `on_uncertain` fails while
    the suite is being built rather than partway through a run.
    """

    def __init__(self, config: Any = None):
        from .config import UncertaintyConfig

        self.config = config or UncertaintyConfig()
        if self.config.on_uncertain not in ON_UNCERTAIN:
            raise GateConfigurationError(
                f"uncertainty.on_uncertain={self.config.on_uncertain!r} — must be one "
                f"of {', '.join(ON_UNCERTAIN)}"
            )

    def interval(self, statistic: Callable[[np.ndarray], float], frame: pd.DataFrame):
        """The bootstrap interval, or None when intervals are off or the data
        cannot support one."""
        if not self.config.compute_intervals:
            return None
        return bootstrap(
            statistic,
            frame,
            samples=self.config.bootstrap_samples,
            level=self.config.confidence_level,
            random_state=self.config.random_state,
            min_rows=self.config.min_rows_for_interval,
        )

    def verdict(
        self,
        interval: Interval | None,
        threshold: float,
        *,
        point: float,
        risk_flag: str,
        blocking: bool = True,
        flag_when: str = ABOVE,
    ) -> Verdict:
        """Read the interval against the threshold, and package the answer."""
        flag, is_blocking, note = resolve(
            interval,
            threshold,
            point=point,
            risk_flag=risk_flag,
            on_uncertain=self.config.on_uncertain,
            blocking=blocking,
            flag_when=flag_when,
        )
        metadata: dict[str, Any] = {"on_uncertain": self.config.on_uncertain}
        if interval is not None:
            metadata.update(interval.as_metadata())
        return Verdict(flag=flag, blocking=is_blocking, note=note, metadata=metadata)

    @staticmethod
    def measured(interval: Interval | None, point: float) -> str:
        """The number as it appears in a detail string, with its interval when
        there is one and an explicit note when there is not."""
        return interval.describe() if interval is not None else f"{point:.3f} (no interval)"


def resolve(
    interval: Interval | None,
    threshold: float,
    *,
    point: float,
    risk_flag: str,
    on_uncertain: str = REVIEW,
    blocking: bool = True,
    flag_when: str = ABOVE,
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
    if flag_when not in DIRECTIONS:
        raise GateConfigurationError(
            f"flag_when must be one of {', '.join(DIRECTIONS)} — got {flag_when!r}"
        )
    breached = (point > threshold) if flag_when == ABOVE else (point < threshold)

    if interval is None:
        return (risk_flag if breached else "OK"), blocking, ""

    decision = interval.decide(threshold, flag_when)
    bad_side, good_side = ("above", "below") if flag_when == ABOVE else ("below", "above")

    if decision == "bad":
        return (
            risk_flag,
            blocking,
            f" — the whole {interval.level:.0%} interval sits {bad_side} {threshold:.3f}",
        )

    if decision == "fine":
        return "OK", blocking, f" — the whole interval sits {good_side} {threshold:.3f}"

    # Straddling. Which *direction* the doubt runs in matters to the reader:
    # "looks clean but could breach" and "looks bad but might not" are
    # different conversations, and a single "straddles the threshold" sentence
    # collapses them. What it means for the pipeline is the caller's policy,
    # not this function's.
    reach = interval.high if flag_when == ABOVE else interval.low
    if not breached:
        straddle_note = (
            f" — the estimate stays the right side of {threshold:.3f} but the interval "
            f"reaches {reach:.3f}, so this sample cannot rule out a breach"
        )
    else:
        straddle_note = (
            f" — the estimate breaches {threshold:.3f} but the interval reaches back to "
            f"{(interval.low if flag_when == ABOVE else interval.high):.3f}, so this "
            "sample cannot confirm a breach"
        )

    if on_uncertain == BLOCK:
        return risk_flag, blocking, straddle_note + " (on_uncertain='block')"
    if on_uncertain == POINT:
        return (
            (risk_flag if breached else "OK"),
            blocking,
            straddle_note + " (accepted: on_uncertain='point')",
        )
    return UNCERTAIN_FLAG, False, straddle_note


__all__ = [
    "ABOVE",
    "BELOW",
    "BLOCK",
    "DIRECTIONS",
    "MIN_ROWS_FOR_INTERVAL",
    "ON_UNCERTAIN",
    "UNCERTAIN_FLAG",
    "POINT",
    "REVIEW",
    "Interval",
    "Uncertainty",
    "Verdict",
    "bootstrap",
    "canonical_order",
    "resolve",
]
