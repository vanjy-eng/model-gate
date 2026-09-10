"""Sampling error: the interval, the three postures, and the one property.

The property, first, because the rest of the design rests on it: **an interval
must not depend on the order of the rows.** `test_invariants.py` already
asserts that permuting the validation set cannot move a verdict, and a
textbook bootstrap breaks it — `rng.integers` under a fixed seed draws the same
*positions*, so a re-sorted frame gets a different resample and, near a
threshold, a different answer. `uncertainty.canonical_order` is what prevents
that, and `test_the_interval_is_invariant_to_row_order` is what keeps it
prevented.

Everything else here is about the three-way reading of where an interval sits,
and about the escape hatch: a team that has looked at the uncertainty and
accepted it must be able to say so without switching the check off.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from bdp_model_gate import GateConfig, StructuredGateContext, UncertaintyConfig
from bdp_model_gate.exceptions import GateConfigurationError
from bdp_model_gate.stats import selection_rate_difference
from bdp_model_gate.structured.fairness import DisparateImpactCheck
from bdp_model_gate.uncertainty import (
    ABOVE,
    BELOW,
    BLOCK,
    POINT,
    REVIEW,
    UNCERTAIN_FLAG,
    Interval,
    Uncertainty,
    bootstrap,
    canonical_order,
    resolve,
)

# Fewer resamples than the 1000 default: these tests assert properties, not
# percentile precision, and 300 keeps the file under two seconds.
SAMPLES = 300


def _book(n, gap, seed=0):
    """`n` rows with a target selection-rate gap of `gap` between two groups."""
    rng = np.random.default_rng(seed)
    groups = np.where(np.arange(n) % 2 == 0, "A", "B")
    rate = np.where(groups == "A", 0.5 + gap / 2, 0.5 - gap / 2)
    return groups, (rng.random(n) < rate).astype(int), (rng.random(n) < 0.5).astype(int)


def _context(groups, y_pred, y_true):
    return StructuredGateContext(
        X=pd.DataFrame({"x": np.arange(len(y_pred), dtype=float)}),
        y_true=y_true,
        y_pred=y_pred,
        protected_df=pd.DataFrame({"gender": groups}),
        predict_fn=lambda frame: np.zeros(len(frame)),
        task="binary",
    )


def _check(**uncertainty):
    """A parity check with the interval settings under test.

    `importorskip` sits here rather than at module scope on purpose. The
    interval machinery in `bdp_model_gate.uncertainty` and the statistic in
    `bdp_model_gate.stats` are numpy-only and must keep working on a **core
    install**, so the tests that exercise them directly still run there.
    `DisparateImpactCheck` reports NOT_APPLICABLE without fairlearn, so only
    the tests that go through the check skip.
    """
    pytest.importorskip("fairlearn", reason="disparate_impact needs the [structured] extra")
    uncertainty.setdefault("bootstrap_samples", SAMPLES)
    config = GateConfig(uncertainty=UncertaintyConfig(**uncertainty))
    return DisparateImpactCheck(config.fairness, config.uncertainty)


# --------------------------------------------------------------------------
# The statistic
# --------------------------------------------------------------------------


def test_the_numpy_parity_difference_agrees_with_fairlearn():
    """The bootstrap cannot call fairlearn — 3.9 ms a call against 22 µs makes
    a thousand resamples eight seconds per check. So the statistic is
    numpy-native, and it has to give the same answer as the library everyone
    else compares against. The same guarantee `rank_auc` carries against
    `sklearn.metrics.roc_auc_score`.
    """
    fairlearn = pytest.importorskip("fairlearn.metrics")
    rng = np.random.default_rng(3)
    for n, n_groups in ((200, 2), (500, 3), (1000, 4)):
        groups = rng.choice([f"g{i}" for i in range(n_groups)], n)
        y_pred = (rng.random(n) < 0.4).astype(int)
        y_true = (rng.random(n) < 0.5).astype(int)
        assert selection_rate_difference(y_pred, groups) == pytest.approx(
            fairlearn.demographic_parity_difference(y_true, y_pred, sensitive_features=groups)
        ), (n, n_groups)


def test_a_resample_that_loses_a_group_is_uninformative_not_an_error():
    """Bootstrapping can draw a resample containing one group. There is no
    spread to measure, which is different from a failure."""
    assert selection_rate_difference(np.array([1, 0, 1]), np.array(["A", "A", "A"])) == 0.0
    assert selection_rate_difference(np.array([]), np.array([])) == 0.0


# --------------------------------------------------------------------------
# The property everything else rests on
# --------------------------------------------------------------------------


def test_canonical_order_is_the_same_sequence_whatever_the_row_order():
    rng = np.random.default_rng(11)
    frame = pd.DataFrame({"a": rng.integers(0, 5, 200), "b": rng.random(200)})
    shuffled = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)

    straight = frame.iloc[canonical_order(frame)].reset_index(drop=True)
    permuted = shuffled.iloc[canonical_order(shuffled)].reset_index(drop=True)
    pd.testing.assert_frame_equal(straight, permuted)


def test_the_interval_is_invariant_to_row_order():
    """The property the design rests on, asserted end to end through the check.

    A bootstrap that resampled positions would give a different interval for a
    re-sorted CSV — and near a threshold, a different verdict. Sorting a file
    must not change whether a model ships.
    """
    groups, y_pred, y_true = _book(800, 0.10, seed=5)
    order = np.random.default_rng(2).permutation(len(y_pred))

    straight = _check().run(_context(groups, y_pred, y_true))[0]
    shuffled = _check().run(_context(groups[order], y_pred[order], y_true[order]))[0]

    assert straight.flag == shuffled.flag
    assert straight.metadata["ci_low"] == pytest.approx(shuffled.metadata["ci_low"])
    assert straight.metadata["ci_high"] == pytest.approx(shuffled.metadata["ci_high"])


def test_the_same_data_gives_the_same_interval_twice():
    """A gate that moves between runs on identical input teaches people to
    re-run it until it passes."""
    groups, y_pred, y_true = _book(600, 0.12, seed=8)
    first = _check().run(_context(groups, y_pred, y_true))[0]
    second = _check().run(_context(groups, y_pred, y_true))[0]
    assert first.metadata == second.metadata
    assert first.detail == second.detail


# --------------------------------------------------------------------------
# The interval behaves like an interval
# --------------------------------------------------------------------------


def test_more_rows_narrow_the_interval():
    """The statistical claim the whole feature makes. If this does not hold,
    nothing downstream of it means anything."""
    widths = []
    for n in (400, 1600, 6400):
        groups, y_pred, y_true = _book(n, 0.10, seed=4)
        result = _check().run(_context(groups, y_pred, y_true))[0]
        widths.append(result.metadata["ci_high"] - result.metadata["ci_low"])
    assert widths[0] > widths[1] > widths[2], widths


def test_the_interval_brackets_the_point_estimate_it_reports():
    groups, y_pred, y_true = _book(1000, 0.20, seed=6)
    m = _check().run(_context(groups, y_pred, y_true))[0].metadata
    assert m["ci_low"] <= m["point"] <= m["ci_high"]
    assert m["point"] == pytest.approx(m["demographic_parity_diff"], abs=1e-3)


def test_the_point_estimate_is_untouched_by_the_interval_machinery():
    """Whatever the interval says, the number the check reported before 0.6.0
    is still the number it reports."""
    groups, y_pred, y_true = _book(900, 0.25, seed=9)
    context = _context(groups, y_pred, y_true)
    with_interval = _check().run(context)[0]
    without = _check(compute_intervals=False).run(context)[0]
    assert with_interval.metadata["demographic_parity_diff"] == pytest.approx(
        without.metadata["demographic_parity_diff"]
    )


# --------------------------------------------------------------------------
# Where the interval sits decides the verdict
# --------------------------------------------------------------------------


def test_an_interval_entirely_above_the_threshold_is_a_finding():
    groups, y_pred, y_true = _book(4000, 0.30, seed=1)
    result = _check().run(_context(groups, y_pred, y_true))[0]
    assert result.flag == "DISPARITY_RISK"
    assert result.metadata["ci_low"] > result.metadata["threshold"]
    assert "whole 95% interval sits above" in result.detail


def test_an_interval_entirely_below_the_threshold_is_clean():
    groups, y_pred, y_true = _book(4000, 0.02, seed=1)
    result = _check().run(_context(groups, y_pred, y_true))[0]
    assert result.flag == "OK"
    assert "whole interval sits below" in result.detail


@pytest.mark.parametrize(
    "mode,expected_flag,blocking",
    [
        (REVIEW, UNCERTAIN_FLAG, False),
        (BLOCK, "DISPARITY_RISK", False),
        (POINT, "OK", False),
    ],
)
def test_the_three_postures_on_identical_straddling_data(mode, expected_flag, blocking):
    """The configurability, and it is not cosmetic: the same data yields three
    different, defensible verdicts.

    600 rows with a true 0.10 gap. The point estimate lands under the
    threshold by sampling luck, and the interval reaches over it.

    - `review` says the sample cannot decide -> a governance conversation
    - `block` takes the precautionary reading -> a finding
    - `point` decides as releases before 0.6.0 did -> clean

    `blocking` is False in all three because fairness checks are non-blocking
    throughout this library by design. `on_uncertain="block"` makes the
    *finding fire*; it does not promote fairness to a build-breaker. On a
    blocking check the same machinery does stop the pipeline.
    """
    groups, y_pred, y_true = _book(600, 0.10, seed=7)
    result = _check(on_uncertain=mode).run(_context(groups, y_pred, y_true))[0]
    assert result.flag == expected_flag
    assert result.blocking is blocking
    assert result.metadata["on_uncertain"] == mode


def test_accepting_the_risk_does_not_mean_hiding_it():
    """`on_uncertain="point"` is "we have seen the uncertainty and decided".
    It is not "do not tell me" — the interval still reaches the detail string
    and the metadata, so the governance session can still have the argument.
    """
    groups, y_pred, y_true = _book(600, 0.10, seed=7)
    result = _check(on_uncertain=POINT).run(_context(groups, y_pred, y_true))[0]

    assert result.flag == "OK"
    assert "ci_low" in result.metadata and "ci_high" in result.metadata
    assert "cannot rule out a breach" in result.detail
    assert "accepted: on_uncertain='point'" in result.detail


def test_the_note_says_which_direction_the_doubt_runs():
    """ "Looks clean but could breach" and "looks bad but might not" are
    different conversations; one sentence about straddling collapses them."""
    under = _check().run(_context(*_book(600, 0.10, seed=7)))[0]
    assert "cannot rule out a breach" in under.detail

    over = resolve(
        Interval(point=0.14, low=0.06, high=0.22, level=0.95, samples=300),
        0.10,
        point=0.14,
        risk_flag="DISPARITY_RISK",
    )
    assert "cannot confirm a breach" in over[2]


# --------------------------------------------------------------------------
# When there is no interval to be had
# --------------------------------------------------------------------------


def test_intervals_can_be_turned_off_entirely():
    """Bootstrapping is not free, and a team that does not want to spend the
    time should not have to disable the check to avoid it."""
    groups, y_pred, y_true = _book(600, 0.10, seed=7)
    result = _check(compute_intervals=False).run(_context(groups, y_pred, y_true))[0]
    assert result.flag == "OK"  # the pre-0.6.0 point-estimate verdict
    assert "no interval" in result.detail
    assert "ci_low" not in result.metadata


def test_too_few_rows_means_no_interval_rather_than_a_fabricated_one():
    """Resampling twelve rows tells you about those twelve rows."""
    groups, y_pred, y_true = _book(20, 0.30, seed=7)
    result = _check(min_rows_for_interval=30).run(_context(groups, y_pred, y_true))[0]
    assert "no interval" in result.detail
    assert "ci_low" not in result.metadata


def test_bootstrap_returns_none_rather_than_a_zero_width_interval():
    """None means "no interval". An interval of zero width would be a
    fabricated certainty, which is the failure this library exists to avoid.
    """
    frame = pd.DataFrame({"a": np.arange(200.0)})

    def always_raises(positions):
        raise ValueError("no")

    assert bootstrap(always_raises, frame, samples=50) is None
    assert bootstrap(lambda positions: float("nan"), frame, samples=50) is None
    # Too few rows.
    assert bootstrap(lambda positions: 1.0, frame.head(5), samples=50) is None


def test_a_statistic_that_mostly_fails_produces_no_interval():
    """Percentiles from a handful of surviving draws are not an interval."""
    frame = pd.DataFrame({"a": np.arange(200.0)})
    calls = {"n": 0}

    def flaky(positions):
        calls["n"] += 1
        if calls["n"] % 25:
            raise ValueError("degenerate resample")
        return 0.5

    assert bootstrap(flaky, frame, samples=100) is None


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def test_an_unknown_posture_fails_while_the_suite_is_being_built():
    """At construction, not partway through a run — the same treatment a
    typo'd metric name gets."""
    with pytest.raises(GateConfigurationError, match="on_uncertain"):
        DisparateImpactCheck(uncertainty=UncertaintyConfig(on_uncertain="ignore"))
    # ...and the helper refuses on its own too, so a check that forgets to
    # route through it cannot skip the validation.
    with pytest.raises(GateConfigurationError, match="on_uncertain"):
        Uncertainty(UncertaintyConfig(on_uncertain="whatever"))


def test_the_default_suite_wires_the_uncertainty_config_through():
    from bdp_model_gate.structured import default_structured_checks

    config = GateConfig(uncertainty=UncertaintyConfig(on_uncertain=BLOCK))
    parity = next(
        c
        for c in default_structured_checks(config, include_plugins=False)
        if c.name == "disparate_impact"
    )
    assert parity.uncertainty.config.on_uncertain == BLOCK


# --------------------------------------------------------------------------
# Interval arithmetic
# --------------------------------------------------------------------------


def test_the_boundary_follows_the_rule_the_checks_already_documented():
    """A value exactly equal to the threshold passes, so the passing side is
    inclusive and the failing side is strict.

    `PerformanceThresholdCheck` has always said a score equal to `min_score`
    clears it, and a gap equal to `disparity_threshold` is not a finding. The
    interval has to agree, or an off-by-one silently changes every borderline
    verdict.
    """
    interval = Interval(point=0.08, low=0.05, high=0.15, level=0.95, samples=300)

    # A ceiling. Flag when the metric exceeds it.
    assert interval.decide(0.15, ABOVE) == "fine"  # 0.15 exactly is not a breach
    assert interval.decide(0.16, ABOVE) == "fine"
    assert interval.decide(0.10, ABOVE) == "uncertain"
    assert interval.decide(0.05, ABOVE) == "uncertain"  # low == threshold, not yet strict
    assert interval.decide(0.04, ABOVE) == "bad"

    # A floor. Flag when the metric falls short.
    assert interval.decide(0.05, BELOW) == "fine"  # 0.05 exactly clears the floor
    assert interval.decide(0.04, BELOW) == "fine"
    assert interval.decide(0.10, BELOW) == "uncertain"
    assert interval.decide(0.16, BELOW) == "bad"

    assert interval.width == pytest.approx(0.10)


def test_a_unanimous_sample_is_not_an_uncertain_one():
    """The case that caught the original boundary rule: a perfect model against
    `min_score=1.0` gives a zero-width interval sitting exactly on the
    threshold. Reporting doubt there would be as wrong as reporting a
    fabricated certainty anywhere else."""
    exact = Interval(point=1.0, low=1.0, high=1.0, level=0.95, samples=300)
    assert exact.decide(1.0, BELOW) == "fine"
    assert exact.decide(1.0, ABOVE) == "fine"
    assert resolve(exact, 1.0, point=1.0, risk_flag="X", flag_when=BELOW)[0] == "OK"


def test_resolve_falls_back_to_the_point_estimate_without_an_interval():
    for point, expected in ((0.2, "DISPARITY_RISK"), (0.05, "OK")):
        flag, blocking, note = resolve(
            None, 0.10, point=point, risk_flag="DISPARITY_RISK", blocking=True
        )
        assert (flag, blocking, note) == (expected, True, "")


# --------------------------------------------------------------------------
# The other direction: a floor, not a ceiling
# --------------------------------------------------------------------------


def test_a_score_floor_reads_the_interval_the_other_way_round():
    """`min_score` is a floor, so the *bad* side is below it. An 0.81 AUC
    against `min_score=0.80` on a small sample is exactly as much noise as a
    0.11 disparity against a 0.10 ceiling, and the reading has to mirror.
    """
    interval = Interval(point=0.81, low=0.74, high=0.88, level=0.95, samples=300)

    flag, _, note = resolve(
        interval, 0.80, point=0.81, risk_flag="PERFORMANCE_RISK", flag_when=BELOW
    )
    assert flag == UNCERTAIN_FLAG
    assert "cannot rule out a breach" in note

    # Whole interval clear of the floor -> clean.
    clear = Interval(point=0.91, low=0.88, high=0.94, level=0.95, samples=300)
    assert resolve(clear, 0.80, point=0.91, risk_flag="X", flag_when=BELOW)[0] == "OK"
    assert "sits above 0.800" in resolve(clear, 0.80, point=0.91, risk_flag="X", flag_when=BELOW)[2]

    # Whole interval under the floor -> a finding, and it blocks.
    short = Interval(point=0.60, low=0.55, high=0.66, level=0.95, samples=300)
    flag, blocking, note = resolve(
        short, 0.80, point=0.60, risk_flag="PERFORMANCE_RISK", flag_when=BELOW, blocking=True
    )
    assert (flag, blocking) == ("PERFORMANCE_RISK", True)
    assert "sits below 0.800" in note


def test_the_same_interval_reads_opposite_ways_on_the_two_directions():
    """The one property that would catch a copy-paste of the wrong direction
    into a check: a floor and a ceiling cannot agree about the same numbers."""
    interval = Interval(point=0.5, low=0.4, high=0.6, level=0.95, samples=300)
    above = resolve(interval, 0.30, point=0.5, risk_flag="X", flag_when=ABOVE)
    below = resolve(interval, 0.30, point=0.5, risk_flag="X", flag_when=BELOW)
    assert above[0] == "X" and below[0] == "OK"


def test_an_unknown_direction_is_refused():
    with pytest.raises(GateConfigurationError, match="flag_when"):
        resolve(None, 0.1, point=0.2, risk_flag="X", flag_when="sideways")


# --------------------------------------------------------------------------
# A degenerate resample is not evidence
# --------------------------------------------------------------------------


def test_a_warning_inside_a_draw_discards_it():
    """The correctness rule that makes warnings-as-errors non-negotiable here.

    A resample can lose the information a statistic needs, and the libraries do
    not agree on how to say so. `roc_auc_score` returns NaN, which `isfinite`
    catches. `average_precision_score` **warns and returns 0.0** — finite,
    plausible and completely fabricated.

    Left in, those zeros drag the lower bound of the interval to zero and
    manufacture uncertainty that is not in the data. Measured on 3 positives
    in 400 rows, where roughly 5% of resamples contain no positive at all:
    the interval is `[0.000, 1.000]` if the draws are kept and
    `[1.000, 1.000]` if they are discarded.
    """
    pytest.importorskip("sklearn")
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(4)
    n = 400
    y = np.zeros(n, dtype=int)
    y[rng.choice(n, 3, replace=False)] = 1
    p = np.clip(y * 0.6 + rng.random(n) * 0.4, 0.001, 0.999)

    interval = bootstrap(
        lambda positions: float(average_precision_score(y[positions], p[positions])),
        pd.DataFrame({"y": y, "p": p}),
        samples=600,
    )
    assert interval is not None
    assert interval.low == pytest.approx(1.0), "a fabricated 0.0 reached the interval"
    assert interval.samples < 600, "the degenerate draws should have been discarded"


def test_the_kept_draw_count_is_reported():
    """How much evidence the range rests on is part of the finding, not an
    implementation detail — a range built on 571 draws of a requested 600 is
    telling you something about the data."""
    frame = pd.DataFrame({"a": np.arange(200.0)})
    calls = {"n": 0}

    def warns_sometimes(positions):
        calls["n"] += 1
        if calls["n"] % 3 == 0:
            warnings.warn("degenerate", UserWarning, stacklevel=1)
        return 0.5

    interval = bootstrap(warns_sometimes, frame, samples=300)
    assert interval is not None
    assert 150 < interval.samples < 300
    assert interval.as_metadata()["bootstrap_samples"] == interval.samples


def test_a_statistic_that_always_warns_yields_no_interval():
    """Safe degradation: the check reports its point estimate and says the
    interval was unavailable, rather than inventing one from nothing."""
    frame = pd.DataFrame({"a": np.arange(200.0)})

    def always_warns(positions):
        warnings.warn("degenerate", UserWarning, stacklevel=1)
        return 0.5

    assert bootstrap(always_warns, frame, samples=100) is None
