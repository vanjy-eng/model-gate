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

import logging
import warnings

import numpy as np
import pandas as pd
import pytest

from bdp_model_gate import GateConfig, StructuredGateContext, UncertaintyConfig
from bdp_model_gate.exceptions import GateConfigurationError
from bdp_model_gate.stats import (
    benjamini_hochberg,
    correlation_ratio,
    selection_rate_difference,
)
from bdp_model_gate.structured.fairness import (
    DisparateImpactCheck,
    ProxyCorrelationCheck,
)
from bdp_model_gate.uncertainty import (
    ABOVE,
    BELOW,
    BLOCK,
    MIN_ROWS_FOR_INTERVAL,
    POINT,
    REVIEW,
    UNCERTAIN_FLAG,
    Interval,
    Uncertainty,
    bootstrap,
    canonical_order,
    permutation_pvalue,
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


# --------------------------------------------------------------------------
# Multiple comparisons: the proxy grid
# --------------------------------------------------------------------------


def test_benjamini_hochberg_matches_the_worked_example():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    q = benjamini_hochberg(p)
    assert q[0] == pytest.approx(0.010)
    assert q[1] == pytest.approx(0.040)
    assert q[-1] == pytest.approx(0.216)
    # Monotone, and a q-value is a probability.
    assert np.all(np.diff(q[np.argsort(p)]) >= -1e-12)
    assert np.all(q <= 1.0) and np.all(q >= 0.0)
    # Degenerate inputs pass through rather than raising.
    assert benjamini_hochberg(np.array([0.03])) == pytest.approx([0.03])
    assert benjamini_hochberg(np.array([])).size == 0


def test_the_vectorised_correlation_ratio_agrees_with_the_loop_it_replaced():
    """`correlation_ratio` was a Python loop over `groups.unique()` with
    boolean indexing, at 1.3 ms per call on 5,000 rows — which made the
    permutation test five seconds instead of half of one. bincount is 11x
    faster and has to give the same answer."""

    def loop_implementation(values, groups):
        overall = values.mean()
        between = sum(
            len(values[groups == g]) * (values[groups == g].mean() - overall) ** 2
            for g in groups.unique()
        )
        total = ((values - overall) ** 2).sum()
        return float(between / total) if total > 0 else 0.0

    rng = np.random.default_rng(0)
    for _ in range(40):
        n, k = int(rng.integers(30, 900)), int(rng.integers(2, 6))
        groups = pd.Series(rng.choice([f"g{i}" for i in range(k)], n))
        values = pd.Series(rng.normal(size=n) + groups.map({f"g{i}": i * 0.4 for i in range(k)}))
        assert correlation_ratio(values, groups) == pytest.approx(
            loop_implementation(values, groups), abs=1e-12
        )


def _proxy_grid(seed, n=45, k=8, n_noise=8, with_real_proxy=False):
    """A grid of pure-noise features against a many-level attribute.

    eta-squared is inflated by the group count — its expectation under the
    null is about (k-1)/(n-1) — so at n=45 with 8 zones a noise column crosses
    the 0.30 threshold about 5% of the time. That is where multiple-comparison
    control earns its keep; on a two-level attribute at any reasonable sample
    size, chance crossings are already vanishingly rare.
    """
    rng = np.random.default_rng(seed)
    region = rng.choice([f"zone{i}" for i in range(k)], n)
    X = pd.DataFrame({f"noise_{i}": rng.normal(size=n) for i in range(n_noise)})
    if with_real_proxy:
        X["territory_factor"] = pd.Series(region).map(
            {f"zone{i}": 0.8 + 0.06 * i for i in range(k)}
        ).to_numpy() + rng.normal(0, 0.02, n)
    return StructuredGateContext(
        X=X,
        y_true=np.tile([0, 1], n // 2 + 1)[:n],
        y_pred=np.linspace(0, 1, n),
        protected_df=pd.DataFrame({"region": region}),
        predict_fn=lambda frame: np.zeros(len(frame)),
        task="binary",
    )


def _proxy_check(**uncertainty):
    uncertainty.setdefault("bootstrap_samples", 1000)
    config = GateConfig(uncertainty=UncertaintyConfig(**uncertainty))
    return ProxyCorrelationCheck(config.fairness, config.uncertainty)


@pytest.mark.real_bootstrap
@pytest.mark.parametrize(
    "mode,expected",
    [("review", UNCERTAIN_FLAG), ("block", "PROXY_RISK"), (POINT, "PROXY_RISK")],
)
def test_a_chance_crossing_is_demoted(mode, expected):
    """Seed 4 puts `noise_1` at eta^2 = 0.332 — over the 0.30 threshold and
    pure noise. Benjamini-Hochberg gives it q = 0.256 across 8 comparisons, so
    `review` routes it to a human instead of reporting a proxy that is not
    there. `point` keeps the pre-0.6.0 effect-size-only verdict.
    """
    results = _proxy_check(on_uncertain=mode).run(_proxy_grid(seed=4))
    finding = next(r for r in results if r.metadata.get("feature") == "noise_1")
    assert finding.flag == expected
    assert finding.metadata["q_value"] > finding.metadata["fdr"]
    assert finding.metadata["n_comparisons"] == 8


@pytest.mark.real_bootstrap
def test_a_real_proxy_survives_the_correction():
    """The other half: control that suppressed genuine findings would be
    worse than no control at all."""
    results = _proxy_check().run(_proxy_grid(seed=3, n=600, n_noise=8, with_real_proxy=True))
    findings = [r for r in results if r.flag == "PROXY_RISK"]
    assert [r.metadata["feature"] for r in findings] == ["territory_factor"]
    assert findings[0].metadata["q_value"] <= findings[0].metadata["fdr"]


@pytest.mark.real_bootstrap
def test_too_few_permutations_to_resolve_the_grid_says_so(caplog):
    """The trap this guard exists for.

    The permutation count bounds the smallest p-value obtainable, and BH
    multiplies the strongest cell's by the number of comparisons — so with
    `m` cells the smallest reachable q is about `m / samples`. If that exceeds
    the FDR, **no cell can ever be significant however real the association
    is**, and every genuine proxy would be reported as UNCERTAIN: a
    confidently wrong verdict wearing humility.
    """
    context = _proxy_grid(seed=3, n=600, n_noise=25, with_real_proxy=True)
    with caplog.at_level(logging.WARNING, logger="bdp_model_gate.fairness"):
        results = _proxy_check(bootstrap_samples=200).run(context)

    assert "need at least 520 permutations" in caplog.text
    finding = next(r for r in results if r.metadata.get("feature") == "territory_factor")
    assert finding.flag == "PROXY_RISK"
    assert "no significance test" in finding.detail
    assert "q_value" not in finding.metadata


def test_turning_intervals_off_falls_back_to_effect_size_alone():
    results = _proxy_check(compute_intervals=False).run(_proxy_grid(seed=4, n_noise=8))
    finding = next(r for r in results if r.metadata.get("feature") == "noise_1")
    assert finding.flag == "PROXY_RISK"
    assert "no significance test" in finding.detail


@pytest.mark.real_bootstrap
def test_a_permutation_pvalue_is_never_exactly_zero():
    """`(1 + hits) / (1 + draws)`: "no permutation beat the observed value" is
    evidence bounded by how many were run, not proof."""
    rng = np.random.default_rng(0)
    n = 300
    groups = rng.choice(list("ABC"), n)
    perfect = pd.Series(groups).map({"A": 0.0, "B": 5.0, "C": 10.0}).to_numpy()

    p = permutation_pvalue(
        lambda base, permuted: correlation_ratio(
            pd.Series(perfect[base]), pd.Series(groups[permuted])
        ),
        pd.DataFrame({"v": perfect, "g": groups}),
        samples=200,
    )
    assert p is not None
    assert 0.0 < p <= 1.0 / 201 + 1e-12


@pytest.mark.real_bootstrap
def test_the_proxy_qvalues_do_not_depend_on_row_order():
    straight = _proxy_grid(seed=4, n_noise=8)
    order = np.random.default_rng(1).permutation(len(straight.X))
    shuffled = StructuredGateContext(
        X=straight.X.iloc[order].reset_index(drop=True),
        y_true=np.asarray(straight.y_true)[order],
        y_pred=np.asarray(straight.y_pred)[order],
        protected_df=straight.protected_df.iloc[order].reset_index(drop=True),
        predict_fn=straight.predict_fn,
        task="binary",
    )
    before = _proxy_check().run(straight)
    after = _proxy_check().run(shuffled)
    assert {r.metadata.get("feature"): r.flag for r in before} == {
        r.metadata.get("feature"): r.flag for r in after
    }
    assert [r.metadata.get("q_value") for r in before] == [r.metadata.get("q_value") for r in after]


# --------------------------------------------------------------------------
# The boundaries themselves
#
# Every test below was written against a specific surviving mutant from
# `scripts/mutmut_decision_surface.py`. These functions are full of thresholds
# -- the minimum rows to resample at all, the minimum successful draws to form
# percentiles from, the tie rule in the p-value -- and a threshold that is
# only ever tested from well inside its own range is a threshold nobody has
# checked. Each is exercised *at* the boundary, where off-by-one lives.
# --------------------------------------------------------------------------


def _counting(values):
    """A statistic returning `values[i]` on its i-th call, then NaN.

    `bootstrap` calls the statistic once for the point estimate before the
    resampling loop, so `values[0]` is the point estimate and the rest are
    draws. Returning NaN rather than raising exercises the `isfinite` filter
    specifically.
    """
    state = {"i": 0}

    def statistic(positions):
        i = state["i"]
        state["i"] += 1
        return values[i] if i < len(values) else float("nan")

    return statistic


def test_exactly_min_rows_is_enough_to_resample():
    """`n < min_rows` refuses; `n == min_rows` must not.

    The mutant is `<` to `<=`, which moves the floor by one row and silently
    drops the interval from every check sitting exactly on it. Nothing else in
    the suite calls `bootstrap` at exactly `min_rows`.
    """
    frame = pd.DataFrame({"a": np.arange(float(MIN_ROWS_FOR_INTERVAL))})
    interval = bootstrap(lambda pos: float(pos.mean()), frame, samples=100)
    assert interval is not None, "an interval should be available at exactly min_rows"


def test_the_hard_floor_is_two_rows_not_three():
    """`max(2, min_rows)` -- the 2 is a real floor, not decoration: one row
    cannot be resampled into a distribution. A caller lowering `min_rows`
    below it should still get an interval at two rows."""
    frame = pd.DataFrame({"a": [1.0, 5.0]})
    interval = bootstrap(lambda pos: float(pos.mean()), frame, min_rows=2, samples=100)
    assert interval is not None, "two rows is the documented hard floor"


def test_the_kept_draw_floor_is_inclusive():
    """ "Fewer than 20 draws is not an interval" means *fewer than*.

    At exactly the floor the interval stands. Two mutants live here -- `<` to
    `<=`, and `max(20, ...)` to `max(21, ...)` -- and both turn a valid
    interval into a silent None.
    """
    # 1 point estimate + exactly 20 finite draws, then NaN for the rest.
    interval = bootstrap(
        _counting([0.5] * 21),
        pd.DataFrame({"a": np.arange(200.0)}),
        samples=200,  # floor = max(20, 200 // 10) = 20
    )
    assert interval is not None, "exactly 20 kept draws is enough"
    assert interval.samples == 20


def test_the_kept_draw_floor_scales_with_the_request():
    """The floor is `max(20, samples // 10)`, so asking for more draws demands
    more of them back. At 220 requested the floor is 22, and 21 is not enough
    -- which is what separates `// 10` from `// 11`."""
    interval = bootstrap(
        _counting([0.5] * 22),  # point estimate + 21 draws
        pd.DataFrame({"a": np.arange(200.0)}),
        samples=220,  # floor = max(20, 22) = 22
    )
    assert interval is None, "21 kept draws is below the floor of 22"


def test_a_permutation_that_raises_is_skipped_not_fatal():
    """`continue`, not `break`.

    One degenerate permutation is one lost draw. Under `break` it ends the
    null distribution wherever it happened to occur, and the p-value is then
    either wrong or -- because the draw count falls below the floor -- absent
    entirely. The mutant is invisible to any test whose statistic never fails.
    """
    state = {"i": 0}

    def fails_once(base, permuted):
        state["i"] += 1
        if state["i"] == 3:
            raise ValueError("degenerate permutation")
        return 0.5

    p = permutation_pvalue(fails_once, pd.DataFrame({"a": np.arange(100.0)}), samples=100)
    assert p is not None, "one failed permutation should not end the test"


def test_a_non_finite_permutation_is_skipped_not_fatal():
    """The same rule for the other way a draw can be useless. Separate test
    because they are separate branches, and a mutant in one is invisible to a
    test that only exercises the other."""
    state = {"i": 0}

    def nan_once(base, permuted):
        state["i"] += 1
        return float("nan") if state["i"] == 3 else 0.5

    p = permutation_pvalue(nan_once, pd.DataFrame({"a": np.arange(100.0)}), samples=100)
    assert p is not None, "one non-finite permutation should not end the test"


def test_every_permutation_tying_the_observed_value_gives_a_p_of_one():
    """The exact arithmetic of `(1 + hits) / (1 + drawn)`, pinned.

    A constant statistic makes every permutation tie the observed value, so
    every draw is a hit and the p-value is exactly 1.0 -- the honest answer:
    nothing about the observation is special. It is the one input where the
    value is exact rather than approximate, which is what makes it able to
    catch a miscounted denominator (`drawn += 2` halves it to ~0.5) and a
    flipped tie tolerance (`observed + 1e-12` stops counting ties at all and
    collapses it to ~0.02).
    """
    p = permutation_pvalue(
        lambda base, permuted: 0.5, pd.DataFrame({"a": np.arange(50.0)}), samples=50
    )
    assert p == 1.0


def test_the_documented_defaults_are_the_shipped_ones():
    """`samples=1000` and `random_state=42` are contract, not preference: the
    docs quote them and reproducibility depends on the seed. Every caller in
    the package passes its own, so nothing else would notice them drifting."""
    import inspect

    for fn in (bootstrap, permutation_pvalue):
        defaults = {
            name: parameter.default for name, parameter in inspect.signature(fn).parameters.items()
        }
        assert defaults["samples"] == 1000, fn.__name__
        assert defaults["random_state"] == 42, fn.__name__
        assert defaults["min_rows"] == MIN_ROWS_FOR_INTERVAL, fn.__name__
