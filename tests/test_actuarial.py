"""Exposure weighting and the actuarial measures, scored against known answers.

Every number asserted here is derivable on paper from the fixture above it.
That is the whole point: the failures this library has shipped were all
plausible numbers that were the wrong number, and a test asserting only that
a check *ran* passes on every one of them.

The file is organised by what kind of wrongness each block catches:

    the statistics       hand-computable values for the primitives
    the four checks      each verdict, each threshold, each skip path
    exposure weighting   that supplying it changes the answer, in the
                         direction and by the amount it should
    invariants           permuting rows, rescaling exposure, rescaling the
                         prediction — the verdict must stay put

`lorenz_gini` gets the most attention because it is the one measure here a
reader cannot sanity-check by eye, and because its sign carries a finding:
negative means the rating structure orders risk *backwards*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bdp_model_gate import ActuarialConfig, ModelGate, StructuredGateContext
from bdp_model_gate.actuarial import (
    actual_over_expected,
    band_edges,
    lorenz_curve,
    lorenz_gini,
    monotonicity_breaks,
    partial_dependence,
    relative_change,
    weighted_mean,
    weighted_quantile,
)
from bdp_model_gate.exceptions import GateConfigurationError, GateValidationError
from bdp_model_gate.metrics import resolve_metric
from bdp_model_gate.structured.actuarial_checks import (
    ActualVsExpectedCheck,
    DislocationCheck,
    MonotonicityCheck,
    RiskDiscriminationCheck,
)
from bdp_model_gate.structured.performance import PerformanceThresholdCheck
from bdp_model_gate.structured.regression_fairness import GroupMeanGapCheck

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _book(y_pred, y_true, *, n=None, exposure=None, groups=None, baseline=None, X=None, **kwargs):
    """A minimal regression context. `task` is always explicit: a two-valued
    target is indistinguishable from a binary one by shape, and half these
    fixtures have exactly two distinct values."""
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_pred) if n is None else n
    return StructuredGateContext(
        X=X if X is not None else pd.DataFrame({"x": np.arange(n, dtype=float)}),
        y_true=np.asarray(y_true, dtype=float),
        y_pred=y_pred,
        exposure=exposure,
        baseline_pred=baseline,
        protected_df=None if groups is None else pd.DataFrame({"g": groups}),
        predict_fn=lambda frame: np.zeros(len(frame)),
        task="regression",
        **kwargs,
    )


@pytest.fixture
def two_band_book():
    """A book whose overall A/E is exactly 1.00 and whose every band is wrong.

    100 policies priced at 100 realise 80 (A/E 0.80); 100 priced at 200
    realise 220 (A/E 1.10). Totals: 30,000 expected against 30,000 actual.
    The cheap half subsidises the dear half, the rate level looks perfect,
    and every individual price is wrong — which is the failure the band
    curve exists to expose and a rate-level check cannot see.
    """
    y_pred = np.concatenate([np.full(100, 100.0), np.full(100, 200.0)])
    y_true = np.concatenate([np.full(100, 80.0), np.full(100, 220.0)])
    return y_pred, y_true


# --------------------------------------------------------------------------
# The statistics
# --------------------------------------------------------------------------


def test_weighted_mean_is_the_hand_computed_average():
    # (1*1 + 3*3) / (1 + 3) = 10/4
    assert weighted_mean([1.0, 3.0], [1.0, 3.0]) == pytest.approx(2.5)
    assert weighted_mean([1.0, 3.0]) == pytest.approx(2.0)


def test_weighted_mean_of_no_exposure_is_not_a_number():
    """NaN, not 0.0. A segment carrying no exposure has no mean, and a
    fabricated zero would read in the report as a finding."""
    assert np.isnan(weighted_mean([1.0, 3.0], [0.0, 0.0]))


def test_equal_weights_reproduce_the_unweighted_quantiles():
    """The weighted path is the only path — `weights_or_ones` sends a book
    with no exposure column through it too, so it must agree with numpy on
    equal weights or every existing verdict moves."""
    values = np.linspace(0, 100, 501)
    quantiles = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    weighted = weighted_quantile(values, quantiles, np.ones(len(values)))
    assert weighted == pytest.approx(np.quantile(values, quantiles), abs=0.25)


def test_exposure_moves_the_quantile_towards_the_heavy_rows():
    """Half the rows, three quarters of the exposure: the median must sit in
    the heavy half. Equal-count deciles are the wrong cut for a motor book."""
    values = np.concatenate([np.zeros(100), np.ones(100)])
    heavy_on_ones = np.concatenate([np.full(100, 1.0), np.full(100, 3.0)])
    assert weighted_quantile(values, [0.5], heavy_on_ones)[0] == pytest.approx(1.0)
    assert weighted_quantile(values, [0.5], np.ones(200))[0] == pytest.approx(0.5, abs=0.01)


def test_actual_over_expected_is_a_ratio_of_totals_not_a_mean_of_ratios():
    # (10*1 + 20*3) / (20*1 + 20*3) = 70/80. The mean of the per-row ratios
    # would be (0.5 + 1.0*3)/4 = 0.875 here by coincidence, so use inputs
    # where the two genuinely differ.
    assert actual_over_expected([10.0, 20.0], [20.0, 20.0], [1.0, 3.0]) == pytest.approx(0.875)
    # totals 30 over 12 = 2.5; the mean of the ratios (10/2 + 20/10)/2 = 3.5
    assert actual_over_expected([10.0, 20.0], [2.0, 10.0]) == pytest.approx(2.5)


def test_actual_over_expected_without_an_expected_total_is_not_a_number():
    assert np.isnan(actual_over_expected([1.0, 2.0], [0.0, 0.0]))


def test_a_constant_prediction_has_a_gini_of_exactly_zero():
    """Every policy tied, so the whole book is one point on the curve and the
    curve is the diagonal. A model that charges the average premium to
    everyone is perfectly calibrated and orders nothing."""
    y_true = np.array([0.0, 1.0, 5.0, 20.0])
    assert lorenz_gini(y_true, np.full(4, 7.0)) == pytest.approx(0.0)


def test_a_correct_ordering_and_its_inverse_are_equal_and_opposite():
    """Four equal-exposure policies, two of which cost 1 and two 0.

    Ordered correctly the curve is (0,0) (.25,0) (.5,0) (.75,.5) (1,1), whose
    area is 0.25, so the Gini is 1 - 2(0.25) = +0.5. Reverse the score and
    the area is 0.75 and the Gini is -0.5. The sign is the finding: a
    negative Gini is what a flipped sign on a rating factor produces, and no
    error metric shows it.
    """
    score = np.array([1.0, 2.0, 3.0, 4.0])
    assert lorenz_gini([0.0, 0.0, 1.0, 1.0], score) == pytest.approx(0.5)
    assert lorenz_gini([1.0, 1.0, 0.0, 0.0], score) == pytest.approx(-0.5)


def test_the_gini_ceiling_is_the_same_function_scored_on_the_outcome():
    """`lorenz_gini(y, y)` is the highest value this book allows, and the
    check reports the model against it. One implementation, so the model's
    Gini and the ceiling cannot be computed two different ways."""
    y_true = np.array([0.0, 0.0, 1.0, 1.0])
    perfect = lorenz_gini(y_true, np.array([1.0, 2.0, 3.0, 4.0]))
    assert perfect == pytest.approx(lorenz_gini(y_true, y_true))


def test_the_gini_depends_only_on_the_ordering_of_the_prediction():
    """Any strictly increasing transform of the score leaves it untouched —
    it is a concentration measure, not an error measure. A book priced in
    naira and the same book priced in thousands must not disagree."""
    rng = np.random.default_rng(4)
    y_true = rng.gamma(2.0, 500.0, 400)
    score = rng.uniform(1, 100, 400)
    base = lorenz_gini(y_true, score)
    assert lorenz_gini(y_true, score * 1000.0) == pytest.approx(base)
    assert lorenz_gini(y_true, score + 5000.0) == pytest.approx(base)
    assert lorenz_gini(y_true, np.log(score)) == pytest.approx(base)


def test_exposure_changes_the_gini():
    """Not a formality: the whole reason the index is exposure-weighted is
    that a one-month policy is not a year's evidence about a rate."""
    y_true = np.array([0.0, 0.0, 1.0, 1.0])
    score = np.array([1.0, 2.0, 3.0, 4.0])
    unweighted = lorenz_gini(y_true, score)
    weighted = lorenz_gini(y_true, score, exposure=np.array([1.0, 1.0, 1.0, 9.0]))
    assert weighted != pytest.approx(unweighted)


def test_the_gini_refuses_inputs_it_is_not_defined_on():
    with pytest.raises(GateConfigurationError, match="non-negative y_true"):
        lorenz_gini([-1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    with pytest.raises(GateConfigurationError, match="non-zero total"):
        lorenz_gini([0.0, 0.0, 0.0], [1.0, 2.0, 3.0])
    with pytest.raises(GateConfigurationError, match="at least two rows"):
        lorenz_gini([1.0], [1.0])
    with pytest.raises(GateConfigurationError, match="equal length"):
        lorenz_gini([1.0, 2.0], [1.0])


def test_the_lorenz_curve_starts_and_ends_on_the_corners():
    x, y = lorenz_curve([0.0, 0.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0])
    assert (x[0], y[0]) == (0.0, 0.0)
    assert x[-1] == pytest.approx(1.0)
    assert y[-1] == pytest.approx(1.0)


def test_monotonicity_breaks_are_relative_to_the_curves_own_range():
    """A dip of 1 on a curve spanning 2 is a 50% break; the same dip on a
    curve spanning 2,000 is noise. One tolerance has to serve a premium in
    naira and a probability alike, so the measure is relative."""
    assert monotonicity_breaks([1.0, 2.0, 3.0], "increasing") == []
    breaks = monotonicity_breaks([1.0, 3.0, 2.0], "increasing")
    assert breaks == [(1, pytest.approx(0.5))]
    assert monotonicity_breaks([3.0, 2.0, 1.0], "decreasing") == []
    assert monotonicity_breaks([3.0, 2.0, 1.0], "increasing")[0][0] == 0


def test_a_flat_curve_cannot_break_monotonicity():
    """No range, so nothing moves against anything. Dividing by the range
    would otherwise produce an infinity for a model that ignores the factor."""
    assert monotonicity_breaks([5.0, 5.0, 5.0], "increasing") == []


def test_monotonicity_rejects_a_direction_it_does_not_know():
    with pytest.raises(GateConfigurationError, match="must be one of"):
        monotonicity_breaks([1.0, 2.0], "up")


def test_partial_dependence_is_the_hand_computed_marginal():
    """`2*a + b` swept over `a`: each point must be `2*value + mean(b)`,
    exactly, because every other column keeps its own values."""
    frame = pd.DataFrame({"a": np.arange(50.0), "b": np.arange(50.0) * 3.0})
    grid = np.array([0.0, 10.0, 20.0])
    curve = partial_dependence(
        lambda df: 2 * df["a"].to_numpy() + df["b"].to_numpy(), frame, "a", grid, max_rows=50
    )
    assert curve == pytest.approx(2 * grid + frame["b"].mean())


def test_relative_change_excludes_a_baseline_of_zero():
    change, defined = relative_change([110.0, 90.0, 100.0], [100.0, 100.0, 0.0])
    assert change[:2] == pytest.approx([0.1, -0.1])
    assert list(defined) == [True, True, False]
    assert np.isnan(change[2])


def test_band_edges_collapse_rather_than_inventing_empty_bands():
    """A prediction constant over half the book must yield fewer bands, not
    several bands holding nothing."""
    values = np.concatenate([np.zeros(100), np.ones(100)])
    assert len(band_edges(values, 10)) < 11


# --------------------------------------------------------------------------
# ActualVsExpectedCheck
# --------------------------------------------------------------------------


def test_the_rate_level_is_the_ratio_of_the_totals():
    """Priced at 100, realised 110: A/E is exactly 1.100, the book is
    under-priced by 10%, and that is past the 5% level tolerance."""
    level = ActualVsExpectedCheck().run(_book(np.full(60, 100.0), np.full(60, 110.0)))[0]
    assert level.metadata["ae"] == pytest.approx(1.1)
    assert level.metadata["deviation"] == pytest.approx(0.1)
    assert level.flag == "AE_LEVEL_RISK"
    assert "under-priced" in level.detail


def test_a_perfect_rate_level_hides_two_broken_bands(two_band_book):
    """The finding this check exists for. Overall A/E is exactly 1.000 —
    every rate-level test passes — while the cheap band runs at 0.80 and the
    dear band at 1.10."""
    y_pred, y_true = two_band_book
    level, bands = ActualVsExpectedCheck().run(_book(y_pred, y_true))

    assert level.flag == "OK"
    assert level.metadata["ae"] == pytest.approx(1.0)

    assert bands.flag == "AE_BAND_RISK"
    assert bands.metadata["worst_band_ae"] == pytest.approx(0.8)
    assert bands.metadata["worst_deviation"] == pytest.approx(0.2)
    assert [b["ae"] for b in bands.metadata["bands"]] == pytest.approx([0.8, 1.1])


@pytest.fixture
def thin_top_band():
    """180 policies priced right, and a top band of 20 priced at half what it
    costs. Ten bands over 200 rows cut at 100 and 200, so the top band holds
    exactly 20 rows — right on the default floor."""
    y_pred = np.concatenate([np.full(180, 100.0), np.full(20, 200.0)])
    y_true = np.concatenate([np.full(180, 100.0), np.full(20, 400.0)])
    return y_pred, y_true


def test_a_band_on_the_floor_is_scored(thin_top_band):
    y_pred, y_true = thin_top_band
    _, bands = ActualVsExpectedCheck(ActuarialConfig(min_band_rows=20)).run(_book(y_pred, y_true))

    assert [b["n_rows"] for b in bands.metadata["bands"]] == [180, 20]
    assert bands.metadata["worst_band_ae"] == pytest.approx(2.0)
    assert bands.flag == "AE_BAND_RISK"


def test_a_band_below_the_floor_is_reported_but_does_not_drive_the_verdict(thin_top_band):
    """The same treatment `min_group_size` gives a three-policy segment. The
    A/E is still printed — a reader should see it — but a ratio the check does
    not trust cannot block a deploy."""
    y_pred, y_true = thin_top_band
    _, bands = ActualVsExpectedCheck(ActuarialConfig(min_band_rows=25)).run(_book(y_pred, y_true))

    thin = bands.metadata["bands"][-1]
    assert thin["n_rows"] == 20
    assert thin["scored"] is False
    assert thin["ae"] == pytest.approx(2.0)  # reported, so a reader can see it
    assert bands.metadata["worst_band_ae"] == pytest.approx(1.0)  # but not scored
    assert bands.flag == "OK"
    assert "1 band(s) not scored" in bands.detail


def test_a_book_with_no_band_big_enough_says_so(thin_top_band):
    y_pred, y_true = thin_top_band
    _, bands = ActualVsExpectedCheck(ActuarialConfig(min_band_rows=500)).run(_book(y_pred, y_true))
    assert bands.flag == "NOT_APPLICABLE"
    assert "min_band_rows=500" in bands.detail
    assert len(bands.metadata["bands"]) == 2  # still reported, just not scored


def test_a_constant_prediction_reports_the_level_only():
    """One distinct predicted value cuts into no bands. Reporting the level
    alone beats inventing a band structure that is not there."""
    results = ActualVsExpectedCheck().run(_book(np.full(60, 100.0), np.linspace(50, 150, 60)))
    assert len(results) == 1
    assert results[0].metadata["measure"] == "level"


def test_exposure_reweights_the_rate_level(two_band_book):
    """The cheap half at a quarter of a year, the dear half at a full year.
    A/E becomes 24,000/22,500 = 1.0667 — past the tolerance the unweighted
    figure of exactly 1.000 sailed through."""
    y_pred, y_true = two_band_book
    exposure = np.concatenate([np.full(100, 0.25), np.full(100, 1.0)])
    level = ActualVsExpectedCheck().run(_book(y_pred, y_true, exposure=exposure))[0]

    assert level.metadata["ae"] == pytest.approx(24_000 / 22_500, abs=1e-4)
    assert level.flag == "AE_LEVEL_RISK"
    assert level.metadata["exposure_weighted"] is True
    assert "exposure-weighted" in level.detail


# --------------------------------------------------------------------------
# RiskDiscriminationCheck
# --------------------------------------------------------------------------


def test_a_model_that_orders_nothing_is_flagged_however_well_calibrated():
    """Charging every policy the book mean gives an A/E of exactly 1.00 and a
    Gini of exactly 0. Calibration and discrimination are independent, and
    only one of them is measured by an error metric."""
    y_true = np.tile([0.0, 40.0], 100)
    context = _book(np.full(200, 20.0), y_true)

    level = ActualVsExpectedCheck().run(context)[0]
    gini = RiskDiscriminationCheck().run(context)[0]

    assert level.flag == "OK" and level.metadata["ae"] == pytest.approx(1.0)
    assert gini.flag == "DISCRIMINATION_RISK"
    assert gini.metadata["gini"] == pytest.approx(0.0)
    assert "no information" in gini.detail


def test_an_inverted_ordering_says_so_rather_than_scoring_low():
    y_true = np.array([1.0, 1.0, 0.0, 0.0])
    result = RiskDiscriminationCheck().run(_book([1.0, 2.0, 3.0, 4.0], y_true))[0]
    assert result.metadata["gini"] == pytest.approx(-0.5)
    assert result.flag == "DISCRIMINATION_RISK"
    assert "inverted" in result.detail


def test_a_good_ordering_is_reported_against_the_ceiling_this_book_allows():
    """ "0.28" is not a judgement anyone can make; "0.28 of a possible 0.52"
    is. The share of the attainable maximum is the number a reviewer reads."""
    rng = np.random.default_rng(12)
    n = 400
    risk = rng.uniform(0.2, 3.0, n)
    y_true = rng.gamma(2.0, risk * 300.0)
    result = RiskDiscriminationCheck().run(_book(risk * 600.0, y_true))[0]

    assert result.flag == "OK"
    assert 0.0 < result.metadata["gini"] < result.metadata["gini_ceiling"]
    assert result.metadata["gini_share_of_ceiling"] == pytest.approx(
        result.metadata["gini"] / result.metadata["gini_ceiling"], abs=1e-3
    )
    assert "of the" in result.detail


def test_a_gini_floor_can_be_raised_above_zero():
    config = ActuarialConfig(min_gini=0.9)
    result = RiskDiscriminationCheck(config).run(_book([1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 1.0, 1.0]))[
        0
    ]
    assert result.metadata["gini"] == pytest.approx(0.5)
    assert result.flag == "DISCRIMINATION_RISK"


# --------------------------------------------------------------------------
# MonotonicityCheck
# --------------------------------------------------------------------------


def _rating_context(coefficient: float, n: int = 120, **kwargs):
    """A premium linear in `prior_claims`, with the sign under test."""
    rng = np.random.default_rng(5)
    X = pd.DataFrame(
        {
            "prior_claims": rng.integers(0, 6, n).astype(float),
            "vehicle_age": rng.integers(0, 20, n).astype(float),
        }
    )

    def predict(frame):
        return 20_000.0 + coefficient * frame["prior_claims"].to_numpy()

    premium = predict(X)
    return StructuredGateContext(
        X=X,
        y_true=premium * 0.9,
        y_pred=premium,
        predict_fn=predict,
        task="regression",
        **kwargs,
    )


def test_a_premium_that_falls_with_prior_claims_is_a_compliance_finding():
    check = MonotonicityCheck(ActuarialConfig(monotonic_features={"prior_claims": "increasing"}))
    result = check.run(_rating_context(-500.0))[0]

    assert result.flag == "MONOTONICITY_RISK"
    assert result.category == "compliance"
    assert result.blocking is True
    assert result.metadata["feature"] == "prior_claims"
    # Every step of a strictly decreasing line breaks an increasing claim.
    assert result.metadata["n_breaks"] == len(result.metadata["grid"]) - 1
    assert "declared increasing" in result.detail


def test_a_premium_that_rises_with_prior_claims_passes():
    check = MonotonicityCheck(ActuarialConfig(monotonic_features={"prior_claims": "increasing"}))
    result = check.run(_rating_context(500.0))[0]
    assert result.flag == "OK"
    assert result.metadata["n_breaks"] == 0
    curve = result.metadata["partial_dependence"]
    assert curve == sorted(curve)


def test_the_partial_dependence_curve_is_the_declared_relationship():
    """The stored curve is not decoration — the plot draws it and the verdict
    came from it, so it has to be the model's actual marginal effect."""
    check = MonotonicityCheck(ActuarialConfig(monotonic_features={"prior_claims": "increasing"}))
    result = check.run(_rating_context(500.0))[0]
    grid = np.asarray(result.metadata["grid"])
    curve = np.asarray(result.metadata["partial_dependence"])
    assert curve == pytest.approx(20_000.0 + 500.0 * grid)


def test_a_misspelled_rating_factor_blocks_rather_than_skipping():
    """The failure mode this library exists to prevent. A typo must not
    produce a green gate on a regulatory constraint nobody checked."""
    check = MonotonicityCheck(ActuarialConfig(monotonic_features={"prior_claim": "increasing"}))
    result = check.run(_rating_context(-500.0))[0]

    assert result.flag == "MONOTONICITY_UNCHECKABLE"
    assert result.blocking is True
    assert "not a column of X" in result.detail
    assert "prior_claims" in result.detail  # names the near miss


def test_a_categorical_rating_factor_cannot_carry_a_direction():
    context = _rating_context(500.0)
    context.X["cover_type"] = ["comprehensive", "third_party"] * (len(context.X) // 2)
    check = MonotonicityCheck(ActuarialConfig(monotonic_features={"cover_type": "increasing"}))
    result = check.run(context)[0]

    assert result.flag == "MONOTONICITY_UNCHECKABLE"
    assert "not numeric" in result.detail


def test_a_constant_factor_is_vacuous_rather_than_satisfied():
    context = _rating_context(500.0)
    context.X["prior_claims"] = 1.0
    check = MonotonicityCheck(ActuarialConfig(monotonic_features={"prior_claims": "increasing"}))
    result = check.run(context)[0]

    assert result.flag == "MONOTONICITY_UNCHECKABLE"
    assert "single value" in result.detail


def test_no_declared_factor_means_nothing_to_check():
    result = MonotonicityCheck().run(_rating_context(-500.0))[0]
    assert result.flag == "NOT_APPLICABLE"
    assert "monotonic_features" in result.detail


def test_an_unknown_direction_fails_while_the_suite_is_being_built():
    """At construction, not six checks into a run — the same treatment a
    typo'd metric name gets."""
    with pytest.raises(GateConfigurationError, match="must be one of"):
        MonotonicityCheck(ActuarialConfig(monotonic_features={"prior_claims": "up"}))


def test_a_classifier_without_probabilities_declines_to_guess():
    """Partial dependence over hard 0/1 labels is a staircase that can look
    monotone while the score underneath it is not."""
    n = 80
    X = pd.DataFrame({"prior_claims": np.tile(np.arange(4.0), n // 4)})
    context = StructuredGateContext(
        X=X,
        y_true=np.tile([0, 1], n // 2),
        y_pred=np.tile([0, 1], n // 2),
        predict_fn=lambda frame: np.zeros(len(frame)),
        task="binary",
    )
    check = MonotonicityCheck(ActuarialConfig(monotonic_features={"prior_claims": "increasing"}))
    result = check.run(context)[0]
    assert result.flag == "NOT_APPLICABLE"
    assert "probabilities" in result.detail


def test_a_classifier_with_probabilities_is_checked_on_the_score():
    n = 200
    X = pd.DataFrame({"prior_claims": np.tile(np.arange(4.0), n // 4)})
    scores = np.clip(0.9 - 0.2 * X["prior_claims"].to_numpy(), 0.01, 0.99)
    context = StructuredGateContext(
        X=X,
        y_true=(np.arange(n) % 3 == 0).astype(int),
        y_pred=scores,
        predict_proba_fn=lambda frame: np.clip(
            0.9 - 0.2 * frame["prior_claims"].to_numpy(), 0.01, 0.99
        ),
        predict_fn=lambda frame: np.zeros(len(frame)),
        task="binary",
    )
    check = MonotonicityCheck(ActuarialConfig(monotonic_features={"prior_claims": "increasing"}))
    result = check.run(context)[0]
    assert result.flag == "MONOTONICITY_RISK"


# --------------------------------------------------------------------------
# DislocationCheck
# --------------------------------------------------------------------------


def _dislocated(rise_rows: int, n: int = 200, exposure=None, groups=None):
    baseline = np.full(n, 100.0)
    y_pred = baseline.copy()
    y_pred[:rise_rows] = 130.0  # a 30% rise, exactly
    return _book(y_pred, baseline * 0.95, exposure=exposure, groups=groups, baseline=baseline)


def test_the_dislocated_share_is_the_share_of_exposure_that_moves():
    result = DislocationCheck().run(_dislocated(60))[0]
    assert result.metadata["rise_share"] == pytest.approx(0.3)
    assert result.metadata["fall_share"] == pytest.approx(0.0)
    assert result.metadata["largest_rise"] == pytest.approx(0.3)
    assert result.flag == "DISLOCATION_RISK"
    assert result.blocking is False  # NEEDS_REVIEW: a re-rate may be correct


def test_a_book_inside_the_tolerance_passes():
    assert DislocationCheck().run(_dislocated(10))[0].flag == "OK"


def test_exposure_reweights_who_counts_as_moving():
    """Sixty of two hundred policies rise, but they are monthly policies:
    30% of rows is 4.1% of exposure, and the conduct question is about
    exposure. 6 / 146, exactly."""
    exposure = np.concatenate([np.full(60, 0.1), np.full(140, 1.0)])
    result = DislocationCheck().run(_dislocated(60, exposure=exposure))[0]
    assert result.metadata["rise_share"] == pytest.approx(6.0 / 146.0, abs=1e-4)
    assert result.flag == "OK"


def test_the_most_affected_protected_group_is_named():
    """A dislocation of 30% spread evenly and one landing entirely on one
    region are different conversations, and the overall figure cannot tell
    them apart."""
    groups = np.array(["A"] * 100 + ["B"] * 100)
    result = DislocationCheck().run(_dislocated(60, groups=groups))[0]

    assert result.metadata["rise_share_by_group"]["g"] == {"A": pytest.approx(0.6), "B": 0.0}
    assert "Most affected: g=A at 60.0%" in result.detail


def test_rows_with_no_baseline_to_move_from_are_excluded_and_counted():
    """A premium going from 0 to 500 is not an increase of any percentage."""
    n = 100
    baseline = np.full(n, 100.0)
    baseline[:10] = 0.0
    y_pred = np.full(n, 130.0)
    result = DislocationCheck().run(_book(y_pred, baseline + 1.0, baseline=baseline))[0]

    assert result.metadata["n_rows_excluded"] == 10
    assert result.metadata["n_rows_compared"] == 90
    assert result.metadata["rise_share"] == pytest.approx(1.0)
    assert "10 row(s) excluded" in result.detail


def test_no_baseline_means_no_comparison_rather_than_a_stand_in():
    result = DislocationCheck().run(_book(np.full(50, 100.0), np.full(50, 90.0)))[0]
    assert result.flag == "NOT_APPLICABLE"
    assert "baseline_pred" in result.detail
    assert "book mean" in result.detail  # says what it refuses to substitute


def test_an_all_zero_baseline_is_not_applicable():
    result = DislocationCheck().run(
        _book(np.full(50, 100.0), np.linspace(80, 120, 50), baseline=np.zeros(50))
    )[0]
    assert result.flag == "NOT_APPLICABLE"
    assert "zero or negative" in result.detail


# --------------------------------------------------------------------------
# Exposure through the metrics and the fairness checks
# --------------------------------------------------------------------------


def test_weighted_error_metrics_are_the_hand_computed_values():
    y_true, y_pred, weights = [10.0, 20.0], [12.0, 20.0], [1.0, 3.0]
    # errors 2 and 0; weighted MSE = (4*1 + 0*3)/4 = 1, so RMSE = 1
    assert resolve_metric("rmse", "regression", exposure=weights).fn(y_true, y_pred) == (
        pytest.approx(1.0)
    )
    # weighted MAE = (2*1 + 0*3)/4 = 0.5
    assert resolve_metric("mae", "regression", exposure=weights).fn(y_true, y_pred) == (
        pytest.approx(0.5)
    )


def test_the_numpy_weighted_metrics_agree_with_scikit_learn():
    """The numpy implementations are what a core install uses, so they have
    to give the same answer as the library everyone else compares against."""
    sklearn_metrics = pytest.importorskip("sklearn.metrics", reason="needs [structured]")
    rng = np.random.default_rng(3)
    y_true = rng.gamma(2.0, 100.0, 300)
    y_pred = y_true * rng.uniform(0.6, 1.4, 300)
    weights = rng.uniform(0.05, 1.0, 300)

    from bdp_model_gate.metrics import _mae_numpy, _r2_numpy, _rmse_numpy

    assert _rmse_numpy(y_true, y_pred, weights) == pytest.approx(
        np.sqrt(sklearn_metrics.mean_squared_error(y_true, y_pred, sample_weight=weights))
    )
    assert _mae_numpy(y_true, y_pred, weights) == pytest.approx(
        sklearn_metrics.mean_absolute_error(y_true, y_pred, sample_weight=weights)
    )
    assert _r2_numpy(y_true, y_pred, weights) == pytest.approx(
        sklearn_metrics.r2_score(y_true, y_pred, sample_weight=weights)
    )


def test_the_report_says_when_a_metric_could_not_take_the_weighting(caplog):
    """Silently dropping the exposure would put an unweighted number in a
    report whose reader believes it is weighted."""
    n = 100
    context = StructuredGateContext(
        X=pd.DataFrame({"x": np.arange(n, dtype=float)}),
        y_true=np.tile([0, 1], n // 2),
        y_pred=np.linspace(0.05, 0.95, n),
        exposure=np.linspace(0.1, 1.0, n),
        predict_fn=lambda frame: np.zeros(len(frame)),
        task="binary",
    )
    from bdp_model_gate import PerformanceConfig

    result = PerformanceThresholdCheck(PerformanceConfig(metric="roc_auc", min_score=0.0)).run(
        context
    )[0]
    assert result.metadata["exposure_weighted"] is False
    assert "NOT exposure-weighted" in result.detail


def test_a_weighted_regression_metric_says_so_in_the_report():
    context = _book(np.full(60, 100.0), np.full(60, 110.0), exposure=np.linspace(0.1, 1.0, 60))
    from bdp_model_gate import PerformanceConfig

    result = PerformanceThresholdCheck(
        PerformanceConfig(metric="mae", max_error=1e9),
    ).run(context)[0]
    assert result.metadata["exposure_weighted"] is True
    assert "exposure-weighted" in result.detail


def test_exposure_reweights_the_regression_fairness_denominator():
    """A predicts 120 on one year of exposure, B predicts 80 on three. The
    group means are unchanged, but the book's own mean moves from 100 to 90,
    so the relative gap goes from 40% to 44.4%."""
    n = 200
    groups = np.where(np.arange(n) % 2 == 0, "A", "B")
    y_pred = np.where(groups == "A", 120.0, 80.0)
    exposure = np.where(groups == "A", 1.0, 3.0)
    context = _book(y_pred, y_pred, groups=groups, exposure=exposure)

    result = GroupMeanGapCheck().run(context)[0]
    assert result.metadata["group_means"] == {"A": 120.0, "B": 80.0}
    assert result.metadata["relative_gap"] == pytest.approx(40 / 90, abs=1e-4)
    assert result.metadata["exposure_weighted"] is True
    assert "exposure-weighted" in result.detail


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exposure,message",
    [
        ([1.0] * 5, "row-aligned"),
        ([1.0] * 59 + [-1.0], "negative"),
        ([0.0] * 60, "every context.exposure value is zero"),
        ([1.0] * 59 + [np.nan], "NaN or infinite"),
        (["a"] * 60, "must be numeric"),
    ],
)
def test_a_bad_exposure_column_is_refused_eagerly(exposure, message):
    context = _book(np.linspace(90, 110, 60), np.linspace(95, 115, 60), exposure=exposure)
    with pytest.raises(GateValidationError, match=message):
        ModelGate(checks=[]).run(context)


@pytest.mark.parametrize(
    "baseline,message",
    [
        ([1.0] * 5, "row-aligned"),
        ([1.0] * 59 + [np.nan], "NaN or infinite"),
        (["a"] * 60, "must be numeric"),
    ],
)
def test_a_bad_baseline_column_is_refused_eagerly(baseline, message):
    context = _book(np.linspace(90, 110, 60), np.linspace(95, 115, 60), baseline=baseline)
    with pytest.raises(GateValidationError, match=message):
        ModelGate(checks=[]).run(context)


def test_exposure_on_a_classification_task_warns_rather_than_silently_ignoring(caplog):
    import logging

    n = 60
    context = StructuredGateContext(
        X=pd.DataFrame({"x": np.arange(n, dtype=float)}),
        y_true=np.tile([0, 1], n // 2),
        y_pred=np.linspace(0.05, 0.95, n),
        exposure=np.ones(n),
        predict_fn=lambda frame: np.zeros(len(frame)),
        task="binary",
    )
    with caplog.at_level(logging.WARNING):
        ModelGate(checks=[]).run(context)
    assert "exposure weighting applies to the regression metrics" in caplog.text.lower()


# --------------------------------------------------------------------------
# Invariants — the verdict must not move for a reason that is not a reason
# --------------------------------------------------------------------------


def test_the_gini_is_invariant_to_row_order():
    rng = np.random.default_rng(8)
    y_true = rng.gamma(2.0, 400.0, 300)
    y_pred = rng.uniform(1, 50, 300)
    exposure = rng.uniform(0.1, 1.0, 300)
    order = rng.permutation(300)

    assert lorenz_gini(y_true[order], y_pred[order], exposure[order]) == pytest.approx(
        lorenz_gini(y_true, y_pred, exposure)
    )


def test_the_gini_is_invariant_to_the_order_of_tied_predictions():
    """Where the score ties, the curve must be a straight line through the
    block however the rows inside it are arranged. A naive cumulative sum
    would let sorting a CSV change the number."""
    y_true = np.array([0.0, 10.0, 0.0, 10.0, 5.0, 5.0])
    tied = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    base = lorenz_gini(y_true, tied)
    for order in ([2, 1, 0, 3, 4, 5], [0, 2, 1, 5, 4, 3]):
        assert lorenz_gini(y_true[order], tied[order]) == pytest.approx(base)


def test_scaling_every_exposure_by_the_same_amount_changes_nothing():
    """Exposure in months and exposure in years are the same book. Weights
    are relative, and a verdict that moved on the unit would be a bug."""
    y_pred, y_true = np.linspace(50, 500, 200), np.linspace(60, 480, 200)
    exposure = np.linspace(0.1, 1.0, 200)
    context = _book(y_pred, y_true, exposure=exposure)
    scaled = _book(y_pred, y_true, exposure=exposure * 12.0)

    for check in (ActualVsExpectedCheck(), RiskDiscriminationCheck()):
        original, rescaled = check.run(context), check.run(scaled)
        assert [r.flag for r in original] == [r.flag for r in rescaled], check.name
        for left, right in zip(original, rescaled):
            # The absolute exposure a band holds is in months rather than
            # years, and says so. Every ratio and every verdict is identical.
            comparable = {k: v for k, v in left.metadata.items() if k != "bands"}
            assert comparable == {k: v for k, v in right.metadata.items() if k != "bands"}, (
                check.name
            )
            for a, b in zip(left.metadata.get("bands", []), right.metadata.get("bands", [])):
                assert a["ae"] == b["ae"] and a["n_rows"] == b["n_rows"]
                assert b["exposure"] == pytest.approx(a["exposure"] * 12.0, rel=1e-3)


def test_the_dislocation_share_is_invariant_to_row_order():
    rng = np.random.default_rng(2)
    order = rng.permutation(200)
    context = _dislocated(60)
    shuffled = _book(
        np.asarray(context.y_pred)[order],
        np.asarray(context.y_true)[order],
        baseline=np.asarray(context.baseline_pred)[order],
    )
    assert DislocationCheck().run(shuffled)[0].metadata["rise_share"] == pytest.approx(
        DislocationCheck().run(context)[0].metadata["rise_share"]
    )


def test_the_partial_dependence_curve_is_invariant_to_row_order():
    """`partial_dependence` samples through `stable_sample`, which selects by
    row content — so a re-sorted validation set yields the same curve, and
    the same verdict on a filed rating constraint."""
    check = MonotonicityCheck(ActuarialConfig(monotonic_features={"prior_claims": "increasing"}))
    context = _rating_context(-500.0)
    rng = np.random.default_rng(6)
    order = rng.permutation(len(context.X))

    shuffled = _rating_context(-500.0)
    shuffled.X = context.X.iloc[order].reset_index(drop=True)

    assert check.run(shuffled)[0].metadata["partial_dependence"] == pytest.approx(
        check.run(context)[0].metadata["partial_dependence"]
    )


def test_renaming_a_band_cannot_change_the_verdict(two_band_book):
    """Multiplying the target and the prediction by the same factor is a
    change of currency, not a change of pricing accuracy: A/E is a ratio."""
    y_pred, y_true = two_band_book
    original = ActualVsExpectedCheck().run(_book(y_pred, y_true))
    rescaled = ActualVsExpectedCheck().run(_book(y_pred * 1000, y_true * 1000))

    assert [r.flag for r in original] == [r.flag for r in rescaled]
    assert original[0].metadata["ae"] == pytest.approx(rescaled[0].metadata["ae"])


def test_a_metric_scikit_learn_does_not_define_is_not_reported_as_a_fallback():
    """A false statement in a governance record, fixed in 0.5.3.

    `rmse`, `mape`, `poisson_deviance` and `lorenz_gini` have no scikit-learn
    equivalent, so the numpy implementation is the *only* implementation. The
    report used to print "[computed without scikit-learn]" beside every RMSE
    on machines where scikit-learn was installed and working.
    """
    pytest.importorskip("sklearn", reason="the claim under test is about sklearn being present")
    for name in ("rmse", "mape", "poisson_deviance", "lorenz_gini"):
        assert resolve_metric(name, "regression").used_fallback_impl is False, name
    # ...while a metric scikit-learn *does* define still reports honestly when
    # it has to stand in, which is what the flag is for.
    from bdp_model_gate.metrics import BUILTIN_METRICS

    assert BUILTIN_METRICS["mae"].sklearn_fn == "mean_absolute_error"
    assert BUILTIN_METRICS["rmse"].sklearn_fn == ""
