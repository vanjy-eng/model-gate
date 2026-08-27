"""Calibration and separation — the two fairness families added in 0.5.0.

Known-answer style throughout: each case is constructed so the correct value
is derivable on paper, then asserted exactly. A calibration check that returns
a plausible-looking number for a badly calibrated model is precisely the
failure mode this project keeps finding.
"""

import numpy as np
import pandas as pd
import pytest

from bdp_model_gate import FairnessConfig, PerformanceConfig, StructuredGateContext
from bdp_model_gate.calibration import (
    brier_decomposition,
    brier_score,
    calibration_curve,
    expected_calibration_error,
)
from bdp_model_gate.exceptions import GateConfigurationError
from bdp_model_gate.groups import iter_protected
from bdp_model_gate.structured.calibration_checks import (
    CalibrationCheck,
    EqualisedOddsCheck,
    SubgroupCalibrationCheck,
)


class Passthrough:
    def predict(self, X):
        return (X["score"].to_numpy() >= 0.5).astype(int)


def _ctx(scores, actuals, protected=None, **kw):
    frame = pd.DataFrame({"score": np.asarray(scores, dtype=float)})
    kw.setdefault("task", "binary")
    return StructuredGateContext(
        model=Passthrough(),
        X=frame,
        y_true=np.asarray(actuals),
        y_pred=np.asarray(scores, dtype=float),
        protected_df=protected,
        **kw,
    )


# --- the metric itself -------------------------------------------------------


def test_perfectly_calibrated_predictions_approach_zero_error():
    """Generate outcomes *from* the stated probabilities: calibration is then
    correct by construction and ECE must be near zero."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 40_000)
    y = rng.binomial(1, p)
    assert expected_calibration_error(y, p, n_bins=10) < 0.01


def test_a_constant_forecast_at_the_base_rate_is_perfectly_calibrated():
    """Predicting 0.3 for everyone when 30% are positive is useless but
    perfectly calibrated — the case that shows ECE alone is not quality."""
    y = np.array([1] * 300 + [0] * 700)
    p = np.full(1000, 0.3)
    assert expected_calibration_error(y, p, n_bins=10) == pytest.approx(0.0, abs=1e-9)

    parts = brier_decomposition(y, p, n_bins=10)
    assert parts["reliability"] == pytest.approx(0.0, abs=1e-9)
    assert parts["resolution"] == pytest.approx(0.0, abs=1e-9)  # learned nothing


def test_a_fixed_offset_produces_exactly_that_calibration_error():
    """Every prediction 0.2 too high, in a single bin: ECE must be 0.2."""
    y = np.array([1] * 500 + [0] * 500)  # base rate 0.5
    p = np.full(1000, 0.7)  # claims 0.7
    assert expected_calibration_error(y, p, n_bins=10) == pytest.approx(0.2, abs=1e-9)


def test_brier_bounds_and_the_decomposition_identity():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 5_000)
    y = rng.binomial(1, p)

    assert brier_score(y, y.astype(float)) == pytest.approx(0.0)
    assert brier_score(y, 1.0 - y) == pytest.approx(1.0)

    parts = brier_decomposition(y, p)
    assert parts["reliability"] - parts["resolution"] + parts["uncertainty"] == pytest.approx(
        parts["binned_brier"], abs=1e-12
    )


def test_probabilities_outside_the_unit_interval_are_refused():
    """A common mistake is passing a decision function or log-odds. Silently
    binning those would produce a confident, meaningless number."""
    y = np.array([0, 1, 0, 1])
    with pytest.raises(GateConfigurationError, match=r"probabilities in \[0, 1\]"):
        expected_calibration_error(y, np.array([-2.0, 3.1, 0.4, 0.6]))


def test_quantile_binning_handles_a_skewed_score_distribution():
    """Uniform bins leave the interesting region nearly empty when scores
    cluster near zero, as fraud and default scores do."""
    rng = np.random.default_rng(2)
    p = rng.beta(1.2, 20, 8_000)  # heavily skewed toward 0
    y = rng.binomial(1, p)

    uniform = calibration_curve(y, p, n_bins=10, strategy="uniform")
    quantile = calibration_curve(y, p, n_bins=10, strategy="quantile")

    assert uniform.populated.sum() < quantile.populated.sum()
    assert quantile.count.min() > 0  # every quantile bin is used


# --- CalibrationCheck --------------------------------------------------------


def test_check_passes_a_calibrated_model_and_flags_an_inflated_one():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.05, 0.95, 6_000)
    calibrated = rng.binomial(1, p)
    config = PerformanceConfig(max_ece=0.05)

    good = CalibrationCheck(config).run(_ctx(p, calibrated))[0]
    assert good.flag == "OK"
    assert good.metadata["ece"] < 0.05

    # Same scores, outcomes generated at half the stated rate.
    inflated = rng.binomial(1, p * 0.5)
    bad = CalibrationCheck(config).run(_ctx(p, inflated))[0]
    assert bad.flag == "CALIBRATION_RISK"
    assert "run high" in bad.detail


def test_check_names_the_direction_of_miscalibration():
    """ "Too high" and "too low" need different fixes, so the detail must say
    which rather than only that a threshold was breached."""
    rng = np.random.default_rng(4)
    p = rng.uniform(0.05, 0.95, 4_000)
    understated = rng.binomial(1, np.clip(p * 1.8, 0, 1))
    result = CalibrationCheck(PerformanceConfig(max_ece=0.02)).run(_ctx(p, understated))[0]
    assert result.flag == "CALIBRATION_RISK"
    assert "run low" in result.detail


def test_check_skips_hard_labels_rather_than_reporting_a_number():
    y = np.array([0, 1] * 200)
    result = CalibrationCheck().run(_ctx(y.astype(float), y))[0]
    assert result.flag == "NOT_APPLICABLE"
    assert "hard labels" in result.detail


# --- SubgroupCalibrationCheck (sufficiency) ----------------------------------


def test_subgroup_calibration_catches_a_group_the_aggregate_hides():
    """Majority calibrated, minority inflated. Overall ECE looks acceptable
    because the majority dominates the average — which is the whole point."""
    rng = np.random.default_rng(5)
    n_major, n_minor = 4_000, 600
    p_major = rng.uniform(0.05, 0.95, n_major)
    p_minor = rng.uniform(0.05, 0.95, n_minor)

    y = np.concatenate(
        [
            rng.binomial(1, p_major),  # calibrated
            rng.binomial(1, p_minor * 0.35),  # badly over-stated
        ]
    )
    p = np.concatenate([p_major, p_minor])
    protected = pd.DataFrame({"segment": ["major"] * n_major + ["minor"] * n_minor})

    overall = CalibrationCheck(PerformanceConfig(max_ece=0.10)).run(_ctx(p, y))[0]
    subgroup = SubgroupCalibrationCheck(FairnessConfig(subgroup_calibration_threshold=0.05)).run(
        _ctx(p, y, protected)
    )[0]

    assert overall.flag == "OK", "the aggregate is expected to look fine here"
    assert subgroup.flag == "SUBGROUP_CALIBRATION_RISK"
    assert subgroup.metadata["worst_calibrated_group"] == "minor"


def test_subgroup_calibration_is_non_blocking():
    """Sufficiency findings need judgement, so they route to review like every
    other fairness check rather than failing a build."""
    assert SubgroupCalibrationCheck().blocking is False


# --- EqualisedOddsCheck (separation) -----------------------------------------


def _separation_frame(tpr_a, tpr_b, fpr_a=0.2, fpr_b=0.2, per_group=500, seed=6):
    """Builds a book with exactly the requested per-group error rates."""
    rng = np.random.default_rng(seed)
    rows = []
    for group, tpr, fpr in (("A", tpr_a, fpr_a), ("B", tpr_b, fpr_b)):
        actual = np.array([1] * (per_group // 2) + [0] * (per_group // 2))
        predicted = np.where(
            actual == 1,
            rng.random(per_group) < tpr,
            rng.random(per_group) < fpr,
        ).astype(float)
        rows.append(pd.DataFrame({"g": group, "y": actual, "p": predicted}))
    return pd.concat(rows, ignore_index=True)


def test_equal_error_rates_pass_both_notions():
    frame = _separation_frame(tpr_a=0.8, tpr_b=0.8)
    results = EqualisedOddsCheck().run(
        _ctx(frame["p"], frame["y"], pd.DataFrame({"g": frame["g"]}))
    )
    assert {r.flag for r in results} == {"OK"}


def test_a_true_positive_rate_gap_is_flagged_as_equal_opportunity():
    """Among applicants who should be approved, group B is far less likely to
    be — the notion lending regulators centre on."""
    frame = _separation_frame(tpr_a=0.9, tpr_b=0.5)
    results = EqualisedOddsCheck().run(
        _ctx(frame["p"], frame["y"], pd.DataFrame({"g": frame["g"]}))
    )
    by_notion = {r.metadata["notion"]: r for r in results}

    assert by_notion["equal_opportunity"].flag == "EQUAL_OPPORTUNITY_RISK"
    assert by_notion["equal_opportunity"].metadata["tpr_difference"] == pytest.approx(0.4, abs=0.1)
    assert by_notion["equalised_odds"].flag == "EQUALISED_ODDS_RISK"


def test_a_false_positive_gap_alone_trips_only_equalised_odds():
    """Equal opportunity ignores false positives entirely; equalised odds does
    not. A model can satisfy the first and fail the second."""
    frame = _separation_frame(tpr_a=0.8, tpr_b=0.8, fpr_a=0.05, fpr_b=0.55)
    by_notion = {
        r.metadata["notion"]: r
        for r in EqualisedOddsCheck().run(
            _ctx(frame["p"], frame["y"], pd.DataFrame({"g": frame["g"]}))
        )
    }
    assert by_notion["equal_opportunity"].flag == "OK"
    assert by_notion["equalised_odds"].flag == "EQUALISED_ODDS_RISK"


def test_separation_needs_ground_truth():
    """The property that distinguishes it from demographic parity."""
    p = np.linspace(0.01, 0.99, 200)
    context = StructuredGateContext(
        model=Passthrough(),
        X=pd.DataFrame({"score": p}),
        y_true=None,
        y_pred=p,
        protected_df=pd.DataFrame({"g": np.resize(["a", "b"], 200)}),
        task="binary",
    )
    result = EqualisedOddsCheck().run(context)[0]
    assert result.flag == "NOT_APPLICABLE"
    assert "ground truth" in result.detail


def test_a_group_with_no_positive_cases_is_skipped_not_divided_by_zero():
    n = 400
    groups = np.where(np.arange(n) % 2 == 0, "a", "b")
    y = np.where(groups == "a", np.resize([0, 1], n), 0)  # group b: no positives
    p = np.resize([0.2, 0.8], n)
    results = EqualisedOddsCheck().run(_ctx(p, y, pd.DataFrame({"g": groups})))
    assert results[0].flag == "NOT_APPLICABLE"


# --- intersectional ----------------------------------------------------------


def test_intersections_are_off_by_default_and_opt_in():
    frame = pd.DataFrame(
        {
            "gender": np.resize(["F", "M"], 400),
            "region": np.resize(["Lagos", "Lagos", "Kano", "Kano"], 400),
        }
    )
    assert [k for k, _ in iter_protected(frame)] == ["gender", "region"]
    assert [k for k, _ in iter_protected(frame, intersectional=True)] == [
        "gender",
        "region",
        "gender × region",
    ]


def test_an_intersection_can_fail_where_both_margins_pass():
    """The reason intersectional checking exists: harm concentrates where two
    attributes meet, and marginal checks are blind to it by construction."""
    rng = np.random.default_rng(7)
    n = 4_000
    # Both attributes must be minorities for the margins to stay clean: the
    # harmed cell moves each margin by the *other* attribute's share, so with
    # a 50/50 split the marginal gap is 0.25 and the demonstration collapses.
    # At 25% each, the margins move ~0.25 while the intersection moves ~1.0.
    gender = rng.choice(["F", "M"], n, p=[0.25, 0.75])
    region = rng.choice(["Kano", "Lagos"], n, p=[0.25, 0.75])
    actual = rng.binomial(1, 0.5, n)

    # Only F-in-Kano is systematically denied.
    harmed = (gender == "F") & (region == "Kano")
    predicted = np.where(harmed & (actual == 1), 0.0, actual.astype(float))

    protected = pd.DataFrame({"gender": gender, "region": region})
    context = _ctx(predicted, actual, protected)

    marginal = EqualisedOddsCheck(
        FairnessConfig(equalised_odds_threshold=0.35, intersectional=False)
    ).run(context)
    joint = EqualisedOddsCheck(
        FairnessConfig(equalised_odds_threshold=0.35, intersectional=True)
    ).run(context)

    assert all(r.flag == "OK" for r in marginal), "each margin alone should look acceptable"
    intersect = [r for r in joint if "×" in r.metadata.get("protected_attr", "")]
    assert intersect, "the intersection should have been evaluated"
    assert any(r.flag != "OK" for r in intersect), "the intersection should be flagged"
