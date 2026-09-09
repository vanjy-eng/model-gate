"""Checks scored against inputs whose correct answer is known by hand.

Every silent failure this library has shipped had the same root cause: no
test knew what the right answer *was*. `DisparateImpactCheck` returned
`0.000` for a maximally discriminatory model and nothing noticed, because
every test asserted only that it returned *something*.

These tests are constructed so the true value is derivable on paper, then
assert equality. If you know the answer, wrong cannot hide.
"""

import numpy as np
import pandas as pd
import pytest

from bdp_model_gate import FairnessConfig, PerformanceConfig, StructuredGateContext
from bdp_model_gate.metrics import ordinal_mae, quadratic_kappa, resolve_metric
from bdp_model_gate.structured.fairness import DisparateImpactCheck, ProxyCorrelationCheck
from bdp_model_gate.structured.performance import PerformanceThresholdCheck
from bdp_model_gate.structured.regression_fairness import (
    CalibrationParityCheck,
    ErrorParityCheck,
    GroupMeanGapCheck,
    LossRatioParityCheck,
)
from bdp_model_gate.structured.security import AdversarialRobustnessCheck


class Constant:
    """Always predicts the same value, whatever the input."""

    def __init__(self, value=1):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


class Threshold:
    """Predicts on one column against a fixed cut."""

    def __init__(self, column="x", cut=0.5):
        self.column, self.cut = column, cut

    def predict(self, X):
        return (X[self.column].to_numpy() >= self.cut).astype(int)


# --- demographic parity ------------------------------------------------------


def test_maximally_unfair_gives_exactly_one():
    """Every M selected, every F rejected. The parity difference is 1.0 by
    definition — there is no other correct answer."""
    pytest.importorskip("fairlearn")
    n = 400
    gender = np.where(np.arange(n) % 2 == 0, "M", "F")
    y_pred = (gender == "M").astype(int)

    result = DisparateImpactCheck().run(
        StructuredGateContext(
            model=Constant(),
            X=pd.DataFrame({"x": np.arange(n, dtype=float)}),
            y_true=np.resize([0, 1], n),
            y_pred=y_pred,
            protected_df=pd.DataFrame({"gender": gender}),
            task="binary",
        )
    )[0]

    assert result.metadata["demographic_parity_diff"] == pytest.approx(1.0)
    assert result.flag == "DISPARITY_RISK"


def test_perfectly_fair_gives_exactly_zero():
    """Identical selection rates in both groups."""
    pytest.importorskip("fairlearn")
    n = 400
    gender = np.where(np.arange(n) % 2 == 0, "M", "F")
    # Every other row selected, alternating independently of gender.
    y_pred = np.resize([1, 1, 0, 0], n)

    result = DisparateImpactCheck().run(
        StructuredGateContext(
            model=Constant(),
            X=pd.DataFrame({"x": np.arange(n, dtype=float)}),
            y_true=np.resize([0, 1], n),
            y_pred=y_pred,
            protected_df=pd.DataFrame({"gender": gender}),
            task="binary",
        )
    )[0]

    # The point estimate is still exactly what it always was, and that is the
    # known answer this test exists for.
    assert result.metadata["demographic_parity_diff"] == pytest.approx(0.0)

    # The *verdict* changed in 0.6.0, and not because the check regressed.
    # 400 rows at a selection rate of 0.5 carry a standard error of 0.05 on
    # the difference, so the 95% interval reaches about 0.11 — past the
    # default disparity_threshold of 0.10. This sample genuinely cannot
    # distinguish a perfectly fair model from one 0.10 out, and the honest
    # answer is to say so rather than to report a clean pass.
    assert result.flag == "UNCERTAIN"
    assert result.blocking is False
    assert "cannot rule out a breach" in result.detail


def test_a_fair_model_reads_clean_once_the_sample_can_show_it():
    """The other side of the previous test, and the pair documents the
    crossover — which is the practical thing a reader needs.

    1.96/sqrt(n) < 0.10 puts the boundary near n = 384, so 1,000 rows is
    comfortably enough for a genuinely fair model to report OK. Below roughly
    500 rows a 0.10 threshold is inside the noise floor whatever the model
    does; above it, clean models stay clean.
    """
    pytest.importorskip("fairlearn")
    n = 1000
    gender = np.where(np.arange(n) % 2 == 0, "M", "F")
    y_pred = np.resize([1, 1, 0, 0], n)

    result = DisparateImpactCheck().run(
        StructuredGateContext(
            model=Constant(),
            X=pd.DataFrame({"x": np.arange(n, dtype=float)}),
            y_true=np.resize([0, 1], n),
            y_pred=y_pred,
            protected_df=pd.DataFrame({"gender": gender}),
            task="binary",
        )
    )[0]

    assert result.metadata["demographic_parity_diff"] == pytest.approx(0.0)
    assert result.flag == "OK"
    assert result.metadata["ci_high"] < result.metadata["threshold"]
    assert "whole interval sits below" in result.detail


@pytest.mark.parametrize("rate_m,rate_f", [(1.0, 0.5), (0.8, 0.2), (0.6, 0.6)])
def test_parity_difference_equals_the_rate_difference(rate_m, rate_f):
    """The metric is defined as |rate(M) - rate(F)|. Assert exactly that."""
    pytest.importorskip("fairlearn")
    per_group = 200
    gender = np.array(["M"] * per_group + ["F"] * per_group)
    y_pred = np.concatenate(
        [
            np.array([1] * int(per_group * rate_m) + [0] * (per_group - int(per_group * rate_m))),
            np.array([1] * int(per_group * rate_f) + [0] * (per_group - int(per_group * rate_f))),
        ]
    )

    result = DisparateImpactCheck().run(
        StructuredGateContext(
            model=Constant(),
            X=pd.DataFrame({"x": np.arange(2 * per_group, dtype=float)}),
            y_true=np.resize([0, 1], 2 * per_group),
            y_pred=y_pred,
            protected_df=pd.DataFrame({"gender": gender}),
            task="binary",
        )
    )[0]
    assert result.metadata["demographic_parity_diff"] == pytest.approx(
        abs(rate_m - rate_f), abs=1e-9
    )


# --- proxy correlation -------------------------------------------------------


def test_feature_that_is_a_pure_function_of_the_attribute_gives_eta_squared_one():
    """When a feature is constant within each group and differs between them,
    all variance is between-group: eta^2 is exactly 1."""
    n = 300
    region = np.where(np.arange(n) % 2 == 0, "Lagos", "Kano")
    X = pd.DataFrame({"branch_km": np.where(region == "Lagos", 2.0, 14.0)})

    result = ProxyCorrelationCheck().run(
        StructuredGateContext(
            model=Constant(),
            X=X,
            y_true=np.resize([0, 1], n),
            y_pred=np.resize([0, 1], n),
            protected_df=pd.DataFrame({"region": region}),
            task="binary",
        )
    )[0]

    assert result.flag == "PROXY_RISK"
    assert result.metadata["proxy_strength"] == pytest.approx(1.0)


def test_feature_independent_of_the_attribute_gives_eta_squared_zero():
    """Identical values in both groups: no between-group variance at all.

    The cycle length must be a multiple of the group period, or the feature
    is correlated with the attribute by construction — which is what a first
    draft of this test got wrong.
    """
    n = 300
    region = np.where(np.arange(n) % 2 == 0, "Lagos", "Kano")
    # Period 4 against a period-2 grouping: each group sees {1.0, 2.0} equally.
    X = pd.DataFrame({"tenure": np.resize([1.0, 1.0, 2.0, 2.0], n)})

    results = ProxyCorrelationCheck(FairnessConfig(proxy_corr_threshold=1e-9)).run(
        StructuredGateContext(
            model=Constant(),
            X=X,
            y_true=np.resize([0, 1], n),
            y_pred=np.resize([0, 1], n),
            protected_df=pd.DataFrame({"region": region}),
            task="binary",
        )
    )
    assert results[0].flag == "OK"


# --- regression fairness -----------------------------------------------------


def _regression_context(y_pred, expected_loss=None, y_true=None, groups=None, n=200):
    groups = np.where(np.arange(n) % 2 == 0, "A", "B") if groups is None else groups
    return StructuredGateContext(
        model=Constant(0.0),
        X=pd.DataFrame({"x": np.arange(n, dtype=float)}),
        y_true=np.asarray(y_true if y_true is not None else y_pred, dtype=float),
        y_pred=np.asarray(y_pred, dtype=float),
        protected_df=pd.DataFrame({"g": groups}),
        expected_loss=expected_loss,
        task="regression",
    )


def test_loss_ratio_parity_equals_the_hand_computed_margins():
    """Group A is loaded 1.5x its expected loss, group B 1.0x. The ratios must
    come back exactly, and the relative gap is (1.5-1.0)/1.25 = 0.4."""
    n = 200
    groups = np.where(np.arange(n) % 2 == 0, "A", "B")
    expected_loss = np.full(n, 1000.0)
    y_pred = np.where(groups == "A", 1500.0, 1000.0)

    result = LossRatioParityCheck().run(
        _regression_context(y_pred, expected_loss=expected_loss, groups=groups, n=n)
    )[0]

    ratios = result.metadata["group_loss_ratio"]
    assert ratios["A"] == pytest.approx(1.5)
    assert ratios["B"] == pytest.approx(1.0)
    assert result.metadata["relative_gap"] == pytest.approx(0.4, abs=1e-3)
    assert result.metadata["highest_margin_group"] == "A"


def test_group_mean_gap_equals_the_hand_computed_spread():
    """A predicts 120, B predicts 80. Overall mean 100, so the relative gap
    is (120-80)/100 = 0.4."""
    n = 200
    groups = np.where(np.arange(n) % 2 == 0, "A", "B")
    y_pred = np.where(groups == "A", 120.0, 80.0)

    result = GroupMeanGapCheck().run(_regression_context(y_pred, groups=groups, n=n))[0]

    assert result.metadata["group_means"]["A"] == pytest.approx(120.0)
    assert result.metadata["group_means"]["B"] == pytest.approx(80.0)
    assert result.metadata["relative_gap"] == pytest.approx(0.4)


def test_error_parity_equals_the_hand_computed_mae_spread():
    """A is off by 30 everywhere, B by 10. Overall MAE 20, gap 20/20 = 1.0."""
    n = 200
    groups = np.where(np.arange(n) % 2 == 0, "A", "B")
    y_true = np.full(n, 100.0)
    y_pred = np.where(groups == "A", 130.0, 110.0)

    result = ErrorParityCheck().run(_regression_context(y_pred, y_true=y_true, groups=groups, n=n))[
        0
    ]

    assert result.metadata["group_mae"]["A"] == pytest.approx(30.0)
    assert result.metadata["group_mae"]["B"] == pytest.approx(10.0)
    assert result.metadata["relative_gap"] == pytest.approx(1.0)
    assert result.metadata["worst_served_group"] == "A"


def test_calibration_parity_signs_the_bias_correctly():
    """A over-predicts by 20, B under-predicts by 20. Over-prediction is
    positive residual, so A must be named as the over-predicted group."""
    n = 200
    groups = np.where(np.arange(n) % 2 == 0, "A", "B")
    y_true = np.full(n, 100.0)
    y_pred = np.where(groups == "A", 120.0, 80.0)

    result = CalibrationParityCheck().run(
        _regression_context(y_pred, y_true=y_true, groups=groups, n=n)
    )[0]

    assert result.metadata["group_bias"]["A"] == pytest.approx(20.0)
    assert result.metadata["group_bias"]["B"] == pytest.approx(-20.0)
    assert result.metadata["most_over_predicted"] == "A"
    assert result.metadata["most_under_predicted"] == "B"


# --- adversarial robustness --------------------------------------------------


def test_constant_model_has_exactly_zero_flip_rate():
    """A model that ignores its input cannot be perturbed into changing its
    mind. Any non-zero flip rate would mean the check is measuring noise."""
    n = 200
    result = AdversarialRobustnessCheck().run(
        StructuredGateContext(
            model=Constant(1),
            X=pd.DataFrame({"x": np.linspace(1, 100, n)}),
            y_true=np.resize([0, 1], n),
            y_pred=np.resize([0, 1], n),
            task="binary",
        )
    )[0]
    assert result.metadata["flip_rate"] == 0.0
    assert result.flag == "OK"


def test_constant_regression_model_has_exactly_zero_relative_shift():
    n = 200
    result = AdversarialRobustnessCheck().run(
        StructuredGateContext(
            model=Constant(42.0),
            X=pd.DataFrame({"x": np.linspace(1, 100, n)}),
            y_true=np.linspace(1, 100, n),
            y_pred=np.full(n, 42.0),
            task="regression",
        )
    )[0]
    assert result.metadata["relative_shift"] == 0.0


# --- metrics -----------------------------------------------------------------


def test_accuracy_bounds_without_scikit_learn():
    """accuracy is numpy-native, so this must hold on a core install too."""
    y = np.array([0, 0, 1, 1])
    assert resolve_metric("accuracy", "binary").fn(y, y) == pytest.approx(1.0)
    assert resolve_metric("accuracy", "binary").fn(y, 1 - y) == pytest.approx(0.0)


def test_roc_auc_bounds():
    """Perfect ranking is 1.0 and perfectly inverted is 0.0, by definition."""
    pytest.importorskip("sklearn", reason="roc_auc needs scikit-learn")
    y = np.array([0, 0, 1, 1])
    auc = resolve_metric("roc_auc", "binary").fn
    assert auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)


def test_regression_metrics_on_hand_computable_values():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])  # errors: +2, -2, +3
    assert resolve_metric("mae", "regression").fn(y_true, y_pred) == pytest.approx(7 / 3)
    assert resolve_metric("rmse", "regression").fn(y_true, y_pred) == pytest.approx(
        np.sqrt((4 + 4 + 9) / 3)
    )
    # MAPE: |2/10| + |2/20| + |3/30| = 0.2 + 0.1 + 0.1, over 3
    assert resolve_metric("mape", "regression").fn(y_true, y_pred) == pytest.approx(0.4 / 3)
    # r2 on a perfect fit is exactly 1
    assert resolve_metric("r2", "regression").fn(y_true, y_true) == pytest.approx(1.0)


def test_ordinal_metrics_on_hand_computable_values():
    order = ["low", "mid", "high"]
    truth = ["high", "high", "mid", "low"]
    # ranks 2,2,1,0 vs 1,2,1,0 -> absolute distances 1,0,0,0 -> mean 0.25
    assert ordinal_mae(truth, ["mid", "high", "mid", "low"], order) == pytest.approx(0.25)
    assert quadratic_kappa(truth, truth, order) == pytest.approx(1.0)


def test_threshold_comparison_is_inclusive_at_the_boundary():
    """A score exactly equal to min_score passes; exactly equal to max_error
    passes. An off-by-one here silently changes every borderline verdict."""
    n = 100
    y = np.resize([0, 1], n)
    context = StructuredGateContext(
        model=Constant(1),
        X=pd.DataFrame({"x": np.arange(n, dtype=float)}),
        y_true=y,
        y_pred=y,
        task="binary",
    )
    exact = PerformanceThresholdCheck(PerformanceConfig(metric="accuracy", min_score=1.0)).run(
        context
    )[0]
    assert exact.metadata["value"] == pytest.approx(1.0)
    assert exact.flag == "OK"

    reg = StructuredGateContext(
        model=Constant(1.0),
        X=pd.DataFrame({"x": np.arange(n, dtype=float)}),
        y_true=np.arange(n, dtype=float),
        y_pred=np.arange(n, dtype=float) + 1.0,
        task="regression",
    )
    boundary = PerformanceThresholdCheck(PerformanceConfig(metric="mae", max_error=1.0)).run(reg)[0]
    assert boundary.metadata["value"] == pytest.approx(1.0)
    assert boundary.flag == "OK"
