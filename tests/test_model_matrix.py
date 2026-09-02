"""Every check against every model family, crossed with every task.

`ShapSubgroupCheck` crashed on `RandomForestClassifier` for a whole release
because the fixtures only ever used one model family, and which array shape
shap returns depends on the estimator. A matrix is the cheapest guard against
that whole class of bug: an API difference in any family surfaces here rather
than in someone's pipeline.

Every flag must land in a known set, and the autouse guard in conftest fails
the test on any unexpected CHECK_ERROR.
"""

import numpy as np
import pandas as pd
import pytest

from bdp_model_gate import (
    ActuarialConfig,
    GateConfig,
    ModelGate,
    PerformanceConfig,
    StructuredGateContext,
)
from bdp_model_gate.structured import default_structured_checks
from bdp_model_gate.structured.actuarial_checks import RiskDiscriminationCheck

sklearn = pytest.importorskip("sklearn", reason="the matrix fits real estimators")

from sklearn.ensemble import (  # noqa: E402
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

VALID_FLAGS = {
    "OK",
    "NOT_APPLICABLE",
    "PROXY_RISK",
    "DISPARITY_RISK",
    "SUBGROUP_IMPACT_RISK",
    "COUNTERFACTUAL_RISK",
    "MEAN_GAP_RISK",
    "ERROR_PARITY_RISK",
    "CALIBRATION_RISK",
    "LOSS_RATIO_RISK",
    "PERFORMANCE_RISK",
    "COMPLIANCE_RISK",
    "ROBUSTNESS_RISK",
    "PII_LEAKAGE_RISK",
    "INJECTION_RISK",
    # 0.5.0 — separation and sufficiency
    "SUBGROUP_CALIBRATION_RISK",
    "EQUAL_OPPORTUNITY_RISK",
    "EQUALISED_ODDS_RISK",
    # 0.5.2 — validation methodology
    "LEAKAGE_RISK",
    "DUPLICATE_ROWS_RISK",
    "SPLIT_OVERLAP_RISK",
    "VALIDATION_STRATEGY_RISK",
    "FEATURE_CONTRACT_RISK",
    "FEATURE_ORDER_RISK",
    "DRIFT_RISK",
    # 0.5.3 — exposure and the actuarial measures
    "AE_LEVEL_RISK",
    "AE_BAND_RISK",
    "DISCRIMINATION_RISK",
    "MONOTONICITY_RISK",
    "MONOTONICITY_UNCHECKABLE",
    "DISLOCATION_RISK",
    # 0.5.4 — prompt injection, properly
    "INJECTION_LEAK",
    "INJECTION_COMPLIANCE",
    "INJECTION_JUDGED",
    "INJECTION_NEEDS_JUDGEMENT",
    "PII_ECHO_RISK",
    "REPORT_INJECTION_RISK",
}

CLASSIFIERS = {
    "LogisticRegression": lambda: LogisticRegression(max_iter=2000),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=25, random_state=0),
    "GradientBoosting": lambda: GradientBoostingClassifier(n_estimators=25, random_state=0),
    "DecisionTree": lambda: DecisionTreeClassifier(max_depth=4, random_state=0),
    "SVC_proba": lambda: SVC(probability=True, random_state=0),
    "Pipeline": lambda: Pipeline(
        [("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))]
    ),
}

REGRESSORS = {
    "LinearRegression": lambda: LinearRegression(),
    "RandomForestRegressor": lambda: RandomForestRegressor(n_estimators=25, random_state=0),
    "GradientBoostingRegressor": lambda: GradientBoostingRegressor(n_estimators=25, random_state=0),
    "Pipeline": lambda: Pipeline([("scale", StandardScaler()), ("reg", LinearRegression())]),
}


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(23)
    n = 260
    region = rng.choice(["Lagos", "Kano"], n)
    X = pd.DataFrame(
        {
            "income": rng.lognormal(11.4, 0.4, n).round(2),
            "age": rng.integers(21, 65, n).astype(float),
            "claims": rng.poisson(0.7, n).astype(float),
            "branch_km": np.where(region == "Lagos", 2.0, 14.0) + rng.normal(0, 1.0, n),
        }
    )
    protected = pd.DataFrame({"region": region})
    latent = np.log(X["income"] / 60_000) - 0.6 * X["claims"]
    y_binary = (latent > latent.median()).astype(int).to_numpy()
    y_multi = pd.cut(latent, 3, labels=["decline", "refer", "accept"]).astype(str).to_numpy()
    y_reg = (X["income"] * 0.1 + 4_000 * X["claims"]).to_numpy()
    return X, protected, y_binary, y_multi, y_reg


MODEL_CARD = {
    "use_case": "credit_scoring",
    "legal_basis": "Contract",
    "data_minimization_justification": "affordability only",
    "training_data_source": "internal book",
    "dpia_completed": True,
    "influences_decision_about_person": True,
    "explainability_method": "SHAP",
}


def _assert_healthy(report, label):
    """Every result must be a flag the library defines, and no check may have
    raised. The autouse conftest guard covers CHECK_ERROR; this adds the
    stricter claim that nothing unknown appeared."""
    assert report.results, f"{label}: the gate produced no results at all"
    unknown = {r.flag for r in report.results} - VALID_FLAGS
    assert not unknown, f"{label}: unknown flag(s) {unknown}"
    assert report.gate_status in {"PASS", "NEEDS_REVIEW", "BLOCKED"}


@pytest.mark.parametrize("name", sorted(CLASSIFIERS))
def test_binary_across_model_families(data, name):
    X, protected, y_binary, _, _ = data
    model = CLASSIFIERS[name]().fit(X, y_binary)

    config = GateConfig(performance=PerformanceConfig(metric="roc_auc", min_score=0.0))
    report = ModelGate(checks=default_structured_checks(config, include_plugins=False)).run(
        StructuredGateContext(
            model=model,
            X=X,
            y_true=y_binary,
            y_pred=model.predict_proba(X)[:, 1],
            protected_df=protected,
            model_card=MODEL_CARD,
            task="binary",
        )
    )
    _assert_healthy(report, f"binary/{name}")


@pytest.mark.parametrize("name", sorted(CLASSIFIERS))
def test_multiclass_across_model_families(data, name):
    """This is the cross-product that would have caught the shap 3-D bug:
    RandomForest returns (rows, features, classes) where others return 2-D."""
    X, protected, _, y_multi, _ = data
    model = CLASSIFIERS[name]().fit(X, y_multi)

    config = GateConfig(performance=PerformanceConfig(metric="balanced_accuracy", min_score=0.0))
    report = ModelGate(checks=default_structured_checks(config, include_plugins=False)).run(
        StructuredGateContext(
            model=model,
            X=X,
            y_true=y_multi,
            y_pred=model.predict(X),
            protected_df=protected,
            model_card=MODEL_CARD,
            task="multiclass",
            class_order=["decline", "refer", "accept"],
        )
    )
    _assert_healthy(report, f"multiclass/{name}")


@pytest.mark.parametrize("name", sorted(REGRESSORS))
def test_regression_across_model_families(data, name):
    """Every regression check against every regressor, with every optional
    pricing input supplied — exposure, an incumbent premium and a declared
    rating constraint. `MonotonicityCheck` in particular calls the model
    `grid_points * max_rows` times through whatever prediction path the family
    exposes, which is the shape that broke SHAP on RandomForest."""
    X, protected, _, _, y_reg = data
    model = REGRESSORS[name]().fit(X, y_reg)
    predictions = model.predict(X)
    rng = np.random.default_rng(5)

    config = GateConfig(
        performance=PerformanceConfig(metric="r2", min_score=-1e9),
        actuarial=ActuarialConfig(monotonic_features={"income": "increasing"}),
    )
    report = ModelGate(checks=default_structured_checks(config, include_plugins=False)).run(
        StructuredGateContext(
            model=model,
            X=X,
            y_true=y_reg,
            y_pred=predictions,
            protected_df=protected,
            expected_loss=np.clip(predictions, 1.0, None),
            exposure=rng.uniform(0.1, 1.0, len(X)),
            baseline_pred=np.clip(predictions, 1.0, None) * rng.uniform(0.8, 1.1, len(X)),
            model_card=MODEL_CARD,
            task="regression",
        )
    )
    _assert_healthy(report, f"regression/{name}")

    by_name = {r.check_name: r for r in report.results}
    for check_name in ("actual_vs_expected", "risk_discrimination", "prediction_dislocation"):
        assert by_name[check_name].flag != "NOT_APPLICABLE", f"{check_name} skipped for {name}"
    assert by_name["monotonicity"].flag in {"OK", "MONOTONICITY_RISK"}


@pytest.mark.parametrize("name", sorted(REGRESSORS))
def test_the_gini_is_reported_for_every_regressor(data, name):
    """A scikit-learn regressor can predict below zero, which the Lorenz curve
    is not defined against. The check must skip with a reason rather than
    raise — and must still report where the prediction is usable."""
    X, _, _, _, y_reg = data
    model = REGRESSORS[name]().fit(X, y_reg)
    result = RiskDiscriminationCheck().run(
        StructuredGateContext(
            model=model,
            X=X,
            y_true=y_reg,
            y_pred=model.predict(X),
            task="regression",
        )
    )[0]
    assert result.flag in {"OK", "DISCRIMINATION_RISK", "NOT_APPLICABLE"}
    if result.flag != "NOT_APPLICABLE":
        assert result.metadata["gini"] <= result.metadata["gini_ceiling"] + 1e-9


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_predict_fn_only_across_tasks(data, task):
    """A closure with no model object at all — the framework-agnostic path."""
    X, protected, y_binary, y_multi, y_reg = data

    if task == "binary":
        y_true, y_pred = y_binary, y_binary.astype(float)
        extra = dict()
        config = PerformanceConfig(metric="accuracy", min_score=0.0)
    elif task == "multiclass":
        y_true, y_pred = y_multi, y_multi
        extra = dict(class_order=["decline", "refer", "accept"])
        config = PerformanceConfig(metric="ordinal_mae", max_error=1e9)
    else:
        y_true, y_pred = y_reg, y_reg
        extra = dict()
        config = PerformanceConfig(metric="mae", max_error=1e12)

    report = ModelGate(
        checks=default_structured_checks(GateConfig(performance=config), include_plugins=False)
    ).run(
        StructuredGateContext(
            X=X,
            y_true=y_true,
            y_pred=y_pred,
            predict_fn=lambda df: np.resize(np.asarray(y_pred), len(df)),
            protected_df=protected,
            model_card=MODEL_CARD,
            task=task,
            **extra,
        )
    )
    _assert_healthy(report, f"predict_fn/{task}")


# --- hostile shapes ----------------------------------------------------------


def test_wide_scale_features_do_not_break_robustness(wide_scale_frame):
    """Seven orders of magnitude between columns. A perturbation derived from a
    single global scale reported a relative shift of ~1448 here."""
    X = wide_scale_frame
    y = (X["risk_score"] > X["risk_score"].median()).astype(int).to_numpy()
    model = LogisticRegression(max_iter=2000).fit(X, y)

    from bdp_model_gate.structured.security import AdversarialRobustnessCheck

    result = AdversarialRobustnessCheck().run(
        StructuredGateContext(
            model=model,
            X=X,
            y_true=y,
            y_pred=model.predict_proba(X)[:, 1],
            task="binary",
        )
    )[0]
    assert result.metadata["flip_rate"] <= 1.0
    assert result.flag in {"OK", "ROBUSTNESS_RISK"}


def test_tiny_groups_are_reported_not_scored(wide_scale_frame, tiny_group_protected):
    """A three-row segment must not produce a wild ratio that reads as a
    finding."""
    X = wide_scale_frame
    predictions = X["sum_insured_ngn"].to_numpy()

    from bdp_model_gate.structured.regression_fairness import GroupMeanGapCheck

    results = GroupMeanGapCheck().run(
        StructuredGateContext(
            model=LinearRegression().fit(X, predictions),
            X=X,
            y_true=predictions,
            y_pred=predictions,
            protected_df=tiny_group_protected,
            task="regression",
        )
    )
    assert results[0].flag == "NOT_APPLICABLE"
    assert "min_group_size" in results[0].detail


def test_severe_imbalance_does_not_break_the_suite(wide_scale_frame, severe_imbalance):
    """A 99.5/0.5 split — where accuracy flatters and some metrics degenerate."""
    X = wide_scale_frame.iloc[:1000] if len(wide_scale_frame) >= 1000 else wide_scale_frame
    y = severe_imbalance[: len(X)]
    if y.sum() == 0:  # keep at least one positive
        y = y.copy()
        y[0] = 1

    model = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X, y)
    config = GateConfig(performance=PerformanceConfig(metric="average_precision", min_score=0.0))
    report = ModelGate(checks=default_structured_checks(config, include_plugins=False)).run(
        StructuredGateContext(
            model=model,
            X=X,
            y_true=y,
            y_pred=model.predict_proba(X)[:, 1],
            model_card=MODEL_CARD,
            task="binary",
        )
    )
    _assert_healthy(report, "imbalance")
