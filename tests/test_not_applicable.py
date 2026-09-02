"""Every skip path, asserted on its reason.

`NOT_APPLICABLE` is treated as OK, so these are the branches where a check
silently does nothing. A check that skips for the wrong reason — or skips when
it should have run — passes every other test in this suite.

So each path is exercised and the reason string checked, which also means the
messages stay useful: they are the only thing a reviewer has to explain why a
governance report is missing a finding.
"""

import numpy as np
import pandas as pd
import pytest

from bdp_model_gate import ModelGate, PerformanceConfig, StructuredGateContext
from bdp_model_gate.structured import default_structured_checks
from bdp_model_gate.structured.actuarial_checks import (
    ActualVsExpectedCheck,
    DislocationCheck,
    MonotonicityCheck,
)
from bdp_model_gate.structured.compliance import ComplianceMappingCheck
from bdp_model_gate.structured.fairness import (
    CounterfactualFlipCheck,
    DisparateImpactCheck,
    ProxyCorrelationCheck,
    ShapSubgroupCheck,
)
from bdp_model_gate.structured.performance import PerformanceThresholdCheck
from bdp_model_gate.structured.regression_fairness import (
    CalibrationParityCheck,
    ErrorParityCheck,
    GroupMeanGapCheck,
    LossRatioParityCheck,
)
from bdp_model_gate.structured.security import (
    AdversarialRobustnessCheck,
    PIILeakageCheck,
    PromptInjectionCheck,
)


class Simple:
    def predict(self, X):
        return (X["x"].to_numpy() > 0.5).astype(int)

    def predict_proba(self, X):
        p = X["x"].to_numpy().clip(0.01, 0.99)
        return np.column_stack([1 - p, p])


@pytest.fixture
def base():
    n = 120
    X = pd.DataFrame({"x": np.linspace(0, 1, n)})
    y = (X["x"] > 0.5).astype(int).to_numpy()
    return X, y


def _ctx(X, y, **kw):
    kw.setdefault("model", Simple())
    kw.setdefault("task", "binary")
    return StructuredGateContext(X=X, y_true=y, y_pred=X["x"].to_numpy(), **kw)


def _only(results):
    assert len(results) == 1, f"expected one result, got {[r.flag for r in results]}"
    return results[0]


# --- missing protected_df ----------------------------------------------------


@pytest.mark.parametrize(
    "check",
    [
        ProxyCorrelationCheck(),
        DisparateImpactCheck(),
        ShapSubgroupCheck(),
        CounterfactualFlipCheck(),
        GroupMeanGapCheck(),
        ErrorParityCheck(),
        CalibrationParityCheck(),
        LossRatioParityCheck(),
    ],
    ids=lambda c: c.name,
)
def test_every_fairness_check_skips_without_protected_df(base, check):
    X, y = base
    task = "regression" if "regression" in type(check).__module__ else "binary"
    result = _only(check.run(_ctx(X, y, task=task, protected_df=None)))
    assert result.flag == "NOT_APPLICABLE"
    assert "protected_df" in result.detail


# --- missing task-specific inputs --------------------------------------------


def test_loss_ratio_skips_without_expected_loss(base):
    X, y = base
    protected = pd.DataFrame({"g": np.resize(["a", "b"], len(X))})
    result = _only(
        LossRatioParityCheck().run(
            _ctx(X, y.astype(float), task="regression", protected_df=protected)
        )
    )
    assert result.flag == "NOT_APPLICABLE"
    assert "expected_loss" in result.detail


@pytest.mark.parametrize(
    "check", [ErrorParityCheck(), CalibrationParityCheck()], ids=lambda c: c.name
)
def test_error_and_calibration_skip_without_y_true(base, check):
    X, _ = base
    protected = pd.DataFrame({"g": np.resize(["a", "b"], len(X))})
    context = StructuredGateContext(
        model=Simple(),
        X=X,
        y_true=None,
        y_pred=X["x"].to_numpy(),
        protected_df=protected,
        task="regression",
    )
    result = _only(check.run(context))
    assert result.flag == "NOT_APPLICABLE"
    assert "y_true" in result.detail


def test_dislocation_skips_without_a_baseline(base):
    """The one input with no possible stand-in: dislocation is a change
    *from* something, and the book mean is a different question."""
    X, y = base
    result = _only(DislocationCheck().run(_ctx(X, y.astype(float), task="regression")))
    assert result.flag == "NOT_APPLICABLE"
    assert "baseline_pred" in result.detail


def test_monotonicity_skips_until_a_constraint_is_declared(base):
    """Nothing to check is not the same as nothing wrong, and the reason has
    to say which — the constraint is a claim about the product."""
    X, y = base
    result = _only(MonotonicityCheck().run(_ctx(X, y.astype(float), task="regression")))
    assert result.flag == "NOT_APPLICABLE"
    assert "monotonic_features" in result.detail


def test_actual_vs_expected_skips_without_realised_outcomes(base):
    X, _ = base
    result = _only(
        ActualVsExpectedCheck().run(
            StructuredGateContext(
                model=Simple(),
                X=X,
                y_true=None,
                y_pred=X["x"].to_numpy(),
                task="regression",
            )
        )
    )
    assert result.flag == "NOT_APPLICABLE"
    assert "y_true" in result.detail


def test_compliance_skips_without_a_model_card(base):
    X, y = base
    result = _only(ComplianceMappingCheck().run(_ctx(X, y, model_card=None)))
    assert result.flag == "NOT_APPLICABLE"
    assert "model_card" in result.detail


def test_prompt_injection_skips_without_a_generative_component(base):
    X, y = base
    result = _only(PromptInjectionCheck().run(_ctx(X, y, generate_fn=None)))
    assert result.flag == "NOT_APPLICABLE"
    assert "generative" in result.detail


def test_performance_skips_without_any_benchmark_data(base):
    X, _ = base
    context = StructuredGateContext(
        model=Simple(),
        X=X,
        y_true=None,
        y_pred=None,
        task="binary",
    )
    result = _only(PerformanceThresholdCheck().run(context))
    assert result.flag == "NOT_APPLICABLE"
    assert "benchmark" in result.detail


def test_counterfactual_skips_without_probabilities(base):
    X, y = base
    protected = pd.DataFrame({"x": np.resize([0.0, 1.0], len(X))})
    context = StructuredGateContext(
        X=X,
        y_true=y,
        y_pred=X["x"].to_numpy(),
        protected_df=protected,
        predict_fn=lambda df: (df["x"].to_numpy() > 0.5).astype(int),
        task="binary",
    )
    result = _only(CounterfactualFlipCheck().run(context))
    assert result.flag == "NOT_APPLICABLE"
    assert "predict_proba_fn" in result.detail


def test_counterfactual_skips_when_no_attribute_is_a_model_input(base):
    """There is nothing to flip — a genuinely different reason from the above,
    and the message must say which."""
    X, y = base
    protected = pd.DataFrame({"region": np.resize(["a", "b"], len(X))})
    result = _only(CounterfactualFlipCheck().run(_ctx(X, y, protected_df=protected)))
    assert result.flag == "NOT_APPLICABLE"
    assert "model inputs" in result.detail


def test_adversarial_skips_without_numeric_features():
    n = 60
    X = pd.DataFrame({"channel": np.resize(["app", "ussd"], n)})
    y = np.resize([0, 1], n)

    class Categorical:
        def predict(self, frame):
            return (frame["channel"].to_numpy() == "app").astype(int)

    result = _only(
        AdversarialRobustnessCheck().run(
            StructuredGateContext(model=Categorical(), X=X, y_true=y, y_pred=y, task="binary")
        )
    )
    assert result.flag == "NOT_APPLICABLE"
    assert "numeric" in result.detail


def test_pii_reports_ok_not_applicable_when_there_are_no_string_columns(base):
    """This one deliberately returns OK rather than NOT_APPLICABLE: having no
    string columns means there is genuinely no PII to leak, which is a pass."""
    X, y = base
    result = _only(PIILeakageCheck().run(_ctx(X, y)))
    assert result.flag == "OK"
    assert "no string columns" in result.detail


# --- task routing ------------------------------------------------------------


@pytest.mark.parametrize(
    "task,expected_skips",
    [
        (
            "binary",
            {
                "group_mean_gap",
                "error_parity",
                "calibration_parity",
                "loss_ratio_parity",
                # 0.5.3: A/E, the Gini and dislocation are all statements
                # about a continuous amount, and mean nothing for a class.
                "actual_vs_expected",
                "risk_discrimination",
                "prediction_dislocation",
            },
        ),
        ("regression", {"disparate_impact", "counterfactual_flip"}),
    ],
)
def test_checks_outside_the_task_are_skipped_with_a_reason(base, task, expected_skips):
    X, y = base
    protected = pd.DataFrame({"g": np.resize(["a", "b"], len(X))})
    config = (
        PerformanceConfig(metric="accuracy", min_score=0.0)
        if task == "binary"
        else PerformanceConfig(metric="mae", max_error=1e12)
    )
    from bdp_model_gate import GateConfig

    report = ModelGate(
        checks=default_structured_checks(GateConfig(performance=config), include_plugins=False)
    ).run(
        _ctx(X, y.astype(float) if task == "regression" else y, task=task, protected_df=protected)
    )

    by_name = {r.check_name: r for r in report.results}
    for name in expected_skips:
        assert by_name[name].flag == "NOT_APPLICABLE", name
        assert task in by_name[name].detail, f"{name} does not say which task it skipped for"


def test_shap_skips_multiclass_without_a_favourable_class(base):
    """Genuinely undecidable: no ordering, no favourable class, more than two
    columns in the SHAP array."""
    pytest.importorskip("shap")
    from sklearn.ensemble import RandomForestClassifier

    n = 150
    rng = np.random.default_rng(2)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = rng.choice(["x", "y", "z"], n)
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)

    result = _only(
        ShapSubgroupCheck().run(
            StructuredGateContext(
                model=model,
                X=X,
                y_true=y,
                y_pred=model.predict(X),
                protected_df=pd.DataFrame({"g": rng.choice(["p", "q"], n)}),
                task="multiclass",  # no class_order, no favourable_classes
            )
        )
    )
    assert result.flag == "NOT_APPLICABLE"
    assert "favourable" in result.detail
