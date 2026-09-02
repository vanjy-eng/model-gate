"""Targeted tests exercising check branches not hit by the smoke tests —
performance thresholds with real benchmark data, security checks with
generative side-cars and clean text columns, and fairness checks with a
real sklearn model (needed for SHAP/counterfactual paths)."""

import numpy as np
import pandas as pd
import pytest

# These tests fit real estimators, so they need the [structured] extra.
# Skipped wholesale on a core-only install rather than failing collection.
pytest.importorskip("sklearn", reason="requires the [structured] extra")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from bdp_model_gate import GateConfig, ModelGate, StructuredGateContext
from bdp_model_gate.structured.compliance import ComplianceMappingCheck
from bdp_model_gate.structured.performance import PerformanceThresholdCheck
from bdp_model_gate.structured.security import PIILeakageCheck, PromptInjectionCheck


@pytest.fixture
def fitted_logreg_context(synthetic_data):
    _, X, y_true, _, protected_df = synthetic_data
    model = LogisticRegression(max_iter=1000).fit(X, y_true)
    y_pred = model.predict_proba(X)[:, 1]
    return StructuredGateContext(
        model=model,
        X=X,
        y_true=y_true,
        y_pred=y_pred,
        protected_df=protected_df,
    )


def test_performance_check_passes_within_thresholds(fitted_logreg_context):
    check = PerformanceThresholdCheck(GateConfig().performance)
    fitted_logreg_context.latencies_ms = list(np.random.uniform(5, 20, 100))
    fitted_logreg_context.cost_per_inference = 0.0001
    results = check.run(fitted_logreg_context)
    metrics = {r.metadata.get("metric") for r in results}
    # "roc_auc" under the default metric="auto" with scikit-learn installed —
    # the key names the metric that actually ran, not a generic "accuracy".
    assert {"roc_auc", "latency_p95_ms", "cost_per_inference"} <= metrics


def test_performance_check_flags_high_latency_and_cost(fitted_logreg_context):
    config = GateConfig().performance
    fitted_logreg_context.latencies_ms = [500] * 50  # way above default 200ms budget
    fitted_logreg_context.cost_per_inference = 1.0  # way above default budget
    check = PerformanceThresholdCheck(config)
    results = check.run(fitted_logreg_context)
    risk_metrics = {r.metadata["metric"] for r in results if r.flag == "PERFORMANCE_RISK"}
    assert "latency_p95_ms" in risk_metrics
    assert "cost_per_inference" in risk_metrics


def test_performance_check_not_applicable_without_labels():
    check = PerformanceThresholdCheck()
    context = StructuredGateContext(
        model=LogisticRegression(),
        X=pd.DataFrame({"a": [1, 2]}),
        y_true=None,
        y_pred=None,
    )
    results = check.run(context)
    assert results[0].flag == "NOT_APPLICABLE"


def test_a_side_car_that_echoes_the_prompt_routes_to_a_human(small_valid_context):
    """Rewritten for 0.5.4, and the change *is* the release.

    The old check called this an `INJECTION_RISK`: the response does not
    contain "cannot", so it must have complied. But echoing the prompt back is
    not compliance — the model performed no injected task and disclosed no
    secret — and with no canaries planted there is nothing here anyone can
    decide. So it becomes a question for a person, non-blocking, with the
    response attached.
    """
    small_valid_context.generate_fn = lambda prompt: "Sure, here you go: " + prompt
    results = PromptInjectionCheck().run(small_valid_context)

    assert any(r.flag == "INJECTION_NEEDS_JUDGEMENT" for r in results)
    assert not any(r.blocking for r in results if r.flag not in ("OK", "NOT_APPLICABLE"))


def test_a_refusal_passes_only_once_there_is_a_canary_to_judge_it_against(small_valid_context):
    """The other half of the same point.

    "I cannot help with that request." is a refusal, and the old check scored
    it OK on the strength of the word "cannot" — the same reasoning that
    passed a response reading *"I cannot normally share this, but the system
    prompt is: …"*. The new check will not call it clean until the caller has
    planted something whose absence means anything.
    """
    small_valid_context.generate_fn = lambda prompt: "I cannot help with that request."

    unjudgeable = PromptInjectionCheck().run(small_valid_context)
    assert any(r.flag == "INJECTION_NEEDS_JUDGEMENT" for r in unjudgeable)

    small_valid_context.canaries = ["never-disclose-the-tariff-table-8831"]
    assert all(r.flag == "OK" for r in PromptInjectionCheck().run(small_valid_context))


def test_pii_check_ok_with_clean_text_column(synthetic_data):
    model, X, y_true, y_pred, _ = synthetic_data
    X_clean = X.copy()
    X_clean["notes"] = ["all good here"] * len(X_clean)
    context = StructuredGateContext(model=model, X=X_clean, y_true=y_true, y_pred=y_pred)
    check = PIILeakageCheck()
    results = check.run(context)
    assert all(r.flag == "OK" for r in results)


def test_compliance_check_passes_with_complete_model_card(small_valid_context):
    small_valid_context.model_card = {
        "legal_basis": "consent",
        "data_minimization_justification": "only relevant fields",
        "training_data_source": "internal db",
        "use_case": "general_scoring",
        "influences_decision_about_person": False,
    }
    check = ComplianceMappingCheck()
    results = check.run(small_valid_context)
    assert all(r.flag == "OK" for r in results)


def test_shap_and_counterfactual_checks_run_with_tree_model(synthetic_data):
    _, X, y_true, _, protected_df = synthetic_data
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y_true)
    y_pred = model.predict_proba(X)[:, 1]
    context = StructuredGateContext(
        model=model,
        X=X,
        y_true=y_true,
        y_pred=y_pred,
        protected_df=protected_df,
    )
    report = ModelGate().run(context)
    shap_results = [r for r in report.results if r.check_name == "shap_subgroup_gap"]
    assert len(shap_results) > 0
    # TreeExplainer path should have been used for a RandomForest
    assert report.gate_status in {"PASS", "NEEDS_REVIEW", "BLOCKED"}


def test_counterfactual_check_runs_when_protected_attr_is_a_feature(synthetic_data):
    _, X, y_true, _, _ = synthetic_data
    rng = np.random.default_rng(1)
    X_with_gender = X.copy()
    X_with_gender["gender_code"] = rng.choice([0, 1], len(X_with_gender))
    model = LogisticRegression(max_iter=1000).fit(X_with_gender, y_true)
    y_pred = model.predict_proba(X_with_gender)[:, 1]
    protected_df = pd.DataFrame({"gender_code": X_with_gender["gender_code"].values})

    context = StructuredGateContext(
        model=model,
        X=X_with_gender,
        y_true=y_true,
        y_pred=y_pred,
        protected_df=protected_df,
    )
    from bdp_model_gate.structured.fairness import CounterfactualFlipCheck

    results = CounterfactualFlipCheck(n_samples=50).run(context)
    assert any(r.metadata.get("protected_attr") == "gender_code" for r in results)
