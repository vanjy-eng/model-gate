"""Validation methodology — the checks that ask whether the evidence is sound.

Known-answer style throughout. Each case is built so the correct verdict is
derivable on paper: a column that *is* the target must be flagged, a column of
noise must not, a frame passed as its own training set must show 100% overlap.

The motivating hole is blunt. Before 0.5.2 nothing stopped a user passing the
training set as the validation set: the gate reported a superb score and
`PASS`, and every fairness figure beside it was measured on data the model had
memorised. The first test here is that exact scenario.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bdp_model_gate import ModelGate, StructuredGateContext
from bdp_model_gate.config import ComplianceConfig, ValidationConfig
from bdp_model_gate.stats import average_ranks, pearson_r, rank_auc
from bdp_model_gate.structured.validation_checks import (
    FeatureContractCheck,
    FeatureDriftCheck,
    LeakageCheck,
    SplitOverlapCheck,
    ValidationStrategyCheck,
)


def _frame(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "income": rng.lognormal(11.0, 0.5, n),
            "age": rng.integers(21, 70, n).astype(float),
            "tenure_months": rng.integers(0, 200, n).astype(float),
        }
    )
    logit = -3.0 + 2.0e-5 * X["income"].to_numpy() + 0.02 * X["tenure_months"].to_numpy()
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    y_pred = 1 / (1 + np.exp(-logit))
    return X, y, y_pred


def _context(X, y, y_pred, **kwargs):
    return StructuredGateContext(
        model=None,
        X=X,
        y_true=y,
        y_pred=y_pred,
        predict_fn=lambda frame: np.zeros(len(frame)),
        **kwargs,
    )


# --------------------------------------------------------------------------
# The statistics, against values known by hand
# --------------------------------------------------------------------------


def test_rank_auc_matches_hand_worked_cases():
    # Perfectly separated: every positive scores above every negative.
    assert rank_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    # Perfectly inverted.
    assert rank_auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == 0.0
    # A constant score cannot separate anything: every pair is a tie, so 0.5.
    assert rank_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5
    # One swapped pair out of 2x2 = 4 comparisons.
    assert rank_auc([0, 1, 0, 1], [0.1, 0.2, 0.3, 0.4]) == 0.75


def test_rank_auc_is_undefined_with_one_class():
    assert np.isnan(rank_auc([1, 1, 1], [0.1, 0.5, 0.9]))
    assert np.isnan(rank_auc([0, 0, 0], [0.1, 0.5, 0.9]))


def test_rank_auc_agrees_with_scikit_learn():
    """The implementation exists so leakage detection survives a core install.
    That is only worth doing if it gives the same answer."""
    roc_auc_score = pytest.importorskip("sklearn.metrics").roc_auc_score
    rng = np.random.default_rng(11)
    for _ in range(5):
        y = rng.integers(0, 2, 400)
        # Rounded on purpose: ties are where a naive rank implementation
        # diverges from a correct one.
        scores = np.round(rng.normal(size=400), 1)
        assert rank_auc(y, scores) == pytest.approx(roc_auc_score(y, scores))


def test_average_ranks_shares_ranks_between_ties():
    # Ranks 2 and 3 are tied, so both take 2.5.
    assert list(average_ranks(np.array([10.0, 20.0, 20.0, 30.0]))) == [1.0, 2.5, 2.5, 4.0]
    # All tied: every rank is the mean of 1..4.
    assert list(average_ranks(np.array([7.0, 7.0, 7.0, 7.0]))) == [2.5] * 4


def test_average_ranks_do_not_depend_on_row_order():
    rng = np.random.default_rng(3)
    values = np.round(rng.normal(size=200), 1)
    order = rng.permutation(len(values))
    assert np.allclose(average_ranks(values)[order], average_ranks(values[order]))


def test_pearson_r_is_zero_for_a_constant_column():
    """np.corrcoef returns NaN with a warning; 0.0 is the honest answer and
    saves every caller a special case."""
    assert pearson_r([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == 0.0


# --------------------------------------------------------------------------
# LeakageCheck
# --------------------------------------------------------------------------


def test_a_copy_of_the_target_is_flagged():
    """The whole point. A column that *is* the outcome must not slip through."""
    X, y, y_pred = _frame()
    X = X.assign(settled_amount=y * 1000.0 + 5.0)  # populated after the outcome
    results = LeakageCheck().run(_context(X, y, y_pred))

    assert [r.flag for r in results] == ["LEAKAGE_RISK"]
    assert results[0].metadata["feature"] == "settled_amount"
    # A perfect copy separates the classes completely: AUC 1.0 -> power 1.0.
    assert results[0].metadata["feature_power"] == pytest.approx(1.0)


def test_ordinary_features_are_not_flagged():
    X, y, y_pred = _frame()
    (result,) = LeakageCheck().run(_context(X, y, y_pred))
    assert result.flag == "OK"
    assert result.metadata["n_features_scored"] == 3


def test_a_weak_model_does_not_make_every_feature_a_leak():
    """The guard that keeps this check usable.

    Against a near-random model, any feature reaches ~100% of its power. Ratio
    alone would flag the lot; the absolute floor is what stops it.
    """
    rng = np.random.default_rng(5)
    n = 300
    X = pd.DataFrame({"noise_a": rng.normal(size=n), "noise_b": rng.normal(size=n)})
    y = rng.integers(0, 2, n)
    y_pred = np.clip(0.5 + rng.normal(0, 0.01, n), 0, 1)  # barely better than a coin

    (result,) = LeakageCheck().run(_context(X, y, y_pred))
    assert result.flag == "OK"

    # Drop the floor and the same data flags — proof the floor is what saved it,
    # not the absence of correlation.
    permissive = LeakageCheck(ValidationConfig(leakage_min_power=0.0, leakage_ratio=0.5))
    assert any(r.flag == "LEAKAGE_RISK" for r in permissive.run(_context(X, y, y_pred)))


def test_leakage_is_found_in_a_regression_target_too():
    rng = np.random.default_rng(7)
    n = 300
    premium = rng.lognormal(10.0, 0.4, n)
    X = pd.DataFrame(
        {
            "vehicle_age": rng.integers(0, 20, n).astype(float),
            "technical_premium": premium * 0.98 + rng.normal(0, 1.0, n),  # the answer
        }
    )
    results = LeakageCheck().run(
        _context(X, premium, premium * 1.02 + rng.normal(0, 500, n), task="regression")
    )
    flagged = [r.metadata["feature"] for r in results if r.flag == "LEAKAGE_RISK"]
    assert flagged == ["technical_premium"]


def test_a_categorical_copy_of_the_target_is_flagged():
    """Leaks are not always numeric — a status string joined back in is the
    classic churn-model version."""
    X, y, y_pred = _frame()
    X = X.assign(closed_reason=np.where(y == 1, "accepted", "declined"))
    results = LeakageCheck().run(_context(X, y, y_pred))
    assert [r.metadata["feature"] for r in results if r.flag == "LEAKAGE_RISK"] == ["closed_reason"]


def test_an_identifier_column_is_not_scored():
    """A near-unique string column puts every row in its own group, which
    drives the correlation ratio to 1 by arithmetic rather than association."""
    X, y, y_pred = _frame()
    X = X.assign(policy_ref=[f"POL-{i:05d}" for i in range(len(X))])
    (result,) = LeakageCheck().run(_context(X, y, y_pred))
    assert result.flag == "OK"


def test_leakage_reports_features_worst_first():
    X, y, y_pred = _frame()
    X = X.assign(leak_partial=y + np.random.default_rng(1).normal(0, 0.30, len(y)), leak_exact=y)
    results = LeakageCheck().run(_context(X, y, y_pred))
    powers = [r.metadata["feature_power"] for r in results]
    assert powers == sorted(powers, reverse=True)
    assert results[0].metadata["feature"] == "leak_exact"


# --------------------------------------------------------------------------
# SplitOverlapCheck — the motivating hole
# --------------------------------------------------------------------------


def _overlap(results):
    return next(r for r in results if r.metadata["check"] == "overlap_with_training")


def _duplicates(results):
    return next(r for r in results if r.metadata["check"] == "duplicates_within_validation")


def test_the_training_set_passed_as_validation_is_caught():
    """Before 0.5.2 this scored AUC 0.99 and PASSed."""
    X, y, y_pred = _frame()
    results = SplitOverlapCheck().run(_context(X, y, y_pred, X_train=X))
    overlap = _overlap(results)
    assert overlap.flag == "SPLIT_OVERLAP_RISK"
    assert overlap.metadata["overlap_fraction"] == 1.0
    assert overlap.metadata["n_overlapping"] == len(X)


def test_a_clean_split_passes():
    X, y, y_pred = _frame(n=400)
    train, validation = X.iloc[:200], X.iloc[200:].reset_index(drop=True)
    results = SplitOverlapCheck().run(_context(validation, y[200:], y_pred[200:], X_train=train))
    assert _overlap(results).flag == "OK"
    assert _overlap(results).metadata["n_overlapping"] == 0


def test_overlap_is_detected_by_content_not_index():
    """A reset index must not hide a shared row, and a shuffle must not
    invent one."""
    X, y, y_pred = _frame(n=400)
    train = X.iloc[:200]
    validation = pd.concat([X.iloc[150:200], X.iloc[200:350]]).reset_index(drop=True)
    results = SplitOverlapCheck().run(_context(validation, y[:200], y_pred[:200], X_train=train))
    assert _overlap(results).metadata["n_overlapping"] == 50


def test_duplicates_within_validation_are_reported_separately():
    """A different defect from a broken split, with a different fix: it
    inflates every metric by weighting one observation twice."""
    X, y, y_pred = _frame(n=100)
    doubled = pd.concat([X, X.iloc[:30]]).reset_index(drop=True)
    y2 = np.concatenate([y, y[:30]])
    p2 = np.concatenate([y_pred, y_pred[:30]])

    results = SplitOverlapCheck().run(_context(doubled, y2, p2))
    duplicates = _duplicates(results)
    assert duplicates.flag == "DUPLICATE_ROWS_RISK"
    assert duplicates.metadata["n_duplicates"] == 30
    # No X_train, so the overlap half of the check skips rather than guessing.
    assert _overlap(results).flag == "NOT_APPLICABLE"


def test_overlap_skips_when_no_columns_are_shared():
    X, y, y_pred = _frame()
    unrelated = pd.DataFrame({"totally": [1.0, 2.0], "different": [3.0, 4.0]})
    results = SplitOverlapCheck().run(_context(X, y, y_pred, X_train=unrelated))
    assert _overlap(results).flag == "NOT_APPLICABLE"
    assert "share no column names" in _overlap(results).detail


# --------------------------------------------------------------------------
# ValidationStrategyCheck
# --------------------------------------------------------------------------


def _card(**extra):
    return {"use_case": "credit_scoring", **extra}


def test_a_random_split_is_flagged_for_a_high_risk_use_case():
    X, y, y_pred = _frame()
    (result,) = ValidationStrategyCheck().run(
        _context(X, y, y_pred, model_card=_card(validation_strategy="random_split"))
    )
    assert result.flag == "VALIDATION_STRATEGY_RISK"
    assert result.metadata["is_high_risk"] is True
    assert result.metadata["is_out_of_time"] is False


def test_an_out_of_time_split_passes_for_a_high_risk_use_case():
    X, y, y_pred = _frame()
    (result,) = ValidationStrategyCheck().run(
        _context(X, y, y_pred, model_card=_card(validation_strategy="out_of_time"))
    )
    assert result.flag == "OK"
    assert result.metadata["is_out_of_time"] is True


def test_a_random_split_is_fine_for_a_low_risk_use_case():
    X, y, y_pred = _frame()
    (result,) = ValidationStrategyCheck().run(
        _context(
            X,
            y,
            y_pred,
            model_card={"use_case": "marketing_propensity", "validation_strategy": "random_split"},
        )
    )
    assert result.flag == "OK"


def test_a_missing_strategy_is_flagged_and_lists_the_options():
    X, y, y_pred = _frame()
    (result,) = ValidationStrategyCheck().run(_context(X, y, y_pred, model_card=_card()))
    assert result.flag == "VALIDATION_STRATEGY_RISK"
    assert "out_of_time" in result.detail and "random_split" in result.detail


def test_an_unrecognised_strategy_is_refused_rather_than_accepted():
    """'holdout' and 'out_of_time' are different claims, and only one of them
    answers the question."""
    X, y, y_pred = _frame()
    (result,) = ValidationStrategyCheck().run(
        _context(X, y, y_pred, model_card=_card(validation_strategy="holdout"))
    )
    assert result.flag == "VALIDATION_STRATEGY_RISK"
    assert result.metadata["check"] == "recognised"


def test_the_out_of_time_requirement_can_be_switched_off():
    X, y, y_pred = _frame()
    check = ValidationStrategyCheck(
        ValidationConfig(require_out_of_time_for_high_risk=False), ComplianceConfig()
    )
    (result,) = check.run(
        _context(X, y, y_pred, model_card=_card(validation_strategy="random_split"))
    )
    assert result.flag == "OK"


def test_no_model_card_skips_with_a_reason():
    X, y, y_pred = _frame()
    (result,) = ValidationStrategyCheck().run(_context(X, y, y_pred))
    assert result.flag == "NOT_APPLICABLE"
    assert "model_card" in result.detail


# --------------------------------------------------------------------------
# FeatureContractCheck
# --------------------------------------------------------------------------


def test_reordered_columns_are_flagged():
    """Harmless for an estimator that reads names, silently wrong for anything
    handed a positional array."""
    X, y, y_pred = _frame()
    shuffled = X[["tenure_months", "income", "age"]]
    (result,) = FeatureContractCheck().run(
        _context(shuffled, y, y_pred, expected_features=list(X.columns))
    )
    assert result.flag == "FEATURE_ORDER_RISK"
    assert result.metadata["expected_order"] == list(X.columns)
    assert result.metadata["actual_order"] == list(shuffled.columns)


def test_matching_columns_in_order_pass():
    X, y, y_pred = _frame()
    (result,) = FeatureContractCheck().run(
        _context(X, y, y_pred, expected_features=list(X.columns))
    )
    assert result.flag == "OK"
    assert result.metadata["expected_order"] is None  # nothing to show


def test_a_missing_or_unexpected_column_is_flagged():
    X, y, y_pred = _frame()
    renamed = X.rename(columns={"age": "applicant_age"})
    (result,) = FeatureContractCheck().run(
        _context(renamed, y, y_pred, expected_features=list(X.columns))
    )
    assert result.flag == "FEATURE_CONTRACT_RISK"
    assert result.metadata["missing"] == ["age"]
    assert result.metadata["unexpected"] == ["applicant_age"]


def test_the_expected_features_come_from_the_model_when_it_knows():
    sklearn = pytest.importorskip("sklearn.linear_model")
    X, y, _ = _frame()
    model = sklearn.LogisticRegression(max_iter=1000).fit(X, y)
    context = StructuredGateContext(
        model=model, X=X[["age", "income", "tenure_months"]], y_true=y, y_pred=model.predict(X)
    )
    (result,) = FeatureContractCheck().run(context)
    assert result.metadata["source"] == "model.feature_names_in_"
    assert result.flag == "FEATURE_ORDER_RISK"


def test_x_train_is_the_last_resort_for_the_expected_columns():
    X, y, y_pred = _frame()
    (result,) = FeatureContractCheck().run(_context(X, y, y_pred, X_train=X))
    assert result.metadata["source"] == "X_train.columns"
    assert result.flag == "OK"


def test_an_unknowable_schema_skips_rather_than_guessing():
    X, y, y_pred = _frame()
    (result,) = FeatureContractCheck().run(_context(X, y, y_pred))
    assert result.flag == "NOT_APPLICABLE"
    assert "expected_features" in result.detail


# --------------------------------------------------------------------------
# FeatureDriftCheck
# --------------------------------------------------------------------------


def test_a_shifted_feature_is_flagged():
    X, y, y_pred = _frame()
    drifted = X.assign(income=X["income"] * 2.0)
    results = FeatureDriftCheck().run(_context(drifted, y, y_pred, X_train=X))
    assert [r.metadata["feature"] for r in results] == ["income"]
    assert results[0].flag == "DRIFT_RISK"


def test_drift_is_non_blocking():
    """An out-of-time holdout *should* differ a little. A gate that hard-fails
    on every seasonal shift gets switched off."""
    assert FeatureDriftCheck().blocking is False


def test_no_drift_between_two_halves_of_one_sample():
    X, y, y_pred = _frame(n=800)
    results = FeatureDriftCheck().run(
        _context(X.iloc[400:].reset_index(drop=True), y[400:], y_pred[400:], X_train=X.iloc[:400])
    )
    assert [r.flag for r in results] == ["OK"]
    assert results[0].metadata["n_features_scored"] == 3


def test_categorical_drift_is_measured_by_total_variation():
    """Two frames sharing no level at all: TVD is exactly 1."""
    n = 100
    train = pd.DataFrame({"region": ["Lagos"] * n})
    live = pd.DataFrame({"region": ["Kano"] * n})
    y = np.tile([0, 1], n // 2)
    results = FeatureDriftCheck().run(_context(live, y, y.astype(float), X_train=train))
    assert results[0].flag == "DRIFT_RISK"
    assert results[0].metadata["shift"] == pytest.approx(1.0)


def test_a_feature_constant_in_training_is_skipped_not_divided_by_zero():
    X, y, y_pred = _frame()
    train = X.assign(flag=1.0)
    live = X.assign(flag=2.0)
    results = FeatureDriftCheck().run(_context(live, y, y_pred, X_train=train))
    # No scale to measure a shift against, so `flag` is skipped — and the
    # other three features are still scored.
    assert all(r.metadata.get("feature") != "flag" for r in results)
    assert results[0].metadata["n_features_scored"] == 3


def test_drift_skips_without_a_training_frame():
    X, y, y_pred = _frame()
    (result,) = FeatureDriftCheck().run(_context(X, y, y_pred))
    assert result.flag == "NOT_APPLICABLE"
    assert "X_train" in result.detail


# --------------------------------------------------------------------------
# In the suite
# --------------------------------------------------------------------------


def test_validation_findings_block_and_lead_the_report():
    """A validation finding says the evidence is unsound, which is a prior
    question to whether the model is any good — so it blocks, and a reader
    meets it first."""
    X, y, y_pred = _frame()
    X = X.assign(settled_amount=y * 1000.0)
    report = ModelGate().run(_context(X, y, y_pred, X_train=X))

    assert report.gate_status == "BLOCKED"
    validation = report.by_category("validation")
    assert {r.check_name for r in validation} == {
        "target_leakage",
        "split_overlap",
        "validation_strategy",
        "feature_contract",
        "feature_drift",
    }
    assert "validation:" in report.summary()
    assert report.summary().index("validation:") < report.summary().index("fairness:")


def test_a_sound_setup_reports_no_validation_findings():
    X, y, y_pred = _frame(n=600)
    train, validation = X.iloc[:300], X.iloc[300:].reset_index(drop=True)
    report = ModelGate().run(
        _context(
            validation,
            y[300:],
            y_pred[300:],
            X_train=train,
            model_card={"use_case": "credit_scoring", "validation_strategy": "out_of_time"},
        )
    )
    assert [r for r in report.by_category("validation") if not r.is_ok] == []
