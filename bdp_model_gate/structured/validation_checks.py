"""Is the evidence behind every other number in this report sound?

Nothing currently stops a user passing the **training set** as the validation
set. The gate would report an AUC of 0.99, a clean calibration curve, and
`PASS`. Every fairness figure beside it would be measured on data the model
has memorised.

That is why these are their own category rather than more performance checks.
A performance finding says *the model is not good enough*. A validation
finding says *you do not yet know whether it is*, which is a different
sentence and a strictly prior one — so `validation` is reported first, and
these checks block.

Five questions:

    LeakageCheck             Does one column do what the whole model does?
    SplitOverlapCheck        Has the model already seen these rows?
    ValidationStrategyCheck  Was the holdout separated in time, or at random?
    FeatureContractCheck     Are the columns the ones the model was fitted on,
                             in the order it expects them?
    FeatureDriftCheck        Does validation still look like training?

All five degrade to `NOT_APPLICABLE` with a reason when their inputs are
absent, and all five work on a core install — a validation set that is
secretly the training set is the last thing that should go unchecked because
scikit-learn is missing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._logging import get_logger
from ..classes import favourable_mask, resolve_favourable
from ..config import ComplianceConfig, ValidationConfig
from ..core.base import BaseCheck, CheckResult
from ..metrics import to_class_labels
from ..stats import correlation_ratio, pearson_r, rank_auc
from ..task import MULTICLASS, REGRESSION, resolve_task

logger = get_logger("validation_checks")

#: Below this many usable rows a per-feature statistic is noise, not evidence.
_MIN_ROWS = 20


def _not_applicable(check: BaseCheck, reason: str) -> list[CheckResult]:
    return [CheckResult(check.name, check.category, "NOT_APPLICABLE", reason, check.blocking)]


def _binary_target(context, task: str) -> np.ndarray | None:
    """`y_true` as 0/1 against the favourable outcome, or None for regression."""
    if task == REGRESSION:
        return None
    if task == MULTICLASS:
        class_order = getattr(context, "class_order", None)
        favourable = resolve_favourable(
            getattr(context, "favourable_classes", None), class_order, MULTICLASS
        )
        if not favourable:
            return None
        labels = to_class_labels(context.y_true, class_order)
        return favourable_mask(labels, favourable).astype(float)
    return np.asarray(context.y_true, dtype=float)


def _model_score(context, task: str) -> np.ndarray | None:
    """`y_pred` as a 1-D vector on the same footing as a single feature.

    Not simply `float(y_pred)`: for multiclass, `y_pred` is class labels or a
    probability matrix, and casting either to float raises. It is reduced the
    same way `_binary_target` reduces `y_true`, so the model and each feature
    are scored against the identical target.

    Hard labels are accepted here even though they make a coarse AUC. A
    coarse comparison is still the right one — the question is whether a
    feature rivals the model, and both sides are measured the same way.
    """
    raw = np.asarray(context.y_pred)
    if task != MULTICLASS:
        return raw.astype(float) if raw.ndim == 1 else None

    class_order = getattr(context, "class_order", None)
    favourable = resolve_favourable(
        getattr(context, "favourable_classes", None), class_order, MULTICLASS
    )
    if not favourable:
        return None
    if raw.ndim == 2:
        if class_order is None:
            return None
        by_model_order = sorted(class_order, key=str)
        columns = [by_model_order.index(c) for c in favourable]
        return raw[:, columns].sum(axis=1).astype(float)
    return favourable_mask(to_class_labels(raw, class_order), favourable).astype(float)


def _power(column: pd.Series, target: np.ndarray, binary: bool) -> float | None:
    """How much of the target one column explains, on a 0–1 scale.

    Deliberately the *same* scale for the feature and for the model, because
    the check is a comparison between them. |2·AUC − 1| for classification —
    a Gini, so 0 is a coin flip rather than 0.5 — and |r| for regression. A
    low-cardinality categorical column is scored by its correlation ratio
    against the target instead, which needs no ordering on its levels.

    Returns None when the column carries nothing to measure.
    """
    usable = column.notna().to_numpy()
    if usable.sum() < _MIN_ROWS:
        return None
    values, y = column[usable], target[usable]

    if values.dtype.kind not in "if":
        # eta, not eta²: the square root puts it on the same footing as |r|,
        # both being the fraction of the target's spread the column accounts
        # for rather than the fraction of its variance.
        if values.nunique() > max(20, len(values) // 10):
            return None  # effectively an identifier; every row its own group
        return float(np.sqrt(correlation_ratio(pd.Series(y, index=values.index), values)))

    numeric = values.astype(float)
    if numeric.std() == 0:
        return 0.0
    if binary:
        auc = rank_auc(y, numeric.to_numpy())
        return None if np.isnan(auc) else abs(2 * auc - 1)
    return abs(pearson_r(numeric.to_numpy(), y))


class LeakageCheck(BaseCheck):
    """Does a single column do what the whole model does?

    The classic signature of a leaked target: a field populated *after* the
    outcome was known and joined back into the training data — a settlement
    amount on a claims-frequency model, a `closed_reason` on a churn model, a
    row identifier that happens to be ordered by outcome. The model learns it,
    scores beautifully offline, and collapses in production where the column
    is empty at scoring time.

    Two conditions must both hold before anything is flagged. The column's
    solo power must reach `leakage_ratio` of the model's, **and** be at least
    `leakage_min_power` in absolute terms. Parity alone is not evidence: on a
    model scoring 0.55, a feature scoring 0.54 reaches 98% of it and means
    nothing at all.
    """

    name = "target_leakage"
    category = "validation"
    blocking = True

    def __init__(self, config: ValidationConfig | None = None):
        self.config = config or ValidationConfig()

    def run(self, context) -> list[CheckResult]:
        if context.y_true is None or context.y_pred is None:
            return _not_applicable(
                self,
                "no y_true/y_pred — leakage is measured by comparing each feature's "
                "solo predictive power against the model's own",
            )

        task = resolve_task(context)
        binary = task != REGRESSION
        if binary:
            reduced = _binary_target(context, task)
            if reduced is None:
                return _not_applicable(
                    self,
                    "multiclass leakage needs context.class_order or "
                    "context.favourable_classes, so the target can be reduced to the "
                    "one outcome a feature might be leaking",
                )
            target = reduced
        else:
            target = np.asarray(context.y_true, dtype=float)

        # The model's own power, on the same scale as each feature's, so the
        # comparison does not depend on which metric happens to be configured.
        scores = _model_score(context, task)
        if scores is None:
            return _not_applicable(
                self,
                "could not reduce y_pred to a single score to compare features "
                "against — for multiclass, set context.class_order",
            )
        model_power = _power(pd.Series(scores, index=context.X.index), target, binary)
        if model_power is None or model_power <= 0:
            return _not_applicable(
                self,
                "the model's predictions carry no signal against y_true, so there is "
                "nothing for a feature to be suspiciously close to",
            )

        suspects = []
        for feature in context.X.columns:
            power = _power(context.X[feature], target, binary)
            if power is None:
                continue
            if power >= self.config.leakage_min_power and power >= (
                self.config.leakage_ratio * model_power
            ):
                suspects.append((feature, power))

        if not suspects:
            return [
                CheckResult(
                    self.name,
                    self.category,
                    "OK",
                    f"no single feature approaches the model's own predictive power "
                    f"({model_power:.3f})",
                    self.blocking,
                    metadata={
                        "model_power": round(model_power, 4),
                        "threshold_ratio": self.config.leakage_ratio,
                        "min_power": self.config.leakage_min_power,
                        "n_features_scored": int(context.X.shape[1]),
                    },
                )
            ]

        return [
            CheckResult(
                self.name,
                self.category,
                "LEAKAGE_RISK",
                detail=(
                    f"{feature} alone reaches {power:.3f} of the target on its own, "
                    f"{power / model_power:.0%} of the whole model's {model_power:.3f} "
                    "— check it is populated before the outcome is known, and is "
                    "available at scoring time"
                ),
                blocking=self.blocking,
                metadata={
                    "feature": feature,
                    "feature_power": round(power, 4),
                    "model_power": round(model_power, 4),
                    "power_ratio": round(power / model_power, 4),
                    "threshold_ratio": self.config.leakage_ratio,
                    "min_power": self.config.leakage_min_power,
                },
            )
            for feature, power in sorted(suspects, key=lambda pair: -pair[1])
        ]


class SplitOverlapCheck(BaseCheck):
    """Has the model already seen the rows it is being graded on?

    Reports two things, which have different causes and different fixes:
    rows shared between `X_train` and `X`, and exact duplicates *within* `X`.
    The first is a broken split. The second inflates every metric by weighting
    one observation twice, and survives even a correct split.

    Matching is by row **content**, not index, so a reset index cannot hide an
    overlap and a shuffled frame cannot invent one.
    """

    name = "split_overlap"
    category = "validation"
    blocking = True

    def __init__(self, config: ValidationConfig | None = None):
        self.config = config or ValidationConfig()

    @staticmethod
    def _digests(frame: pd.DataFrame) -> np.ndarray:
        return pd.util.hash_pandas_object(frame, index=False).to_numpy()

    def run(self, context) -> list[CheckResult]:
        validation = context.X
        results = []

        duplicates = int(validation.duplicated().sum())
        duplicate_fraction = duplicates / len(validation)
        results.append(
            CheckResult(
                self.name,
                self.category,
                "OK"
                if duplicate_fraction <= self.config.max_duplicate_fraction
                else "DUPLICATE_ROWS_RISK",
                detail=(
                    f"{duplicates} duplicate row(s) within the validation set "
                    f"({duplicate_fraction:.1%}, max "
                    f"{self.config.max_duplicate_fraction:.0%}) — duplicates weight the "
                    "same observation more than once in every metric"
                ),
                blocking=self.blocking,
                metadata={
                    "check": "duplicates_within_validation",
                    "n_duplicates": duplicates,
                    "duplicate_fraction": round(duplicate_fraction, 4),
                    "threshold": self.config.max_duplicate_fraction,
                },
            )
        )

        train = context.X_train
        if train is None or train.empty:
            results.append(
                CheckResult(
                    self.name,
                    self.category,
                    "NOT_APPLICABLE",
                    "no X_train supplied — supply the training frame and this check "
                    "reports how many validation rows the model has already seen",
                    self.blocking,
                    metadata={"check": "overlap_with_training"},
                )
            )
            return results

        shared_columns = [c for c in validation.columns if c in train.columns]
        if not shared_columns:
            results.append(
                CheckResult(
                    self.name,
                    self.category,
                    "NOT_APPLICABLE",
                    "X and X_train share no column names, so rows cannot be compared "
                    "— see the feature_contract check",
                    self.blocking,
                    metadata={"check": "overlap_with_training"},
                )
            )
            return results

        seen = set(self._digests(train[shared_columns]).tolist())
        validation_digests = self._digests(validation[shared_columns])
        overlapping = int(np.fromiter((d in seen for d in validation_digests), bool).sum())
        fraction = overlapping / len(validation)

        note = ""
        if duplicates and overlapping:
            # Worth saying out loud: with few categorical levels, identical
            # feature vectors recur legitimately and inflate this number.
            note = (
                " — the validation set also contains duplicates, so some of this may be "
                "coincidental repetition rather than a broken split"
            )

        results.append(
            CheckResult(
                self.name,
                self.category,
                "OK" if fraction <= self.config.max_split_overlap else "SPLIT_OVERLAP_RISK",
                detail=(
                    f"{overlapping} of {len(validation)} validation rows ({fraction:.1%}, "
                    f"max {self.config.max_split_overlap:.0%}) also appear in X_train, "
                    f"compared over {len(shared_columns)} shared column(s){note}"
                ),
                blocking=self.blocking,
                metadata={
                    "check": "overlap_with_training",
                    "n_overlapping": overlapping,
                    "overlap_fraction": round(fraction, 4),
                    "threshold": self.config.max_split_overlap,
                    "n_columns_compared": len(shared_columns),
                },
            )
        )
        return results


class ValidationStrategyCheck(BaseCheck):
    """Was the holdout separated in time, or at random?

    A random split asks "can the model predict a policy it has not seen?" An
    out-of-time split asks "can it predict *next quarter*?" — which is the
    only question that matters for a model about to be applied to future
    business, and the one a random split silently answers yes to while the
    real answer is no. Seasonality, inflation and portfolio mix all leak
    backwards through a random split.

    So `model_card["validation_strategy"]` is required, and for the high-risk
    use cases — pricing, underwriting, credit scoring, claims decisioning — it
    has to be out-of-time.

    This is the one check here that cannot be verified from the data. It
    records a claim, which is what a model card is for: an untrue answer here
    is a signed statement, not an oversight.
    """

    name = "validation_strategy"
    category = "validation"
    blocking = True

    def __init__(
        self,
        config: ValidationConfig | None = None,
        compliance: ComplianceConfig | None = None,
    ):
        self.config = config or ValidationConfig()
        self.compliance = compliance or ComplianceConfig()

    def run(self, context) -> list[CheckResult]:
        model_card = context.model_card
        if not model_card:
            return _not_applicable(
                self,
                "no model_card supplied — the validation strategy is a claim the card "
                "records, not something measurable from the data",
            )

        declared = str(model_card.get("validation_strategy") or "").strip().lower()
        use_case = str(model_card.get("use_case") or "").lower()
        high_risk = any(hr in use_case for hr in self.compliance.high_risk_use_cases)

        if not declared:
            return [
                CheckResult(
                    self.name,
                    self.category,
                    "VALIDATION_STRATEGY_RISK",
                    detail=(
                        "model_card.validation_strategy missing — state how the holdout "
                        f"was produced, one of: "
                        f"{', '.join(self.config.accepted_validation_strategies)}"
                    ),
                    blocking=self.blocking,
                    metadata={"check": "declared", "is_high_risk": high_risk},
                )
            ]

        if declared not in self.config.accepted_validation_strategies:
            return [
                CheckResult(
                    self.name,
                    self.category,
                    "VALIDATION_STRATEGY_RISK",
                    detail=(
                        f"model_card.validation_strategy={declared!r} is not a recognised "
                        f"strategy — use one of: "
                        f"{', '.join(self.config.accepted_validation_strategies)}. "
                        "'holdout' and 'out_of_time' are different claims and only one "
                        "of them is checkable"
                    ),
                    blocking=self.blocking,
                    metadata={"check": "recognised", "declared": declared},
                )
            ]

        out_of_time = declared in self.config.out_of_time_strategies
        needs_out_of_time = (
            high_risk and self.config.require_out_of_time_for_high_risk and not out_of_time
        )
        return [
            CheckResult(
                self.name,
                self.category,
                "VALIDATION_STRATEGY_RISK" if needs_out_of_time else "OK",
                detail=(
                    f"{use_case or 'this use case'} is high-risk and was validated with "
                    f"a {declared!r} split — a random split cannot tell you whether the "
                    "model holds up on next period's business, which is what it will be "
                    "applied to"
                    if needs_out_of_time
                    else f"validated with a {declared!r} split"
                    + (" (out-of-time)" if out_of_time else "")
                ),
                blocking=self.blocking,
                metadata={
                    "check": "out_of_time_for_high_risk",
                    "declared": declared,
                    "is_out_of_time": out_of_time,
                    "is_high_risk": high_risk,
                },
            )
        ]


class FeatureContractCheck(BaseCheck):
    """Are these the columns the model was fitted on, in the right order?

    Silent column reordering is a classic production failure: a model handed a
    positional array scores confidently against the wrong features and nothing
    raises. scikit-learn checks names when it is given a DataFrame, but a
    `predict_fn` doing `df.values` does not, and neither does a booster fed a
    numpy array.

    The expected list is taken from the first of these that exists:
    `context.expected_features`, `model.feature_names_in_`, the booster's own
    feature names, then `X_train.columns`.
    """

    name = "feature_contract"
    category = "validation"
    blocking = True

    def __init__(self, config: ValidationConfig | None = None):
        self.config = config or ValidationConfig()

    @staticmethod
    def _expected(context) -> tuple[list[str], str] | None:
        """(feature names, where they came from), or None if unknowable."""
        declared = getattr(context, "expected_features", None)
        if declared is not None:
            return [str(c) for c in declared], "context.expected_features"

        model = context.model
        names = getattr(model, "feature_names_in_", None)  # scikit-learn >= 1.0
        if names is not None:
            return [str(c) for c in names], "model.feature_names_in_"

        names = getattr(model, "feature_name_", None)  # LightGBM sklearn API
        if names:
            return [str(c) for c in names], "model.feature_name_"

        booster = getattr(model, "get_booster", None)
        if callable(booster):
            try:
                names = getattr(booster(), "feature_names", None)
            except Exception:  # pragma: no cover - a booster that refuses to be read
                names = None
            if names:
                return [str(c) for c in names], "the model's booster"

        if context.X_train is not None and not context.X_train.empty:
            return [str(c) for c in context.X_train.columns], "X_train.columns"
        return None

    def run(self, context) -> list[CheckResult]:
        expected = self._expected(context)
        if expected is None:
            return _not_applicable(
                self,
                "cannot tell what the model was fitted on — supply context.X_train, or "
                "context.expected_features for a model whose schema this library "
                "cannot read (a remote endpoint, say)",
            )
        names, source = expected
        actual = [str(c) for c in context.X.columns]

        missing = [c for c in names if c not in actual]
        extra = [c for c in actual if c not in names]
        common_order_differs = [c for c in actual if c in names] != [
            c for c in names if c in actual
        ]

        if missing or extra:
            problems = []
            if missing:
                problems.append(f"missing from X: {', '.join(missing)}")
            if extra:
                problems.append(f"not seen in training: {', '.join(extra)}")
            return [
                CheckResult(
                    self.name,
                    self.category,
                    "FEATURE_CONTRACT_RISK",
                    detail=(
                        f"X does not match the {len(names)} feature(s) recorded in "
                        f"{source} — " + "; ".join(problems)
                    ),
                    blocking=self.blocking,
                    metadata={
                        "check": "feature_set",
                        "source": source,
                        "missing": missing,
                        "unexpected": extra,
                        "n_expected": len(names),
                    },
                )
            ]

        return [
            CheckResult(
                self.name,
                self.category,
                "FEATURE_ORDER_RISK" if common_order_differs else "OK",
                detail=(
                    f"X has the right {len(names)} features but in a different order from "
                    f"{source} — harmless for an estimator that reads column names, "
                    "silently wrong for anything handed a positional array"
                    if common_order_differs
                    else f"X matches the {len(names)} feature(s) recorded in {source}, in order"
                ),
                blocking=self.blocking,
                metadata={
                    "check": "feature_order",
                    "source": source,
                    "n_features": len(names),
                    "expected_order": names if common_order_differs else None,
                    "actual_order": actual if common_order_differs else None,
                },
            )
        ]


class FeatureDriftCheck(BaseCheck):
    """Does the validation set still look like what the model was trained on?

    Train-serve skew: a feature whose distribution has moved between the two
    frames. The model is being graded on a population it was not fitted to,
    which is not necessarily wrong — an out-of-time holdout *should* differ a
    little, and that is the point of one — but it is something a reviewer has
    to see rather than infer.

    Non-blocking for that reason. Drift warrants a look, not a hard stop, and
    a gate that fails the build on every seasonal shift gets switched off.

    Numeric features are compared by standardised mean shift, so the threshold
    is unitless. Categorical features are compared by total variation distance
    between their frequency tables.
    """

    name = "feature_drift"
    category = "validation"
    blocking = False

    def __init__(self, config: ValidationConfig | None = None):
        self.config = config or ValidationConfig()

    def _numeric_shift(self, train: pd.Series, live: pd.Series) -> float | None:
        spread = float(train.std())
        if not np.isfinite(spread) or spread == 0:
            return None  # constant in training; a shift has no scale to be measured against
        return abs(float(live.mean()) - float(train.mean())) / spread

    def _categorical_shift(self, train: pd.Series, live: pd.Series) -> float:
        train_share = train.value_counts(normalize=True)
        live_share = live.value_counts(normalize=True)
        levels = train_share.index.union(live_share.index)
        return float(
            0.5
            * sum(
                abs(float(live_share.get(level, 0.0)) - float(train_share.get(level, 0.0)))
                for level in levels
            )
        )

    def run(self, context) -> list[CheckResult]:
        train = context.X_train
        if train is None or train.empty:
            return _not_applicable(
                self,
                "no X_train supplied — train-serve skew needs the training frame to "
                "compare against",
            )

        shared = [c for c in context.X.columns if c in train.columns]
        if not shared:
            return _not_applicable(
                self, "X and X_train share no column names — see the feature_contract check"
            )

        drifted, scored = [], 0
        for feature in shared:
            reference, live = train[feature].dropna(), context.X[feature].dropna()
            if len(reference) < _MIN_ROWS or len(live) < _MIN_ROWS:
                continue
            numeric = reference.dtype.kind in "if" and live.dtype.kind in "if"
            if numeric:
                shift = self._numeric_shift(reference, live)
                limit = self.config.drift_threshold
                unit = "standardised mean shift"
            else:
                shift = self._categorical_shift(reference, live)
                limit = self.config.categorical_drift_threshold
                unit = "total variation distance"
            if shift is None:
                continue
            scored += 1
            if shift > limit:
                drifted.append((feature, shift, limit, unit))

        if not scored:
            return _not_applicable(
                self,
                f"no shared feature had at least {_MIN_ROWS} usable rows in both frames",
            )
        if not drifted:
            return [
                CheckResult(
                    self.name,
                    self.category,
                    "OK",
                    f"no drift above threshold across {scored} shared feature(s)",
                    self.blocking,
                    metadata={"n_features_scored": scored},
                )
            ]

        return [
            CheckResult(
                self.name,
                self.category,
                "DRIFT_RISK",
                detail=(
                    f"{feature}: {unit} {shift:.3f} between training and validation "
                    f"(max {limit}) — the model is being graded on a different population "
                    "from the one it was fitted to"
                ),
                blocking=self.blocking,
                metadata={
                    "feature": feature,
                    "shift": round(shift, 4),
                    "threshold": limit,
                    "measure": unit,
                    "n_features_scored": scored,
                },
            )
            for feature, shift, limit, unit in sorted(drifted, key=lambda row: -row[1])
        ]


__all__ = [
    "FeatureContractCheck",
    "FeatureDriftCheck",
    "LeakageCheck",
    "SplitOverlapCheck",
    "ValidationStrategyCheck",
]
