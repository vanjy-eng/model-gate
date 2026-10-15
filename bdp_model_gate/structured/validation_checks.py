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
from ..config import ComplianceConfig, UncertaintyConfig, ValidationConfig
from ..core.base import BaseCheck, CheckResult
from ..exceptions import GateConfigurationError
from ..metrics import to_class_labels
from ..stats import correlation_ratio, pearson_r, rank_auc
from ..task import MULTICLASS, REGRESSION, resolve_task
from ..uncertainty import ABOVE, Uncertainty

logger = get_logger("validation_checks")

#: Below this many usable rows a per-feature statistic is noise, not evidence.
_MIN_ROWS = 20
#: `ValidationConfig.drift_statistic` accepts only these.
_DRIFT_STATISTICS = ("standardised_shift", "psi")
#: Dtype kinds `FeatureDriftCheck` treats as numeric for either statistic:
#: signed/unsigned int, float, bool. A uint or bool column used to fall
#: through to the categorical branch, which is wrong for it specifically
#: under PSI — every distinct value becomes its own bin.
_NUMERIC_DTYPE_KINDS = "ifub"
#: Floor applied to a bin's proportion before taking its log, so an empty bin
#: in either frame doesn't divide by zero or take log(0). Standard fix for
#: this statistic; at 1e-4 it cannot move a verdict near either PSI band.
_PSI_EPSILON = 1e-4


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

    Two statistics, chosen by `ValidationConfig.drift_statistic`:

    - `"standardised_shift"` (default) — a numeric feature is compared by
      standardised mean shift, so the threshold is unitless; a categorical
      feature by total variation distance between frequency tables. Point
      estimate only, unchanged from before PSI existed.
    - `"psi"` — the Population Stability Index, the measure credit-risk
      auditors recognise by name for a *feature's* distribution (the
      score's own drift is usually called PSI too and the per-feature
      version CSI — see the docs), with the convention's own fixed bands
      (0.10 "worth a look", 0.25 "significant") applying to numeric and
      categorical features alike. A numeric feature is cut into
      `psi_bins` quantile bins (deciles, by default) using edges taken from
      the *training* distribution, so a live value outside that range still
      falls in the nearest bin rather than being dropped; a categorical
      feature uses its existing category levels as the bins, no cutting
      needed. A feature that reaches the moderate band without crossing the
      significant one is still named, alongside an otherwise-clean result,
      rather than silently folded into `OK` — catching drift early, before
      it becomes severe, is the reason to reach for PSI over the default
      statistic.

      Binned PSI is biased upward under the null at small samples — two
      identical distributions can read `DRIFT_RISK` most of the time at
      n=20-50 — and this is handled two ways, because they fix different
      halves of it. Below a per-feature bias floor (derived from the
      closed-form approximation of that bias, `(bins-1)*(1/n_train +
      1/n_live)`), the feature is skipped rather than scored: a plain
      percentile bootstrap does **not** fix this on its own, because
      resampling an already-biased small draw mostly reproduces the same
      bias rather than correcting for it. Above the floor, the
      significant-threshold verdict **is** bootstrapped, the same way
      twelve other checks already are, so a straddling interval reports
      `UNCERTAIN` rather than a confident finding the data cannot support.

      A category present in live data but never seen in training is
      reported by name, as its own fact, rather than folded into PSI
      through the epsilon floor — that floor is only correct for a known
      level's count being small in one frame, and an unseen level is a
      different claim entirely.
    """

    name = "feature_drift"
    category = "validation"
    blocking = False

    def __init__(
        self,
        config: ValidationConfig | None = None,
        uncertainty: UncertaintyConfig | None = None,
    ):
        self.config = config or ValidationConfig()
        if self.config.drift_statistic not in _DRIFT_STATISTICS:
            raise GateConfigurationError(
                f"validation.drift_statistic={self.config.drift_statistic!r} — must be "
                f"one of {', '.join(_DRIFT_STATISTICS)}"
            )
        if self.config.psi_bins < 2:
            raise GateConfigurationError(
                f"validation.psi_bins={self.config.psi_bins!r} — must be at least 2, or "
                "every numeric feature falls in one bin and PSI is always 0"
            )
        if self.config.psi_moderate_threshold > self.config.psi_significant_threshold:
            raise GateConfigurationError(
                f"validation.psi_moderate_threshold="
                f"{self.config.psi_moderate_threshold!r} exceeds "
                f"psi_significant_threshold={self.config.psi_significant_threshold!r} — "
                "the moderate band would never be reachable"
            )
        self.uncertainty = Uncertainty(uncertainty)

    def _numeric_shift(self, train: pd.Series, live: pd.Series) -> float | None:
        spread = float(train.std())
        if not np.isfinite(spread) or spread == 0:
            return None  # constant in training; a shift has no scale to be measured against
        return abs(float(live.mean()) - float(train.mean())) / spread

    @staticmethod
    def _shares(train: pd.Series, live: pd.Series) -> tuple[pd.Series, pd.Series, pd.Index]:
        train_share = train.value_counts(normalize=True)
        live_share = live.value_counts(normalize=True)
        levels = train_share.index.union(live_share.index)
        return train_share, live_share, levels

    def _categorical_shift(self, train: pd.Series, live: pd.Series) -> float:
        train_share, live_share, levels = self._shares(train, live)
        return float(
            0.5
            * sum(
                abs(float(live_share.get(level, 0.0)) - float(train_share.get(level, 0.0)))
                for level in levels
            )
        )

    @staticmethod
    def _psi_from_shares(train_share: pd.Series, live_share: pd.Series, levels) -> float:
        """PSI over aligned proportion tables, flooring each at `_PSI_EPSILON`
        first so a level present in both frames but absent in one resample
        cannot divide by zero or take log(0). Correct only for levels that
        genuinely occur in both frames — a level training never saw at all
        is excluded from `levels` by the caller, not floored here; see
        `_unseen_categories`."""
        train_p = np.array(
            [max(float(train_share.get(level, 0.0)), _PSI_EPSILON) for level in levels]
        )
        live_p = np.array(
            [max(float(live_share.get(level, 0.0)), _PSI_EPSILON) for level in levels]
        )
        return float(np.sum((live_p - train_p) * np.log(live_p / train_p)))

    @staticmethod
    def _unseen_categories(
        train_share: pd.Series, live_share: pd.Series
    ) -> list[tuple[str, float]]:
        """Levels present in live data that never occurred in training, with
        their live share, strongest first. `_PSI_EPSILON` must not decide
        whether these count as drift — see `_categorical_psi_with_unseen`."""
        return sorted(
            (
                (str(level), float(live_share[level]))
                for level in live_share.index
                if level not in train_share.index and live_share[level] > 0
            ),
            key=lambda pair: -pair[1],
        )

    def _categorical_psi_with_unseen(
        self, train: pd.Series, live: pd.Series
    ) -> tuple[float, float, list[tuple[str, float]], int]:
        """`(shift, known_level_psi, unseen_categories, n_levels)`.

        `shift` — what gets read against the moderate/significant bands and
        bootstrapped — is the worse of two different claims: PSI computed
        only over levels training actually saw (where the epsilon floor is
        the right tool), and the total live share sitting in levels training
        never saw at all (where it is not, so that mass is read directly
        rather than floored and logged). `n_levels` is the categorical
        analogue of a numeric feature's bin count, for `_psi_bias_floor`.
        """
        train_share, live_share, _ = self._shares(train, live)
        known_levels = train_share.index
        known_psi = self._psi_from_shares(train_share, live_share, known_levels)
        unseen = self._unseen_categories(train_share, live_share)
        unseen_share = sum(share for _, share in unseen)
        return max(known_psi, unseen_share), known_psi, unseen, len(known_levels)

    @staticmethod
    def _psi_bias_floor(n_levels: int, n_train: int, n_live: int) -> float:
        """Binned/leveled PSI's own upward bias under the null — roughly
        `(levels - 1) * (1/n_train + 1/n_live)`, a closed-form approximation
        rather than a resampling result, so it is checked directly instead
        of discovered empirically each run.

        This exists *beside* bootstrapping, not instead of it, because a
        plain percentile bootstrap does not fix it: at the sample sizes this
        bites hardest, resampling the same small, already-skewed draw mostly
        reproduces the same skew rather than correcting for it (verified —
        two identical n=30 samples still confidently read `DRIFT_RISK` under
        the bootstrapped verdict alone, PSI 1.7 with a tight interval).
        Below this floor, the feature is skipped rather than scored as if
        its bias were sampling noise.
        """
        return (n_levels - 1) * (1.0 / n_train + 1.0 / n_live)

    def _psi_numeric(
        self, train: pd.Series, live: pd.Series, bins: int
    ) -> tuple[float, int] | None:
        """`(psi, n_bins_used)`, or None. `n_bins_used` can be less than
        `psi_bins` when quantile edges collapse under a zero-inflated or
        otherwise repeated-value feature — worth knowing, since PSI's scale
        depends on how many bins it was computed over."""
        edges = np.unique(train.quantile(np.linspace(0.0, 1.0, bins + 1)).to_numpy())
        if len(edges) < 2:
            return None  # constant in training; no spread to bin on
        edges = edges.astype(float)
        edges[0], edges[-1] = -np.inf, np.inf
        bin_edges = pd.Index(edges)
        train_binned = pd.cut(train, bins=bin_edges, include_lowest=True)
        live_binned = pd.cut(live, bins=bin_edges, include_lowest=True)
        train_share = train_binned.value_counts(normalize=True)
        live_share = live_binned.value_counts(normalize=True)
        psi = self._psi_from_shares(train_share, live_share, train_share.index)
        return psi, len(edges) - 1

    def _psi_bootstrap_statistic(self, joint: pd.DataFrame, numeric: bool, bins: int):
        """A callable over row *positions* into `joint` (columns `value`,
        `is_live`), suitable for `Uncertainty.interval` — recomputes the same
        point statistic on whatever each resample contains, bin edges (for a
        numeric feature) included, so the interval reflects edge-estimation
        noise along with everything else."""

        def statistic(positions: np.ndarray) -> float:
            sub = joint.iloc[positions]
            tr = sub.loc[~sub["is_live"], "value"]
            lv = sub.loc[sub["is_live"], "value"]
            if len(tr) < 2 or len(lv) < 2:
                raise ValueError("degenerate resample: a group vanished")
            if numeric:
                result = self._psi_numeric(tr, lv, bins)
                if result is None:
                    raise ValueError("degenerate resample: no spread to bin on")
                return result[0]
            shift, _, _, _ = self._categorical_psi_with_unseen(tr, lv)
            return shift

        return statistic

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

        use_psi = self.config.drift_statistic == "psi"
        findings, moderate, scored = [], [], 0
        insufficient_for_psi: list[tuple[str, int, float]] = []

        for feature in shared:
            reference, live = train[feature].dropna(), context.X[feature].dropna()
            if len(reference) < _MIN_ROWS or len(live) < _MIN_ROWS:
                continue
            numeric = (
                reference.dtype.kind in _NUMERIC_DTYPE_KINDS
                and live.dtype.kind in _NUMERIC_DTYPE_KINDS
            )

            if not use_psi:
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
                    findings.append(
                        CheckResult(
                            self.name,
                            self.category,
                            "DRIFT_RISK",
                            detail=(
                                f"{feature}: {unit} {shift:.3f} between training and "
                                f"validation (max {limit}) — the model is being graded "
                                "on a different population from the one it was fitted to"
                            ),
                            blocking=self.blocking,
                            metadata={
                                "feature": feature,
                                "shift": round(shift, 4),
                                "threshold": limit,
                                "measure": unit,
                                "n_features_scored": None,  # filled in once scored is final
                            },
                        )
                    )
                continue

            # drift_statistic == "psi"
            extra_meta: dict = {}
            note_suffix = ""
            if numeric:
                result = self._psi_numeric(reference, live, self.config.psi_bins)
                if result is None:
                    continue
                shift, n_levels = result
                extra_meta["n_bins_used"] = n_levels
            else:
                shift, known_psi, unseen, n_levels = self._categorical_psi_with_unseen(
                    reference, live
                )
                extra_meta["known_level_psi"] = round(known_psi, 4)
                if unseen:
                    unseen_share = sum(s for _, s in unseen)
                    named = ", ".join(f"{lvl!r} ({s:.1%})" for lvl, s in unseen[:3])
                    more = f", +{len(unseen) - 3} more" if len(unseen) > 3 else ""
                    plural = "y" if len(unseen) == 1 else "ies"
                    note_suffix = (
                        f" — {len(unseen)} categor{plural} never seen in training make up "
                        f"{unseen_share:.1%} of live data: {named}{more}"
                    )
                    extra_meta["unseen_category_share"] = round(unseen_share, 4)
                    extra_meta["unseen_categories"] = [lvl for lvl, _ in unseen[:5]]

            bias_floor = self._psi_bias_floor(n_levels, len(reference), len(live))
            if bias_floor >= self.config.psi_moderate_threshold:
                # PSI's own small-sample bias would sit at or above the
                # moderate band on its own, for *this many* bins or levels —
                # not enough data to tell a real shift from the statistic's
                # own noise floor, so this feature is skipped rather than
                # scored as if that bias were evidence.
                insufficient_for_psi.append((feature, n_levels, bias_floor))
                continue
            scored += 1

            joint = pd.DataFrame(
                {
                    "value": pd.concat([reference, live], ignore_index=True),
                    "is_live": [False] * len(reference) + [True] * len(live),
                }
            )
            interval = self.uncertainty.interval(
                self._psi_bootstrap_statistic(joint, numeric, self.config.psi_bins),
                joint,
            )
            verdict = self.uncertainty.verdict(
                interval,
                self.config.psi_significant_threshold,
                point=shift,
                risk_flag="DRIFT_RISK",
                blocking=self.blocking,
                flag_when=ABOVE,
            )

            if verdict.flag == "OK":
                if shift >= self.config.psi_moderate_threshold:
                    moderate.append((feature, shift))
                continue

            findings.append(
                CheckResult(
                    self.name,
                    self.category,
                    verdict.flag,
                    detail=(
                        f"{feature}: PSI {self.uncertainty.measured(interval, shift)} "
                        f"between training and validation (max "
                        f"{self.config.psi_significant_threshold}) — the model is being "
                        "graded on a different population from the one it was fitted to"
                        f"{verdict.note}{note_suffix}"
                    ),
                    blocking=verdict.blocking,
                    metadata={
                        "feature": feature,
                        "shift": round(shift, 4),
                        "threshold": self.config.psi_significant_threshold,
                        "measure": "PSI",
                        "n_features_scored": None,
                        **extra_meta,
                        **verdict.metadata,
                    },
                )
            )

        if not scored:
            if insufficient_for_psi:
                worst = max(insufficient_for_psi, key=lambda row: row[2])
                return _not_applicable(
                    self,
                    f"{len(insufficient_for_psi)} shared feature(s) had enough rows for "
                    f"{_MIN_ROWS}-row scoring but not enough for a trustworthy PSI at "
                    f"their bin/level count — e.g. {worst[0]} ({worst[1]} bins/levels) "
                    f"would carry a small-sample bias of about {worst[2]:.3f} on its own, "
                    f"at or above psi_moderate_threshold={self.config.psi_moderate_threshold}. "
                    "Lower psi_bins, gather more rows, or use drift_statistic="
                    "'standardised_shift' instead.",
                )
            return _not_applicable(
                self,
                f"no shared feature had at least {_MIN_ROWS} usable rows in both frames",
            )

        for finding in findings:
            finding.metadata["n_features_scored"] = scored

        if not findings:
            detail = f"no drift above threshold across {scored} shared feature(s)"
            metadata: dict = {"n_features_scored": scored}
            if insufficient_for_psi:
                detail += (
                    f"; {len(insufficient_for_psi)} other feature(s) had too few rows for "
                    "a trustworthy PSI at their bin/level count and were skipped rather "
                    "than scored"
                )
                metadata["insufficient_for_psi"] = [
                    {"feature": f, "levels": lv, "bias_floor": round(b, 4)}
                    for f, lv, b in insufficient_for_psi
                ]
            if moderate:
                moderate.sort(key=lambda pair: -pair[1])
                named = ", ".join(f"{feature} ({psi:.3f})" for feature, psi in moderate)
                lo, hi = self.config.psi_moderate_threshold, self.config.psi_significant_threshold
                detail += (
                    f"; {len(moderate)} feature(s) sit in PSI's {lo:.2f}–{hi:.2f} "
                    f"'moderate' band, worth a look before it becomes significant: {named}"
                )
                metadata["moderate_band"] = [
                    {"feature": feature, "psi": round(psi, 4)} for feature, psi in moderate
                ]
            return [
                CheckResult(
                    self.name,
                    self.category,
                    "OK",
                    detail,
                    self.blocking,
                    metadata=metadata,
                )
            ]

        return sorted(findings, key=lambda r: -r.metadata["shift"])


__all__ = [
    "FeatureContractCheck",
    "FeatureDriftCheck",
    "LeakageCheck",
    "SplitOverlapCheck",
    "ValidationStrategyCheck",
]
