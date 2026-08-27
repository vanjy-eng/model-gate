"""Calibration and separation — the two fairness families the suite lacked.

Fairness splits into three families that cannot all hold at once:

    independence   selection rates match across groups  (demographic parity)
    separation     error rates match across groups      (equalised odds)
    sufficiency    a given score means the same thing   (calibration)
                   for every group

Before 0.5.0 this library measured only **independence**. That is not a
neutral position: demographic parity ignores `y_true` entirely, so a model can
achieve perfect parity by being wrong in compensating directions, and a
reader seeing one green check has no way to know two other notions were never
tested.

Calibration, TPR balance and FPR balance are mutually incompatible except in
degenerate cases (Kleinberg–Mullainathan–Raghavan 2016; Chouldechova 2017).
The suite therefore reports all three and names the trade-off, rather than
picking one on the user's behalf.
"""

from __future__ import annotations

import numpy as np

from .._logging import get_logger
from ..calibration import brier_decomposition, expected_calibration_error
from ..classes import favourable_mask, resolve_favourable
from ..config import FairnessConfig, PerformanceConfig
from ..core.base import BaseCheck, CheckResult
from ..groups import iter_protected
from ..metrics import to_class_labels
from ..task import CLASSIFICATION_TASKS, MULTICLASS, resolve_task

logger = get_logger("calibration_checks")


def _positive_probability(context, task: str) -> np.ndarray | None:
    """Probability of the favourable outcome, as a 1-D vector, or None.

    Calibration is only meaningful against probabilities. Hard labels give a
    degenerate curve, so the checks skip rather than report a number that
    looks like calibration and is not.
    """
    raw = np.asarray(context.y_pred)

    if task == MULTICLASS:
        class_order = getattr(context, "class_order", None)
        favourable = resolve_favourable(
            getattr(context, "favourable_classes", None), class_order, MULTICLASS
        )
        if class_order is None or not favourable:
            return None
        if raw.ndim == 2:
            by_model_order = sorted(class_order, key=str)
            columns = [by_model_order.index(c) for c in favourable]
            return raw[:, columns].sum(axis=1).astype(float)
        return None  # multiclass hard labels carry no probability to calibrate

    if raw.ndim != 1:
        return None
    values = raw.astype(float)
    if values.min() < 0.0 or values.max() > 1.0:
        return None
    # All-0/1 predictions are hard labels wearing a float dtype.
    if np.all(np.isin(values, (0.0, 1.0))):
        return None
    return values


def _binary_actuals(context, task: str) -> np.ndarray:
    """Ground truth as 0/1 against the favourable outcome."""
    if task == MULTICLASS:
        class_order = getattr(context, "class_order", None)
        favourable = resolve_favourable(
            getattr(context, "favourable_classes", None), class_order, MULTICLASS
        )
        labels = to_class_labels(context.y_true, class_order)
        return favourable_mask(labels, favourable or []).astype(float)
    return np.asarray(context.y_true, dtype=float)


_NO_PROBABILITIES = (
    "no probability predictions available — calibration needs y_pred to be "
    "probabilities in [0, 1], not hard labels. For multiclass, also set "
    "context.class_order so the favourable outcome can be identified."
)


class CalibrationCheck(BaseCheck):
    """Do stated probabilities match observed frequencies?

    Reports Expected Calibration Error and the Brier decomposition.
    Discrimination and calibration are independent: a model can rank perfectly
    while every probability it emits is twice too high, which scores well on
    ROC AUC and misprices every policy.

    Blocking, because it sits in the performance category — but with a
    deliberately permissive default. Plenty of good models are uncalibrated by
    construction, and a gate that blocks all of them gets switched off.
    Tighten `max_ece` for pricing.
    """

    name = "calibration"
    category = "performance"
    blocking = True
    supported_tasks = CLASSIFICATION_TASKS

    def __init__(self, config: PerformanceConfig | None = None):
        self.config = config or PerformanceConfig()

    def run(self, context) -> list[CheckResult]:
        task = resolve_task(context)
        probabilities = _positive_probability(context, task)
        if probabilities is None or context.y_true is None:
            return [
                CheckResult(
                    self.name, self.category, "NOT_APPLICABLE", _NO_PROBABILITIES, self.blocking
                )
            ]

        actuals = _binary_actuals(context, task)
        ece = expected_calibration_error(
            actuals,
            probabilities,
            n_bins=self.config.n_calibration_bins,
            strategy=self.config.calibration_strategy,
        )
        parts = brier_decomposition(actuals, probabilities, n_bins=self.config.n_calibration_bins)

        flag = "OK" if ece <= self.config.max_ece else "CALIBRATION_RISK"
        direction = ""
        if flag != "OK":
            mean_predicted = float(np.mean(probabilities))
            direction = (
                " — predictions run high on average"
                if mean_predicted > parts["base_rate"]
                else " — predictions run low on average"
            )

        return [
            CheckResult(
                self.name,
                self.category,
                flag,
                detail=(
                    f"expected calibration error={ece:.4f} (max {self.config.max_ece})"
                    f"{direction}; reliability={parts['reliability']:.4f}, "
                    f"resolution={parts['resolution']:.4f}"
                ),
                blocking=self.blocking,
                metadata={
                    "ece": round(ece, 5),
                    "threshold": self.config.max_ece,
                    "n_bins": self.config.n_calibration_bins,
                    "strategy": self.config.calibration_strategy,
                    **{k: round(v, 5) for k, v in parts.items()},
                },
            )
        ]


class SubgroupCalibrationCheck(BaseCheck):
    """Sufficiency: does a score of 0.7 mean the same thing for every group?

    A model can be well calibrated overall and badly miscalibrated for a
    minority group — the aggregate hides it, because the majority dominates
    the average. That is a fairness failure, not merely a performance one:
    it means the same stated risk carries a different real risk depending on
    who the applicant is.

    The regression analogue is `CalibrationParityCheck`, which compares mean
    residuals rather than binned frequencies.
    """

    name = "subgroup_calibration"
    category = "fairness"
    blocking = False
    supported_tasks = CLASSIFICATION_TASKS

    def __init__(self, config: FairnessConfig | None = None):
        self.config = config or FairnessConfig()

    def run(self, context) -> list[CheckResult]:
        if context.protected_df is None or context.protected_df.empty:
            return [
                CheckResult(
                    self.name,
                    self.category,
                    "NOT_APPLICABLE",
                    "no protected_df supplied",
                    self.blocking,
                )
            ]
        task = resolve_task(context)
        probabilities = _positive_probability(context, task)
        if probabilities is None or context.y_true is None:
            return [
                CheckResult(
                    self.name, self.category, "NOT_APPLICABLE", _NO_PROBABILITIES, self.blocking
                )
            ]

        actuals = _binary_actuals(context, task)
        results = []

        for label, groups in iter_protected(
            context.protected_df, self.config.intersectional, self.config.min_group_size
        ):
            per_group = {}
            for value in groups.unique():
                mask = np.asarray(groups == value)
                if mask.sum() < self.config.min_group_size:
                    continue
                per_group[str(value)] = expected_calibration_error(
                    actuals[mask],
                    probabilities[mask],
                    n_bins=self.config.n_calibration_bins,
                )
            if len(per_group) < 2:
                continue

            worst = max(per_group, key=lambda g: per_group[g])
            best = min(per_group, key=lambda g: per_group[g])
            gap = per_group[worst] - per_group[best]
            flag = (
                "SUBGROUP_CALIBRATION_RISK"
                if gap > self.config.subgroup_calibration_threshold
                else "OK"
            )
            results.append(
                CheckResult(
                    self.name,
                    self.category,
                    flag,
                    detail=(
                        f"{label}: calibration error spans {per_group[best]:.4f} ({best}) "
                        f"to {per_group[worst]:.4f} ({worst}) — gap {gap:.4f} "
                        f"(max {self.config.subgroup_calibration_threshold})"
                    ),
                    blocking=self.blocking,
                    metadata={
                        "protected_attr": label,
                        "group_ece": {k: round(v, 5) for k, v in per_group.items()},
                        "ece_gap": round(gap, 5),
                        "threshold": self.config.subgroup_calibration_threshold,
                        "worst_calibrated_group": worst,
                    },
                )
            )

        return results or [
            CheckResult(
                self.name,
                self.category,
                "NOT_APPLICABLE",
                "no protected attribute had two groups of at least "
                f"min_group_size={self.config.min_group_size} rows",
                self.blocking,
            )
        ]


class EqualisedOddsCheck(BaseCheck):
    """Separation: do error rates match across groups?

    Two results per attribute:

    - **equal opportunity** — the true-positive-rate difference. Among people
      who *should* be approved, are all groups equally likely to be? This is
      the notion lending regulators most often centre on.
    - **equalised odds** — the larger of the TPR and FPR differences. Stricter,
      and the one that matters when a false positive is costly too.

    Unlike demographic parity, both condition on the ground truth, so a model
    cannot satisfy them by being wrong in compensating directions.
    """

    name = "equalised_odds"
    category = "fairness"
    blocking = False
    supported_tasks = CLASSIFICATION_TASKS

    def __init__(self, config: FairnessConfig | None = None):
        self.config = config or FairnessConfig()

    @staticmethod
    def _rates(actuals: np.ndarray, predicted: np.ndarray) -> tuple[float, float] | None:
        """(TPR, FPR) for one group, or None when a rate is undefined."""
        positives = actuals == 1
        negatives = ~positives
        if not positives.any() or not negatives.any():
            return None
        return (
            float(predicted[positives].mean()),
            float(predicted[negatives].mean()),
        )

    def run(self, context) -> list[CheckResult]:
        if context.protected_df is None or context.protected_df.empty:
            return [
                CheckResult(
                    self.name,
                    self.category,
                    "NOT_APPLICABLE",
                    "no protected_df supplied",
                    self.blocking,
                )
            ]
        if context.y_true is None:
            return [
                CheckResult(
                    self.name,
                    self.category,
                    "NOT_APPLICABLE",
                    "no y_true supplied — separation conditions on the ground "
                    "truth, which is what distinguishes it from demographic parity",
                    self.blocking,
                )
            ]

        task = resolve_task(context)
        actuals = _binary_actuals(context, task)

        probabilities = _positive_probability(context, task)
        if probabilities is not None:
            predicted = (probabilities >= self.config.decision_threshold).astype(float)
        elif task == MULTICLASS:
            favourable = resolve_favourable(
                getattr(context, "favourable_classes", None),
                getattr(context, "class_order", None),
                MULTICLASS,
            )
            if not favourable:
                return [
                    CheckResult(
                        self.name,
                        self.category,
                        "NOT_APPLICABLE",
                        "multiclass separation needs context.class_order or "
                        "context.favourable_classes to know which outcome is good",
                        self.blocking,
                    )
                ]
            labels = to_class_labels(context.y_pred, getattr(context, "class_order", None))
            predicted = favourable_mask(labels, favourable).astype(float)
        else:
            predicted = np.asarray(context.y_pred, dtype=float)

        results = []
        for label, groups in iter_protected(
            context.protected_df, self.config.intersectional, self.config.min_group_size
        ):
            rates = {}
            for value in groups.unique():
                mask = np.asarray(groups == value)
                if mask.sum() < self.config.min_group_size:
                    continue
                pair = self._rates(actuals[mask], predicted[mask])
                if pair is not None:
                    rates[str(value)] = pair
            if len(rates) < 2:
                continue

            tprs = {g: r[0] for g, r in rates.items()}
            fprs = {g: r[1] for g, r in rates.items()}
            tpr_gap = max(tprs.values()) - min(tprs.values())
            fpr_gap = max(fprs.values()) - min(fprs.values())
            threshold = self.config.equalised_odds_threshold

            results.append(
                CheckResult(
                    self.name,
                    self.category,
                    "EQUAL_OPPORTUNITY_RISK" if tpr_gap > threshold else "OK",
                    detail=(
                        f"{label}: true-positive-rate difference={tpr_gap:.3f} "
                        f"(max {threshold}) — among those who should be approved, "
                        f"{min(tprs, key=lambda g: tprs[g])} is least likely to be"
                    ),
                    blocking=self.blocking,
                    metadata={
                        "protected_attr": label,
                        "notion": "equal_opportunity",
                        "tpr_difference": round(tpr_gap, 4),
                        "group_tpr": {g: round(v, 4) for g, v in tprs.items()},
                        "threshold": threshold,
                    },
                )
            )
            results.append(
                CheckResult(
                    self.name,
                    self.category,
                    "EQUALISED_ODDS_RISK" if max(tpr_gap, fpr_gap) > threshold else "OK",
                    detail=(
                        f"{label}: equalised odds difference={max(tpr_gap, fpr_gap):.3f} "
                        f"(max {threshold}) — TPR gap {tpr_gap:.3f}, FPR gap {fpr_gap:.3f}"
                    ),
                    blocking=self.blocking,
                    metadata={
                        "protected_attr": label,
                        "notion": "equalised_odds",
                        "equalised_odds_difference": round(max(tpr_gap, fpr_gap), 4),
                        "tpr_difference": round(tpr_gap, 4),
                        "fpr_difference": round(fpr_gap, 4),
                        "group_fpr": {g: round(v, 4) for g, v in fprs.items()},
                        "threshold": threshold,
                    },
                )
            )

        return results or [
            CheckResult(
                self.name,
                self.category,
                "NOT_APPLICABLE",
                "no protected attribute had two groups with both a positive and a "
                f"negative case and at least min_group_size={self.config.min_group_size} rows",
                self.blocking,
            )
        ]


__all__ = ["CalibrationCheck", "EqualisedOddsCheck", "SubgroupCalibrationCheck"]
