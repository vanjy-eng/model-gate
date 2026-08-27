"""Fairness checks for continuous-output models.

Demographic parity has no regression analogue — there is no "selected"
class to count — so this module asks four different questions, each
answering something the others cannot:

    GroupMeanGapCheck      Does one group receive systematically higher
                           predictions? Raw level difference.
    ErrorParityCheck       Is the model materially *worse* for one group?
                           Quality of service, independent of level.
    CalibrationParityCheck Does one group's prediction systematically
                           over- or under-shoot its realised outcome?
    LossRatioParityCheck   Is one group charged a higher margin over its
                           own expected loss? The actuarial question.

The distinction matters most in insurance. A pricing model *should* charge
more in a higher-loss segment — that is risk-based pricing, not
discrimination — so a raw mean gap flags legitimate rating differences and
will be noisy on its own. Loss-ratio parity is the one that isolates
unfairness from actuarially justified variation, which is why it is worth
supplying `context.expected_loss` when you have it.

Every gap is measured *relative* to the overall figure, so a single
threshold works whether the target is a naira premium or a claim count.
Groups smaller than `FairnessConfig.min_group_size` are reported but not
scored: a three-policy segment produces wild ratios that read as findings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._logging import get_logger
from ..config import FairnessConfig
from ..core.base import BaseCheck, CheckResult
from ..groups import group_series
from ..task import REGRESSION

logger = get_logger("regression_fairness")

#: Guard for the relative-gap denominators below.
_EPSILON = 1e-12


def _no_protected(check: BaseCheck) -> list[CheckResult]:
    return [
        CheckResult(
            check.name,
            check.category,
            "NOT_APPLICABLE",
            "no protected_df supplied",
            check.blocking,
        )
    ]


def _relative_gap(values: pd.Series, reference: float) -> float:
    """Spread across groups, as a fraction of the overall figure."""
    if len(values) < 2:
        return 0.0
    return float((values.max() - values.min()) / (abs(reference) + _EPSILON))


def _usable_groups(protected: pd.Series, min_group_size: int) -> tuple[list, list[tuple[str, int]]]:
    """Splits group labels into those large enough to score and those not."""
    counts = protected.value_counts()
    usable = [g for g, n in counts.items() if n >= min_group_size]
    too_small = [(str(g), int(n)) for g, n in counts.items() if n < min_group_size]
    return usable, too_small


def _small_group_note(too_small: list[tuple[str, int]], min_group_size: int) -> str:
    if not too_small:
        return ""
    listed = ", ".join(f"{g} (n={n})" for g, n in too_small)
    return f" [not scored, below min_group_size={min_group_size}: {listed}]"


class _RegressionFairnessCheck(BaseCheck):
    """Shared plumbing: regression-only, non-blocking, needs protected_df."""

    category = "fairness"
    blocking = False
    supported_tasks = (REGRESSION,)

    def __init__(self, config: FairnessConfig | None = None):
        self.config = config or FairnessConfig()

    def _per_group(self, context, statistic):
        """Applies `statistic(mask)` to each sufficiently large group of each
        protected attribute. Yields (attr, Series-of-group-values, note)."""
        for attr in context.protected_df.columns:
            protected = context.protected_df[attr]
            usable, too_small = _usable_groups(protected, self.config.min_group_size)
            if len(usable) < 2:
                logger.debug(
                    "%s: attribute %r has fewer than two groups of at least %d rows — skipping",
                    self.name,
                    attr,
                    self.config.min_group_size,
                )
                continue
            values = pd.Series(
                {g: statistic(np.asarray(protected == g)) for g in usable}, dtype=float
            )
            yield attr, values, _small_group_note(too_small, self.config.min_group_size)


class GroupMeanGapCheck(_RegressionFairnessCheck):
    """Relative spread in mean prediction across protected groups.

    The bluntest of the four. On a risk-priced model a gap here is expected
    and not by itself evidence of unfairness — read it alongside
    LossRatioParityCheck, which says whether the gap is justified by cost.
    """

    name = "group_mean_gap"

    def run(self, context) -> list[CheckResult]:
        if context.protected_df is None or context.protected_df.empty:
            return _no_protected(self)

        y_pred = np.asarray(context.y_pred, dtype=float)
        overall = float(np.mean(y_pred))
        results = []

        for attr, means, note in self._per_group(context, lambda m: float(np.mean(y_pred[m]))):
            gap = _relative_gap(means, overall)
            flag = "MEAN_GAP_RISK" if gap > self.config.mean_gap_threshold else "OK"
            hi, lo = means.idxmax(), means.idxmin()
            results.append(
                CheckResult(
                    self.name,
                    self.category,
                    flag,
                    detail=(
                        f"{attr}: mean prediction spans {means.min():,.2f} ({lo}) to "
                        f"{means.max():,.2f} ({hi}) — {gap:.1%} of the overall mean "
                        f"{overall:,.2f}{note}"
                    ),
                    blocking=self.blocking,
                    metadata={
                        "protected_attr": attr,
                        "relative_gap": round(gap, 4),
                        "threshold": self.config.mean_gap_threshold,
                        "group_means": {str(k): round(v, 4) for k, v in means.items()},
                        "highest_group": str(hi),
                        "lowest_group": str(lo),
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


class ErrorParityCheck(_RegressionFairnessCheck):
    """Relative spread in per-group prediction error (MAE).

    Answers a quality-of-service question rather than a pricing one: a group
    the model simply predicts worse for is being under-served, however fair
    the average price looks.
    """

    name = "error_parity"

    def run(self, context) -> list[CheckResult]:
        if context.protected_df is None or context.protected_df.empty:
            return _no_protected(self)
        if context.y_true is None:
            return [
                CheckResult(
                    self.name,
                    self.category,
                    "NOT_APPLICABLE",
                    "no y_true supplied — error parity needs realised outcomes",
                    self.blocking,
                )
            ]

        y_true = np.asarray(context.y_true, dtype=float)
        y_pred = np.asarray(context.y_pred, dtype=float)
        abs_err = np.abs(y_true - y_pred)
        overall = float(np.mean(abs_err))
        results = []

        for attr, errors, note in self._per_group(context, lambda m: float(np.mean(abs_err[m]))):
            gap = _relative_gap(errors, overall)
            flag = "ERROR_PARITY_RISK" if gap > self.config.error_parity_threshold else "OK"
            worst = errors.idxmax()
            results.append(
                CheckResult(
                    self.name,
                    self.category,
                    flag,
                    detail=(
                        f"{attr}: mean absolute error spans {errors.min():,.2f} to "
                        f"{errors.max():,.2f} (worst: {worst}) — {gap:.1%} of the overall "
                        f"MAE {overall:,.2f}{note}"
                    ),
                    blocking=self.blocking,
                    metadata={
                        "protected_attr": attr,
                        "relative_gap": round(gap, 4),
                        "threshold": self.config.error_parity_threshold,
                        "group_mae": {str(k): round(v, 4) for k, v in errors.items()},
                        "worst_served_group": str(worst),
                    },
                )
            )
        return results or [
            CheckResult(
                self.name,
                self.category,
                "NOT_APPLICABLE",
                "no protected attribute had two sufficiently large groups",
                self.blocking,
            )
        ]


class CalibrationParityCheck(_RegressionFairnessCheck):
    """Per-group bias: does one group's prediction systematically over- or
    under-shoot its realised outcome?

    Distinct from error parity, which is scale-free about direction. A group
    can have perfectly typical error magnitude while being consistently
    over-predicted — systematically overcharged, in a pricing model.
    """

    name = "calibration_parity"

    def run(self, context) -> list[CheckResult]:
        if context.protected_df is None or context.protected_df.empty:
            return _no_protected(self)
        if context.y_true is None:
            return [
                CheckResult(
                    self.name,
                    self.category,
                    "NOT_APPLICABLE",
                    "no y_true supplied — calibration needs realised outcomes",
                    self.blocking,
                )
            ]

        y_true = np.asarray(context.y_true, dtype=float)
        y_pred = np.asarray(context.y_pred, dtype=float)
        overall_actual = float(np.mean(y_true))
        residual = y_pred - y_true  # positive = over-prediction
        results = []

        for attr, bias, note in self._per_group(context, lambda m: float(np.mean(residual[m]))):
            gap = _relative_gap(bias, overall_actual)
            flag = "CALIBRATION_RISK" if gap > self.config.calibration_threshold else "OK"
            over, under = bias.idxmax(), bias.idxmin()
            results.append(
                CheckResult(
                    self.name,
                    self.category,
                    flag,
                    detail=(
                        f"{attr}: prediction bias spans {bias.min():,.2f} ({under}, "
                        f"under-predicted) to {bias.max():,.2f} ({over}, over-predicted) "
                        f"— {gap:.1%} of the overall actual mean {overall_actual:,.2f}{note}"
                    ),
                    blocking=self.blocking,
                    metadata={
                        "protected_attr": attr,
                        "relative_gap": round(gap, 4),
                        "threshold": self.config.calibration_threshold,
                        "group_bias": {str(k): round(v, 4) for k, v in bias.items()},
                        "most_over_predicted": str(over),
                        "most_under_predicted": str(under),
                    },
                )
            )
        return results or [
            CheckResult(
                self.name,
                self.category,
                "NOT_APPLICABLE",
                "no protected attribute had two sufficiently large groups",
                self.blocking,
            )
        ]

    def plot(self, context, results=None, ax=None):
        """Actual over expected, by predicted band, per group.

        A mean residual is one number for the whole book, and a book is not
        uniform. RMSE says "wrong by 25,000"; this says "under-priced in the
        top decile, and only for one group" — which is the difference between
        a model that needs recalibrating and a model that needs withdrawing.

        Bands are quantiles of the prediction, shared across groups, so the
        lines are comparable. A ratio above 1 means the realised outcome
        exceeded the prediction: under-priced.
        """
        from ..plots import require_plotting, worst_result
        from ..plots.style import RULE, caption, categorical, markers, new_axes

        require_plotting()
        if context.protected_df is None or context.y_true is None:
            return None
        results = self.run(context) if results is None else results
        finding = worst_result(results, "relative_gap")
        if finding is None:
            return None

        attribute = finding.metadata["protected_attr"]
        protected = group_series(context.protected_df, attribute, self.config.min_group_size)
        if protected is None:
            return None
        scored = list(finding.metadata["group_bias"])

        y_true = np.asarray(context.y_true, dtype=float)
        y_pred = np.asarray(context.y_pred, dtype=float)

        # Quantile bands over the whole book, not per group: per-group edges
        # would put a different slice of business on each x position and the
        # lines would not be comparable, which is the entire point of the plot.
        n_bands = min(10, max(3, len(y_pred) // (5 * max(len(scored), 1))))
        edges = np.unique(np.quantile(y_pred, np.linspace(0, 1, n_bands + 1)))
        if len(edges) < 3:
            return None
        band = np.clip(np.digitize(y_pred, edges[1:-1]), 0, len(edges) - 2)

        ax = new_axes(ax)
        ax.axhline(1.0, color=RULE, linewidth=1.2, linestyle="--", zorder=1)

        centres = np.arange(len(edges) - 1)
        for colour, marker, value in zip(categorical(len(scored)), markers(len(scored)), scored):
            mask = np.asarray(protected.astype(str) == value)
            ratios, positions = [], []
            for b in centres:
                cell = mask & (band == b)
                predicted_total = y_pred[cell].sum()
                # A band a group barely occupies produces a ratio driven by
                # two policies. Leave the gap in the line rather than draw it.
                if cell.sum() < 5 or abs(predicted_total) <= _EPSILON:
                    continue
                ratios.append(float(y_true[cell].sum() / predicted_total))
                positions.append(b)
            if positions:
                ax.plot(positions, ratios, color=colour, marker=marker, label=str(value), zorder=2)

        ax.set_xticks(centres)
        ax.set_xticklabels([f"{edges[b]:,.0f}–\n{edges[b + 1]:,.0f}" for b in centres], fontsize=8)
        # Keep break-even inside the frame even when no band comes near it —
        # a chart cropped to the data hides how far off the whole book is.
        low, high = ax.get_ylim()
        ax.set_ylim(min(low, 0.95), max(high, 1.05))
        ax.set_xlabel("predicted value, by band")
        ax.set_ylabel("actual ÷ expected")
        ax.set_title(f"Actual against expected by band, split on {attribute}")
        ax.legend(loc="best")
        caption(
            ax,
            "the dashed line is break-even. Above it the outcome beat the prediction "
            "(under-predicted);\nbelow it the prediction was too high. A group drifting "
            "in one band only is a segment problem.",
        )
        return ax


class LossRatioParityCheck(_RegressionFairnessCheck):
    """Margin parity: is one group charged more *relative to its own
    expected loss* than another?

    This is the actuarially meaningful fairness test for a pricing model.
    Charging a higher premium in a higher-loss segment is risk-based pricing;
    charging a higher **margin** over expected loss is not justified by cost,
    and is what this check isolates.

    Requires `context.expected_loss` — a per-row expected loss, technical
    premium or pure premium, row-aligned to X. Without it the check reports
    NOT_APPLICABLE rather than falling back to a raw price comparison, which
    would answer a different question under the same name.
    """

    name = "loss_ratio_parity"

    def run(self, context) -> list[CheckResult]:
        if context.protected_df is None or context.protected_df.empty:
            return _no_protected(self)
        if context.expected_loss is None:
            return [
                CheckResult(
                    self.name,
                    self.category,
                    "NOT_APPLICABLE",
                    "no expected_loss supplied — margin parity needs a per-row expected "
                    "loss (or technical premium) to compare the prediction against",
                    self.blocking,
                )
            ]

        expected = np.asarray(context.expected_loss, dtype=float)
        y_pred = np.asarray(context.y_pred, dtype=float)

        positive = expected > 0
        n_dropped = int((~positive).sum())
        if n_dropped:
            logger.warning(
                "%s: ignoring %d row(s) with a non-positive expected_loss — the margin "
                "ratio is undefined there",
                self.name,
                n_dropped,
            )
        if not positive.any():
            return [
                CheckResult(
                    self.name,
                    self.category,
                    "NOT_APPLICABLE",
                    "every expected_loss value is non-positive — no margin to compare",
                    self.blocking,
                )
            ]

        ratio = np.full(len(y_pred), np.nan)
        ratio[positive] = y_pred[positive] / expected[positive]
        overall = float(np.nanmean(ratio))
        results = []

        def group_ratio(mask):
            selected = ratio[mask & positive]
            return float(np.mean(selected)) if selected.size else float("nan")

        for attr, ratios, note in self._per_group(context, group_ratio):
            ratios = ratios.dropna()
            if len(ratios) < 2:
                continue
            gap = _relative_gap(ratios, overall)
            flag = "LOSS_RATIO_RISK" if gap > self.config.loss_ratio_threshold else "OK"
            hi, lo = ratios.idxmax(), ratios.idxmin()
            results.append(
                CheckResult(
                    self.name,
                    self.category,
                    flag,
                    detail=(
                        f"{attr}: premium-to-expected-loss ratio spans {ratios.min():.3f} "
                        f"({lo}) to {ratios.max():.3f} ({hi}) — {gap:.1%} of the overall "
                        f"ratio {overall:.3f}; {hi} carries the higher margin over its "
                        f"own expected cost{note}"
                    ),
                    blocking=self.blocking,
                    metadata={
                        "protected_attr": attr,
                        "relative_gap": round(gap, 4),
                        "threshold": self.config.loss_ratio_threshold,
                        "group_loss_ratio": {str(k): round(v, 4) for k, v in ratios.items()},
                        "highest_margin_group": str(hi),
                        "lowest_margin_group": str(lo),
                        "rows_ignored": n_dropped,
                    },
                )
            )
        return results or [
            CheckResult(
                self.name,
                self.category,
                "NOT_APPLICABLE",
                "no protected attribute had two sufficiently large groups with a "
                "positive expected_loss",
                self.blocking,
            )
        ]

    def plot(self, context, results=None, ax=None):
        """Charged premium against expected loss, one point per policy.

        The scalar says the margin gap is 18%. It cannot say *where*. A
        uniform vertical offset between two groups is a flat loading — argue
        about it, but it is one decision. A fan that opens at the top of the
        book is a gap concentrated in high-value risks, which is a different
        finding with a different remedy.

        The 45° line is break-even: on it, premium equals expected loss.
        """
        from ..plots import require_plotting, worst_result
        from ..plots.style import RULE, caption, categorical, markers, new_axes

        require_plotting()
        if context.protected_df is None or context.expected_loss is None:
            return None
        results = self.run(context) if results is None else results
        finding = worst_result(results, "relative_gap")
        if finding is None:
            return None

        attribute = finding.metadata["protected_attr"]
        protected = group_series(context.protected_df, attribute, self.config.min_group_size)
        if protected is None:
            return None
        scored = list(finding.metadata["group_loss_ratio"])

        expected = np.asarray(context.expected_loss, dtype=float)
        y_pred = np.asarray(context.y_pred, dtype=float)
        positive = expected > 0
        if not positive.any():
            return None

        ax = new_axes(ax, figsize=(6.0, 5.2))
        ceiling = float(max(expected[positive].max(), y_pred[positive].max()))
        ax.plot([0, ceiling], [0, ceiling], color=RULE, linewidth=1.4, linestyle="--", zorder=1)

        for colour, marker, value in zip(categorical(len(scored)), markers(len(scored)), scored):
            cell = positive & np.asarray(protected.astype(str) == value)
            if not cell.any():
                continue
            ratio = finding.metadata["group_loss_ratio"][value]
            ax.scatter(
                expected[cell],
                y_pred[cell],
                color=colour,
                marker=marker,
                s=18,
                alpha=0.55,
                linewidth=0,
                label=f"{value} — mean ratio {ratio:.2f}",
                zorder=2,
            )
            # The group's own mean ratio as a ray from the origin: the line the
            # scalar in the report describes, drawn over the points it came from.
            ax.plot(
                [0, ceiling], [0, ceiling * ratio], color=colour, linewidth=1.1, alpha=0.9, zorder=3
            )

        ax.set_xlim(0, ceiling * 1.02)
        ax.set_ylim(0, max(ceiling, float(y_pred[positive].max())) * 1.02)
        ax.set_xlabel("expected loss")
        ax.set_ylabel("predicted premium")
        ax.set_title(f"Premium against expected loss, split on {attribute}")
        ax.legend(loc="upper left")
        caption(
            ax,
            "the dashed 45° line is break-even; each group's ray is its mean margin.\n"
            "Parallel rays are a flat loading. Diverging rays are a gap that grows with "
            "the size of the risk.",
        )
        return ax


__all__ = [
    "CalibrationParityCheck",
    "ErrorParityCheck",
    "GroupMeanGapCheck",
    "LossRatioParityCheck",
]
