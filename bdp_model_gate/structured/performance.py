"""Performance/cost thresholds that must pass before promotion."""

from __future__ import annotations

from typing import Any

import numpy as np

from .._logging import get_logger
from ..actuarial import exposure_array
from ..config import PerformanceConfig
from ..core.base import BaseCheck, CheckResult
from ..exceptions import GateConfigurationError
from ..metrics import (
    ResolvedMetric,
    resolve_metric,
    to_class_labels,
    to_hard_labels,
    validate_metric,
)
from ..task import MULTICLASS, resolve_task

logger = get_logger("performance")


class PerformanceThresholdCheck(BaseCheck):
    """Hard gate on model score, p95 latency, and cost-per-inference.

    The score metric is whatever `PerformanceConfig.metric` names — see
    `bdp_model_gate.metrics`. Which metric actually ran is recorded in the
    result's detail string and metadata, so a report always states what
    `min_score` was compared against.

    latencies_ms and cost_per_inference are optional on the context — if
    neither is supplied, only the score is checked; if the score inputs are
    also unavailable the check reports NOT_APPLICABLE rather than failing.
    """

    name = "performance_thresholds"
    category = "performance"
    blocking = True

    def __init__(self, config: PerformanceConfig | None = None):
        self.config = config or PerformanceConfig()
        # Fail at construction time on a typo'd metric name, rather than
        # partway through a gate run. Dependency availability is checked
        # lazily in _score(), so building the suite never needs sklearn.
        # Task is unknown at construction time, so only the name is checked
        # here; metric/task compatibility is verified in run().
        self._context = None
        validate_metric(self.config.metric)

    def _score(self, y_true: Any, y_pred: Any, task: str) -> tuple[ResolvedMetric, float]:
        """Scores the model with the configured metric.

        Raises GateConfigurationError if an explicitly requested metric
        isn't available; ModelGate turns that into a blocking CHECK_ERROR
        so the pipeline stops rather than proceeding on a substituted score.
        """
        metric = resolve_metric(
            self.config.metric,
            task,
            average=self.config.average,
            class_order=getattr(self._context, "class_order", None),
            exposure=exposure_array(self._context),
        )
        if not metric.needs_hard_labels:
            y_pred_eval = y_pred
        elif task == MULTICLASS:
            # Binarising at a 0.5 threshold is meaningless with more than two
            # classes; reduce a probability matrix by argmax instead.
            y_pred_eval = to_class_labels(y_pred, getattr(self._context, "class_order", None))
        else:
            y_pred_eval = to_hard_labels(y_pred, self.config.decision_threshold)
        return metric, float(metric.fn(y_true, y_pred_eval))

    def _threshold_for(self, metric: ResolvedMetric) -> tuple[float, str, bool]:
        """Picks the threshold that matches the metric's direction.

        Returns (threshold, config field name, passed-comparison-is-`>=`).
        An error metric with no `max_error` set is a configuration error, not
        a silent pass: the whole point of the gate is the comparison.
        """
        if metric.greater_is_better:
            return self.config.min_score, "min_score", True
        if self.config.max_error is None:
            raise GateConfigurationError(
                f"performance.metric={metric.name!r} is an error metric (lower is "
                "better), so it is gated with performance.max_error — which is unset. "
                "There is no sensible default: a ceiling depends on the scale of your "
                "target. Set max_error, or choose a higher-is-better metric such as 'r2'."
            )
        return self.config.max_error, "max_error", False

    def _score_result(self, context, task: str) -> CheckResult:
        metric, score = self._score(context.y_true, context.y_pred, task)
        threshold, threshold_field, higher_passes = self._threshold_for(metric)
        passed = score >= threshold if higher_passes else score <= threshold
        bound = "min" if higher_passes else "max"

        notes = []
        if metric.is_fallback:
            notes.append("fell back from the preferred metric — scikit-learn not installed")
        if metric.used_fallback_impl:
            notes.append("computed without scikit-learn")
        # Said out loud rather than left in metadata: an exposure-weighted RMSE
        # and an unweighted one are different numbers, and a reader comparing
        # this report against last quarter's needs to know which they have.
        if metric.exposure_weighted:
            notes.append("exposure-weighted")
        elif getattr(context, "exposure", None) is not None:
            notes.append("NOT exposure-weighted — this metric takes no per-row weight")
        suffix = f" [{'; '.join(notes)}]" if notes else ""

        logger.debug(
            "scored with metric=%s value=%.4f %s=%s fallback=%s",
            metric.name,
            score,
            threshold_field,
            threshold,
            metric.is_fallback,
        )

        return CheckResult(
            self.name,
            self.category,
            "OK" if passed else "PERFORMANCE_RISK",
            detail=f"{metric.name}={score:.4f} ({bound} {threshold}){suffix}",
            blocking=self.blocking,
            metadata={
                "metric_kind": "score",
                "metric": metric.name,
                "value": round(score, 4),
                "threshold": threshold,
                "threshold_field": threshold_field,
                "greater_is_better": metric.greater_is_better,
                "metric_is_fallback": metric.is_fallback,
                "exposure_weighted": metric.exposure_weighted,
            },
        )

    def plot(self, context, results=None, ax=None):
        """Confusion matrix in the caller's own class order.

        Only drawn where the classes are *ordered* — `context.class_order`
        set, three or more classes. `quadratic_kappa` penalises rank distance
        squared and then reports one number, which hides direction entirely:
        a model that sends accepts to decline and one that sends them to refer
        can score alike, and only one of those is a scandal. Keeping the
        caller's ordering on both axes is what makes distance from the
        diagonal readable as severity.

        A binary matrix is four numbers the detail line already carries, so
        this returns None there rather than charting a table.
        """
        from ..plots import require_plotting
        from ..plots.style import ACCENT, caption, new_axes, ring_cell, sharpen_colourbar

        _, sns = require_plotting()
        if context.y_true is None or context.y_pred is None:
            return None
        class_order = list(getattr(context, "class_order", None) or ())
        if len(class_order) < 3 or resolve_task(context) != MULTICLASS:
            return None

        import pandas as pd

        actual = pd.Series(to_class_labels(context.y_true, class_order)).astype(str)
        predicted = pd.Series(to_class_labels(context.y_pred, class_order)).astype(str)
        labels = [str(c) for c in class_order]
        counts = (
            pd.crosstab(actual, predicted)
            .reindex(index=labels, columns=labels, fill_value=0)
            .astype(int)
        )
        # Normalise by row, so a rare class is not rendered invisible by a
        # common one — the recall of the smallest band is usually the finding.
        rates = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

        ax = new_axes(ax, figsize=(1.6 + 0.95 * len(labels), 1.4 + 0.85 * len(labels)))
        sns.heatmap(
            rates,
            ax=ax,
            annot=counts,
            fmt="d",
            cmap="crest",
            vmin=0.0,
            vmax=1.0,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "share of the true class"},
        )
        sharpen_colourbar(ax)

        for i in range(len(labels)):  # ring the diagonal: correct, distance zero
            ring_cell(ax, i, i, ACCENT)
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        ax.set_title("Where the errors land — shading is the row's share, labels are counts")
        ax.tick_params(labelrotation=0)
        caption(
            ax,
            "distance from the ringed diagonal is how wrong the error was.\n"
            "quadratic_kappa squares that distance and reports one number, which hides "
            "the direction.",
        )
        return ax

    def run(self, context) -> list[CheckResult]:
        results = []
        # Stashed so _score can reach class_order without threading the whole
        # context through every helper.
        self._context = context
        task = resolve_task(context)

        if context.y_true is not None and context.y_pred is not None:
            results.append(self._score_result(context, task))

        if context.latencies_ms is not None:
            p95 = float(np.percentile(context.latencies_ms, 95))
            flag = "OK" if p95 <= self.config.max_latency_ms_p95 else "PERFORMANCE_RISK"
            results.append(
                CheckResult(
                    self.name,
                    self.category,
                    flag,
                    detail=f"p95 latency={p95:.2f}ms (max {self.config.max_latency_ms_p95}ms)",
                    blocking=self.blocking,
                    metadata={
                        "metric_kind": "latency",
                        "metric": "latency_p95_ms",
                        "value": round(p95, 2),
                        "threshold": self.config.max_latency_ms_p95,
                    },
                )
            )

        if context.cost_per_inference is not None:
            cost = context.cost_per_inference
            flag = "OK" if cost <= self.config.max_cost_per_inference else "PERFORMANCE_RISK"
            results.append(
                CheckResult(
                    self.name,
                    self.category,
                    flag,
                    detail=f"cost/inference={cost:.5f} (max {self.config.max_cost_per_inference})",
                    blocking=self.blocking,
                    metadata={
                        "metric_kind": "cost",
                        "metric": "cost_per_inference",
                        "value": round(cost, 5),
                        "threshold": self.config.max_cost_per_inference,
                    },
                )
            )

        return results or [
            CheckResult(
                self.name,
                self.category,
                "NOT_APPLICABLE",
                "no performance benchmark data supplied",
                self.blocking,
            )
        ]
