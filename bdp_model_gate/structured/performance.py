"""Performance/cost thresholds that must pass before promotion."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._logging import get_logger
from ..actuarial import exposure_array
from ..config import PerformanceConfig, UncertaintyConfig
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
from ..uncertainty import ABOVE, BELOW, Uncertainty

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

    def __init__(
        self,
        config: PerformanceConfig | None = None,
        uncertainty: UncertaintyConfig | None = None,
    ):
        self.config = config or PerformanceConfig()
        self.uncertainty = Uncertainty(uncertainty)
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
        return metric, float(metric.fn(y_true, self._as_metric_expects(y_pred, metric, task)))

    def _as_metric_expects(self, y_pred: Any, metric: ResolvedMetric, task: str) -> Any:
        """`y_pred` in the shape the metric wants.

        Split out of `_score` so the bootstrap resamples exactly what the
        point estimate scored — binarised once, not once per draw, and never
        by a second code path that could round differently.
        """
        if not metric.needs_hard_labels:
            return y_pred
        if task == MULTICLASS:
            # Binarising at a 0.5 threshold is meaningless with more than two
            # classes; reduce a probability matrix by argmax instead.
            return to_class_labels(y_pred, getattr(self._context, "class_order", None))
        return to_hard_labels(y_pred, self.config.decision_threshold)

    def _prepared(self, context, metric: ResolvedMetric, task: str):
        """`(y_true, y_pred)` as the metric will see them, for the bootstrap."""
        return context.y_true, self._as_metric_expects(context.y_pred, metric, task)

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
        bound = "min" if higher_passes else "max"

        # `min_score` is a floor, so the *bad* side is below it. Reading the
        # interval the same way round as a disparity ceiling would invert
        # every verdict here — see `uncertainty.resolve`.
        flag_when = BELOW if higher_passes else ABOVE
        y_true, y_pred_eval = self._prepared(context, metric, task)
        interval = None
        if y_pred_eval is not None:
            arrays = {"y_true": np.asarray(y_true)}
            evaluated = np.asarray(y_pred_eval)
            if evaluated.ndim == 1:
                # A probability matrix has no one-column-per-row form to hash,
                # and re-deriving it per resample would cost more than the
                # interval is worth. Multiclass reports its point estimate.
                arrays["y_pred"] = evaluated
                exposure = exposure_array(context)
                if exposure is not None:
                    arrays["exposure"] = exposure

                def scored(positions, metric=metric):
                    return float(
                        metric.fn(arrays["y_true"][positions], arrays["y_pred"][positions])
                    )

                interval = self.uncertainty.interval(scored, pd.DataFrame(arrays))

        verdict = self.uncertainty.verdict(
            interval,
            threshold,
            point=score,
            risk_flag="PERFORMANCE_RISK",
            blocking=self.blocking,
            flag_when=flag_when,
        )

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

        measured = (
            f"{score:.4f} {interval.describe().split(' at ')[0].split(' ', 1)[1]}"
            if interval is not None
            else f"{score:.4f}"
        )
        return CheckResult(
            self.name,
            self.category,
            verdict.flag,
            detail=(f"{metric.name}={measured} ({bound} {threshold}){suffix}{verdict.note}"),
            blocking=verdict.blocking,
            metadata={
                "metric_kind": "score",
                "metric": metric.name,
                "value": round(score, 4),
                "threshold": threshold,
                "threshold_field": threshold_field,
                "greater_is_better": metric.greater_is_better,
                "metric_is_fallback": metric.is_fallback,
                "exposure_weighted": metric.exposure_weighted,
                **verdict.metadata,
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
            latencies = np.asarray(context.latencies_ms, dtype=float)
            p95 = float(np.percentile(latencies, 95))
            # A p95 from fifty timings is a noisy number, and 205 ms against a
            # 200 ms budget is the same coin-flip as an 0.11 disparity against
            # a 0.10 ceiling. Its own frame, because latencies are a benchmark
            # sample and are not row-aligned to anything.
            interval = self.uncertainty.interval(
                lambda positions: float(np.percentile(latencies[positions], 95)),
                pd.DataFrame({"latency_ms": latencies}),
            )
            verdict = self.uncertainty.verdict(
                interval,
                self.config.max_latency_ms_p95,
                point=p95,
                risk_flag="PERFORMANCE_RISK",
                blocking=self.blocking,
            )
            reported = (
                f"{p95:.2f}ms [{interval.low:.2f}, {interval.high:.2f}]"
                if interval is not None
                else f"{p95:.2f}ms"
            )
            results.append(
                CheckResult(
                    self.name,
                    self.category,
                    verdict.flag,
                    detail=(
                        f"p95 latency={reported} "
                        f"(max {self.config.max_latency_ms_p95}ms){verdict.note}"
                    ),
                    blocking=verdict.blocking,
                    metadata={
                        "metric_kind": "latency",
                        "metric": "latency_p95_ms",
                        "value": round(p95, 2),
                        "threshold": self.config.max_latency_ms_p95,
                        **verdict.metadata,
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
