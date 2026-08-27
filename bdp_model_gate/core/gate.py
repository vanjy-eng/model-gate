"""Orchestrator that runs a set of checks against a context and returns a GateReport."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, Callable

from .._logging import get_logger
from ..exceptions import GateValidationError
from ..task import resolve_task, supports
from .base import BaseCheck, CheckResult
from .report import GateReport
from .validation import validate_structured_context

logger = get_logger("gate")

#: Per-modality input validators. `ModelGate` itself is modality-agnostic;
#: it dispatches on `context.modality` so the unstructured suite can add
#: its own validator here without touching the orchestrator.
VALIDATORS: dict[str, Callable[[Any], None]] = {
    "structured": validate_structured_context,
}


class ModelGate:
    """Runs a list of governance checks against a context and aggregates a GateReport.

    If no checks are supplied, defaults to the full structured-data check
    suite (built-in checks plus any registered via the plugin entry-point
    group — see `bdp_model_gate.registry`). Pass a custom `checks` list to
    run a subset, add your own checks (subclass BaseCheck), or reorder
    blocking behavior.
    """

    def __init__(self, checks: Sequence[BaseCheck] | None = None, config=None):
        from ..config import GateConfig  # local import avoids a hard cycle at module load

        self.config = config or GateConfig()
        self.checks = list(checks) if checks is not None else self._default_checks()

    def _default_checks(self) -> list[BaseCheck]:
        from ..structured import default_structured_checks

        return default_structured_checks(self.config)

    def run(self, context) -> GateReport:
        self._validate(context)  # raises GateValidationError on bad input — fails fast

        # Resolved once per run, not per check: inference logs a line, and
        # every check must agree on what the task is.
        task = resolve_task(context)

        results: list[CheckResult] = []
        for check in self.checks:
            check_name = getattr(check, "name", check.__class__.__name__)
            if not supports(check, task):
                logger.debug("check=%s skipped — does not support task=%s", check_name, task)
                results.append(
                    CheckResult(
                        check_name=check_name,
                        category=getattr(check, "category", "unknown"),
                        flag="NOT_APPLICABLE",
                        detail=(
                            f"check does not apply to a {task} task (supports: "
                            f"{', '.join(getattr(check, 'supported_tasks', ()))})"
                        ),
                        blocking=getattr(check, "blocking", True),
                        duration_ms=0.0,
                    )
                )
                continue
            start = time.perf_counter()
            try:
                check_results = check.run(context)
                for r in check_results:
                    r.duration_ms = round((time.perf_counter() - start) * 1000, 2)
                results.extend(check_results)
                n_flags = sum(1 for r in check_results if not r.is_ok)
                logger.debug(
                    "check=%s duration_ms=%.1f flags=%d",
                    check_name,
                    (time.perf_counter() - start) * 1000,
                    n_flags,
                )
            except Exception as exc:  # a broken check shouldn't crash the whole gate
                logger.warning("check=%s raised an exception: %r", check_name, exc)
                results.append(
                    CheckResult(
                        check_name=check_name,
                        category=getattr(check, "category", "unknown"),
                        flag="CHECK_ERROR",
                        detail=f"check raised an exception: {exc!r}",
                        blocking=True,
                        duration_ms=round((time.perf_counter() - start) * 1000, 2),
                    )
                )

        metric, score = self._headline_score(results)
        report = GateReport(results=results, model_metric=metric, model_score=score, task=task)
        # Handed to the report so `to_html()` can ask each check to draw
        # itself. Deliberately not part of the serialised record — see the
        # field comments on GateReport.
        report._checks = self.checks
        report._context = context
        logger.info(
            "gate_status=%s task=%s n_flags=%d metric=%s score=%s",
            report.gate_status,
            task,
            len(report.flags),
            metric,
            score,
        )
        return report

    @staticmethod
    def _validate(context) -> None:
        modality = getattr(context, "modality", "structured")
        validator = VALIDATORS.get(modality)
        if validator is None:
            raise GateValidationError(
                f"no input validator registered for modality {modality!r} — "
                f"known modalities: {', '.join(sorted(VALIDATORS))}"
            )
        validator(context)

    @staticmethod
    def _headline_score(results: Sequence[CheckResult]) -> tuple[str | None, float | None]:
        """Lifts the model's headline score out of the performance check.

        Reading it from the check rather than recomputing here means the
        report always names whichever metric was actually configured,
        instead of asserting an AUC the gate never gated on.
        """
        for r in results:
            if r.category == "performance" and r.metadata.get("metric_kind") == "score":
                return r.metadata.get("metric"), r.metadata.get("value")
        return None, None
