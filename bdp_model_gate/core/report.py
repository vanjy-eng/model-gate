"""Aggregated results of a gate run."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from typing import Any

from .base import CheckResult


@dataclass
class GateReport:
    """Aggregated results of a gate run.

    `model_metric`/`model_score` are the headline score lifted from the
    performance check — whichever metric was configured, not always AUC.
    Both are None if no performance check ran or the score was unavailable.
    """

    results: list[CheckResult]
    model_metric: str | None = None
    model_score: float | None = None
    #: The resolved prediction task this run was graded as. Recorded because
    #: a report read months later must say what it assumed the model was.
    task: str | None = None

    #: Set by `ModelGate.run` so `to_html()` can ask each check to draw
    #: itself. Excluded from the constructor, the repr, equality and
    #: `to_dict()`: a report is an archival record of findings, and neither
    #: the check objects nor the validation set belong in one. They are a
    #: convenience for rendering in the session that produced the report,
    #: nothing more.
    _checks: Any = field(default=None, init=False, repr=False, compare=False)
    _context: Any = field(default=None, init=False, repr=False, compare=False)

    @property
    def model_auc(self) -> float | None:
        """Deprecated. Returns `model_score` only when the configured
        metric really was ROC AUC, and None otherwise — reading it as
        "the AUC" was only ever correct by coincidence.
        """
        warnings.warn(
            "GateReport.model_auc is deprecated — use model_score together with "
            "model_metric, which names the metric the score came from.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.model_score if self.model_metric == "roc_auc" else None

    @property
    def flags(self) -> list[CheckResult]:
        """All non-OK, non-NOT_APPLICABLE results."""
        return [r for r in self.results if not r.is_ok]

    def by_category(self, category: str) -> list[CheckResult]:
        return [r for r in self.results if r.category == category]

    @property
    def gate_status(self) -> str:
        """BLOCKED if any blocking check failed, NEEDS_REVIEW if only
        non-blocking checks failed, PASS otherwise."""
        failing = self.flags
        if any(r.blocking for r in failing):
            return "BLOCKED"
        if failing:
            return "NEEDS_REVIEW"
        return "PASS"

    @property
    def total_duration_ms(self) -> float:
        return round(sum(r.duration_ms or 0.0 for r in self.results), 2)

    def to_dict(self) -> dict:
        by_cat: dict = {}
        for r in self.results:
            by_cat.setdefault(r.category, []).append(
                {
                    "check_name": r.check_name,
                    "flag": r.flag,
                    "detail": r.detail,
                    "blocking": r.blocking,
                    "metadata": r.metadata,
                    "duration_ms": r.duration_ms,
                }
            )
        return {
            "gate_status": self.gate_status,
            "task": self.task,
            "model_metric": self.model_metric,
            "model_score": self.model_score,
            # retained for consumers of the pre-0.2 report format; populated
            # only when the configured metric actually was ROC AUC
            "model_auc": self.model_score if self.model_metric == "roc_auc" else None,
            "n_flags": len(self.flags),
            "total_duration_ms": self.total_duration_ms,
            "results_by_category": by_cat,
        }

    def to_json(self, path: str | None = None, indent: int = 2) -> str:
        payload = json.dumps(self.to_dict(), indent=indent, default=str)
        if path:
            with open(path, "w") as f:
                f.write(payload)
        return payload

    def to_html(
        self,
        path: str | None = None,
        checks: Any = None,
        context: Any = None,
        title: str = "Model gate report",
        include_plots: bool = True,
    ) -> str:
        """Renders the report as one self-contained HTML page.

        A `NEEDS_REVIEW` verdict asks a person to decide, and `to_json()`
        hands that person a blob. This hands them a page: the verdict, every
        finding, and each check's plot beside the number it explains — no
        network, no JavaScript, nothing to install to open it.

        Plots need the checks and the data they scored, which `ModelGate.run`
        attaches to the report for you. Pass `checks` and `context` explicitly
        to override, or `include_plots=False` for text only. Without the
        `[plots]` extra the page renders text-only by itself.
        """
        from ..reporting import render_html

        page = render_html(
            self,
            checks=checks if checks is not None else self._checks,
            context=context if context is not None else self._context,
            title=title,
            include_plots=include_plots,
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(page)
        return page

    def summary(self) -> str:
        task_note = f", {self.task}" if self.task else ""
        lines = [f"Gate status: {self.gate_status} ({self.total_duration_ms:.0f}ms{task_note})"]
        if self.model_metric is not None and self.model_score is not None:
            lines.append(f"  {self.model_metric}: {self.model_score:.4f}")
        # Validation leads: a finding there means the numbers below it were
        # measured on evidence that does not support them.
        for category in ("validation", "performance", "compliance", "security", "fairness"):
            cat_flags = [r for r in self.by_category(category) if not r.is_ok]
            lines.append(f"  {category}: {len(cat_flags)} flag(s)")
        return "\n".join(lines)
