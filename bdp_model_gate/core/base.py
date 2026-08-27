"""Core interfaces shared by every governance check, regardless of data modality."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..task import ALL_TASKS


@dataclass
class CheckResult:
    """The outcome of a single governance check.

    `flag` is "OK", "NOT_APPLICABLE" (check skipped — e.g. optional input
    missing or optional dependency not installed), "CHECK_ERROR" (the check
    raised an exception), or a check-specific risk string such as
    "PROXY_RISK" or "PII_LEAKAGE_RISK".
    """

    check_name: str
    category: str
    flag: str
    detail: str = ""
    blocking: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    duration_ms: float | None = None

    @property
    def is_ok(self) -> bool:
        return self.flag in ("OK", "NOT_APPLICABLE")


class BaseCheck:
    """Interface every governance check implements.

    Subclasses set `name`, `category`, `blocking` and optionally
    `supported_tasks` as class attributes, and implement `run(context)`. `blocking=True` means a failing flag from
    this check should block promotion outright; `blocking=False` routes a
    failure to human review instead (used for checks that need judgment,
    like fairness flags that may be false positives).
    """

    name: str = "base_check"
    category: str = "fairness"
    blocking: bool = True
    #: Prediction tasks this check can meaningfully run against. The gate
    #: reports NOT_APPLICABLE for anything else rather than letting the check
    #: produce a confident but meaningless number. Defaults to every task, so
    #: third-party plugins written before 0.3.0 keep running unchanged.
    supported_tasks: tuple[str, ...] = ALL_TASKS

    def run(self, context: Any) -> list[CheckResult]:
        raise NotImplementedError(f"{self.__class__.__name__} must implement run()")

    def plot(self, context: Any, results: Any = None, ax: Any = None) -> Any:
        """Draw this check's finding, or return None if it has none to draw.

        Optional. The report renderer calls `plot()` on every check and uses
        whatever comes back, so there is nothing to declare or register — an
        override is the whole opt-in.

        Implementations take an optional matplotlib `Axes` and **return it**,
        which is what lets a caller compose these into their own figure and
        restyle the result. Draw only where the shape matters: a check whose
        finding is genuinely one number should leave this alone rather than
        chart it.

        Anything a plot needs beyond `CheckResult.metadata` is recomputed here
        rather than stored on the result, so the archival JSON does not carry
        presentation data most consumers never read. A check that scored a
        *subsample* must redraw the same rows the finding came from — use
        `bdp_model_gate._sampling.stable_sample`, which is content-addressed
        and so returns the same rows by construction.
        """
        return None
