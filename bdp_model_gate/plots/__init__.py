"""Optional plotting for the checks that produce a shape, not just a number.

Plots are drawn only where a check collapses a distribution to a scalar **and
the shape is what a reviewer needs to judge**. Latency, cost and model-card
completeness are genuinely scalars; charting them would be decoration.

The contract with your own plotting code is deliberately narrow: every
`plot()` takes an optional matplotlib `Axes` and returns it. We draw onto your
canvas and hand it back, so these compose into your figures and can be
restyled. This package does not replace your plotting library.

`matplotlib` and `seaborn` live in the `[plots]` extra. Without them the
plotting calls raise `GateConfigurationError` naming the extra, and the HTML
report renders text-only — the same degradation shap and fairlearn already
follow.
"""

from __future__ import annotations

from typing import Any

from ..exceptions import GateConfigurationError

_MISSING = (
    "plotting needs matplotlib and seaborn — install them with "
    '`pip install "bdp-model-gate[plots]"`. Every other part of the gate works '
    "without them."
)


def require_plotting() -> tuple[Any, Any]:
    """Returns (pyplot, seaborn), or explains how to get them.

    Imported lazily rather than at module load so that importing
    `bdp_model_gate` never costs a matplotlib import, which is slow and pulls
    a font cache on first use.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:  # pragma: no cover - exercised in the core-install CI job
        raise GateConfigurationError(_MISSING) from exc
    return plt, sns


def plotting_available() -> bool:
    """Whether the `[plots]` extra is installed. Used by the report renderer
    to degrade to text rather than fail."""
    try:
        require_plotting()
    except GateConfigurationError:
        return False
    return True


def worst_result(results: Any, key: str) -> Any:
    """The result carrying the largest `key`, or None.

    A `plot()` is handed one Axes, and a check may have scored six protected
    attributes. Where only one can be drawn, draw the one the reader is being
    asked to judge. Taking the first attribute instead would quietly hide the
    finding on any report whose verdict came from the last one.
    """
    scored = [r for r in (results or []) if key in getattr(r, "metadata", {})]
    return max(scored, key=lambda r: r.metadata[key]) if scored else None


from .style import (  # noqa: E402  - must follow require_plotting
    CATEGORICAL,
    VERDICT_COLOURS,
    apply_style,
    new_axes,
)

__all__ = [
    "CATEGORICAL",
    "VERDICT_COLOURS",
    "apply_style",
    "new_axes",
    "plotting_available",
    "require_plotting",
    "worst_result",
]
