"""bdp_model_gate — automated pre-deployment ML model governance.

Runs validation, fairness, performance, compliance and security checks against a
trained model before it's promoted to production, and produces a single
GateReport with a PASS / NEEDS_REVIEW / BLOCKED status you can wire into CI.

Quickstart:

    from bdp_model_gate import StructuredGateContext, ModelGate

    context = StructuredGateContext(
        model=my_model, X=X_val, y_true=y_val, y_pred=y_pred,
        protected_df=protected_val,          # optional — enables fairness checks
        latencies_ms=benchmark_latencies,    # optional — enables performance checks
        cost_per_inference=0.0008,           # optional
        model_card=my_model_card,            # optional — enables compliance checks
        generate_fn=None,                    # optional — set if there's a generative side-car
    )

    report = ModelGate().run(context)
    print(report.summary())
    report.to_json("gate_report.json")

Unstructured-data (text/image/audio) support is planned — see
`bdp_model_gate.unstructured` for the reserved, not-yet-implemented interface.
"""

from .config import (
    ActuarialConfig,
    ComplianceConfig,
    FairnessConfig,
    GateConfig,
    PerformanceConfig,
    SecurityConfig,
)
from .core import BaseCheck, CheckResult, GateReport, ModelGate, StructuredGateContext
from .task import ALL_TASKS, BINARY, MULTICLASS, REGRESSION

__version__ = "0.5.4"


def run_structured_gate(model, X, y_true, y_pred, protected_df=None, **kwargs) -> GateReport:
    """Convenience one-shot function: builds a StructuredGateContext and runs
    the default check suite in a single call."""
    context = StructuredGateContext(
        model=model,
        X=X,
        y_true=y_true,
        y_pred=y_pred,
        protected_df=protected_df,
        **kwargs,
    )
    return ModelGate().run(context)


__all__ = [
    "ALL_TASKS",
    "BINARY",
    "MULTICLASS",
    "REGRESSION",
    "BaseCheck",
    "CheckResult",
    "GateReport",
    "ModelGate",
    "StructuredGateContext",
    "GateConfig",
    "ActuarialConfig",
    "FairnessConfig",
    "PerformanceConfig",
    "ComplianceConfig",
    "SecurityConfig",
    "run_structured_gate",
    "__version__",
]
