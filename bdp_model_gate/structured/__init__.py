from __future__ import annotations

from ..config import GateConfig
from .calibration_checks import (
    CalibrationCheck,
    EqualisedOddsCheck,
    SubgroupCalibrationCheck,
)
from .compliance import ComplianceMappingCheck
from .fairness import (
    CounterfactualFlipCheck,
    DisparateImpactCheck,
    ProxyCorrelationCheck,
    ShapSubgroupCheck,
)
from .performance import PerformanceThresholdCheck
from .regression_fairness import (
    CalibrationParityCheck,
    ErrorParityCheck,
    GroupMeanGapCheck,
    LossRatioParityCheck,
)
from .security import (
    AdversarialRobustnessCheck,
    PIILeakageCheck,
    PromptInjectionCheck,
)
from .validation_checks import (
    FeatureContractCheck,
    FeatureDriftCheck,
    LeakageCheck,
    SplitOverlapCheck,
    ValidationStrategyCheck,
)


def default_structured_checks(config: GateConfig | None = None, include_plugins: bool = True):
    """The full default check suite for structured-data models.

    Every check for every task is included; `ModelGate` reports
    NOT_APPLICABLE for the ones that do not apply to the resolved task rather
    than omitting them, so a report shows what was skipped and why.

    If `include_plugins` is True (default), also appends any checks
    registered via the `bdp_model_gate.checks` entry-point group — see
    `bdp_model_gate.registry`.
    """
    config = config or GateConfig()
    checks = [
        # Validation first. A finding here says the evidence behind every
        # other number in the report is unsound, which is a strictly prior
        # question to whether the model is any good.
        LeakageCheck(config.validation),
        SplitOverlapCheck(config.validation),
        ValidationStrategyCheck(config.validation, config.compliance),
        FeatureContractCheck(config.validation),
        FeatureDriftCheck(config.validation),
        ProxyCorrelationCheck(config.fairness),
        DisparateImpactCheck(config.fairness),
        ShapSubgroupCheck(config.fairness),
        CounterfactualFlipCheck(config.fairness),
        # Separation and sufficiency. Reported alongside demographic parity
        # because the three are mutually incompatible — presenting only one
        # would make the choice silently.
        EqualisedOddsCheck(config.fairness),
        SubgroupCalibrationCheck(config.fairness),
        GroupMeanGapCheck(config.fairness),
        ErrorParityCheck(config.fairness),
        CalibrationParityCheck(config.fairness),
        LossRatioParityCheck(config.fairness),
        PerformanceThresholdCheck(config.performance),
        CalibrationCheck(config.performance),
        ComplianceMappingCheck(config.compliance),
        AdversarialRobustnessCheck(config.security),
        PIILeakageCheck(config.security),
        PromptInjectionCheck(config.security),
    ]
    if include_plugins:
        from ..registry import discover_plugin_checks

        checks.extend(discover_plugin_checks())
    return checks


__all__ = [
    "FeatureContractCheck",
    "FeatureDriftCheck",
    "LeakageCheck",
    "SplitOverlapCheck",
    "ValidationStrategyCheck",
    "CalibrationCheck",
    "CalibrationParityCheck",
    "EqualisedOddsCheck",
    "SubgroupCalibrationCheck",
    "ErrorParityCheck",
    "GroupMeanGapCheck",
    "LossRatioParityCheck",
    "ProxyCorrelationCheck",
    "DisparateImpactCheck",
    "ShapSubgroupCheck",
    "CounterfactualFlipCheck",
    "PerformanceThresholdCheck",
    "ComplianceMappingCheck",
    "AdversarialRobustnessCheck",
    "PIILeakageCheck",
    "PromptInjectionCheck",
    "default_structured_checks",
]
