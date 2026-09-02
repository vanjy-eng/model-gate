"""Configuration dataclasses for every gate category. Override any field to
tune thresholds per model/use case; defaults are reasonable starting points,
not regulatory guidance."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

from .metrics import AUTO, MetricSetting


@dataclass
class FairnessConfig:
    """Thresholds for the fairness checks.

    `decision_threshold` turns continuous predictions into class labels for
    `DisparateImpactCheck`, which measures selection *rates* and so needs
    hard classes. Predictions already in {0, 1} are used as-is.

    Every gap threshold here is **relative**, not absolute: each is measured
    as a fraction of the corresponding overall figure — the overall mean,
    the overall error, or the mean absolute SHAP contribution. That is what
    lets one default be meaningful for a premium in naira, a claim count and
    a probability alike. An absolute threshold in the units of the model
    output flags nothing on one scale and everything on another.
    `min_group_size` guards against a three-policy segment producing a wild
    ratio that reads as a fairness finding.
    """

    disparity_threshold: float = 0.10  # max demographic parity difference
    decision_threshold: float = 0.5  # cutoff for binarising y_pred before parity
    # --- regression fairness (see bdp_model_gate.structured.regression_fairness) ---
    # All four are *relative* gaps, expressed as a fraction of the overall
    # figure, so one set of defaults works whether the target is naira
    # premiums or claim counts.
    # --- separation and sufficiency (see bdp_model_gate.structured.fairness) ---
    equalised_odds_threshold: float = 0.10  # max TPR or FPR difference across groups
    subgroup_calibration_threshold: float = 0.05  # max ECE difference across groups
    n_calibration_bins: int = 10  # bins per group for subgroup calibration
    # Harm concentrates at intersections: a model can look fair on gender and
    # on region while failing badly for women in one region. Off by default
    # because the joint groups are smaller and the reading needs care.
    intersectional: bool = False
    mean_gap_threshold: float = 0.10  # max relative gap in group mean prediction
    error_parity_threshold: float = 0.20  # max relative gap in per-group error
    calibration_threshold: float = 0.10  # max relative predicted-vs-actual gap per group
    loss_ratio_threshold: float = 0.10  # max relative gap in premium-over-expected-loss
    min_group_size: int = 30  # groups smaller than this are reported, not scored
    proxy_corr_threshold: float = 0.30  # eta^2 above this = proxy risk
    shap_gap_threshold: float = 0.50  # max cross-group SHAP gap, relative to mean |contribution|
    counterfactual_shift_threshold: float = 0.05  # max prediction shift on attribute flip


@dataclass
class PerformanceConfig:
    """Thresholds for the performance gate.

    `metric` selects how the model is scored — a name from
    `bdp_model_gate.metrics.BUILTIN_METRICS` ("roc_auc", "accuracy", "f1",
    "precision", "recall", "balanced_accuracy", "average_precision"), a
    `fn(y_true, y_pred) -> float` callable of your own, or "auto" to use
    whichever of roc_auc/accuracy the installed dependencies support. Under
    "auto" a fallback is logged and named in the report, never silent.

    Metrics point in different directions. Higher-is-better metrics
    (roc_auc, f1, r2, ...) are gated with `min_score`; error metrics where
    lower is better (rmse, mae, mape, poisson_deviance) are gated with
    `max_error`, which has no default because a sensible ceiling depends
    entirely on the scale of your target. Configuring an error metric
    without `max_error` is a GateConfigurationError rather than a guess.

    `min_score`/`max_error` are interpreted against whichever metric ran, so
    set the metric and its threshold together.

    `average` is the multiclass averaging strategy for `f1`, `precision` and
    `recall` (scikit-learn's `average=`). It defaults to "macro", which
    weights every class equally — so a rarely predicted "decline" counts as
    much as a common "accept". Use "weighted" to weight by support instead.
    Ignored for binary and regression. `decision_threshold` is used to binarize continuous
    predictions for metrics that need hard class labels; it's ignored for
    ranking metrics like roc_auc and for custom callables.
    """

    metric: MetricSetting = AUTO
    min_score: float = 0.80
    max_error: float | None = None
    # Calibration. Deliberately permissive: plenty of good models are
    # uncalibrated by construction (an SVM's decision function, say), and a
    # gate that blocks all of them gets switched off. Tighten it for pricing,
    # where a systematically inflated probability misprices every policy.
    max_ece: float = 0.10
    n_calibration_bins: int = 10
    calibration_strategy: str = "uniform"  # or "quantile", for skewed scores
    decision_threshold: float = 0.5
    average: str = "macro"
    max_latency_ms_p95: float = 200.0
    max_cost_per_inference: float = 0.002

    @property
    def min_accuracy(self) -> float:
        """Deprecated alias for `min_score`.

        The old name was misleading: the threshold was compared against
        ROC AUC whenever scikit-learn was installed, and against accuracy
        otherwise. Kept working so existing configs and CLI `--config`
        files don't break.
        """
        warnings.warn(
            "PerformanceConfig.min_accuracy is deprecated — use min_score, and set "
            "PerformanceConfig.metric to name the metric it applies to.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.min_score

    @min_accuracy.setter
    def min_accuracy(self, value: float) -> None:
        warnings.warn(
            "PerformanceConfig.min_accuracy is deprecated — use min_score, and set "
            "PerformanceConfig.metric to name the metric it applies to.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.min_score = value


@dataclass
class ComplianceConfig:
    required_model_card_fields: list[str] = field(
        default_factory=lambda: [
            "legal_basis",
            "data_minimization_justification",
            "training_data_source",
        ]
    )
    high_risk_use_cases: list[str] = field(
        default_factory=lambda: [
            "pricing",
            "claims_decisioning",
            "credit_scoring",
            "underwriting",
        ]
    )


@dataclass
class ValidationConfig:
    """Thresholds for the checks that ask whether the evaluation itself is sound.

    Nothing else in the report means anything if these fail. A model scored
    on its own training data reports a superb AUC and a clean `PASS`, and
    every fairness and calibration number beside it is measured on data the
    model has memorised.

    `leakage_min_power` is the guard that keeps the leak check usable. Parity
    between a feature and the model is only suspicious when the model is
    actually good: on a model scoring 0.55, a feature scoring 0.54 reaches
    99% of it and means nothing. Both conditions must hold before anything is
    flagged.
    """

    #: A feature is a suspected leak when its solo discriminative power
    #: reaches this fraction of the whole model's — one column should not do
    #: what the model does.
    leakage_ratio: float = 0.95
    #: ...and only when that power is high in absolute terms. Expressed on the
    #: same 0–1 scale as the ratio: |2·AUC − 1| for classification, |r| for
    #: regression. 0.80 is a Gini of 0.80, which no ordinary single feature
    #: reaches honestly.
    leakage_min_power: float = 0.80
    #: Rows appearing in both the training and validation frames, as a
    #: fraction of the validation set. Not zero by default: identical feature
    #: vectors legitimately recur in data with few categorical levels.
    max_split_overlap: float = 0.01
    #: Exact duplicate rows *within* the validation set. Duplicates inflate
    #: every metric by weighting the same observation twice.
    max_duplicate_fraction: float = 0.05
    #: Standardised mean shift between training and validation for a numeric
    #: feature — the difference in means over the training standard
    #: deviation, so it is unitless. 0.25 is a small-to-moderate effect.
    drift_threshold: float = 0.25
    #: For a categorical feature: total variation distance between the two
    #: frequency tables, on 0–1.
    categorical_drift_threshold: float = 0.15
    #: Values of `model_card["validation_strategy"]` that count as a holdout
    #: separated in *time* rather than at random.
    out_of_time_strategies: list[str] = field(
        default_factory=lambda: ["out_of_time", "temporal", "forward_chaining", "walk_forward"]
    )
    #: Everything else the field is allowed to say. An unrecognised value is
    #: flagged rather than accepted, because "holdout" and "out_of_time" are
    #: different claims and only one of them is checkable.
    accepted_validation_strategies: list[str] = field(
        default_factory=lambda: [
            "out_of_time",
            "temporal",
            "forward_chaining",
            "walk_forward",
            "random_split",
            "stratified_split",
            "cross_validation",
            "grouped_split",
        ]
    )
    #: A random split is the wrong test for a model that will be applied to
    #: next quarter's business. Enforced only for the high-risk use cases in
    #: `ComplianceConfig`.
    require_out_of_time_for_high_risk: bool = True


@dataclass
class ActuarialConfig:
    """Thresholds for the pricing measures — actual-vs-expected, monotonicity
    and dislocation.

    These are the three questions a pricing review asks that a general
    regression suite does not. **Is the level right** (A/E), **is the shape
    defensible** (monotone in each rating factor), and **who does the change
    hurt** (dislocation against the incumbent).

    All of them read `context.exposure` when it is supplied. See
    `bdp_model_gate.actuarial` for the convention: `exposure` is the weight
    a row's observation deserves, and it belongs on a target expressed as a
    *rate*.
    """

    #: Prediction bands for the A/E curve, cut at equal *exposure* rather
    #: than equal row counts. Ten is the pricing convention (deciles).
    n_bands: int = 10
    #: How far the whole book's A/E may sit from 1.0. Tighter than the band
    #: tolerance on purpose: an overall A/E of 1.10 means the book is
    #: under-priced by ten percent, which is a single, unambiguous defect.
    max_overall_ae_deviation: float = 0.05
    #: How far any one band's A/E may sit from 1.0. Looser, because a decile
    #: is a smaller sample and some scatter is expected.
    max_band_ae_deviation: float = 0.10
    #: A band thinner than this is reported but not scored — the same
    #: treatment `min_group_size` gives a three-policy segment.
    min_band_rows: int = 20

    #: Floor on the Lorenz Gini. Zero, and deliberately: there is no
    #: defensible universal target — a motor book and a fire book discriminate
    #: to different degrees for reasons that have nothing to do with model
    #: quality — but a Gini at or below zero says the rating structure orders
    #: risk no better than chance, or backwards, and that is a finding on any
    #: book. Raise it to your own portfolio's benchmark if you have one.
    min_gini: float = 0.0

    #: Rating factors the model's output must move monotonically in, as
    #: `{"prior_claims": "increasing"}`. Empty by default: the constraint is
    #: a claim about your product and its regulator, and no library can guess
    #: it. `MonotonicityCheck` reports NOT_APPLICABLE until you state one.
    monotonic_features: dict[str, str] = field(default_factory=dict)
    #: A step against the declared direction is a violation once it exceeds
    #: this fraction of the curve's own range. Relative, so one value works
    #: for a premium in naira and a probability alike.
    monotonicity_tolerance: float = 0.02
    #: Points on each partial-dependence curve, at quantiles of the feature.
    monotonicity_grid_points: int = 10
    #: Rows scored per grid point. The curve costs
    #: `grid_points * max_rows` predictions, and a gate is not the place to
    #: spend a million of them.
    monotonicity_max_rows: int = 200

    #: A relative move against `context.baseline_pred` at or beyond this is
    #: a "dislocation". 25% is the figure most conduct reviews use.
    dislocation_threshold: float = 0.25
    #: Share of exposure allowed above that threshold before the check asks
    #: for a human. Non-blocking either way — a dislocated book may be
    #: entirely correct, and the question is whether it was intended.
    max_dislocated_share: float = 0.10


@dataclass
class SecurityConfig:
    adversarial_epsilon: float = 0.02
    adversarial_flip_rate_threshold: float = 0.05
    # Regression has no notion of a "flipped" prediction — every perturbation
    # moves a continuous output. Sensitivity is measured instead as the mean
    # relative change in prediction, which must stay proportionate to the
    # size of the perturbation.
    adversarial_max_relative_shift: float = 0.10
    # Ordinal classification: the mean rank distance a prediction may move
    # under perturbation. A flip from accept to decline is two steps; to
    # refer, one. Only applied when context.class_order is set.
    adversarial_max_rank_shift: float = 0.10
    pii_patterns: dict[str, str] = field(
        default_factory=lambda: {
            "email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
            "phone_ng": r"(\+?234|0)[789]\d{9}\b",
            "nin_bvn": r"\b\d{10,11}\b",  # broad on purpose — Nigerian NIN/BVN are 10-11 digits
        }
    )
    # --- prompt injection (see bdp_model_gate.injection) -----------------
    #: How much of the corpus to fire. 1 is the six-prompt smoke set, one
    #: attack per family, and the default **because these calls are metered**:
    #: every prompt is a billed generation, on each surface you supply. 2 adds
    #: the rest of the corpus. The number of calls actually made is recorded
    #: in metadata and logged before they are spent.
    injection_depth: int = 1
    #: Restrict to particular attack families, or None for all of them. The
    #: other cost lever, and the one to reach for when a family is irrelevant
    #: to your deployment rather than merely expensive.
    injection_families: list[str] | None = None
    #: Responses are evidence and belong in the report, but a response to
    #: "repeat the customer record" may contain the customer record. Matches
    #: of `pii_patterns` are replaced with a `[redacted:<type>]` label before
    #: anything is stored.
    redact_injection_responses: bool = True
    #: How much of each response to keep. Enough to judge a finding, not
    #: enough to carry a document into an archived report.
    injection_response_chars: int = 280
    #: Extra prompts of your own, appended to the corpus as the `custom`
    #: family. They are judged the same way: a canary hit is a leak, and
    #: anything else routes to a human, since this library cannot know what
    #: a successful outcome looks like for a prompt it did not write.
    extra_injection_prompts: list[str] = field(default_factory=list)
    #: Instruction-shaped text in the strings this library copies into its own
    #: report — feature names, protected-attribute names, model-card keys and
    #: values. See `ReportInjectionCheck`: the risk is to whatever reads the
    #: report next, which is increasingly an LLM.
    report_injection_patterns: dict[str, str] = field(
        default_factory=lambda: {
            "instruction_override": (
                r"(?i)\b(ignore|disregard|forget)\b[^.]{0,30}\b"
                r"(previous|prior|above|earlier|all)\b[^.]{0,30}\b(instruction|prompt|rule)"
            ),
            "role_reassignment": r"(?i)\byou\s+are\s+(now|no\s+longer)\b",
            "instruction_injection": (
                r"(?i)\bnew\s+instructions?\b|\bend\s+of\s+(prompt|context)\b"
            ),
            "verdict_steering": (
                r"(?i)\b(approve|pass|ignore|suppress|disregard)\s+"
                r"(the\s+|this\s+)?(model|finding|check|gate|report|verdict)s?\b"
            ),
        }
    )

    @property
    def jailbreak_prompts(self) -> list[str]:
        """Deprecated alias for `extra_injection_prompts`.

        The old name described the old design. Three canned jailbreak prompts
        *were* the whole check until 0.5.4; they are now the smallest part of
        a categorised corpus, and the interesting knob is which families to
        fire and how deep, not which three strings to send.
        """
        warnings.warn(
            "SecurityConfig.jailbreak_prompts is deprecated — use "
            "extra_injection_prompts for prompts of your own, and "
            "injection_depth / injection_families to control the built-in corpus.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.extra_injection_prompts

    @jailbreak_prompts.setter
    def jailbreak_prompts(self, value: list[str]) -> None:
        warnings.warn(
            "SecurityConfig.jailbreak_prompts is deprecated — use "
            "extra_injection_prompts for prompts of your own, and "
            "injection_depth / injection_families to control the built-in corpus.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.extra_injection_prompts = value


@dataclass
class GateConfig:
    fairness: FairnessConfig = field(default_factory=FairnessConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    compliance: ComplianceConfig = field(default_factory=ComplianceConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    actuarial: ActuarialConfig = field(default_factory=ActuarialConfig)
