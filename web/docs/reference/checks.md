# The checks

Sixteen built-in checks. Each declares a **category**, whether it is
**blocking**, and which **tasks** it supports. Nine also draw a chart — see
[Plots](plots.md).

## Fairness — non-blocking

Flags route to <span class="verdict-review">NEEDS_REVIEW</span>, not
<span class="verdict-blocked">BLOCKED</span>. All need `protected_df`.

| Check | Tasks | Flags |
|---|---|---|
| `proxy_correlation` | all | `PROXY_RISK` |
| `disparate_impact` | binary, multiclass | `DISPARITY_RISK` |
| `shap_subgroup_gap` | all | `SUBGROUP_IMPACT_RISK` |
| `counterfactual_flip` | binary, multiclass | `COUNTERFACTUAL_RISK` |
| `equalised_odds` | binary, multiclass | `EQUAL_OPPORTUNITY_RISK`, `EQUALISED_ODDS_RISK` |
| `subgroup_calibration` | binary, multiclass | `SUBGROUP_CALIBRATION_RISK` |

The last two are new in 0.5.0 and cover *separation* and *sufficiency*; the
four above them all measure *independence*. See
[Fairness: three families](../tasks/fairness.md) — the three are mutually
incompatible, so reporting one without the others makes a choice silently.

### `proxy_correlation`

Correlation ratio (η²) between each numeric feature and each protected
attribute. Catches a feature that reconstructs an attribute the model was
never given — the most common way "fairness through unawareness" fails.

Attributes with 10 or more distinct values are skipped as effectively
continuous.

### `disparate_impact`

Demographic parity difference via `fairlearn`. Needs hard class labels, so
continuous predictions are binarised at `decision_threshold`. For multiclass,
predictions are collapsed to "landed in `favourable_classes`".

### `shap_subgroup_gap`

Per-feature SHAP contribution gap across groups — catches a feature that looks
fair on average but drives outcomes differently for a subgroup. Uses
`TreeExplainer` for tree models, else the generic explainer, else
`model.predict` as a black box. If shap cannot cope it reports
`NOT_APPLICABLE` rather than blocking.

### `counterfactual_flip`

Flips a protected attribute that *is* a model input and measures the mean
prediction shift. Reports `NOT_APPLICABLE` when no protected attribute appears
in `X` — there is nothing to flip.

## Fairness, regression — non-blocking

Replace demographic parity, which has no continuous analogue. All measured
relative to the overall figure.

| Check | Question | Also needs |
|---|---|---|
| `loss_ratio_parity` | higher margin over own expected loss? | `expected_loss` |
| `group_mean_gap` | systematically higher predictions? | — |
| `error_parity` | materially less accurate for a group? | `y_true` |
| `calibration_parity` | over- or under-shoots reality? | `y_true` |

See [Regression](../tasks/regression.md#why-loss-ratio-parity-is-the-one-that-matters)
for why the margin question is the one that separates discrimination from
risk-based pricing.

## Performance — blocking

`performance_thresholds` emits up to three results, each independent:

| Result | Gated with | Needs |
|---|---|---|
| model score | `min_score` or `max_error` | `y_true`, `y_pred` |
| p95 latency | `max_latency_ms_p95` | `latencies_ms` |
| cost per inference | `max_cost_per_inference` | `cost_per_inference` |

With none of those inputs it reports `NOT_APPLICABLE` rather than passing
vacuously.

### `calibration`

Do the stated probabilities match observed frequencies? Expected Calibration
Error against `max_ece`, plus Murphy's decomposition into reliability,
resolution and uncertainty.

Independent of discrimination: a model can rank perfectly while every
probability is twice too high. Blocking, but with a permissive default —
see [Fairness: three families](../tasks/fairness.md#overall-calibration-calibration).

## Compliance — blocking

`compliance_mapping` reads `context.model_card` and enforces three things:

1. **Required fields present** — `required_model_card_fields`
2. **DPIA** — a `use_case` matching `high_risk_use_cases` requires
   `dpia_completed: true`
3. **Explainability** — if `influences_decision_about_person` (which defaults
   to true for a high-risk use case), `explainability_method` must be
   documented

A model card is a plain dict:

```python
model_card = {
    "use_case": "credit_scoring",
    "legal_basis": "Contractual necessity (NDPA 2023, s.25(1)(b))",
    "data_minimization_justification": "Affordability signals only.",
    "training_data_source": "Loan book 2021-2025, consented at origination",
    "dpia_completed": True,
    "influences_decision_about_person": True,
    "explainability_method": "SHAP, surfaced in the adverse-action notice",
}
```

## Security — blocking

### `adversarial_robustness`

Perturbs numeric features and measures how much the prediction moves. Each
feature is stepped relative to **its own** magnitude, so a column in the
millions does not swamp a single-digit one.

| Task | Measures | Threshold |
|---|---|---|
| classification | class flip rate | `adversarial_flip_rate_threshold` |
| ordinal | also mean rank distance | `adversarial_max_rank_shift` |
| regression | mean relative prediction shift | `adversarial_max_relative_shift` |

Attack direction, strongest first: `gradient_fn` → `coef_` → random noise.
See [Any model](../models.md#gradients-make-robustness-real).

Deterministic: seeded via `random_state` (default 42), so the same model and
data always produce the same verdict.

### `pii_leakage`

Regex-scans string columns for identifiers that should have been hashed or
tokenised upstream. Defaults cover email, Nigerian phone numbers and
NIN/BVN-shaped values; `pii_patterns` is a plain dict you can replace.

### `prompt_injection`

Only relevant with a generative side-car — an explanation writer, a chatbot.
Supply it as `generate_fn` and the check fires canned jailbreak prompts at it,
looking for refusal. Without one it reports `NOT_APPLICABLE`.

## Flags

| Flag | Meaning |
|---|---|
| `OK` | passed |
| `NOT_APPLICABLE` | skipped — treated as OK |
| `CHECK_ERROR` | the check raised; always blocking |
| risk string | check-specific; blocking per the check |
