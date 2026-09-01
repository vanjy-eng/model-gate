# The checks

Twenty-one built-in checks across five categories. Each declares a
**category**, whether it is **blocking**, and which **tasks** it supports.
Nine also draw a chart — see [Plots](plots.md).

## Validation — blocking

**Reported first, and blocking, because these are a prior question.** A
performance finding says *the model is not good enough*. A validation finding
says *you do not yet know whether it is* — and if one fires, every number
below it was measured on evidence that does not support it.

The motivating hole: nothing used to stop you passing the **training set** as
the validation set. The gate reported a superb score, a clean calibration
curve and `PASS`, and every fairness figure beside it was measured on data the
model had memorised.

| Check | Needs | Flags |
|---|---|---|
| `target_leakage` | `y_true`, `y_pred` | `LEAKAGE_RISK` |
| `split_overlap` | `X_train` for the overlap half | `SPLIT_OVERLAP_RISK`, `DUPLICATE_ROWS_RISK` |
| `validation_strategy` | `model_card` | `VALIDATION_STRATEGY_RISK` |
| `feature_contract` | `X_train` or `expected_features` | `FEATURE_CONTRACT_RISK`, `FEATURE_ORDER_RISK` |
| `feature_drift` | `X_train` | `DRIFT_RISK` (non-blocking) |

All five work on a **core install**. A validation set that is secretly the
training set is the last thing that should go unchecked because scikit-learn
is missing.

### `target_leakage`

Does a single column do what the whole model does? That is the signature of a
field populated *after* the outcome was known and joined back in — a
settlement amount on a claims-frequency model, a `closed_reason` on a churn
model. The model learns it, scores beautifully offline, and collapses in
production where the column is empty at scoring time.

Each feature's solo predictive power is measured on the same 0–1 scale as the
model's own: |2·AUC − 1| for classification, |r| for regression, and the
correlation ratio for a low-cardinality categorical column. Two conditions
must both hold before anything is flagged — the feature must reach
`leakage_ratio` of the model's power **and** clear `leakage_min_power` in
absolute terms. Parity alone is not evidence: against a model scoring 0.55, a
feature scoring 0.54 reaches 98% of it and means nothing.

Near-unique string columns are skipped rather than scored, since an
identifier puts every row in its own group and drives the statistic to 1 by
arithmetic.

### `split_overlap`

Two separate findings, with different causes and different fixes:

- **Rows shared between `X_train` and `X`** — a broken split.
- **Exact duplicates within `X`** — survives a correct split, and inflates
  every metric by weighting one observation twice.

Matching is by row **content**, not index, so a reset index cannot hide an
overlap and a shuffle cannot invent one. `max_split_overlap` is not zero by
default: identical feature vectors legitimately recur in data with few
categorical levels, and the check says so when both findings fire together.

### `validation_strategy`

A random split asks "can the model predict a policy it has not seen?" An
out-of-time split asks "can it predict **next quarter**?" — the only question
that matters for a model about to be applied to future business, and the one a
random split silently answers yes to when the real answer is no. Seasonality,
inflation and portfolio mix all leak backwards through a random split.

So `model_card["validation_strategy"]` is required, and for the high-risk use
cases it must be out-of-time. An unrecognised value is flagged rather than
accepted — `holdout` and `out_of_time` are different claims and only one of
them answers the question.

This is the one check here that cannot be verified from the data. It records a
claim, which is what a model card is for: an untrue answer is a signed
statement, not an oversight.

### `feature_contract`

Are these the columns the model was fitted on, in the order it expects? Silent
column reordering is a classic production failure — a model handed a
positional array scores confidently against the wrong features and nothing
raises. scikit-learn checks names when given a DataFrame; a `predict_fn` doing
`df.values` does not, and neither does a booster fed a numpy array.

The expected list comes from the first of these that exists:
`context.expected_features`, `model.feature_names_in_`, the booster's own
feature names, then `X_train.columns`.

### `feature_drift`

Train-serve skew: a feature whose distribution has moved between the two
frames. Numeric features are compared by standardised mean shift (unitless);
categorical features by total variation distance between their frequency
tables.

**Non-blocking**, unlike its four neighbours. An out-of-time holdout *should*
differ a little — that is the point of one — so drift warrants a look rather
than a hard stop, and a gate that fails the build on every seasonal shift gets
switched off.

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

!!! warning "Treat this one as a smoke test, not a security assessment"
    It sends three well-known prompts and decides whether the model refused by
    substring-matching six English words. Three consequences worth knowing
    before you rely on it:

    - **A response that leaks while apologising passes.** "I cannot normally
      share this, but the system prompt is …" contains *cannot*, so it scores
      `OK`.
    - **A refusal in other words, or another language, fails.** "That request
      is out of scope" has no marker, and the check blocks.
    - **Indirect injection is not tested at all** — untrusted text arriving as
      *data* (a claim description, a customer email) rather than as the user
      turn, which is the realistic attack against a regulated pipeline.

    A rewrite is scheduled for **0.5.4**: canary-based leak detection instead
    of refusal detection, a separate indirect-injection surface, and a
    categorised corpus. See [`ROADMAP.md`](https://github.com/vanjy-eng/model-gate/blob/main/ROADMAP.md).
    Until then, consider setting `blocking=False` on this check if a false
    alarm stopping a deploy is worse for you than a missed finding.

## Flags

| Flag | Meaning |
|---|---|
| `OK` | passed |
| `NOT_APPLICABLE` | skipped — treated as OK |
| `CHECK_ERROR` | the check raised; always blocking |
| risk string | check-specific; blocking per the check |
