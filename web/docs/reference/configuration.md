# Configuration

`GateConfig` nests one dataclass per category. Defaults are reasonable
starting points chosen to be useful — **not** positions on what any regulator
requires.

```python
from bdp_model_gate import GateConfig
from bdp_model_gate.structured import default_structured_checks

config = GateConfig()
config.performance.metric = "roc_auc"
config.performance.min_score = 0.85
config.fairness.disparity_threshold = 0.05

report = ModelGate(checks=default_structured_checks(config)).run(context)
```

## Validation

| Field | Default | Notes |
|---|---|---|
| `leakage_ratio` | `0.95` | a feature is suspect at this fraction of the model's own predictive power |
| `leakage_min_power` | `0.80` | ...and only if it clears this absolutely. Both must hold |
| `max_split_overlap` | `0.01` | validation rows also present in `X_train` |
| `max_duplicate_fraction` | `0.05` | exact duplicate rows *within* the validation set |
| `drift_threshold` | `0.25` | standardised mean shift, numeric features |
| `categorical_drift_threshold` | `0.15` | total variation distance, categorical features |
| `out_of_time_strategies` | `out_of_time`, `temporal`, `forward_chaining`, `walk_forward` | values that count as separated in time |
| `accepted_validation_strategies` | the four above plus `random_split`, `stratified_split`, `cross_validation`, `grouped_split` | anything else is flagged, not accepted |
| `require_out_of_time_for_high_risk` | `True` | applies to `ComplianceConfig.high_risk_use_cases` |

`leakage_min_power` is the field that keeps the leak check usable, and the one
to reach for if it is noisy. Parity between a feature and the model only means
something when the model is good: against a model scoring 0.55, a feature
scoring 0.54 reaches 98% of it and means nothing at all. Both conditions are
required precisely so a weak model does not make every column look like a
leak.

`max_split_overlap` is deliberately not zero. Identical feature vectors recur
legitimately in data with few categorical levels, and the check says so when
duplicates and overlap fire together.

```python
config.validation.leakage_min_power = 0.90  # quieter on a strong model
config.validation.max_split_overlap = 0.0  # zero tolerance
config.validation.require_out_of_time_for_high_risk = False
```

## Performance

| Field | Default | Notes |
|---|---|---|
| `metric` | `"auto"` | a name, a callable, or `"auto"` |
| `min_score` | `0.80` | for higher-is-better metrics |
| `max_error` | `None` | for error metrics — **required** when one is selected |
| `decision_threshold` | `0.5` | binarises continuous predictions for label metrics |
| `average` | `"macro"` | multiclass averaging for f1 / precision / recall |
| `max_ece` | `0.10` | calibration; permissive on purpose |
| `n_calibration_bins` | `10` | |
| `calibration_strategy` | `"uniform"` | or `"quantile"`, for skewed scores |
| `max_latency_ms_p95` | `200.0` | |
| `max_cost_per_inference` | `0.002` | |

## Fairness

| Field | Default | Applies to |
|---|---|---|
| `disparity_threshold` | `0.10` | demographic parity |
| `decision_threshold` | `0.5` | binarising for parity |
| `proxy_corr_threshold` | `0.30` | η² for proxy correlation |
| `shap_gap_threshold` | `0.50` | relative to the mean absolute SHAP contribution |
| `counterfactual_shift_threshold` | `0.05` | |
| `mean_gap_threshold` | `0.10` | regression, relative |
| `error_parity_threshold` | `0.20` | regression, relative |
| `calibration_threshold` | `0.10` | regression, relative |
| `loss_ratio_threshold` | `0.10` | regression, relative |
| `equalised_odds_threshold` | `0.10` | max TPR or FPR difference (separation) |
| `subgroup_calibration_threshold` | `0.05` | max ECE difference (sufficiency) |
| `intersectional` | `False` | also evaluate pairwise attribute combinations |
| `min_group_size` | `30` | groups below this are reported, not scored |

!!! note "All gap thresholds are relative"
    Each is a fraction of the corresponding overall figure — the overall mean,
    the overall error, or the mean absolute SHAP contribution. That is what
    lets one default work for a premium in naira, a claim count and a
    probability alike.

    `shap_gap_threshold = 0.50` means "flag a feature whose cross-group gap is
    worth half a typical contribution". Before 0.4.2 it was absolute and in
    target units, which flagged nearly every feature on a money target.

## Compliance

| Field | Default |
|---|---|
| `required_model_card_fields` | `legal_basis`, `data_minimization_justification`, `training_data_source` |
| `high_risk_use_cases` | `pricing`, `claims_decisioning`, `credit_scoring`, `underwriting` |

Both are plain lists — replace them for your own regime:

```python
config.compliance.high_risk_use_cases += ["bnpl_limit_setting"]
config.compliance.required_model_card_fields = ["lawful_basis", "retention_period"]
```

## Actuarial

The pricing measures — actual-vs-expected, the Gini, monotonicity and
dislocation. See [Insurance pricing](../tasks/insurance.md) for what each one
answers.

| Field | Default | Notes |
|---|---|---|
| `n_bands` | `10` | A/E bands, cut at equal **exposure** rather than equal row counts |
| `max_overall_ae_deviation` | `0.05` | how far the whole book's A/E may sit from 1.0 |
| `max_band_ae_deviation` | `0.10` | ...and any one band. Looser: a decile is a smaller sample |
| `min_band_rows` | `20` | a thinner band is reported, not scored |
| `min_gini` | `0.0` | floor on the Lorenz Gini |
| `monotonic_features` | `{}` | `{"prior_claims": "increasing"}` — nothing is checked until you declare it |
| `monotonicity_tolerance` | `0.02` | a step against the direction, as a fraction of the curve's range |
| `monotonicity_grid_points` | `10` | points on each partial-dependence curve |
| `monotonicity_max_rows` | `200` | rows scored per grid point |
| `dislocation_threshold` | `0.25` | a relative move at or beyond this is a dislocation |
| `max_dislocated_share` | `0.10` | share of exposure allowed past it before asking for a human |

Two of these defaults are choices worth understanding rather than tuning.

`min_gini = 0.0` is not a quality target and is not meant to be read as one.
There is no defensible universal figure — a motor book and a fire book
discriminate to different degrees for reasons that have nothing to do with
model quality — but a Gini at or below zero says the rating structure orders
risk no better than chance, or backwards, and that is a finding on any book.
Raise it to your own portfolio's benchmark if you have one.

`monotonic_features` is empty because the constraint is a claim about your
product and its regulator. A declared factor the check cannot evaluate — a
misspelled column, a categorical one, one that is constant on the validation
set — **blocks** rather than skipping, since a typo would otherwise produce a
green gate on an unverified regulatory constraint.

`monotonicity_grid_points * monotonicity_max_rows` is the number of extra
predictions each declared factor costs. At the defaults that is 2,000 rows
scored per factor.

```python
config.actuarial.monotonic_features = {"prior_claims": "increasing"}
config.actuarial.max_overall_ae_deviation = 0.02  # tighter for a filed rate
config.actuarial.dislocation_threshold = 0.15
```

## Security

| Field | Default | Notes |
|---|---|---|
| `adversarial_epsilon` | `0.02` | perturbation size, relative per feature |
| `adversarial_flip_rate_threshold` | `0.05` | classification |
| `adversarial_max_relative_shift` | `0.10` | regression |
| `adversarial_max_rank_shift` | `0.10` | ordinal |
| `pii_patterns` | email, Nigerian phone, NIN/BVN | a `dict[str, regex]` |
| `injection_depth` | `1` | six prompts, one per attack family. `2` fires the whole corpus |
| `injection_families` | `None` | narrow the corpus to the families relevant to your deployment |
| `redact_injection_responses` | `True` | replace `pii_patterns` matches in a stored response with `[redacted:<type>]` |
| `injection_response_chars` | `280` | how much of each response to keep as evidence |
| `extra_injection_prompts` | `[]` | prompts of your own, fired as the `custom` family |
| `report_injection_patterns` | four patterns | instruction-shaped text in feature names and model-card values |

```python
config.security.pii_patterns = {
    **config.security.pii_patterns,
    "iban": r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b",
}
config.security.injection_depth = 2  # the whole corpus; costs more calls
config.security.injection_families = ["encoding", "context_flooding"]
```

!!! warning "`injection_depth` is a bill, not a thoroughness dial"
    Every corpus prompt is a billed generation **on every surface you
    supply** — so depth 2 against both `generate_fn` and `inject_fn` is
    roughly thirty calls per gate run. The count is logged at INFO before the
    calls are made, and recorded in each result's metadata after.

The two things that actually decide whether the injection check can reach a
verdict are not thresholds at all: they are `context.canaries` and
`context.inject_fn`. See [Generative side-cars](../security.md).

## From a file

`--config` accepts JSON, YAML or TOML. CLI flags take precedence over the
file, so a pipeline can pin one threshold inline.

```yaml
performance:
  metric: rmse
  max_error: 45000.0
fairness:
  loss_ratio_threshold: 0.08
actuarial:
  max_overall_ae_deviation: 0.02
  monotonic_features:
    prior_claims: increasing
    deductible: decreasing
security:
  adversarial_epsilon: 0.01
```

YAML needs `pyyaml`; TOML needs `tomli` on Python < 3.11.

## Deprecated

| Old | New | Since |
|---|---|---|
| `PerformanceConfig.min_accuracy` | `min_score` + `metric` | 0.2.1 |
| `GateReport.model_auc` | `model_metric` + `model_score` | 0.2.1 |
| `SecurityConfig.jailbreak_prompts` | `extra_injection_prompts` | 0.5.4 |

All three still work and emit a `DeprecationWarning`.

`min_accuracy` was misleading: it was compared against ROC AUC when
scikit-learn was installed and accuracy when it was not, with nothing in the
report saying which.

`jailbreak_prompts` described the old design. Three canned prompts *were* the
whole injection check until 0.5.4; they are now the smallest part of a
categorised corpus, and the interesting knobs are which families to fire and
how deep rather than which three strings to send.
