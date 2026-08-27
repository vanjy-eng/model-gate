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

## Security

| Field | Default | Notes |
|---|---|---|
| `adversarial_epsilon` | `0.02` | perturbation size, relative per feature |
| `adversarial_flip_rate_threshold` | `0.05` | classification |
| `adversarial_max_relative_shift` | `0.10` | regression |
| `adversarial_max_rank_shift` | `0.10` | ordinal |
| `pii_patterns` | email, Nigerian phone, NIN/BVN | a `dict[str, regex]` |
| `jailbreak_prompts` | three canned prompts | a list of strings |

```python
config.security.pii_patterns = {
    **config.security.pii_patterns,
    "iban": r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b",
}
```

## From a file

`--config` accepts JSON, YAML or TOML. CLI flags take precedence over the
file, so a pipeline can pin one threshold inline.

```yaml
performance:
  metric: rmse
  max_error: 45000.0
fairness:
  loss_ratio_threshold: 0.08
security:
  adversarial_epsilon: 0.01
```

YAML needs `pyyaml`; TOML needs `tomli` on Python < 3.11.

## Deprecated

| Old | New | Since |
|---|---|---|
| `PerformanceConfig.min_accuracy` | `min_score` + `metric` | 0.2.1 |
| `GateReport.model_auc` | `model_metric` + `model_score` | 0.2.1 |

Both still work and emit a `DeprecationWarning`. `min_accuracy` was
misleading: it was compared against ROC AUC when scikit-learn was installed
and accuracy when it was not, with nothing in the report saying which.
