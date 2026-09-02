# BDP Model Gate

[![PyPI](https://img.shields.io/pypi/v/bdp-model-gate)](https://pypi.org/project/bdp-model-gate/)
[![Python versions](https://img.shields.io/pypi/pyversions/bdp-model-gate)](https://pypi.org/project/bdp-model-gate/)
[![License: MIT](https://img.shields.io/pypi/l/bdp-model-gate)](LICENSE)

**📖 [Documentation](https://vanjy-eng.github.io/model-gate/)** — guide, API reference and runnable examples.

Automated pre-deployment ML model governance: fairness, performance,
compliance, security and validation-methodology checks, run as a single
gate that gives you a
`PASS` / `NEEDS_REVIEW` / `BLOCKED` status to wire into CI before a model
is promoted to production.

Covers **structured data models** for **binary classification, multiclass
(including ordinal) and regression**. Unstructured
(text, image, audio) support is planned — see `bdp_model_gate.unstructured`
for the reserved interface.

## Install

Available on [PyPI](https://pypi.org/project/bdp-model-gate/):

```bash
# core (context/report/gate objects only — no check logic that needs ML libs)
pip install bdp-model-gate

# structured-data checks (fairlearn, shap, scikit-learn) — install this for real use
pip install bdp-model-gate[structured]

# charts, and charts in the HTML report (matplotlib, seaborn)
pip install bdp-model-gate[plots]

# for running the test suite
pip install bdp-model-gate[dev]
```

Compliance and security checks (model card validation, adversarial
robustness, PII scanning, prompt-injection testing) work with just the core
install. Fairness checks need `fairlearn`/`shap`, and every performance
metric except `accuracy` needs `scikit-learn` — install the `structured`
extra to get all of it. On a core-only install the default `metric="auto"`
falls back to `accuracy` and says so loudly; see
[Choosing the performance metric](#choosing-the-performance-metric).

`plots` is separate again, and optional: without it `report.to_html()` still
writes a full report, just without the charts.

## Quickstart

```python
from bdp_model_gate import StructuredGateContext, ModelGate

context = StructuredGateContext(
    model=my_model,
    X=X_val,
    y_true=y_val,
    y_pred=y_pred,
    protected_df=protected_val,  # optional — enables fairness checks
    X_train=X_train,  # optional — enables split-overlap and drift checks
    latencies_ms=benchmark_latencies,  # optional — enables performance checks
    cost_per_inference=0.0008,  # optional
    model_card=my_model_card,  # optional — enables compliance checks
    generate_fn=None,  # optional — set if there's a generative side-car
)

report = ModelGate().run(context)
print(report.summary())
report.to_json("gate_report.json")
report.to_html("gate_report.html")  # the page a reviewer reads

if report.gate_status == "BLOCKED":
    raise SystemExit("Model failed governance gate — see gate_report.json")
```

`model` can be a scikit-learn estimator, a Keras model, a LightGBM or
XGBoost sklearn-API model, or your own class — anything with `.predict()`.
For a PyTorch module, a raw `Booster` or a remote endpoint, pass a function
instead; see [Any model, not just scikit-learn](#any-model-not-just-scikit-learn).

Or the one-liner:

```python
from bdp_model_gate import run_structured_gate

report = run_structured_gate(model, X_val, y_val, y_pred, protected_df=protected_val)
```

## What each category checks

**Validation** (blocking — and reported first)
- `LeakageCheck` — a feature whose solo predictive power rivals the whole
  model's, the signature of a leaked target
- `SplitOverlapCheck` — rows the model has already seen, plus duplicates
  within the validation set
- `ValidationStrategyCheck` — was the holdout separated in time, or at
  random? Out-of-time is required for high-risk use cases
- `FeatureContractCheck` — the columns the model was fitted on, in the order
  it expects them
- `FeatureDriftCheck` — train-serve skew (non-blocking: an out-of-time
  holdout *should* differ a little)

Nothing used to stop you passing the **training set** as the validation set.
The gate reported a superb score and `PASS`, and every fairness figure beside
it was measured on data the model had memorised. A performance finding says
the model is not good enough; a validation finding says you do not yet know
whether it is, which is a prior question — so these block, and lead the
report.

**Actuarial — pricing** (blocking, except dislocation)
- `ActualVsExpectedCheck` — did the book collect what it needed to? The level,
  and then the same ratio band by band, because an overall A/E of 1.00 is
  routinely produced by a model subsidising its worst risks out of its best
- `RiskDiscriminationCheck` — the exposure-weighted **Lorenz Gini**: does the
  rating structure order risk at all? A negative value means the ordering is
  *inverted*, which no error metric shows
- `MonotonicityCheck` — does premium still rise with prior claims? The filed
  constraint, checked empirically by partial dependence. A declared factor that
  cannot be evaluated blocks rather than skipping
- `DislocationCheck` — replacing an incumbent, how much of the book moves by
  more than a quarter, and which group carries it? **Non-blocking**: a
  dislocated book may be entirely correct, and that is a judgement

These are what `context.exposure` exists for. On a rate target an unweighted
metric answers a different question: a policy written for one month and one
written for twelve are not equal evidence about a claims rate, and an
unweighted RMSE says they are.

**Fairness** (non-blocking by default — routes to `NEEDS_REVIEW`, since some
flags need human judgment)
- `ProxyCorrelationCheck` — input features that correlate with a protected attribute
- `DisparateImpactCheck` — outcome-level demographic parity
- `ShapSubgroupCheck` — features whose SHAP contribution differs across groups
- `CounterfactualFlipCheck` — prediction shift when a protected attribute is flipped
- `EqualisedOddsCheck` — *separation*: equal opportunity (the TPR gap) and
  equalised odds (the wider of the TPR and FPR gaps)
- `SubgroupCalibrationCheck` — *sufficiency*: does a score of 0.7 carry the
  same real risk for every group?

The first four measure *independence*. All three families cannot hold at once
whenever base rates differ between groups, so the suite reports each of them
rather than picking one silently.

**Fairness — regression** (non-blocking; see [Regression models](#regression-models))
- `LossRatioParityCheck` — margin charged over each group's own expected loss
- `GroupMeanGapCheck` — raw spread in mean prediction across groups
- `ErrorParityCheck` — is the model materially worse for one group?
- `CalibrationParityCheck` — systematic over- or under-prediction per group

**Performance** (blocking)
- `PerformanceThresholdCheck` — model score on a metric you choose, p95
  latency, cost-per-inference. See [Choosing the performance metric](#choosing-the-performance-metric).
- `CalibrationCheck` — do the stated probabilities match observed
  frequencies? Discrimination and calibration are independent: a model can
  rank perfectly while every probability it emits is twice too high.

**Compliance** (blocking)
- `ComplianceMappingCheck` — model card completeness, DPIA trigger for
  high-risk use cases, explainability requirement for models affecting a person

**Security** (blocking)
- `AdversarialRobustnessCheck` — prediction flip rate under small feature perturbation
- `PIILeakageCheck` — regex scan of string columns for PII patterns
- `PromptInjectionCheck` — canned jailbreak prompts against any generative side-car

## Customizing thresholds

```python
from bdp_model_gate import GateConfig
from bdp_model_gate.structured import default_structured_checks
from bdp_model_gate import ModelGate

config = GateConfig()
config.performance.metric = "roc_auc"
config.performance.min_score = 0.85
config.fairness.disparity_threshold = 0.05

gate = ModelGate(checks=default_structured_checks(config))
report = gate.run(context)
```

## Choosing the performance metric

`PerformanceConfig.metric` decides what the model is scored on, and
`min_score` is the threshold that score must clear. Set the two together —
`min_score` means nothing on its own.

```python
config = GateConfig()
config.performance.metric = "f1"  # what to measure
config.performance.min_score = 0.75  # what it has to beat
```

Built-in names: `roc_auc`, `average_precision`, `accuracy`,
`balanced_accuracy`, `f1`, `precision`, `recall`. All except `accuracy`
require scikit-learn (the `structured` extra).

**Label-based metrics need hard classes.** `accuracy`, `balanced_accuracy`,
`f1`, `precision`, and `recall` binarize continuous `y_pred` at
`config.performance.decision_threshold` (default `0.5`). Predictions already
in `{0, 1}` are left alone. Ranking metrics (`roc_auc`,
`average_precision`) use the raw scores and ignore the threshold.

**Your own metric.** Any `fn(y_true, y_pred) -> float` works, and is called
with `y_pred` exactly as you supplied it — no thresholding, since only you
know what your metric expects:

```python
from sklearn.metrics import fbeta_score


def f2(y_true, y_pred):
    return fbeta_score(y_true, (y_pred >= 0.3).astype(int), beta=2)


config.performance.metric = f2  # reported under the name "f2"
```

**`"auto"` (the default)** uses `roc_auc` when scikit-learn is installed and
falls back to `accuracy` when it isn't. The fallback is never silent: it's
logged at `WARNING`, marked `metric_is_fallback: true` in the result
metadata, and spelled out in the check's detail string. A score is only
comparable to `min_score` if you know which metric produced it, so the
report always names it:

```json
{
  "gate_status": "PASS",
  "model_metric": "roc_auc",
  "model_score": 0.9132
}
```

Naming a metric explicitly opts out of fallback entirely — if
`metric="roc_auc"` can't run, the gate reports a blocking `CHECK_ERROR`
rather than quietly scoring you on something else. A typo'd metric name
raises `GateConfigurationError` as soon as the check is constructed.

From the CLI, `--metric`, `--min-score`, and `--decision-threshold` do the
same thing, and take precedence over a `--config` file:

```bash
bdp-model-gate --model model.joblib --data validation.csv --target-col label \
  --metric f1 --min-score 0.75 --output gate_report.json
```

> **Migrating from 0.1.0:** `min_accuracy` is now `min_score`, and the old
> name was misleading — it was compared against ROC AUC whenever
> scikit-learn was installed, and accuracy otherwise. `min_accuracy` still
> works (in Python and in `--config` files) but emits a `DeprecationWarning`.
> Likewise `GateReport.model_auc` is superseded by `model_metric` /
> `model_score`, and now returns `None` unless the metric really was AUC.

## Writing your own check

```python
from bdp_model_gate import BaseCheck, CheckResult


class MyCustomCheck(BaseCheck):
    name = "my_custom_check"
    category = "compliance"  # validation | fairness | performance | compliance | security
    blocking = True

    def run(self, context):
        # inspect context.model, context.X, context.model_card, etc.
        return [CheckResult(self.name, self.category, "OK", "looks fine", self.blocking)]


gate = ModelGate(checks=[MyCustomCheck()])
```

## Using it as a pre-deployment CI/CD gate

Installing the package gives you an `bdp-model-gate` console script, meant to
run as a **pre-deployment step** — after a model is trained/built, before
it's promoted to a registry or prod endpoint. It is not intended to run on
every PR.

```bash
bdp-model-gate \
  --model model.joblib \
  --data validation.csv \
  --target-col label \
  --protected protected.csv \
  --model-card model_card.json \
  --cost-per-inference 0.0008 \
  --output gate_report.json
```

Exit codes are chosen so a pipeline can distinguish three outcomes:

| Exit code | Status | Pipeline behavior |
|---|---|---|
| `0` | `PASS` | proceed to deploy automatically |
| `2` | `NEEDS_REVIEW` | stop and require a human sign-off (fairness flags need judgment) |
| `1` | `BLOCKED` | hard fail — performance, compliance, or security check failed |

A ready-to-adapt **Azure Pipelines** example is in
[`ci_examples/azure-pipelines.model-gate.yml`](ci_examples/azure-pipelines.model-gate.yml),
and a **GitHub Actions** equivalent (a reusable `workflow_call` workflow) is in
[`ci_examples/github-actions.model-gate.yml`](ci_examples/github-actions.model-gate.yml).
Both structure this as three stages/jobs: run the gate, a manual-approval
step gated behind exit code `2` (GitHub Environments / Azure Environments
with required reviewers), and a deploy step that only runs if the gate
passed outright or was manually approved. Point them at wherever your
training pipeline publishes `model.joblib` / `validation.csv` /
`protected.csv` / `model_card.json` as a build artifact.

Config overrides for the CLI can be JSON, YAML, or TOML — pick whichever
matches your repo's conventions:

```yaml
# config.yaml
performance:
  metric: f1
  min_score: 0.85
  decision_threshold: 0.5
fairness:
  disparity_threshold: 0.05
```

```bash
bdp-model-gate --model model.joblib --data validation.csv --target-col label \
  --config config.yaml --output gate_report.json
```

YAML configs need `pip install pyyaml` (or `bdp-model-gate[dev]`, which
already includes it); TOML needs `tomli` on Python < 3.11 (3.11+ has
`tomllib` built in).

Pass `-v`/`--verbose` for debug-level logging (per-check timing, which
checks ran/skipped and why) — the library uses the standard `logging`
module throughout, so it composes with whatever logging setup your
pipeline already has.

## The report a reviewer reads

`PASS` and `BLOCKED` need no page — the pipeline acts on the exit code.
`NEEDS_REVIEW` delegates the decision to a person, and that person should not
be handed a JSON blob.

```python
report.to_html("gate-report.html", title="Retail credit scorecard v4")
```

One self-contained file. No script, no stylesheet, no font, no image fetched
from anywhere: a governance record gets emailed, filed and reopened years
later, and every external reference is a way for it to stop rendering. It
opens offline and prints clean.

Charts are inlined as SVG rather than `<img src="data:...">`, so they inherit
the page's CSS — one render reads correctly in light and dark — and stay
sharp on paper.

### Thirteen checks draw; twelve deliberately do not

A check draws only where it **collapses a distribution to a scalar and the
shape is what you need to judge**. Latency, cost and model-card completeness
are genuinely scalars, and a binary confusion matrix is four numbers the
detail line already carries — charting those would be decoration.

| Plot | Check | What the number cannot say |
|---|---|---|
| Reliability curve | `calibration` | two models with the same ECE can be wrong in opposite directions |
| Reliability per group | `subgroup_calibration` | where the aggregate hides a minority |
| TPR/FPR bars | `equalised_odds` | which notion is failed, and by how much |
| η² heatmap | `proxy_correlation` | replaces a forty-row table; the eye finds the hot cell |
| Threshold sweep | `disparate_impact` | whether the verdict survives a different cutoff |
| Actual-vs-expected by band | `calibration_parity` | *where* in the book the pricing is wrong |
| Loss-ratio scatter | `loss_ratio_parity` | whether the margin gap is flat or grows with the risk |
| Ordinal confusion | `performance_thresholds` | the direction of the error, which `quadratic_kappa` hides |
| Robustness sweep | `adversarial_robustness` | a cliff versus a slope |
| A/E by band | `actual_vs_expected` | one bad decile versus a tilt across all of them |
| Lorenz curve | `risk_discrimination` | how much of the attainable discrimination was captured |
| Partial dependence | `monotonicity` | whether the curve dips once or sags through the middle |
| Change histogram | `prediction_dislocation` | a bump past the threshold versus a long tail |

The robustness sweep is opt-in — `AdversarialRobustnessCheck(plot_sweep=True)`
— because each point re-scores the sample, which is a real bill against a
metered endpoint.

### Composing into your own figures

Every `plot()` takes an optional matplotlib `Axes` and returns **the same
one**. We draw onto your canvas and hand it back; this library does not
replace your plotting stack.

```python
import matplotlib.pyplot as plt
from bdp_model_gate.structured.calibration_checks import (
    CalibrationCheck,
    SubgroupCalibrationCheck,
)

fig, (left, right) = plt.subplots(1, 2, figsize=(11, 5))
CalibrationCheck().plot(context, ax=left)
SubgroupCalibrationCheck().plot(context, ax=right)
fig.savefig("fairness.svg")
```

Override `plot()` on your own check and the report picks it up — discovery is
by override, so there is nothing to register.

### It degrades; it never fails

No `[plots]` extra, no context, or a `plot()` that raises: you lose a chart,
never a finding. A renderer that threw and took the results with it would be
worse than the JSON it replaces.

## Extending with plugins

Third-party packages can register additional checks without forking this
library, via the `bdp_model_gate.checks` entry-point group:

```toml
# in your plugin package's pyproject.toml
[project.entry-points."bdp_model_gate.checks"]
my_check = "my_package.checks:MyCustomCheck"
```

Once installed alongside `bdp-model-gate`, `default_structured_checks()`
picks it up automatically (pass `include_plugins=False` to opt out). A
plugin that fails to import or isn't a `BaseCheck` subclass is logged and
skipped rather than crashing the gate.

## Error handling

Bad inputs fail fast with a clear message rather than a confusing
exception from deep inside a check:

```python
from bdp_model_gate import ModelGate, StructuredGateContext
from bdp_model_gate.exceptions import GateValidationError

try:
    report = ModelGate().run(context)
except GateValidationError as exc:
    print(f"Fix your inputs: {exc}")
```

Validation covers: the model exposes `.predict()`, `X` is a non-empty
DataFrame, `y_true`/`y_pred`/`X` are aligned in length, `y_true` has at
least two classes, `protected_df` is row-aligned and has no all-NaN
columns, `model_card` is a dict, `generate_fn` is callable, and
`latencies_ms` has no negative values.

## Any model, not just scikit-learn

Nothing here imports a deep-learning framework. Instead of requiring a
particular object shape, the gate accepts a plain function:

```python
import torch

net.eval()

context = StructuredGateContext(
    X=X_val,
    y_true=y_val,
    y_pred=y_pred,
    task="regression",
    # DataFrame in, array out — your function owns tensor conversion,
    # device placement and batching.
    predict_fn=lambda df: net(torch.tensor(df.values, dtype=torch.float32)).detach().numpy(),
)
```

`model` is optional: a remote scoring endpoint has no model object at all,
so `predict_fn` alone is a complete context. A bare callable also works as
`model=`, so the two routes are interchangeable.

| Field | Type | Unlocks |
|---|---|---|
| `predict_fn` | `fn(DataFrame) -> array` | everything; takes precedence over `model` |
| `predict_proba_fn` | `fn(DataFrame) -> array` | `CounterfactualFlipCheck` |
| `gradient_fn` | `fn(DataFrame) -> (n_rows, n_features)` | a real targeted adversarial attack |

**Probability shapes are normalised.** A Keras sigmoid returns `(n, 1)`,
scikit-learn returns `(n, 2)`, and a custom model might return `(n,)`. All
three mean the same thing and are reduced to one positive-class vector, so
you don't have to know which the library expects. A genuinely multiclass
`(n, k)` output is refused with a clear message rather than silently sliced.

**Gradients make the robustness check real.** `AdversarialRobustnessCheck`
prefers true per-row gradients, falls back to `coef_` for linear models, and
only then to random noise. Supplying `gradient_fn` turns a weak random probe
into a targeted attack; the method used is recorded in the result metadata.

```python
context.gradient_fn = lambda df: compute_input_gradients(net, df)  # -> (n, n_features)
```

### From the CLI

`joblib` only reads pickles, so `--model-loader` names a function that
returns a model or a scoring callable. Your loader does the framework
import:

```python
# mypkg/serving.py
def load_scorer():
    net = torch.load("model.pt")
    net.eval()
    return lambda df: net(torch.tensor(df.values).float()).detach().numpy()
```

```bash
bdp-model-gate --model-loader "mypkg.serving:load_scorer" \
  --data validation.csv --target-col realised_loss --task regression \
  --metric rmse --max-error 5000 --output gate_report.json
```

> Note: `roc_auc`, `average_precision`, `balanced_accuracy`, `f1`,
> `precision` and `recall` still need scikit-learn. That is a *metrics*
> dependency, not a model one — the regression metrics and `accuracy` are
> numpy-native and work on a core install.

## Multiclass and ordinal models

Set `task="multiclass"`. If the classes have a natural ordering — an
underwriting decision, a risk tier — supply `class_order` too, listed from
least to most favourable:

```python
context = StructuredGateContext(
    model=underwriter,
    X=X_val,
    y_true=decisions,
    y_pred=predicted,
    protected_df=protected_val,
    task="multiclass",
    class_order=["decline", "refer", "accept"],  # marks it ordinal
    favourable_classes=["accept"],  # defaults to the last entry
)

config = GateConfig()
config.performance.metric = "quadratic_kappa"
config.performance.min_score = 0.70
```

### Why ordering matters

Plain multiclass metrics count errors. They cannot see that predicting
**decline** on an application that should have been **accepted** is worse
than predicting **refer** — both are simply "one mistake". For an
underwriting gate that distinction is the whole point.

Two metrics use the ordering:

| Metric | Direction | What it measures |
|---|---|---|
| `ordinal_mae` | lower better (`max_error`) | mean error in *rank* space — decline-for-accept is 2, refer-for-accept is 1 |
| `quadratic_kappa` | higher better (`min_score`) | chance-corrected agreement, penalising a disagreement by the **square** of its rank distance |

Ordering also sharpens the robustness check. `AdversarialRobustnessCheck`
reports the mean rank *distance* a prediction moves under perturbation
alongside the flip rate, because two models can flip at an identical rate
while one wobbles by a single rank and the other swings across the scale.

For nominal problems with no ordering, omit `class_order` — `accuracy`,
`balanced_accuracy`, `f1`, `precision` and `recall` all work, averaged per
`config.performance.average` (default `"macro"`, which weights every class
equally so a rare "decline" counts as much as a common "accept").

### Fairness needs a favourable outcome

Demographic parity counts a selected class. With three outcomes, which one
counts as selected is a judgement the data cannot supply, so
`favourable_classes` decides it — defaulting to the most favourable entry of
`class_order`, and reporting `NOT_APPLICABLE` when neither is given rather
than guessing.

The choice genuinely changes what you measure. "Was accepted" and "was not
declined" are different questions:

```python
context.favourable_classes = ["accept"]  # were they approved?
context.favourable_classes = ["accept", "refer"]  # were they spared a decline?
```

`CounterfactualFlipCheck` measures the shift in P(favourable outcome), and
`ShapSubgroupCheck` explains the favourable class column — so it answers
"does this feature push some groups away from being accepted?" rather than
averaging across unrelated classes.

> `roc_auc` and `average_precision` stay binary-only. Their multiclass forms
> need a full probability matrix, which the `y_pred` contract does not
> carry, so they are refused rather than quietly approximated.

From the CLI:

```bash
bdp-model-gate --model underwriter.joblib --data validation.csv \
  --target-col decision --task multiclass \
  --class-order "decline,refer,accept" --favourable-classes accept \
  --metric quadratic_kappa --min-score 0.70 --output gate_report.json
```

## Regression models

Set `task` and the suite reconfigures itself. Classification-only checks
report `NOT_APPLICABLE` rather than being dropped, so the report still shows
what was skipped and why.

```python
from bdp_model_gate import GateConfig, ModelGate, StructuredGateContext

context = StructuredGateContext(
    model=pricing_model,
    X=X_val,
    y_true=realised_loss,
    y_pred=quoted_premium,
    protected_df=protected_val,
    expected_loss=technical_premium,  # enables loss-ratio parity
    task="regression",
)

config = GateConfig()
config.performance.metric = "rmse"
config.performance.max_error = 5000.0  # error metrics use max_error
```

`task` defaults to `"auto"`, which infers from `y_true` and **logs what it
inferred**. Set it explicitly for anything you gate on: a claims-frequency
target of 0/1/2/3 is indistinguishable from a four-class problem by shape.

**Metrics.** `rmse`, `mae`, `mape`, `poisson_deviance` (for count targets
like claims frequency) and `r2`. All are implemented in numpy, so they work
on a core install. `"auto"` picks `r2`, because an RMSE default threshold
would be meaningless without knowing whether the target is naira or claims.

**Thresholds have a direction.** Higher-is-better metrics use `min_score`;
error metrics use `max_error`. There is no default `max_error` — a ceiling
depends entirely on your target's scale — so configuring an error metric
without one raises `GateConfigurationError` instead of passing silently.

### Fairness without a "selected" class

Demographic parity counts a favourable class, which a continuous target does
not have. Four checks replace it, and the distinction matters most in
insurance:

| Check | Question | Needs |
|---|---|---|
| `LossRatioParityCheck` | Is one group charged a higher **margin over its own expected loss**? | `expected_loss` |
| `GroupMeanGapCheck` | Does one group get systematically higher predictions? | — |
| `ErrorParityCheck` | Is the model materially less accurate for one group? | `y_true` |
| `CalibrationParityCheck` | Does one group's prediction over- or under-shoot reality? | `y_true` |

A pricing model *should* charge more in a higher-loss segment — that is
risk-based pricing, not discrimination — so `GroupMeanGapCheck` on its own
flags legitimate rating differences and will be noisy. `LossRatioParityCheck`
is the one that isolates unfairness from actuarially justified variation, by
comparing the **margin** each group is charged over its own expected cost.
It needs `context.expected_loss` (a per-row expected loss, technical premium
or pure premium) and reports `NOT_APPLICABLE` without it rather than
silently answering the raw-price question under the same name.

All four gaps are measured relative to the overall figure, so one threshold
works across scales, and groups smaller than `FairnessConfig.min_group_size`
(default 30) are reported but not scored — a three-policy segment otherwise
produces a wild ratio that reads as a finding.

All four are also **exposure-weighted** when `context.exposure` is supplied,
and each detail string says whether it was. `min_group_size` still counts
*rows*: three policies are three policies however long they ran.

Adversarial robustness also changes shape: a "prediction flip" is
meaningless for a continuous output (every perturbation moves it), so
regression measures the mean relative prediction shift against
`SecurityConfig.adversarial_max_relative_shift`.

From the CLI:

```bash
bdp-model-gate --model pricing.joblib --data validation.csv \
  --target-col realised_loss --task regression \
  --expected-loss-col technical_premium \
  --exposure-col earned_vehicle_years \
  --baseline-col premium_v3 \
  --metric lorenz_gini --min-score 0.15 --output gate_report.json
```

## Roadmap

See [`ROADMAP.md`](ROADMAP.md) for the detail and the decisions behind each.

| Release | Theme |
|---|---|
| **0.5.3** | Exposure weighting, actual-vs-expected, monotonicity and dislocation — the actuarial measures. |
| **0.5.4** | Prompt injection, properly — canary-based leak detection, indirect injection, and a real attack corpus. |
| **0.6.0** | Confidence intervals on every metric, plus pinned lint tooling. |
| **0.6.1** | Release automation — publish on tag via Trusted Publishing, PyPI behind a required reviewer. |
| **1.0.0** | A public, subclassable `ModelAdapter`. |
| Later | Unstructured data (text/image/audio). |

Each release ships as a complete slice: implementation, tests, example
notebooks, and the documentation pages that describe it.

## Contributing

Issues, fixes, checks, docs and examples are all welcome. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for the development setup, the testing
standards, and how to add a check or a plot.

```bash
pip install -e ".[dev,structured,plots,yaml,toml]"

ruff check .              # lint
ruff format .             # format
mypy bdp_model_gate       # type check
pytest -q                 # test (85% coverage floor enforced)
```

One thing worth knowing before you read further: the failure this project
guards against is not a crash but a **confident, wrong, green number**. That
shapes most of the conventions, and `CONTRIBUTING.md` explains them.

## Examples

Seven runnable notebooks live in [`examples/`](examples/), committed with
outputs so they read without being run:

| Notebook | Covers |
|---|---|
| [01 binary classification](examples/01_binary_classification_sklearn.ipynb) | credit scoring, and the library end to end — **start here** |
| [02 multiclass and ordinal](examples/02_multiclass_ordinal_sklearn.ipynb) | underwriting accept / refer / decline |
| [03 regression](examples/03_regression_sklearn.ipynb) | motor premium, claims severity and frequency |
| [04 PyTorch and friends](examples/04_any_framework_classification.ipynb) | `predict_fn`, `gradient_fn`, remote endpoints |
| [05 boosters and the CLI](examples/05_boosters_and_cli.ipynb) | XGBoost `Booster`, `--model-loader` |
| [06 reports and plots](examples/06_reports_and_plots.ipynb) | the thirteen charts, and the HTML report |
| [07 insurance pricing](examples/07_insurance_pricing_end_to_end.ipynb) | exposure, A/E, the Gini, monotonicity and dislocation on one motor book |

See [`CHANGELOG.md`](https://github.com/vanjy-eng/model-gate/blob/main/CHANGELOG.md) for release history.
