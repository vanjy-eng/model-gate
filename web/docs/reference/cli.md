# Command line

Installing the package provides a `bdp-model-gate` console script, intended as
a **pre-deployment step** — after training, before promotion. Not a per-PR
check.

```bash
bdp-model-gate \
  --model model.joblib \
  --data validation.csv \
  --target-col label \
  --protected protected.csv \
  --model-card model_card.json \
  --task binary \
  --metric roc_auc --min-score 0.80 \
  --output gate_report.json
```

## Exit codes

| Code | Status | Pipeline should |
|---|---|---|
| `0` | <span class="verdict-pass">PASS</span> | deploy |
| `2` | <span class="verdict-review">NEEDS_REVIEW</span> | pause for approval |
| `1` | <span class="verdict-blocked">BLOCKED</span> | hard fail |

Three codes, not two, so a pipeline can tell *deploy* from *ask a human*.

## Options

### Model

| Flag | Notes |
|---|---|
| `--model PATH` | a joblib-serialised model |
| `--model-loader "pkg.mod:factory"` | a function returning a model or `fn(DataFrame) -> array` |

Mutually exclusive. Use `--model-loader` for anything joblib cannot
unpickle — PyTorch checkpoints, Keras SavedModel, ONNX, a remote endpoint.
**Your loader does the framework import**, so this package needs no
deep-learning dependency.

```python
# mypkg/serving.py
def load_scorer():
    net = torch.load("model.pt")
    net.eval()
    return lambda df: net(torch.tensor(df.values).float()).detach().numpy()
```

### Data

| Flag | Notes |
|---|---|
| `--data PATH` | CSV of validation data — **required** |
| `--target-col NAME` | ground-truth column — **required** |
| `--protected PATH` | CSV of protected attributes, row-aligned |
| `--train-data PATH` | CSV of the **training** features — unlocks split-overlap and drift |
| `--expected-loss-col NAME` | column holding per-row expected loss |
| `--exposure-col NAME` | column holding per-row exposure — weights the regression metrics and the pricing checks |
| `--baseline-col NAME` | column holding the incumbent model's prediction — unlocks the dislocation check |
| `--latencies PATH` | one latency in ms per line |
| `--cost-per-inference FLOAT` | |
| `--model-card PATH` | JSON model card |
| `--generate-loader "pkg.mod:factory"` | a factory returning `fn(str) -> str` — the generative side-car, direct surface |
| `--inject-loader "pkg.mod:factory"` | a factory returning `fn(payload) -> str` — the indirect surface, where retrieved content goes |
| `--canaries-file PATH` | strings that must never appear in generated output, one per line |

The three column flags name columns of `--data` that are **not** features, and
each is read out and then dropped from `X`. Leaving last quarter's premium in
the feature frame would hand the model its own answer, which is the leak
`target_leakage` exists to find.

`--exposure-col` belongs on a target expressed as a *rate* — claims per
vehicle-year, loss cost per sum-insured-year. Leave it off when the target is
a per-policy total, where the exposure is already inside the value. See
[Insurance pricing](../tasks/insurance.md).

```bash
bdp-model-gate \
  --model premium_v4.joblib \
  --data motor_holdout.csv \
  --target-col realised_loss \
  --exposure-col earned_vehicle_years \
  --baseline-col premium_v3 \
  --expected-loss-col technical_premium \
  --task regression --metric lorenz_gini --min-score 0.15 \
  --config pricing_gate.yaml
```

The injection check was Python-only until 0.5.4 — there is no way to put a
callable on a command line. The three flags above follow `--model-loader`:
**your factory does the SDK import and the credential handling**, so neither is
a dependency of a governance gate.

```bash
bdp-model-gate \
  --model claims_model.joblib \
  --data validation.csv --target-col declined \
  --generate-loader "mypkg.sidecar:load_chat" \
  --inject-loader "mypkg.sidecar:load_claim_summariser" \
  --canaries-file canaries.txt \
  --config gate.yaml
```

`--canaries-file` is a file rather than a flag on purpose: a canary is usually
a sentence from a system prompt, and putting that on a command line puts it in
the shell history and the CI log of every run. One per line, `#` for a
comment. Without canaries the leak probes can only be routed to a human — see
[Generative side-cars](../security.md).

`--train-data` reads columns and distributions only, never labels: the target
column is dropped if present, and the two frames need no row alignment. Give
it and the [validation checks](checks.md#validation-blocking) can tell you
whether the model has already seen the rows it is being graded on.

### Task and scoring

| Flag | Notes |
|---|---|
| `--task {auto,binary,multiclass,regression}` | default `auto`, which infers and logs |
| `--class-order "a,b,c"` | ascending favourability; marks the problem ordinal |
| `--favourable-classes "accept"` | defaults to the last of `--class-order` |
| `--metric NAME` | see [Configuration](configuration.md) |
| `--min-score FLOAT` | higher-is-better metrics |
| `--max-error FLOAT` | error metrics — required when one is selected |
| `--decision-threshold FLOAT` | default `0.5` |
| `--average {macro,micro,weighted}` | multiclass f1 / precision / recall |

### Other

| Flag | Notes |
|---|---|
| `--config PATH` | JSON, YAML or TOML; **CLI flags win** |
| `--output PATH` | default `gate_report.json` |
| `-v`, `--verbose` | debug logging — per-check timing, what ran and why |

## In CI

```yaml
- name: Model governance gate
  id: gate
  continue-on-error: true
  run: |
    bdp-model-gate --model model.joblib --data validation.csv \
      --target-col label --protected protected.csv \
      --model-card model_card.json --output gate_report.json

- name: Block on hard failure
  if: steps.gate.outcome == 'failure'
  run: exit 1

# exit 2 -> route to an environment with required reviewers
```

Ready-to-adapt GitHub Actions and Azure Pipelines examples ship in
[`ci_examples/`](https://github.com/vanjy-eng/model-gate/tree/main/ci_examples).
Both structure it as three stages: run the gate, a manual approval gated
behind exit code `2`, and a deploy that runs only if the gate passed outright
or was approved.
