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
| `--latencies PATH` | one latency in ms per line |
| `--cost-per-inference FLOAT` | |
| `--model-card PATH` | JSON model card |

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
