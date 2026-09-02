# Getting started

## Install

=== "Full (recommended)"

    ```bash
    pip install "bdp-model-gate[structured]"
    ```

    Pulls scikit-learn, fairlearn and shap, which the fairness checks and
    most metrics need.

=== "Core only"

    ```bash
    pip install bdp-model-gate
    ```

    Compliance, PII and prompt-injection checks work with just this. Checks
    needing the extra libraries report `NOT_APPLICABLE` rather than failing,
    and regression metrics still work — they are implemented in numpy.

Python 3.9 – 3.13.

## Gate a model

```python
from bdp_model_gate import ModelGate, StructuredGateContext

context = StructuredGateContext(
    model=my_model,
    X=X_val,
    y_true=y_val,
    y_pred=y_pred,  # positive-class probability, for binary
    protected_df=protected_val,  # optional — enables fairness
    model_card=my_model_card,  # optional — enables compliance
    task="binary",
)

report = ModelGate().run(context)
print(report.summary())
report.to_json("gate_report.json")  # for the system that files it
report.to_html("gate_report.html")  # for the person who signs it off
```

```text
Gate status: NEEDS_REVIEW (2741ms, binary)
  roc_auc: 0.8637
  performance: 0 flag(s)
  compliance: 0 flag(s)
  security: 0 flag(s)
  fairness: 4 flag(s)
```

Only `model` (or a `predict_fn`), `X`, `y_true` and `y_pred` are required.
Every other field is optional, and omitting one makes the checks that need it
report `NOT_APPLICABLE` instead of failing. See
[Concepts](concepts.md#graceful-degradation).

## Read the verdict

```python
if report.gate_status == "BLOCKED":
    raise SystemExit("Model failed governance — see gate_report.json")

for flag in report.flags:
    print(flag.check_name, flag.detail)
```

| Status | Meaning | Your pipeline should |
|---|---|---|
| <span class="verdict-pass">PASS</span> | nothing flagged | deploy automatically |
| <span class="verdict-review">NEEDS_REVIEW</span> | only non-blocking flags | pause for human sign-off |
| <span class="verdict-blocked">BLOCKED</span> | a blocking check failed | hard-fail the build |

## From the command line

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

Exit codes are `0`, `2` and `1` for the three statuses, so a pipeline can tell
*deploy* from *ask a human*. Full options in the [CLI reference](reference/cli.md).

## Next

- [Concepts](concepts.md) — how the pieces fit
- Your task: [binary](tasks/binary.md) · [multiclass](tasks/multiclass.md) · [regression](tasks/regression.md)
- [Reports and plots](reference/reports.md) — what a reviewer actually reads
- [Examples](examples/index.md) — eight runnable notebooks
