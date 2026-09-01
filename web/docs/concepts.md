# Concepts

Four objects and one contract.

## The context

`StructuredGateContext` bundles everything the checks might need — the model,
the validation data, and whatever optional inputs you have. It is a plain
dataclass; nothing is computed at construction.

```python
context = StructuredGateContext(
    model=my_model,
    X=X_val,
    y_true=y_val,
    y_pred=y_pred,
    protected_df=protected_val,
    latencies_ms=benchmark_latencies,
    cost_per_inference=0.0008,
    model_card=my_model_card,
    expected_loss=technical_premium,
    generate_fn=my_llm_explainer,
    task="binary",
)
```

## Checks

Each check answers one question and returns a list of `CheckResult`. A result
carries a **flag**:

| Flag | Meaning |
|---|---|
| `OK` | passed |
| `NOT_APPLICABLE` | skipped — a needed input or dependency is missing, or the check does not apply to this task |
| `CHECK_ERROR` | the check raised; **always** treated as blocking |
| a risk string | `PROXY_RISK`, `PII_LEAKAGE_RISK`, `LOSS_RATIO_RISK`, … |

`result.is_ok` treats `OK` and `NOT_APPLICABLE` as fine, so a skipped check
never blocks a deploy.

## Graceful degradation

**The gate grades what you give it.** This is the contract the whole library
is built around:

- No `protected_df` → fairness checks report `NOT_APPLICABLE`
- No `model_card` → the compliance check reports `NOT_APPLICABLE`
- No `expected_loss` → loss-ratio parity reports `NOT_APPLICABLE`
- No `X_train` → split overlap and drift report `NOT_APPLICABLE`
- `task="regression"` → classification-only checks report `NOT_APPLICABLE`
- shap not installed → the SHAP check reports `NOT_APPLICABLE`

Skipped checks stay **in the report** rather than disappearing, each with the
reason. A reviewer can see what was not evaluated, which matters more than it
sounds: a governance report that silently omits a check is worse than one that
says it was skipped.

## Is the evidence sound?

Every other number in a report rests on one assumption: that the validation
set is a fair test. Nothing used to check it.

Pass the **training set** as the validation set and the gate would report an
AUC of 0.99, a clean calibration curve and `PASS`. Every fairness figure
beside it would be measured on data the model had memorised. The verdict would
be confident, green, and worthless.

So `validation` is its own category, it **blocks**, and it is reported
**first**. The distinction is worth stating plainly:

> A **performance** finding says *the model is not good enough*.
> A **validation** finding says *you do not yet know whether it is*.

The second is a prior question. If it fires, nothing underneath it means what
it appears to mean.

```python
context = StructuredGateContext(
    model=model,
    X=X_val,
    y_true=y_val,
    y_pred=y_pred,
    X_train=X_train,  # unlocks split overlap and train-serve skew
    model_card={
        "use_case": "credit_scoring",
        "validation_strategy": "out_of_time",  # required, and checked
        ...
    },
)
```

`X_train` does not need to be row-aligned to anything — only its columns and
distributions are read, never its labels. Without it, two of the five checks
report `NOT_APPLICABLE` and the other three still run.

Two of these deserve a note here rather than only in the
[reference](reference/checks.md).

**A random split is the wrong test for a pricing model.** It asks "can the
model predict a policy it has not seen?" when the question is "can it predict
*next quarter*?" — and seasonality, inflation and portfolio mix all leak
backwards through a random split. For the high-risk use cases the gate
requires an out-of-time holdout and records the claim in the model card.

**Drift is the exception that does not block.** An out-of-time holdout
*should* differ a little from training; that is the point of one. A gate that
hard-fails on every seasonal shift gets switched off, so drift routes to a
reviewer instead.

## The verdict

`GateReport.gate_status` applies one rule:

1. Any **blocking** check flagged → <span class="verdict-blocked">BLOCKED</span>
2. Otherwise any flag at all → <span class="verdict-review">NEEDS_REVIEW</span>
3. Otherwise → <span class="verdict-pass">PASS</span>

Validation, performance, compliance and security are blocking. **Fairness is
not.**

That is a design stance, not an oversight. A proxy-correlation finding might
be a genuine problem or a legitimate rating factor, and only someone who knows
the business can say which. A gate that hard-fails on it gets switched off; a
gate that routes it to a reviewer gets used.

## Tasks

`context.task` is `"auto"`, `"binary"`, `"multiclass"` or `"regression"`.
`"auto"` infers from `y_true` and **logs what it inferred**.

Inference is genuinely ambiguous — a claims-frequency target of 0/1/2/3 is
indistinguishable from a four-class problem by shape alone — so the guess is
never silent. Set `task` explicitly for anything you gate on.

Each check declares `supported_tasks`, so a check that cannot answer
meaningfully for your task says so instead of producing a confident number.

## Reports

```python
report.gate_status  # "PASS" | "NEEDS_REVIEW" | "BLOCKED"
report.task  # what it was graded as
report.model_metric  # which metric produced the score
report.model_score
report.flags  # non-OK, non-NOT_APPLICABLE results
report.by_category("fairness")
report.to_json("gate_report.json")  # the archival record
report.to_html("gate_report.html")  # the page a reviewer reads
```

`model_metric` sits next to `model_score` deliberately. A score is
uninterpretable without knowing what produced it, so the report always names
it — including when a metric fell back to a substitute.

The two renderings are for two readers. `to_json` is the record a system
files; `to_html` is one self-contained page — verdict, every finding, and a
chart beside each number whose shape is what has to be judged. See
[Reports](reference/reports.md).
