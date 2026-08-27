# Fairness: three families

Fairness is not one measurement. It splits into three families, and the
central fact about them is that **they cannot all hold at once**.

| Family | Question | Check |
|---|---|---|
| **Independence** | Do selection rates match across groups? | `disparate_impact` |
| **Separation** | Do *error* rates match across groups? | `equalised_odds` |
| **Sufficiency** | Does a given score mean the same thing for every group? | `subgroup_calibration` |

Before 0.5.0 this library measured only independence. That was not a neutral
position — demographic parity ignores `y_true` entirely, so a model can
achieve perfect parity by being wrong in compensating directions, and a reader
seeing one green check had no way to know two other notions were never tested.

## Independence — `disparate_impact`

Demographic parity: the share of each group predicted positive. It needs hard
class labels, so continuous predictions are binarised at
`FairnessConfig.decision_threshold`. For multiclass, "positive" means landing
in `context.favourable_classes`.

Its weakness is that it never looks at the ground truth. A model that approves
50% of every group satisfies it perfectly, whether or not those are the right
50%.

## Separation — `equalised_odds`

Conditions on the ground truth, which is what independence lacks. Two results
per attribute:

- **Equal opportunity** — the true-positive-rate difference. *Among applicants
  who should be approved, is every group equally likely to be?* This is the
  notion lending regulators most often centre on.
- **Equalised odds** — the larger of the TPR and FPR differences. Stricter, and
  the one that matters when a false positive is costly too.

A model can satisfy the first and fail the second: equal opportunity ignores
false positives entirely.

## Sufficiency — `subgroup_calibration`

Does a score of 0.7 carry the same real risk for every group? A model can be
well calibrated overall and badly miscalibrated for a minority, because the
majority dominates the average and hides it.

This is a fairness failure, not merely a performance one — it means the same
stated risk means something different depending on who the applicant is.

The regression analogue is [`calibration_parity`](regression.md), which
compares mean residuals rather than binned frequencies.

## The impossibility

!!! warning "You cannot satisfy all three"
    Calibration, TPR balance and FPR balance are **mathematically
    incompatible** whenever the base rate differs between groups, except in
    degenerate cases (Kleinberg–Mullainathan–Raghavan 2016; Chouldechova
    2017).

This is not a limitation of the tool. It is a property of the problem, and it
means **choosing a fairness notion is a policy decision**, not a technical one.

The gate's job is to stop that choice being made silently. It reports all
three and names the trade-off; which one you gate on belongs to whoever signs
the model off.

[Notebook 01](../examples/01_binary_classification_sklearn.ipynb) demonstrates
it rather than asserting it: rescaling each group's scores to equalise
selection rates moves the parity gap from 0.365 to 0.010 while the calibration
gap goes from 0.067 to 0.131. Independence bought, sufficiency spent.

A tool that reported only demographic parity would let you "fix" a model by
making its scores mean different things for different people, and call that
progress.

## Intersections

Every check above treats each protected attribute independently. **Harm
concentrates where attributes meet**, and marginal checks are blind to it by
construction — a model can look acceptable on gender and on region while
failing badly for women in one region.

```python
config.fairness.intersectional = True
```

Turns on pairwise combinations, reported as `gender × region`. Off by default
because joint groups are smaller and the reading needs more care; an
intersection without two groups of at least `min_group_size` is skipped with a
log line rather than scored.

Only pairs are generated. Three-way intersections fragment a validation set
faster than any realistic `min_group_size` tolerates, and a disparity computed
over four rows is worse than none.

## Overall calibration — `calibration`

Distinct from fairness: do the stated probabilities match reality *at all*?

Discrimination and calibration are independent properties. A model can rank
perfectly — AUC 1.0 — while every probability it emits is twice too high,
which scores beautifully and misprices every policy.

Reported as Expected Calibration Error plus Murphy's decomposition:

| Term | Meaning | Direction |
|---|---|---|
| `reliability` | distance from observed frequency | lower better — recalibration fixes this |
| `resolution` | how much predictions vary from the base rate | **higher** better |
| `uncertainty` | the base rate's own variance | a floor; a property of the problem |

`resolution` is the one people forget. A model predicting the base rate for
everyone is **perfectly calibrated and completely useless** — and neither ECE
nor the Brier score says so alone.

This check is blocking, since it sits in the performance category, but
`max_ece` defaults to a permissive `0.10`. Plenty of good models are
uncalibrated by construction, and a gate that blocks all of them gets switched
off. Tighten it for pricing.

!!! tip "Skewed scores"
    Fraud and default scores cluster near zero, which leaves uniform bins
    nearly empty where it matters. Set
    `config.performance.calibration_strategy = "quantile"` for equal-count
    bins instead.
