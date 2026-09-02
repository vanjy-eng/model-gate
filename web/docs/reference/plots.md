# Plots

We are not replacing your plotting library. Fourteen checks draw a chart, and
each one exists because a scalar loses something a reviewer needs.

```bash
pip install "bdp-model-gate[plots]"
```

Without the extra, `plot()` raises `GateConfigurationError` naming it and the
[HTML report](reports.md) renders text-only — the same way shap and fairlearn
already degrade.

## The contract

```python
ax = check.plot(context, results=None, ax=None)
```

An `Axes` in, the **same** `Axes` out. That is the whole composition story: lay
out your own figure and pass each cell in.

```python
import matplotlib.pyplot as plt
from bdp_model_gate.structured.calibration_checks import (
    CalibrationCheck,
    SubgroupCalibrationCheck,
)

fig, (left, right) = plt.subplots(1, 2, figsize=(11, 5))
CalibrationCheck().plot(context, ax=left)
SubgroupCalibrationCheck().plot(context, ax=right)
fig.suptitle("Sufficiency, overall and by region")
fig.savefig("fairness.svg")
```

Everything after that is yours — restyle it, relabel it, drop it into a slide.

`results` is optional. Pass the results you already have and the plot draws
those; omit it and the check runs itself.

## What gets drawn, and why

Plot where a check collapses a distribution to a scalar **and the shape is
what you need to judge**. Latency, cost and model-card completeness are
genuinely scalars; charting them would be decoration. A binary confusion
matrix is four numbers the detail line already carries.

| Plot | Check | What the number cannot say |
|---|---|---|
| Reliability curve | `calibration` | two models with an identical ECE can be miscalibrated in opposite directions |
| Reliability per group | `subgroup_calibration` | where the aggregate hides a minority |
| TPR/FPR bars | `equalised_odds` | which notion the model fails, and by how much |
| η² heatmap | `proxy_correlation` | replaces a forty-row table; the eye finds the hot cell |
| Threshold sweep | `disparate_impact` | whether the verdict survives a small change of cutoff |
| Actual-vs-expected by band | `calibration_parity` | "wrong by 25,000" versus "under-priced in the top decile" |
| Loss-ratio scatter | `loss_ratio_parity` | whether the margin gap is flat or grows with the risk |
| Ordinal confusion | `performance_thresholds` | `quadratic_kappa` hides *direction* |
| Robustness sweep | `adversarial_robustness` | flat-then-collapse is a different risk from linear decay |
| A/E by band | `actual_vs_expected` | one bad decile is a segment to recalibrate; a tilt across all of them is a rating structure that does not hold |
| Lorenz curve | `risk_discrimination` | a Gini earned on one dreadful decile and one spread across the book are the same number |
| Partial dependence | `monotonicity` | "not monotone" does not say whether the curve dips once or sags through the middle |
| Change histogram | `prediction_dislocation` | a tight bump past the threshold is a rounding decision; a long tail is a conduct problem |
| Injection bars | `prompt_injection` | *which* attack family, and on *which* surface — the finding a single score erases |

Six deserve a longer note.

### The threshold sweep

A parity difference is computed at one cutoff, and a cutoff is a cliff edge:
0.49 and 0.51 can land on opposite sides of the verdict. The sweep answers the
question a reviewer actually has — *is this pass robust, or did the cutoff
happen to fall in a good place?* A curve that peaks just beside the marked
point is a pass you should not bank on.

### Actual against expected, by band

A mean residual is one number for the whole book, and a book is not uniform. A
group whose ratio sits at 1.0 across nine bands and 0.7 in the tenth has a
segment problem, not a pricing problem, and only the banded view says so.

### The Lorenz curve, against its ceiling

The honest question about a Gini is not "is 0.28 good?" but "how much of what
was available did the model capture?". No rating structure can predict which
individual policy crashes, so the attainable maximum is well below 1.0 and
varies by class of business. The chart draws the ceiling — the same curve
sorted by the realised outcome — behind the model's own, which is what makes
the shaded gap readable.

### Partial dependence, with the breaks marked

Drawn only when something broke. A compliant curve is a straight-ish line the
detail string already describes in full, and charting it would be decoration.
When a filed constraint *is* violated, the shape is the remedy: a single dip in
a thin cell is a segment to refit, and a sag through the middle of the book is
a structural problem.

### The injection bars, direct against indirect

The finding this check exists to surface is not "injection risk", it is
*which family, on which surface*. A side-car hardened against a customer
typing "ignore previous instructions" and wide open to the same text arriving
inside a claim description is the common case, and any single score erases
it. Two bars per family put it in one glance.

Probes routed to a human are excluded from the bars, so **a short bar is not
the same as a clean one** — the caption says so, because the alternative is a
chart that reads as a pass when nothing could be judged.

### The robustness sweep — opt in

```python
AdversarialRobustnessCheck(plot_sweep=True)
```

Off by default. Every other plot reads data already in hand; this one
re-scores the sample at each epsilon, which is real money against a metered
endpoint. When it is off, `plot()` returns `None` and the report simply omits
the chart.

## A chart may not contradict the number beside it

This is the rule the release was built around. A plot that disagrees with the
finding printed next to it is worse than no plot: it is a second, more
persuasive claim with nothing checking it.

Two checks score a **subsample** for speed. A plot that re-sampled would draw
different rows than the verdict came from. Three things prevent it:

- **`stable_sample` is content-addressed.** It selects rows by their contents,
  not their position, so the same data in a different order yields the same
  sample — and the plot gets the rows the finding came from by construction.
- **One implementation, not two.** The perturbation core lives in
  `AdversarialRobustnessCheck._measure(context, epsilon)`, which both `run()`
  and `plot()` call. `ProxyCorrelationCheck._grid()` is the same idea: the
  heatmap and the findings read one object.
- **Tests read the values back off the `Axes`** and assert them against
  `metadata` — bar heights against `group_tpr`, ray slopes against
  `group_loss_ratio`, the ECE rebuilt from the plotted points and their marker
  areas, and the Gini re-integrated from the drawn Lorenz curve.

Three plots take the strongest available form of this: the A/E bars, the
monotonicity curve and the injection bars are read **straight out of
`metadata`**, so the chart is the finding rather than a second computation of
it. Where a finding is a small table, storing it and drawing from it beats
recomputing.

For the injection bars that is not merely tidier. Redrawing would mean firing
the corpus at a metered endpoint a second time.

If you write your own `plot()`, hold it to the same standard: recompute from
the check's own helper, or assert the drawn value against `metadata`.

## Style

The palette matches this site, and two colour systems are kept strictly apart.

**Semantic** — pass, review, blocked. These *mean* something. A group must
never borrow them: a green bar that happens to be group A, sitting next to a
green verdict pill, is a misread waiting to happen.

**Categorical** — for groups. Okabe–Ito, the standard colour-blind-safe
qualitative palette, because roughly 8% of men have some colour vision
deficiency and a gate report is a document a regulator may read.

Colour is never the only encoding. Series carry marker shapes, paired bars
carry hatching, and heatmap cells that were reported are ringed — these
reports get printed in greyscale.

```python
from bdp_model_gate.plots.style import (
    CATEGORICAL,
    VERDICT_COLOURS,
    apply_style,
    categorical,
    markers,
)

apply_style()  # house rcParams, applied by every plot() anyway
colours = categorical(4)  # four CVD-safe hues
shapes = markers(4)  # the shapes to pair them with
```

`apply_style()` sets rcParams rather than using a style context, so a figure
composed from several plots stays consistent — and your own call comes last,
so it wins.

## Writing a `plot()` for your own check

Override the method. There is nothing to register: the report renderer calls
`plot()` on every check and uses whatever comes back.

```python
from bdp_model_gate.core.base import BaseCheck, CheckResult


class TenureParityCheck(BaseCheck):
    name = "tenure_parity"
    category = "fairness"
    blocking = False

    def run(self, context): ...  # returns [CheckResult(..., metadata={"group_rate": {...}})]

    def plot(self, context, results=None, ax=None):
        from bdp_model_gate.plots import require_plotting, worst_result
        from bdp_model_gate.plots.style import caption, categorical, new_axes

        require_plotting()
        results = self.run(context) if results is None else results
        finding = worst_result(results, "rate_gap")
        if finding is None:
            return None  # nothing to draw is not an error

        rates = finding.metadata["group_rate"]
        ax = new_axes(ax)
        ax.bar(list(rates), list(rates.values()), color=categorical(len(rates)))
        caption(ax, "each bar is a group's rate; the gap is what was reported.")
        return ax
```

Four rules:

1. **Return `None` rather than an empty frame** when the inputs are missing.
   Most checks have nothing to draw most of the time.
2. **Never raise.** The report catches it and prints a note, but the reviewer
   still loses the chart.
3. **Recompute; do not store.** Small per-group dicts in `metadata` are
   *findings* and stay. Anything array-sized — bin edges, curve points,
   per-row SHAP — is recomputed at plot time, so the archival JSON does not
   carry presentation data most consumers never read.
4. **Use seaborn's axes-level functions only** (`barplot`, `lineplot`,
   `heatmap`, `scatterplot`, `histplot`). The figure-level ones (`relplot`,
   `catplot`, `displot`) build their own Figure and do not accept `ax`, which
   breaks the composition contract.

## API

::: bdp_model_gate.plots
    options:
      members: [require_plotting, plotting_available, worst_result]

::: bdp_model_gate.plots.style
    options:
      members: [apply_style, new_axes, categorical, markers, verdict_colour, caption, ring_cell, sharpen_colourbar, themeable_svg]
