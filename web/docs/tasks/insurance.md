# Insurance pricing

A general regression report tells a pricing committee that the tariff is
"wrong by 129,697 naira on average". Nobody has ever been able to act on that
sentence.

What a pricing review asks instead:

1. **Did the book collect what it needed to?** — actual against expected.
2. **Is the shortfall everywhere, or in one band?** — A/E by decile.
3. **Does the tariff order risk at all?** — the Lorenz Gini.
4. **Does premium still rise with prior claims?** — monotonicity, because that
   is what was filed.
5. **Who gets a 30% increase?** — dislocation against the incumbent.

An RMSE answers none of them. It is also symmetric about zero, so a model that
over-charges half the book and under-charges the other half scores exactly
like one that is right everywhere.

Everything on this page works on a **core install** — no scikit-learn
required — and is regression-only, with one exception: `monotonicity` also
applies to binary classification, where the curve is drawn over probabilities.
See
[`07_insurance_pricing_end_to_end.ipynb`](../examples/07_insurance_pricing_end_to_end.ipynb)
for the whole thing on one motor book.

```python
context = StructuredGateContext(
    predict_fn=price_v4,  # the deployed tariff, business rules included
    X=X_val,
    y_true=loss_cost,  # realised cost per exposure-year
    y_pred=quoted_premium,
    exposure=earned_vehicle_years,  # the target is a RATE
    baseline_pred=premium_v3,  # what this replaces
    expected_loss=technical_premium,  # enables loss-ratio parity
    protected_df=protected_val,
    task="regression",
)

config = GateConfig(
    performance=PerformanceConfig(metric="lorenz_gini", min_score=0.15),
    actuarial=ActuarialConfig(monotonic_features={"prior_claims": "increasing"}),
)
```

## Exposure is a correctness question

A general-purpose regression suite treats every row as one observation. An
insurance book does not work that way. **A policy written for one month and a
policy written for twelve are not equal evidence about a claims rate, and an
unweighted RMSE says they are.**

That is closer to a bug in a pricing gate than a missing feature, which is why
`context.exposure` reaches the regression metrics, all four regression
fairness checks, and the whole actuarial suite.

### The convention, stated once

**`y_true` and `y_pred` must be on the same basis as each other, and
`exposure` is how much weight the row's observation deserves.**

| Your target | Supply `exposure`? |
|---|---|
| a **rate** — claims per vehicle-year, loss cost per sum-insured-year | **yes** |
| a per-policy **total** — the premium charged, the loss paid | no; the exposure is already inside the value |

Under either convention the A/E ratio is
`sum(w · actual) / sum(w · expected)`, which is why one formula serves both.

On a real book the difference is not a rounding difference. From notebook 07:

| | unweighted | exposure-weighted |
|---|---|---|
| RMSE | 129,697 | 95,188 |
| actual / expected | 1.021 | 1.092 |

Both move, in opposite directions, for the same reason. The unweighted RMSE is
inflated by short policies — a one-month policy with a claim has a *loss cost*
of twelve times that claim, and counting it once exaggerates the error. The
unweighted A/E happens to sit closer to break-even, so a report quoting it
would say the book is 2% out when it is 9% out.

### What it does not do

- **`min_group_size` still counts rows.** It exists to stop a three-policy
  segment producing a ratio, and three policies are three policies however
  long they ran.
- **Classification metrics ignore it**, and the report says so —
  `[NOT exposure-weighted — this metric takes no per-row weight]` appears in
  the detail line rather than the weighting being dropped silently.
- **A callable of your own never receives it.** Its signature is unknown, and
  passing an unexpected keyword would turn a working metric into a
  `CHECK_ERROR`. Apply the weighting inside your function.
- **Scaling every exposure by the same amount changes nothing.** Months and
  years are the same book; the weights are relative.

## Gate the scoring function, not the estimator

A tariff is rarely just an estimator. Rate caps, capped loadings, retention
discounts and minimum premiums live in code *around* the model, and they are
where a filed constraint usually breaks.

```python
def price(frame):
    scored = frame.copy()
    scored["prior_claims"] = scored["prior_claims"].clip(upper=2.0)  # capped loading
    base = np.clip(booster.predict(scored), 1_000.0, None)
    premium = base.copy()
    premium[base >= RATE_CAP] *= 0.80  # rate cap
    premium[frame["prior_claims"].to_numpy() >= 3] *= 0.85  # retention discount
    return premium
```

Pass that as `predict_fn` and every check that scores the model scores the
tariff. Gating `booster` alone tests a model nobody prices with — and in
notebook 07 the whole monotonicity violation and a third of the top decile's
shortfall come out of exactly these three lines rather than the estimator.

`predict_fn` always wins over `model` for prediction. Supplying `model` as
well is still worth doing: `shap_subgroup_gap` reads it to pick
`TreeExplainer`, which is exact and fast for a booster where the black-box
explainer re-scores the book thousands of times.

## The level, and then the shape

`actual_vs_expected` reports two findings because they have different causes
and different fixes.

The **level** is `sum(actual) / sum(expected)` over the whole book, and it is
the one number a committee can act on immediately: 1.10 says the book is
under-priced by ten percent.

The **shape** is what a single ratio hides. An overall A/E of exactly 1.00 is
routinely produced by a model subsidising its worst risks out of its best —
the rate level looks perfect and every individual price is wrong.

Bands are cut at equal **exposure** rather than equal row counts. On a book
with a mixed exposure profile the two differ substantially: equal-count
deciles put most of the year's risk in whichever decile happens to hold the
annual policies.

!!! tip "The shape can invert the remedy"
    In notebook 07 the A/E climbs almost monotonically from 0.33 in the
    cheapest decile to 1.82 in the dearest — a tariff that does not spread
    enough. "Under-priced by nine percent" invites a nine percent rate
    increase across the book, which would over-charge the seven deciles that
    are *already* over-priced in order to fix the one that is not.

    The finding is a rating structure that needs to spread further, not a rate
    level to move. Only the banded view says so.

A band holding fewer than `min_band_rows` rows is reported but not scored —
the same treatment `min_group_size` gives a three-policy segment.

## Discrimination is a separate question

Calibration and discrimination are independent, and a pricing review needs
both.

A tariff that charges every policy the book average has a perfect A/E and is
useless: it collects the right total and distributes it at random. On a book
where four fifths of policies have no claim, the mean is not even a bad guess,
so every error metric scores it respectably.

`risk_discrimination` reports the **exposure-weighted Lorenz Gini**. Policies
are sorted from cheapest to dearest prediction, and the curve plots cumulative
share of exposure against cumulative share of realised loss.

- **0** — the ordering carries nothing.
- **Negative** — the ordering is *inverted*: the policies priced highest carry
  the lower loss per unit of exposure. That is a finding, not a poor score,
  and it is what a sign error in a rating factor produces. Reversing an
  ordering leaves the average error almost unchanged, so no error metric shows
  it.

### Read it against the ceiling

No rating structure can predict which individual policy crashes, so the
attainable maximum is well below 1.0 and varies by class of business. The
check reports the model against `lorenz_gini(y_true, y_true)` — the same
function called with the actuals as the score, so there is no second
implementation to disagree with the first.

| | Gini | share of ceiling |
|---|---|---|
| v3 (incumbent) | 0.363 | 43% |
| v4 (proposed) | 0.564 | 67% |
| ceiling — sorted by outcome | 0.835 | 100% |

That table is the business case for a new tariff, and it is one line long.
"0.28 of a possible 0.52" is a judgement a reviewer can make; "0.28" alone is
not.

Also available as a gateable metric — `performance.metric = "lorenz_gini"`,
compared against `min_score`. For a **binary** target the Gini is exactly
`2 · roc_auc − 1`, so use `roc_auc` there rather than a second name for the
same quantity.

## The constraint that was filed

Filed rates carry structural claims: premium rises with prior claims, falls
with a higher deductible, rises with sum insured. A gradient booster fitted on
a thin cell will happily violate one, and **nothing else in a validation
report notices** — the model scores well, the book prices sensibly on average,
and one segment of policyholders is charged less for being worse risks.

```python
config.actuarial.monotonic_features = {
    "prior_claims": "increasing",
    "deductible": "decreasing",
}
```

Checked empirically, by partial dependence: each declared factor is swept
across the quantiles of its own distribution while every other column keeps
its real joint distribution. No constraint on the model class, no assumption
of linearity, and it works against a remote endpoint through `predict_fn`.

Nothing is checked until you declare something. The constraint is a claim
about *your* product and its regulator, and no library can guess which factors
carry one or in which direction.

!!! warning "A declared factor that cannot be tested blocks"
    A misspelled column, a categorical one, or one that is constant on the
    validation set reports `MONOTONICITY_UNCHECKABLE` and **blocks** rather
    than skipping quietly — the message names the near miss.

    A typo in a rating-factor name would otherwise produce a green gate on an
    unverified regulatory constraint, which is exactly the confident-and-wrong
    outcome this library exists to prevent.

## Who moves

The question a pricing committee actually asks about a replacement is not "is
it more accurate?". That is settled before a gate runs. It is *"how many
policyholders see a rise above 25%, and are they anyone in particular?"*. A
tariff can be better on every statistical measure and still be undeployable
because of who it re-prices.

`prediction_dislocation` needs `context.baseline_pred` — the incumbent's price
for the same rows, or the previous version of this model — and reports:

- the share of **exposure** moving by at least `dislocation_threshold` in each
  direction
- the 95th percentile move and the largest rise
- the rise share per protected group, naming the most affected

**Non-blocking, deliberately.** A dislocated book may be entirely correct —
that is often the point of a re-rate — and no threshold can decide whether
this profile is acceptable. The check's job is to put the number and the
affected group in front of a person.

Two notes on reading it. **Read the 95th percentile, not the maximum**: the
largest relative rise usually comes from a handful of policies the incumbent
priced at its floor. And rows whose baseline is zero or negative are excluded
and counted — a premium moving from 0 to 500 is not an increase of any
percentage. Without a baseline the check reports `NOT_APPLICABLE` rather than
treating the book mean as a stand-in, which would answer a different question.

## Territory rating and the fairness flags

If `region` is a protected attribute and a territory factor derived from it is
a filed rating factor, `proxy_correlation` will report η² = 1.000. That is
correct and not a finding: territory rating is **explicit**, not a proxy, and
this is what the check looks like when the attribute is a declared factor.

`group_mean_gap` and `error_parity` fire for the same structural reason. A
pricing model *should* charge more in a higher-loss territory — that is
risk-based pricing, not discrimination — so a raw mean gap flags legitimate
rating differences and is noisy on its own.

**`loss_ratio_parity` is the check that separates the two**, and the one to
read first on a pricing book. It asks whether a group is charged a higher
*margin over its own expected cost*, which is the question a regulator asks.
It needs `context.expected_loss` — a per-row technical or pure premium — and
reports `NOT_APPLICABLE` without it rather than silently answering the
raw-price question under the same name.

The rule of thumb: read `loss_ratio_parity` first, and treat `group_mean_gap`
and `error_parity` as context for it rather than as findings in their own
right.

## Where to go next

- [The checks](../reference/checks.md) — every flag and what triggers it
- [Configuration](../reference/configuration.md#actuarial) — every threshold
- [Regression](regression.md) — metric directions, and the four regression
  fairness notions
- [`07_insurance_pricing_end_to_end.ipynb`](../examples/07_insurance_pricing_end_to_end.ipynb)
  — all of the above on one book
