# Roadmap

Planned work, with enough detail that decisions already taken do not get
re-argued. Shipped releases are in [`CHANGELOG.md`](CHANGELOG.md).

## How each release is delivered

Every entry below ships as a complete slice, not just code:

1. **Its own branch**, named in the entry.
2. **Implementation**, with the tests the 0.4.2 guards now demand — known
   answers, invariants, and a place in the model-family matrix.
3. **Examples**: a new notebook, or updates to the existing ones, executed via
   `examples/run_all.sh` so the committed outputs are true.
4. **Web**: the affected pages under `web/docs/`, and the landing page where a
   capability claim changes.
5. **A pull request to `main`** — the `main-lock` ruleset requires one, and CI
   plus the docs build must be green before merge.

A release is not done when the code works. It is done when someone who has
never seen it can find it, read why it exists, and run it.

Current release: **0.4.2**. In progress: **0.5.0**.

> **This file tracks what should happen.** What already happened lives in
> [`CHANGELOG.md`](CHANGELOG.md), and shipped entries are removed from here
> rather than marked done. The only release appearing in both is the one
> currently being built.

---

## 0.5.0 — Calibration and separation

**Branch:** `feat/calibration-separation`

The suite currently implements exactly one of the three fairness families —
*independence* (demographic parity). It has neither *separation* (equal error
rates) nor *sufficiency* (calibration). Shipping parity alone silently makes a
choice on the user's behalf.

### Calibration

- **`CalibrationCheck`** for classification. `CalibrationParityCheck` exists
  but is regression-only, so a classifier's calibration is currently
  unmeasured. Expected Calibration Error plus a Brier decomposition.
- **Calibration by subgroup** — a model can be well-calibrated overall and
  badly miscalibrated for a minority group. That is a fairness finding, not a
  performance one.

For credit scoring and pricing this often matters more than discrimination: a
model with AUC 0.86 whose probabilities are systematically twice too high
misprices every policy while scoring well on everything measured today.

### Separation

- **`EqualisedOddsCheck`** — TPR and FPR differences across groups.
  `fairlearn` already provides `equalized_odds_difference` and
  `true_positive_rate_difference`.
- **Equal opportunity** (TPR parity alone), which is what lending regulators
  most often centre on.

Demographic parity ignores `y_true` entirely, so a model can achieve perfect
parity by being wrong in compensating directions.

### State the impossibility

Calibration, TPR balance and FPR balance are mutually incompatible except in
degenerate cases (Kleinberg–Mullainathan–Raghavan 2016; Chouldechova 2017).
The report must present all three and name the trade-off rather than let a
reader assume the one they happen to read is *the* answer. This is as much a
documentation decision as a code one.

### Also

- **Intersectional fairness** — the suite checks each attribute independently,
  but harm concentrates at intersections. Joint `gender × region` groups,
  guarded by `min_group_size`.

### Deliverables

- Examples: extend `01_binary_classification_sklearn` with a calibration and
  separation section; the impossibility trade-off is best shown, not told.
- Web: new `docs/tasks/fairness.md` covering the three families; update
  `docs/reference/checks.md` and `docs/reference/configuration.md`.

---

## 0.5.1 — Plotting and the HTML report

**Branch:** `feat/plots-and-report`

A `NEEDS_REVIEW` verdict means a human must judge. Today that human gets a
JSON blob. This release gives them a page.

### The principle

Plot where a check collapses a distribution to a scalar **and the shape is
what you need to judge**. Latency, cost and model-card completeness are
genuinely scalars; plotting them is decoration.

| Plot | Why the number is not enough |
|---|---|
| Reliability curve, per subgroup | two models with identical ECE can be miscalibrated in opposite ways |
| Actual-vs-expected by band | RMSE says "wrong by 25k"; A/E says "under-priced in the top decile" |
| Threshold sweep with the verdict marked | shows whether the verdict survives a small change of cutoff |
| Ordinal confusion heatmap | `quadratic_kappa` hides *direction*; accept↔decline is not refer↔accept |
| Loss-ratio scatter with the 45° line | shows whether the margin gap is uniform or concentrated |
| Proxy η² heatmap, feature × attribute | replaces a 40-row table; the eye finds the hot cell |
| Robustness curve over epsilon | flat-then-collapse is a different risk from linear decay |

Grouped, three pages tell a story no single chart does: a **fairness page**
(calibration, selection rate, error rate as small multiples — where the
impossibility trade-off becomes visible), a **pricing page** (A/E beside
loss-ratio), and a **decision page** (threshold sweep).

### Design

- **An optional `plot()` method on `BaseCheck`**, not a subclass and not free
  functions. A subclass forces every check author into an inheritance
  decision; free functions drift from the check they illustrate. A method
  keeps the plot beside the logic that produced the number.
- **Signature `plot(self, context, results=None, ax=None) -> Axes`.** Taking
  and returning an `Axes` is the whole "we are not replacing your plotting
  library" contract — we draw onto your canvas and hand it back.
- **Discovery by override**, not a capability flag: the renderer calls
  `plot()` and uses whatever returns something. Nothing new for plugin
  authors to maintain.
- **Recompute rather than store.** Small per-group dicts already in
  `metadata` (`group_loss_ratio`, `group_means`, `group_mae`) are *findings*
  and stay. Anything array-sized — bin edges, curve points, per-row SHAP — is
  recomputed at plot time, so the archival JSON does not carry presentation
  data most consumers never read.

### Library: seaborn

Chosen over bare matplotlib. Its declared requirements are matplotlib, numpy
and pandas — **no scipy** — and since numpy and pandas already ship, the
marginal cost over matplotlib is about 3MB.

Two constraints that follow:

- **Axes-level functions only** (`barplot`, `lineplot`, `heatmap`,
  `scatterplot`, `histplot`). Seaborn's *figure-level* functions (`relplot`,
  `catplot`, `displot`) build their own Figure and do not accept `ax`, which
  would break the composition contract.
- `seaborn.color_palette("colorblind")` gives the CVD-safe categorical
  palette without hand-rolling one.

Rejected: **plotly** and **altair** need JavaScript at render time. The output
here is an archival record that gets printed, emailed and read years later —
JS breaks all three and adds megabytes per report. If an interactive
exploration dashboard is ever wanted, that is a second renderer, not a
replacement.

### Styling and accessibility

- A `style.mplstyle` carrying the site palette — deep teal `#0e5c55`, cool
  green-grey neutrals, IBM Plex Sans/Mono — so plots match the docs.
- **SVG, inlined into the HTML** rather than `<img src="data:...">`. Inlined
  SVG inherits page CSS, so the report is theme-aware light/dark from one
  render.
- Semantic colours (pass / review / blocked) stay **out** of the categorical
  cycle; groups must never borrow verdict hues.
- **Never encode by colour alone** — pair with marker shape or hatching, since
  these get printed greyscale.

### Guard against the obvious trap

`AdversarialRobustnessCheck` and `CounterfactualFlipCheck` score a
*subsample*. A plot that recomputes must draw the same rows the finding came
from, or the picture contradicts the number it illustrates — the exact class
of silent wrongness 0.4.2 exists to prevent. `stable_sample` is
content-addressed and deterministic, so this holds by construction; a test
must assert the plotted value equals the value in `metadata`.

Expensive plots — the robustness curve re-scores at several epsilons, which
costs real money against a metered endpoint — are opt-in, not default.

### Deliverables

- `matplotlib` and `seaborn` in a new `[plots]` extra. Without it, `plot()`
  raises `GateConfigurationError` naming the extra and the report renders
  text-only, exactly as shap and fairlearn already degrade.
- `GateReport.to_html()`, self-contained, plots inline.
- Examples: a new `06_reports_and_plots.ipynb`; add report rendering to
  `01`.
- Web: new `docs/reference/plots.md` and `docs/reference/reports.md`; link
  from the landing page, since "produces a report a reviewer can sign" is a
  capability claim.

---

## 0.5.2 — Validation methodology and leakage

**Branch:** `feat/validation-checks`

Nothing currently stops a user passing **training data** as the validation
set. The gate would report AUC 0.99 and `PASS`. For a governance tool that is
a serious hole, and the checks are cheap.

- **`LeakageCheck`** — flag a feature whose solo predictive power approaches
  the full model's, the classic signature of a leaked target.
- **Duplicate rows across splits**, when both frames are supplied.
- **`validation_strategy` in the model card**, and a requirement that
  high-risk use cases (`pricing`, `underwriting`, `credit_scoring`,
  `claims_decisioning`) use an **out-of-time** holdout. A random split is the
  wrong test for a pricing model, and no other tool will say so.
- **Feature-list and order match** between `X` and what the model was trained
  on. Silent column reordering is a classic production failure.
- **Train-serve skew** — promote the `FeatureDriftCheck` currently living only
  in the docs into the real suite.

### Deliverables

- Examples: extend `03_regression_sklearn` with an out-of-time split;
  demonstrate the leak check catching a planted leaked feature.
- Web: `docs/reference/checks.md`, and a validation section in
  `docs/concepts.md`.

---

## 0.5.3 — Exposure and actuarial measures

**Branch:** `feat/actuarial`

The suite is aimed at pricing and claims but lacks the measures the domain
actually uses.

- **`context.exposure`** — insurance metrics must be exposure-weighted. An
  unweighted RMSE treats a one-month policy like a twelve-month one. This is
  closer to a bug in the regression suite than a missing feature.
- **Actual-vs-expected by prediction band** — the standard pricing validation,
  and more informative than RMSE.
- **Gini / Somers' D**, the industry convention for pricing discrimination.
- **`MonotonicityCheck`** — regulators frequently require premium to be
  monotone in a rating factor; more claims must not mean cheaper. Checkable
  empirically via partial dependence, and a violation is a compliance finding
  rather than a performance one. No comparable tool does this.
- **`context.baseline_pred`** and **dislocation analysis** — when a model
  replaces an incumbent, the governance question is "how many policyholders
  see a >25% increase?". A baseline also enables report diffing across
  releases: *did fairness regress since v3?*

### Deliverables

- Examples: a new `07_insurance_pricing_end_to_end.ipynb` carrying exposure,
  A/E, monotonicity and dislocation on one book.
- Web: new `docs/tasks/insurance.md`; update the landing page, since this is
  the domain wedge.

---

## 0.6.0 — Confidence intervals, and pinned tooling

**Branch:** `feat/uncertainty`

Two unrelated pieces of hygiene, bundled because both are about trusting what
the tool tells you.

### Confidence intervals

Every check today compares a **point estimate to a fixed threshold with no
notion of sampling error**. `min_group_size = 30` is a crude stand-in: for a
proportion near 0.5, n=30 gives a standard error of about 0.09, so the
*difference* of two such proportions carries an SE near 0.13 — against a
`disparity_threshold` of 0.10, the verdict is noise.

A gate that flips on resampling teaches people to route around it.

- Bootstrap intervals on the fairness and performance metrics.
- Flag on the **interval** relative to the threshold, not the point — or at
  minimum report it so a human can judge.
- **A split-stability test**: halve the validation set at random and assert
  the verdict agrees. It would fail today, and it belongs beside the
  permutation-invariance test that found the sampling bug.
- **Multiple-comparison control.** Proxy correlation tests every numeric
  feature against every attribute; with twenty comparisons at α=0.05 you
  expect a false positive by chance. Benjamini–Hochberg or Holm.

Sequencing note: intervals arriving after the plots means forest plots and
error bars are a second pass over 0.5.1's work. Known and accepted.

### Tooling pinning (formerly 0.4.3)

Unpinned linters change their verdict on unchanged code, producing a
confusing red build months later on an unrelated PR.

- Pin `ruff` and `mypy` **exactly** in a dedicated `lint` extra, separate from
  `dev`.
- **Reconcile pre-commit with CI.** `.pre-commit-config.yaml` pins ruff
  `v0.13.2` while CI installs the latest — a developer running pre-commit and
  CI can disagree *today*.
- A weekly, non-blocking **"latest tooling" job**, so upgrades surface as a
  decision rather than a surprise.
- Bump `actions/*` past the Node 20 deprecation warnings.
- A constraints file, so a lint run is byte-reproducible.

### Also

- **A mutation kill-rate floor.** Baseline is **35.5%** measured in CI (1115
  killed of 3139 with a verdict, from 3430 generated), reproducing a local
  35.6%. Once stable, `mutation_report.py --min-kill-rate` turns it into a
  threshold. Survivors cluster in `structured/fairness` (506),
  `structured/security` (318) and `metrics` (272).

### Deliverables

- Examples: intervals shown on every fairness plot in `06`; the
  split-stability demonstration.
- Web: an uncertainty section in `docs/concepts.md`; update
  `docs/reference/configuration.md`.

---

## 0.6.1 — Release automation (formerly 0.4.4)

**Branch:** `feat/release-automation`

Publishing is manual. Two of the silent failures were caught *only* by
installing the published artifact, and v0.3.0 and v0.3.1 were both tagged on
commits with a red Python 3.9 job. Fitting, for a tool that exists to gate
deploys.

### Decisions already made

- **Trigger on the tag, not on merge to `main`.** Not every merge is a
  release, and the artifact published is then the commit that was tagged.
- **Trusted Publishing (OIDC)**, not API tokens. Nothing to rotate or leak.
- **Protection on a GitHub Environment, not the branch.** Branch protection
  guards what *merges*; only an Environment stands between a tag and PyPI.
  `main-lock` already covers the branch half.

### Pipeline

```
tag v* pushed
  ├─ build        sdist + wheel, tested against the installed artifact
  ├─ testpypi     environment: testpypi — no approval
  ├─ smoke-test   fresh venv, install from TestPyPI, run a real gate
  └─ pypi         environment: pypi — REQUIRED REVIEWER
```

### Guards

- **Tag/version consistency** — fail if the tag does not match
  `pyproject.toml`.
- **Publish only on a green matrix for that commit.**
- **Smoke-test from TestPyPI first.** The 3.9 import failure lived in the
  *published wheel*; the shap/numpy clash only appeared on a clean install.

### Manual prerequisites (repo admin, not code)

- [ ] Register repo, workflow filename and environment as a **trusted
      publisher on PyPI**
- [ ] The same on **TestPyPI** — a separate registration
- [ ] Create `testpypi` and `pypi` **Environments**, reviewer required on
      `pypi`
- [ ] Fix the **`release tags` ruleset**: it currently restricts `creation` on
      `~ALL` tags with no bypass, which forbids creating any tag at all. Drop
      the `creation` rule and keep `deletion` and `non_fast_forward` — that
      gives what the name suggests, and prevents a repeat of the withdrawn
      v0.3.0/v0.3.1 tags.

Two things to know: if the workflow **filename** changes, publishing breaks
until the trusted-publisher config is updated; and **version collisions are
permanent on both indexes**, so a failed release means bumping the patch.

---

## Later

- **A public, subclassable `ModelAdapter` (1.0.0).** The extension point is a
  plain callable for now, which covers every case with less ceremony; a named
  class earns its place once someone needs batching, retries or auth on a
  serving layer.
- **Unstructured data** (text / image / audio).
  `bdp_model_gate.unstructured` reserves the shape and raises
  `NotImplementedError` until it lands. Deliberately *after* the statistical
  work above: broadening the modality before deepening the statistics would
  trade a defensible niche for a shallow generalist.
