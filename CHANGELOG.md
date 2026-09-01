# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.5.2] - 2026-09-01

Validation methodology — is the evidence behind the report sound?

Nothing in this library stopped you passing the **training set** as the
validation set. The gate reported an AUC of 0.99, a clean calibration curve
and `PASS`, and every fairness figure beside it was measured on data the model
had memorised. For a governance tool that is a serious hole, and the checks
are cheap.

### A fifth category, and it goes first
`validation` is a new category rather than five more performance checks,
because the two say different things:

> A **performance** finding says *the model is not good enough*.
> A **validation** finding says *you do not yet know whether it is*.

The second is a prior question. If it fires, nothing underneath it means what
it appears to mean — so these block, and `validation` is reported before
everything else in both `summary()` and the HTML report.

### Added
- **`LeakageCheck`** — does a single column do what the whole model does? The
  signature of a field populated *after* the outcome was known and joined back
  in: a settlement amount on a claims-frequency model, a `closed_reason` on a
  churn model. Each feature's solo power is measured on the same 0–1 scale as
  the model's own — |2·AUC − 1| for classification, |r| for regression, the
  correlation ratio for a categorical column — and **two** conditions must
  hold before anything is flagged. Parity alone is not evidence: against a
  model scoring 0.55, a feature scoring 0.54 reaches 98% of it and means
  nothing, so `leakage_min_power` sets an absolute floor beside the ratio.
- **`SplitOverlapCheck`** — rows shared between `X_train` and `X`, and exact
  duplicates *within* `X`. Different causes, different fixes, so two findings:
  the first is a broken split, the second survives a correct one and inflates
  every metric by weighting an observation twice. Matched by row **content**,
  so a reset index cannot hide an overlap and a shuffle cannot invent one.
- **`ValidationStrategyCheck`** — a random split asks "can the model predict a
  policy it has not seen?" when the question is "can it predict *next
  quarter*?". Seasonality, inflation and portfolio mix all leak backwards
  through a random split. `model_card["validation_strategy"]` is now required,
  and must be out-of-time for the high-risk use cases. An unrecognised value
  is flagged rather than accepted — `holdout` and `out_of_time` are different
  claims and only one of them is checkable.
- **`FeatureContractCheck`** — the right columns, in the right order. Silent
  reordering is a classic production failure: scikit-learn checks names when
  given a DataFrame, but a `predict_fn` doing `df.values` does not. The
  expected list comes from `expected_features`, `model.feature_names_in_`, a
  booster's own names, or `X_train.columns`, in that order.
- **`FeatureDriftCheck`** — train-serve skew, promoted out of the docs into
  the real suite. **Non-blocking**, unlike its four neighbours: an
  out-of-time holdout *should* differ a little, and a gate that hard-fails on
  every seasonal shift gets switched off.
- **`context.X_train`** and **`context.expected_features`**, plus
  `--train-data` on the CLI. `X_train` needs no row alignment — only its
  columns and distributions are read, never its labels.
- **`ValidationConfig`**, and **`bdp_model_gate.stats`**: `rank_auc`,
  `average_ranks`, `correlation_ratio` (moved out of `structured/fairness.py`)
  and `pearson_r`.

### All five work on a core install
A validation set that is secretly the training set is the last thing that
should go unchecked because scikit-learn is missing, so the AUC these need is
computed in numpy via the Mann–Whitney statistic with proper tie correction.

It is **not** exposed as `metrics.roc_auc`, which is still documented as
needing scikit-learn: quietly making a published metric numpy-native as a side
effect of an unrelated release would be a contract change nobody asked for.
`tests/test_validation_checks.py` asserts the implementation agrees with
`sklearn.metrics.roc_auc_score` on tied data, which is where a naive rank
implementation diverges.

### Tests
39 new known-answer tests, plus five metamorphic properties: a leak measured
in naira is the same leak measured in thousands (rank AUC depends only on
ordering), renaming a column cannot change whether it leaks, overlap is
counted by content and not position, and swapping the two frames cannot change
*whether* a feature drifted.

### Changed
- The default suite is 21 checks across five categories, up from 16 across
  four.
- `GateReport.summary()` and the HTML report both lead with `validation`.
- `web/docs/extending.md` no longer uses `FeatureDriftCheck` as its teaching
  example, since that name now belongs to a real check.

## [0.5.1] - 2026-08-27

Plots, and a report a reviewer can actually read.

`NEEDS_REVIEW` is a verdict that delegates to a human, and until now that
human received a JSON blob. This release gives them a page — and gives the
checks whose finding is a *shape* a way to show it.

### Added
- **`BaseCheck.plot(context, results=None, ax=None)`** — optional, discovered
  by override alone, so there is nothing for a plugin author to register. It
  takes and returns a matplotlib `Axes`: we draw onto your canvas and hand it
  back, which is the whole "we are not replacing your plotting library"
  contract. Nine checks implement it; the rest deliberately do not.
- **`GateReport.to_html(path=None)`** — one self-contained HTML file. No
  script, no stylesheet, no font, no image fetched from anywhere: a governance
  record is emailed, filed and reopened years later, and every external
  reference is a way for it to stop rendering. Charts are inlined as SVG
  rather than `<img src="data:...">`, so they inherit the page's CSS and one
  render reads correctly in light and dark — and stays sharp in print.
- **A `[plots]` extra** — matplotlib and seaborn. Without it `plot()` raises
  `GateConfigurationError` naming the extra and the report renders text-only,
  exactly as shap and fairlearn already degrade.
- **`bdp_model_gate.plots`** — `require_plotting`, `plotting_available`,
  `worst_result`, and a `style` module carrying the documentation site's
  palette. Two colour systems kept strictly apart: **semantic** (pass /
  review / blocked) and **categorical** (Okabe–Ito, colour-blind safe) for
  groups. A group never borrows a verdict hue, and colour is never the only
  encoding — series carry marker shapes and bars carry hatching, because
  these reports get printed in greyscale.
- **`bdp_model_gate.groups.group_series`** — recovers the exact split a result
  was reported under, intersections included, so a plot cannot illustrate a
  different grouping from the one that was scored.

### The nine plots, and why each one is not a number
| Plot | Check | What the scalar cannot say |
|---|---|---|
| Reliability curve | `CalibrationCheck` | two models with the same ECE can be miscalibrated in opposite ways |
| Reliability per group | `SubgroupCalibrationCheck` | where the aggregate hides a minority |
| TPR/FPR bars | `EqualisedOddsCheck` | which notion the model fails, and by how much |
| eta-squared heatmap | `ProxyCorrelationCheck` | replaces a forty-row table; the eye finds the hot cell |
| Threshold sweep | `DisparateImpactCheck` | whether the verdict survives a small change of cutoff |
| Actual-vs-expected by band | `CalibrationParityCheck` | "wrong by 25k" versus "under-priced in the top decile" |
| Loss-ratio scatter | `LossRatioParityCheck` | whether the margin gap is flat or grows with the risk |
| Ordinal confusion | `PerformanceThresholdCheck` | `quadratic_kappa` hides *direction* |
| Robustness sweep | `AdversarialRobustnessCheck` | flat-then-collapse is a different risk from linear decay |

Latency, cost and model-card completeness are genuinely scalars. They are not
plotted, and a binary confusion matrix is not plotted either: four numbers the
detail line already carries.

### A chart may not contradict the number beside it
The obvious trap in this release: `AdversarialRobustnessCheck` scores a
*subsample*, so a plot that re-sampled would illustrate different rows than
the verdict came from — the same class of silent wrongness 0.4.2 exists to
prevent. Three structural answers, not more tests of the same kind:

- The perturbation core moved out of `run()` into `_measure(context, epsilon)`,
  so the sweep and the verdict are **one implementation**. The curve passes
  through the reported point by construction.
- `ProxyCorrelationCheck` gained `_grid()` for the same reason: `run()` and
  the heatmap read one object, so a cool cell can never sit beside a report
  line calling that pair a proxy.
- `tests/test_plots.py` reads values **back off the Axes** and asserts them
  against `metadata` — bar heights against `group_tpr`, ray slopes against
  `group_loss_ratio`, and the ECE rebuilt from the plotted points and marker
  areas. A separate test permutes the rows and asserts the robustness curve is
  unmoved.

### Notes
- The robustness sweep is **opt-in** (`AdversarialRobustnessCheck(plot_sweep=True)`):
  each point re-scores the sample, which is a real bill against a metered
  endpoint. Every other plot reads data already in hand.
- `ModelGate.run` attaches the checks and context to the report so
  `to_html()` works without re-supplying them. Both are excluded from the
  constructor, the repr, equality and `to_dict()` — a report is an archival
  record of findings, and neither the check objects nor the validation set
  belong in one.
- Fonts are resolved against what is installed before matplotlib is told about
  them; naming a missing family costs a warning per text element drawn.
- Heatmap colour bars are un-rasterised. matplotlib embeds a base64 PNG in
  them by default, which was the one soft edge on an otherwise vector page.

### Changed
- The LICENSE copyright holder is now Vangelis Oden. Still MIT.
- Examples: new `06_reports_and_plots.ipynb`; notebook 01 now ends by writing
  a report.
- Web: new `reference/plots.md` and `reference/reports.md`.

## [0.5.0] - 2026-08-27

Calibration and separation — the two fairness families the suite lacked.

Before this release the library measured only **independence** (demographic
parity). That was not a neutral position: parity ignores `y_true` entirely, so
a model can satisfy it by being wrong in compensating directions, and a reader
seeing one green check had no way to know two other notions were never tested.

### Added
- **`EqualisedOddsCheck`** — *separation*. Reports two notions per attribute:
  **equal opportunity** (the true-positive-rate difference — among applicants
  who should be approved, is every group equally likely to be?) and
  **equalised odds** (the larger of the TPR and FPR differences). Both
  condition on the ground truth, which is exactly what parity does not.
- **`SubgroupCalibrationCheck`** — *sufficiency*. Does a score of 0.7 carry
  the same real risk for every group? A model can be well calibrated overall
  and badly miscalibrated for a minority, because the majority dominates the
  average and hides it.
- **`CalibrationCheck`** — is the model calibrated at all? Discrimination and
  calibration are independent: a model can rank perfectly while every
  probability it emits is twice too high, scoring beautifully and mispricing
  every policy. Blocking, but `max_ece` defaults to a permissive `0.10` —
  plenty of good models are uncalibrated by construction, and a gate that
  blocks all of them gets switched off.
- **`bdp_model_gate.calibration`** — numpy-native, so it works on a core
  install: `calibration_curve` (uniform or quantile bins), ECE, Brier, and
  Murphy's decomposition into reliability, resolution and uncertainty.
  `resolution` is the term people forget: a model predicting the base rate for
  everyone is perfectly calibrated and completely useless, and neither ECE nor
  the Brier score says so alone. Both `brier` and `binned_brier` are returned,
  because the decomposition identity holds exactly only over the binned
  forecast — a number that almost adds up is worse than two that are labelled.
- **Intersectional fairness** via `FairnessConfig.intersectional`. Harm
  concentrates where attributes meet, and marginal checks are blind to it by
  construction. `bdp_model_gate.groups.iter_protected` yields pairwise
  combinations so every group-based check gets this from one place. Off by
  default; only pairs are generated, since three-way intersections fragment a
  validation set faster than any realistic `min_group_size` tolerates.
- Config: `max_ece`, `n_calibration_bins`, `calibration_strategy`,
  `equalised_odds_threshold`, `subgroup_calibration_threshold`,
  `intersectional`.
- `tests/test_calibration.py` — 18 known-answer tests. A constant forecast at
  the base rate has an ECE of exactly 0; a uniform 0.2 offset has an ECE of
  exactly 0.2; a group with no positive cases is skipped rather than divided
  by zero.

### The impossibility, demonstrated rather than asserted
Calibration, TPR balance and FPR balance are mutually incompatible whenever
base rates differ between groups (Kleinberg–Mullainathan–Raghavan 2016;
Chouldechova 2017). The suite therefore reports all three and names the
trade-off rather than picking one silently.

Notebook 01 now forces it: rescaling each group's scores to equalise selection
rates moves the parity gap from **0.365 to 0.010** while the subgroup
calibration gap goes from **0.067 to 0.131**. Independence bought, sufficiency
spent. A tool reporting only demographic parity would let you "fix" a model by
making its scores mean different things for different people, and call that
progress.

### Changed
- The default suite is 16 checks, up from 13.
- Examples: notebook 01 gains sections on the three families, overall
  calibration, and intersections.
- Web: new `docs/tasks/fairness.md`; `reference/checks.md` and
  `reference/configuration.md` updated.


### Added
- Previous/next controls on every documentation page. Material's stock
  `navigation.footer` omits a control at the ends of the nav, which makes the
  footer move between pages; `web/overrides/partials/footer.html` renders both
  positions always and marks the unavailable direction as a disabled `<span>`
  — no href, out of the tab order, `aria-disabled`, and dimmed.

### Changed
- Roadmap re-sequenced for 0.5.x and 0.6.x, putting statistical depth before
  breadth. Tooling pinning (was 0.4.3) folds into 0.6.0 alongside confidence
  intervals; release automation (was 0.4.4) becomes 0.6.1.

## [0.4.2] - 2026-08-26

Robustness of the checks themselves. Six silent failures have shipped and been
fixed, and every one passed the test suite at the time — so this release
attacks that pattern rather than adding more of the same tests. Two more bugs
surfaced while building it.

### Added
- **Suite-wide `CHECK_ERROR` guard** (`tests/conftest.py`). An autouse fixture
  wraps `ModelGate.run` and fails any test whose report contains a
  `CHECK_ERROR` it did not declare via `@pytest.mark.expect_check_error`. This
  alone would have caught the `ShapSubgroupCheck` crash on
  `RandomForestClassifier`: the test that missed it asserted
  `len(results) > 0`, which passes, because an exception becomes a blocking
  *result* rather than propagating.
- **`tests/test_known_answers.py`** — 17 tests on inputs whose correct answer
  is derivable by hand: parity difference exactly `1.0` and `0.0`, η² exactly
  `1.0` for a feature that is a pure function of the attribute, hand-computed
  loss ratios, group MAEs, calibration signs, and metric values. If you know
  the answer, wrong cannot hide.
- **`tests/test_invariants.py`** — 15 metamorphic tests asserting properties
  that must hold for *any* input: row-permutation invariance, group and class
  relabelling, feature-scale invariance, `y`-scaling behaviour (`rmse`/`mae`
  scale by *k*, `r2` and `mape` do not), and monotonicity — making a model
  strictly more unfair must never lower the measured disparity.
- **`tests/test_model_matrix.py`** — 22 tests crossing six classifier
  families, four regressor families and a `predict_fn` closure with all three
  tasks, asserting every flag lands in a known set. The shap 3-D bug was
  reachable only through a family the fixtures never used.
- **`tests/test_properties.py`** — 14 `hypothesis` tests over the numeric
  core: error metrics non-negative and symmetric, `rmse >= mae`, `r2 <= 1`,
  ordinal bounds, and every binary probability shape reducing to one column.
- **`tests/test_not_applicable.py`** — 21 tests asserting the *reason* on
  every skip path. Those branches are where a check silently does nothing, so
  a check skipping for the wrong reason passes every other test.
- **Hostile fixtures** — features spanning seven orders of magnitude, a
  three-row protected group, and a 99.5/0.5 class split.
- **Mutation testing**, configured under `[tool.mutmut]` and running as an
  advisory CI job — the only technique that answers "would my tests have
  noticed?", which is the question this release exists to answer. First
  measured baseline: **35.6%** (1118 killed of 3139 with a verdict, from 3430
  generated). Advisory until that is stable, then a floor via
  `scripts/mutation_report.py --min-kill-rate`.
- **`scripts/mutation_report.py`**, which exists because the first two
  attempts at this job both reported a number that was not true. mutmut runs
  the tests from inside `mutants/`, so a partial source copy left
  `bdp_model_gate` an incomplete package whose imports fell back to the
  installed one — every mutant survived, which reads as catastrophic but means
  nothing ran. And `mutmut results` lists **only survivors**, so counting its
  statuses gives a 0% kill rate whatever the truth. The script parses the
  run's own tally, takes monotonic maxima so redrawn progress lines cannot mix
  groups, and **fails when too few mutants got a verdict** — so the job can no
  longer go green having done nothing.

### Fixed
- **`FairnessConfig.shap_gap_threshold` was absolute**, in the units of the
  model output, while the four regression fairness thresholds were relative.
  SHAP values inherit the target's scale, so a threshold sensible for a
  probability flagged essentially every feature on a premium model — 12 of 17
  fairness findings in the regression example. It is now measured **relative
  to the mean absolute contribution**, and the default moves from `0.15` to
  `0.50`, meaning "this feature's cross-group gap is worth half a typical
  contribution". Calibrated against both a probability-scale and a
  naira-scale model, where the real proxy sits at 1.4–2.2 and noise below
  0.03. The result metadata gains `relative_gap` and `shap_scale`.
- **Subsampling depended on row order.** `AdversarialRobustnessCheck` and
  `CounterfactualFlipCheck` used `DataFrame.sample(random_state=...)`, which
  is reproducible for a fixed frame but selects by **position** — so the same
  data in a different order produced a different subsample and could produce a
  different verdict. Sorting a CSV should not decide whether a model ships.
  New `bdp_model_gate._sampling.stable_sample` selects by row **content**, so
  permuting the input cannot change which rows are scored. Found by the
  permutation-invariance test, not by inspection.

### Changed
- Project homepage now points at <https://vanjy-eng.github.io/model-gate/>,
  with the documentation as a separate `Documentation` URL. Takes effect on
  this upload; 0.4.1 was published with the old value.
- `hypothesis` and `mutmut` added to the `dev` extra.

### Added — the project website
- **`web/`** — a hand-built landing page with MkDocs Material documentation
  beneath it, mirroring how `pandas.pydata.org` is assembled, deployed to
  GitHub Pages by `.github/workflows/docs.yml`. `README.md` had reached 604
  lines and 15 top-level sections; someone wanting regression had to scroll
  past binary classification, metric selection, custom checks and plugins.
- Three things keep it from going stale: `mkdocstrings` generates the API
  reference from the docstrings covering 89% of the public API so it cannot
  drift; `build.sh` copies notebooks from `examples/` rather than keeping a
  second copy, leaving `run_all.sh` the single source of truth; and
  `mkdocs build --strict` fails on a broken internal link or a page missing
  from the nav.
- Written for an external audience — banks, insurers, any organisation with a
  working data-science team. NDPA/NDPR defaults are presented as
  *configurable defaults* rather than the product's premise, so a reader in
  another regime sees themselves in the hero.


## [0.4.1] - 2026-08-26

Example notebooks — and one real bug that writing them exposed.

### Added
- **`examples/` — five notebooks**, each committed with outputs:
  `01` binary classification (credit scoring), `02` multiclass and ordinal
  (underwriting accept/refer/decline), `03` regression (motor premium, claims
  severity and frequency), `04` PyTorch / Keras-shaped / remote endpoints,
  `05` XGBoost `XGBClassifier` and native `Booster` plus `--model-loader`.
  `01` is the on-ramp and covers the core machinery; the rest focus on what
  their task or framework changes.
- `examples/run_all.sh` — re-executes every notebook and fails on the first
  error, including a cell that *records* an error without failing the run.
  The notebooks are validated at release time rather than in CI, so this
  makes that a one-liner.

### Fixed
- **The gradient-directed adversarial attack was weaker than random noise**,
  which makes "targeted" meaningless. Two separate causes, both introduced
  with `gradient_fn` in 0.3.1:
  - The step was applied only along `+gradient`, so it could never flip a row
    already predicted positive, while the random path perturbs in both
    directions. Both signs are now tried.
  - The gradient was normalised to a unit vector before scaling, so one
    epsilon was spread across all features and each moved by only
    `epsilon/sqrt(n)` — a materially smaller perturbation than the random
    path applies. The step is now sign-of-gradient at full epsilon, an
    FGSM-style attack.

  Measured on the notebook 04 network: at the default `epsilon=0.02` the
  directed attack now finds a 4.5% flip rate where random noise finds
  **zero**; previously it found *fewer* flips than random at every epsilon
  tested. The same signed step is applied to the `coef_` path.

  A regression test now asserts that the directed attack finds strictly more
  flips than the random one.

### Notes
- `FairnessConfig.shap_gap_threshold` is **absolute and in the units of the
  model output**, unlike the four regression fairness thresholds which are
  relative. On a model predicting naira in the tens of thousands the default
  of `0.15` flags essentially every feature — 12 of 17 fairness flags in the
  regression example before it was rescaled. Notebook `03` documents the
  workaround; making it relative is queued for 0.4.2.
- Notebooks `04` and `05` are separate because XGBoost and PyTorch link
  different OpenMP runtimes on macOS and segfault in the same process.


## [0.4.0] - 2026-08-26

Multiclass support, including **ordinal** problems such as underwriting
accept / refer / decline, where a decline-vs-accept error costs more than
refer-vs-accept. Nothing here is breaking.

### Added
- **`context.class_order`** — class labels in ascending order of
  favourability, e.g. `["decline", "refer", "accept"]`. Supplying it marks
  the problem as ordinal. Omit it for a genuinely nominal problem.
- **`context.favourable_classes`** — which outcomes count as a positive
  result for demographic parity. Defaults to the most favourable entry of
  `class_order` (and logs that it inferred). With neither, the multiclass
  parity check reports `NOT_APPLICABLE` rather than picking a class
  arbitrarily: which outcome is favourable is a judgement the data cannot
  supply.
- **Ordinal metrics**, both numpy-native and requiring `class_order`:
  - `ordinal_mae` — mean absolute error in *rank* space. Predicting
    "decline" on an "accept" case is two steps wrong; "refer" is one.
  - `quadratic_kappa` — Cohen's kappa with quadratic weights, the standard
    ordinal agreement measure. Penalises a disagreement by the *square* of
    its rank distance, so a two-step error costs four times a one-step one.
- **`PerformanceConfig.average`** (default `"macro"`) — the multiclass
  averaging strategy for `f1`, `precision` and `recall`. Macro weights every
  class equally, so a rarely predicted "decline" counts as much as a common
  "accept". `metric="auto"` resolves to `balanced_accuracy` for multiclass,
  since accuracy flatters a model that never predicts the rare class.
- **`SecurityConfig.adversarial_max_rank_shift`** — for ordinal problems,
  `AdversarialRobustnessCheck` now reports the mean rank *distance* a
  prediction moves under perturbation alongside the flip rate. Two models
  can flip at an identical rate while one wobbles by a single rank and the
  other swings across the scale; a bare flip rate cannot tell them apart.
- `ModelAdapter.predict_proba_matrix` — the full `(n_rows, n_classes)`
  matrix, which the multiclass checks need.
- CLI `--class-order`, `--favourable-classes` and `--average`.
- `bdp_model_gate.classes`, and `tests/test_multiclass.py` — 28 tests.

### Changed
- `DisparateImpactCheck` supports multiclass by collapsing predictions to a
  "landed in a favourable class" indicator, which is what a selection rate
  means once there are more than two outcomes.
- `CounterfactualFlipCheck` supports multiclass, measuring the shift in
  P(favourable outcome). It was binary-only before.
- `ShapSubgroupCheck` reduces a multiclass SHAP array to the favourable
  class column, so it answers "does this feature push some groups away from
  being accepted?" rather than averaging across unrelated classes. It
  previously reported `NOT_APPLICABLE` for any problem with more than two
  classes.
- `roc_auc` and `average_precision` are now explicitly binary-only. Their
  multiclass forms need a full probability matrix, which the `y_pred`
  contract does not carry, so they are refused rather than approximated.


## [0.3.2] - 2026-08-26

### Fixed
- **The package could not be imported on Python 3.9**, which is inside the
  supported range (`requires-python = ">=3.9"`). `config.py` gained
  `max_error: float | None` in 0.3.0 but has no
  `from __future__ import annotations`, so the PEP 604 union was evaluated
  at runtime — a dataclass field annotation, so it fired on import and took
  every test module down at collection with
  `TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'`.
  Invisible on 3.10+, where PEP 604 is native. Broke the 3.9 CI job on both
  the 0.3.0 and 0.3.1 pushes; every other job passed.

### Changed
- Ruff now selects the `FA` rules, which flag PEP 604 and PEP 585 syntax
  used without `from __future__ import annotations`. `FA102` reproduces the
  failure above statically, so lint catches it on any interpreter instead
  of only the 3.9 CI job. This was the actual gap: `target-version = "py39"`
  alone does not imply those checks.
- Added a test that walks the package AST for the same pattern, so the guard
  survives a change to the lint configuration.


## [0.3.1] - 2026-08-26

> **Never released.** The tag for this version was withdrawn: the code cannot be imported on Python 3.9, which is inside `requires-python`. Everything below ships in [0.3.2] instead.

Any model, not just scikit-learn-shaped ones. Neural networks, raw
boosters, custom classes and remote scoring endpoints are all gateable.
Nothing in this release is breaking — an sklearn-style `model=` keeps
working exactly as before.

### Added
- **`predict_fn`, `predict_proba_fn` and `gradient_fn`** on
  `StructuredGateContext`. Each is a plain `fn(DataFrame) -> array`. The
  boundary is deliberately "DataFrame in, array out": your function owns
  tensor conversion, device placement, batching and auth, so this package
  never imports a deep-learning framework and never guesses at a dtype.
- **`context.model` is now optional.** A remote scoring endpoint has no
  model object at all — `predict_fn` alone is a complete context. A bare
  callable is also accepted as `model=`, so the two routes are
  interchangeable rather than a trap.
- **Probability-shape normalisation.** Binary classifiers emit
  `(n, 2)` (scikit-learn), `(n, 1)` (a Keras sigmoid) or `(n,)` depending on
  the framework; all three now reduce to one positive-class vector. A
  genuinely multiclass `(n, k)` output is refused with a clear message
  rather than silently sliced.
- **`gradient_fn` powers a real targeted attack.**
  `AdversarialRobustnessCheck` now prefers true per-row gradients, falls
  back to `coef_` for linear models, and only then to random noise — so a
  differentiable model gets a meaningful robustness probe instead of a weak
  one. The method used is recorded in the result metadata as `gradient-fn`,
  `gradient-directed` or `random`.
- **CLI `--model-loader "package.module:factory"`.** joblib only reads
  pickles, which rules out `.pt` checkpoints, Keras SavedModel directories,
  ONNX graphs and endpoints. Name a function that returns a model or a
  scoring callable and your loader does the framework import.
  Mutually exclusive with `--model`.
- `tests/test_model_agnostic.py` — 19 tests covering all three supply
  routes, every binary probability shape, a torch-shaped model (callable,
  array-in, column-vector-out) with no torch dependency, gradient
  precedence, and the CLI loader.

### Changed
- **Checks no longer touch `context.model` directly.** A new internal
  `bdp_model_gate.model.ModelAdapter` is the single place that knows how to
  call a model. Previously five call sites each made their own
  scikit-learn assumption — `.predict()`, a two-column `.predict_proba()`,
  or `.coef_` — which is what made anything else unusable.
- `CounterfactualFlipCheck` works for any model that can produce
  probabilities, not only one with a `.predict_proba()` method. It was
  `NOT_APPLICABLE` for every Keras and PyTorch model before.
- `ShapSubgroupCheck` explains through the adapter, so it works on a
  `predict_fn`-only context where there is no model object to introspect.

### Notes
- `roc_auc`, `average_precision`, `balanced_accuracy`, `f1`, `precision`
  and `recall` still require scikit-learn. That is a *metrics* dependency,
  not a model one — the regression metrics and `accuracy` are numpy-native
  and work on a core install.
- A public, subclassable `ModelAdapter` is deferred to 1.0.0. Until then
  the extension point is a plain callable.


## [0.3.0] - 2026-08-26

> **Never released.** The tag for this version was withdrawn: the code cannot be imported on Python 3.9, which is inside `requires-python`. Everything below ships in [0.3.2] instead.

Regression support, and the task abstraction it needed. Multiclass follows
in 0.4.0; example notebooks in 0.4.1.

### Added
- **Prediction tasks.** `StructuredGateContext.task` accepts `"auto"`
  (default), `"binary"`, `"multiclass"` or `"regression"`. `"auto"` infers
  from `y_true` and **logs what it inferred** — inference is genuinely
  ambiguous, since a claims-frequency target of 0/1/2/3 is indistinguishable
  from a four-class problem by shape alone. Set it explicitly for anything
  you gate on. New module `bdp_model_gate.task`.
- **`BaseCheck.supported_tasks`.** Each check declares the tasks it can
  meaningfully run against; `ModelGate` reports `NOT_APPLICABLE` for the
  rest instead of letting them produce a confident, meaningless number. It
  defaults to every task, so plugins written before 0.3.0 keep working.
- **Regression metrics:** `rmse`, `mae`, `mape`, `poisson_deviance`
  (for claims-frequency counts) and `r2`. All are implemented in numpy, so
  they work on a core install and are unaffected by scikit-learn's churn
  around `mean_squared_error(squared=False)`. `metric="auto"` resolves to
  `r2` for regression — an RMSE default would mean nothing without knowing
  whether the target is naira premiums or claim counts.
- **`PerformanceConfig.max_error`.** Error metrics are lower-is-better, so
  they are gated with `max_error` while higher-is-better metrics keep using
  `min_score`; each metric declares its own direction and the report names
  which comparison ran. `max_error` has no default, and configuring an error
  metric without it raises `GateConfigurationError` rather than passing
  silently — the comparison is the entire point of the gate.
- **Four regression fairness checks** in
  `bdp_model_gate.structured.regression_fairness`, each answering something
  the others cannot:
  - `LossRatioParityCheck` — is one group charged a higher **margin over its
    own expected loss**? The actuarially meaningful test for a pricing model:
    charging more in a higher-loss segment is risk-based pricing, charging a
    higher margin is not justified by cost. Needs the new
    `context.expected_loss`.
  - `GroupMeanGapCheck` — raw spread in mean prediction across groups.
  - `ErrorParityCheck` — is the model materially worse for one group?
  - `CalibrationParityCheck` — does one group's prediction systematically
    over- or under-shoot its realised outcome?

  All are relative to the overall figure, so one threshold works for naira
  premiums and claim counts alike, and groups below
  `FairnessConfig.min_group_size` (default 30) are reported but not scored.
- `StructuredGateContext.expected_loss` — per-row expected loss or technical
  premium, row-aligned to `X`.
- `GateReport.task`, in `summary()` and the JSON report: a report read months
  later must state what it assumed the model was.
- CLI `--task`, `--expected-loss-col` and `--max-error`.
- `tests/test_regression.py` — 31 tests covering task inference, metric
  direction, all four fairness notions and the validation rules.

### Fixed
- **The CLI silently mis-scored non-binary models.** `_predict` always took
  `predict_proba(X)[:, 1]`, which for a three-class underwriting model is
  just P(class 1) scored as though it were the positive class. It now picks
  a task-appropriate prediction and logs when it declines the probability
  path.
- **`AdversarialRobustnessCheck` would have blocked every regression model.**
  It measured a class flip rate, and any perturbation moves a continuous
  output, so the rate was ~1.0 by construction. Regression now measures the
  mean relative prediction shift against
  `SecurityConfig.adversarial_max_relative_shift`.
- Regression targets are now validated as numeric and finite, with an error
  message that points at `context.task` when the target looks categorical.
- **`AdversarialRobustnessCheck` perturbed features off a single global
  scale.** The gradient-directed path derived one step size from the mean
  magnitude across *all* numeric columns, so the largest column dominated:
  with a sum-insured column in the millions beside a 0–10 risk score, the
  risk score was shoved by thousands. That is not the "small relative
  perturbation" the check documents. Each feature is now stepped relative to
  its own magnitude, matching what the random path already did. Present
  since 0.1.0; it inflated flip rates for classification too, but only
  became glaring on regression, where it reported a relative prediction
  shift of ~1448 and blocked a perfectly stable linear model.
- **`ShapSubgroupCheck` hard-blocked on models exposing only `.predict()`.**
  shap's generic `Explainer` wants a callable or an estimator it recognises,
  but this library's own validation requires nothing more than `.predict()`.
  Such a model raised, and `ModelGate` converts an exception into a
  *blocking* `CHECK_ERROR` — so a non-blocking fairness check could stop a
  deploy. It now falls back to explaining `model.predict` (the documented
  black-box pattern) and, if shap still cannot cope, reports
  `NOT_APPLICABLE` instead of raising.

### Changed
- `DisparateImpactCheck` and `CounterfactualFlipCheck` are classification-
  only and report `NOT_APPLICABLE` for regression; the regression suite
  covers that ground instead.
- `AUTO_PREFERENCE` is superseded by `AUTO_PREFERENCE_BY_TASK`. The old name
  remains as an alias for the binary order.


### Changed
- Install instructions now point at PyPI. The package is published at
  <https://pypi.org/project/bdp-model-gate/>, so the TestPyPI
  `--extra-index-url` dance in `examples/` is gone; the example notebook's
  committed outputs were regenerated against `0.2.1` installed from PyPI.
- Added `Programming Language :: Python :: 3.9`–`3.13` classifiers. PyPI
  generates its "python versions" badge from these rather than from
  `requires-python`, so without them the badge reads only "3". Takes effect
  on the next upload — 0.2.1 is already published without them.
- README gained PyPI/Python/licence badges and a pointer to the example
  notebook.

## [0.2.1] - 2026-08-26

### Fixed
- **`DisparateImpactCheck` silently reported `OK` for probability
  predictions.** Demographic parity compares selection rates, counting
  predictions equal to `1`. A probability is never exactly `1`, so every
  group's selection rate came out as `0` and the parity difference was
  always exactly `0.000` — the check passed no matter how skewed the model
  was. Verified against a maximally discriminatory model (every man
  selected, no women): the check reported `0.000` where the true parity
  difference is `1.000`. This mattered in practice because the documented
  quickstart passes `model.predict_proba(X)[:, 1]`, so the default usage
  disabled the check. `y_pred` is now binarised at the new
  `FairnessConfig.decision_threshold` (default `0.5`); predictions already
  in `{0, 1}` are untouched, so callers passing hard labels see no change.
- **`ShapSubgroupCheck` crashed on `RandomForestClassifier`.** shap returns
  a 3-D `(rows, features, classes)` array for some classifiers and 2-D for
  others, and which you get changed across shap versions. Building a
  DataFrame from the 3-D form raised `ValueError: Must pass 2-d input`,
  which the gate surfaced as a blocking `CHECK_ERROR`. Binary output is now
  reduced to the positive class; genuine multiclass reports
  `NOT_APPLICABLE` rather than guessing at a class.
- **The `structured` extra resolved to a combination that could not
  import.** `shap>=0.44,<0.47` is incompatible with `numpy>=2.0`, which the
  `numpy>=1.23,<3.0` range permits, so a fresh install produced shap 0.46
  alongside numpy 2.x and `import shap` raised `TypeError: Converting
  'np.inexact' or 'np.floating' to a dtype not allowed`. Since
  `ShapSubgroupCheck` catches only `ImportError`, this surfaced as a
  blocking `CHECK_ERROR` rather than a graceful skip. The floor is now
  `shap>=0.48` (imports cleanly against numpy 2.x, ships cp313 wheels) and
  the ceiling `<0.50`, which raised its own Python floor to 3.11 — above
  this package's `requires-python`.

### Added
- `FairnessConfig.decision_threshold` — the cutoff used to binarise
  continuous predictions for `DisparateImpactCheck`. Reported in that
  check's result metadata so a report states what it measured against.
- `examples/bdp_model_gate_walkthrough.ipynb` — an end-to-end notebook
  covering the full public API, committed with outputs.
- `tests/test_fairness_fixes.py` — regression tests for both fairness bugs,
  including one that asserts on the result *flag* rather than merely that
  results exist. Asserting existence is why the SHAP crash went unnoticed:
  routed through `ModelGate`, a raising check still produces a result.

### Changed
- CI test matrix extended to Python 3.13, now that `shap>=0.48` supports it.


## [0.2.0] - 2026-08-26

### Added
- **Configurable performance metric.** `PerformanceConfig.metric` selects
  what the model is scored on: a built-in name (`roc_auc`,
  `average_precision`, `accuracy`, `balanced_accuracy`, `f1`, `precision`,
  `recall`), a `fn(y_true, y_pred) -> float` callable, or `"auto"`. A new
  `bdp_model_gate.metrics` module owns resolution.
- `PerformanceConfig.decision_threshold` (default `0.5`) — binarizes
  continuous predictions for metrics that need hard class labels.
  Predictions already in `{0, 1}` are untouched; ranking metrics ignore it.
- `GateReport.model_metric` / `model_score`, and a `model_metric` /
  `model_score` pair in `to_dict()` / `to_json()`. `summary()` now prints
  the headline score.
- CLI `--metric`, `--min-score`, and `--decision-threshold` flags, which
  take precedence over `--config` file values.
- `AdversarialRobustnessCheck(random_state=...)` to control its sampling
  and perturbation seed.
- `ModelGate` now dispatches input validation on `context.modality` via a
  `VALIDATORS` registry, instead of hardcoding the structured validator —
  an unknown modality raises `GateValidationError` rather than being
  silently validated as structured.
- `.github/workflows/ci.yml`, `.pre-commit-config.yaml`, and `.gitignore`,
  which the README described but the repo didn't contain. CI gained a
  core-install job (no `structured` extra) that exercises the
  graceful-degradation and metric-fallback paths, and a build job.
- Tests: `tests/test_metrics.py` covering metric selection, fallback
  visibility, thresholding and the deprecation aliases; plus tests for
  adversarial determinism, coefficient alignment, and modality dispatch.
- Eager input validation (`GateValidationError`) — bad inputs now fail
  fast with a clear message instead of an opaque exception from inside a check.
- Structured `logging` throughout the library and CLI, replacing `print()`.
  CLI gained a `-v/--verbose` flag.
- Per-check timing (`CheckResult.duration_ms`, `GateReport.total_duration_ms`).
- Plugin system: third-party checks can be registered via the
  `bdp_model_gate.checks` entry-point group and are picked up automatically
  by `default_structured_checks()`.
- CLI `--config` now accepts YAML and TOML in addition to JSON.
- `ShapSubgroupCheck` now uses `shap.TreeExplainer` automatically for
  tree-based models (much faster and exact, vs. the generic explainer).
- `AdversarialRobustnessCheck` now uses a gradient-directed perturbation
  for linear models (via `model.coef_`) instead of always using isotropic
  random noise; falls back to random perturbation for black-box models.
- `py.typed` marker (PEP 561) — type checkers now see this package's
  annotations.
- Full type-hint pass across the codebase; `mypy` and `ruff` configured
  and added to `dev` extras.
- Expanded test suite: input-validation tests, edge cases (models without
  `predict_proba`, all-categorical features, a check that raises), plugin
  registry test, and `pytest-cov` with an 85% coverage floor enforced in CI.
- `tests/conftest.py` with shared fixtures (previously duplicated per test file).
- CI workflow (`.github/workflows/ci.yml`) running lint, type-check, and
  tests on every push/PR — separate from the pre-deployment gate workflows.

### Changed
- Renamed package from `mlgate` to `bdp_model_gate` (distribution name
  `bdp-model-gate`, CLI command `bdp-model-gate`).
- Dependency version ranges are now upper-bounded, not just floored.

### Deprecated
- `PerformanceConfig.min_accuracy` → use `min_score` and set `metric` to
  name what it applies to. The old name still works, in Python and in
  `--config` files, but emits a `DeprecationWarning`; the CLI additionally
  logs a rename notice.
- `GateReport.model_auc` → use `model_metric` + `model_score`. It now
  returns `None` unless the configured metric really was `roc_auc`, rather
  than mislabelling another metric's score as an AUC. The `model_auc` key
  remains in the JSON report for existing consumers, under the same rule.

### Fixed
- **`AdversarialRobustnessCheck` was not deterministic.** Its random
  perturbation used unseeded `np.random`, so the same model and data could
  produce a different flip rate — and a different gate verdict — between
  runs. Now seeded via `random_state` (default 42), matching the sampling
  already done elsewhere in the check.
- **`AdversarialRobustnessCheck` misaligned linear coefficients.**
  `coef_` is laid out over every column the model was fitted on, but was
  indexed by position among the *numeric* columns — so any non-numeric
  column ahead of a numeric one applied the wrong feature's weight. Now
  indexed by position in `X.columns`, with an explicit bail-out to random
  perturbation when the lengths don't line up. Multiclass `coef_` (one row
  per class) also falls back, rather than being flattened into a
  meaningless direction.
- **The performance gate silently changed metrics.** `min_accuracy` was
  compared against ROC AUC when scikit-learn was installed and against
  accuracy when it wasn't, with nothing in the report saying which. The
  metric is now explicit, and any fallback is logged at `WARNING`, flagged
  in the result metadata, and named in the detail string.
- `GateReport.model_auc` was populated by recomputing ROC AUC in the gate,
  independently of what the performance check actually gated on.
- Docstrings still referring to the pre-rename `mlgate` package name.
- Removed the empty, unused `bdp_model_gate/ci_examples/` directory (the
  real examples live in the top-level `ci_examples/`).

## [0.1.0] - 2026-08-22

### Added
- Initial release: fairness (proxy correlation, disparate impact, SHAP
  subgroup gaps, counterfactual flip), performance thresholds, NDPA/NDPR
  compliance mapping, and security checks (adversarial robustness, PII
  leakage, prompt injection).
- `ModelGate` / `GateReport` / `StructuredGateContext` core API.
- `bdp-model-gate` CLI for CI/CD use.
- Azure Pipelines and GitHub Actions pre-deployment gate examples.

[Unreleased]: https://github.com/vanjy-eng/model-gate/compare/v0.5.2...HEAD
[0.5.2]: https://github.com/vanjy-eng/model-gate/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/vanjy-eng/model-gate/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/vanjy-eng/model-gate/compare/v0.4.2...v0.5.0
[0.4.2]: https://github.com/vanjy-eng/model-gate/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/vanjy-eng/model-gate/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/vanjy-eng/model-gate/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/vanjy-eng/model-gate/compare/v0.2.1...v0.3.2
[0.3.1]: https://github.com/vanjy-eng/model-gate/compare/b02ebe0...561ef52
[0.3.0]: https://github.com/vanjy-eng/model-gate/compare/v0.2.1...b02ebe0
[0.2.1]: https://github.com/vanjy-eng/model-gate/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/vanjy-eng/model-gate/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/vanjy-eng/model-gate/releases/tag/v0.1.0
