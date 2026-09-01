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
4. **Web**: the affected pages under `web/docs/`, and the landing page — its
   version stamp every time, its checks grid and capability claims whenever
   those change. The site is part of the release, not a follow-up; see the
   checklist in [`CONTRIBUTING.md`](CONTRIBUTING.md#the-site-ships-with-the-release).
   `tests/test_package.py` enforces the version stamp and check coverage.
5. **A pull request to `main`** — the `main-lock` ruleset requires one, and CI
   plus the docs build must be green before merge.

A release is not done when the code works. It is done when someone who has
never seen it can find it, read why it exists, and run it.

Current release: **0.5.2**. Next: **0.5.3**.

> **This file tracks what should happen.** What already happened lives in
> [`CHANGELOG.md`](CHANGELOG.md), and shipped entries are removed from here
> rather than marked done. The only release appearing in both is the one
> currently being built.

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

- **A mutation kill-rate floor.** Baseline is **42.0%** measured in CI at
  0.5.2 (1758 killed of 4185 with a verdict, from 7004 generated), against
  42.7% at 0.5.1 and 35.5% at 0.4.2.

  Read the trend, not the number. 0.5.2 killed **286 more mutants** than 0.5.1
  and still scored lower, because the run is time-boxed to 25 minutes and a
  release that adds code adds mutants faster than the box gets through them.
  Generated mutants went 3430 → 5959 → 7004 across those three releases while
  the wall clock stayed fixed.

  That has a consequence for this item: a floor set on the *rate* will fail on
  an ordinary release that happens to add a lot of code. Either raise the time
  box first, or floor the **absolute kill count** instead. Decide that before
  turning `mutation_report.py --min-kill-rate` on.
- **Executed documentation code blocks.** The prose snippets under `web/docs/`
  are not run by anything, so an API change can leave them wrong while
  `mkdocs build --strict` still passes — the generated API reference and the
  notebooks are both guarded, and this is the remaining gap. Belongs with the
  tooling work rather than the intervals.

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
