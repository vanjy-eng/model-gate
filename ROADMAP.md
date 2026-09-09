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
5. **A pull request to `main`**, with CI and the docs build green before
   merge. Both are conventions rather than enforcement today — see
   [Before the next release](#before-the-next-release--the-repository-itself),
   which is about fixing that.

A release is not done when the code works. It is done when someone who has
never seen it can find it, read why it exists, and run it.

Current release: **0.5.4**. Next: **0.6.0**.

> **This file tracks what should happen.** What already happened lives in
> [`CHANGELOG.md`](CHANGELOG.md), and shipped entries are removed from here
> rather than marked done. The only release appearing in both is the one
> currently being built.

---

## Before the next release — the repository itself

Not a release, and not code: three findings about this repository's own
configuration, checked against the GitHub API on 2026-09-09. The first one is
live.

### `main-lock` has never protected `main`

The ruleset exists, is `active`, and applies to nothing. Its ref condition is:

```
refs/heads/"main"
```

Those double quotes are literal characters in an fnmatch pattern, so it can
only ever match a branch named `"main"` — quotes included — and the branch is
called `main`. `GET /repos/{owner}/{repo}/rules/branches/main` returns `[]`,
and the ruleset reports `current_user_can_bypass: "never"`, so the empty
answer is not bypass permission masking the rules. There are none.

Which means, right now: **direct pushes to `main` are not blocked,
force-pushes are not blocked, and no review is required.** Every release so
far went through a PR by convention, not by enforcement.

> A guard with a hole in it is worse than none, because it is trusted. That
> sentence is already in `CONTRIBUTING.md` about the version-stamp guard, and
> it applies here to `CONTRIBUTING.md` itself, which states that "direct
> pushes are blocked by a ruleset". They are not. Fix the ruleset and the
> sentence becomes true; until then it is a false claim in the contributor
> documentation.

Fix: change the condition to `refs/heads/main` — or better `~DEFAULT_BRANCH`,
which cannot be typo'd.

### There is no required-status-check rule, and that is the one that matters

Even with the pattern fixed, `main-lock` carries `deletion`,
`non_fast_forward`, `creation`, `update`, `pull_request`, `code_scanning` and
`code_quality` — and **no `required_status_checks`**. So a PR could be merged
with a red matrix.

That is not hypothetical: v0.3.0 and v0.3.1 were both tagged on commits with a
red Python 3.9 job, and both tags had to be withdrawn. Requiring a PR stops
nobody from merging a broken one.

Add `required_status_checks` naming the jobs, exactly:

- `Lint and type-check`
- `Test (Python 3.9)` … `Test (Python 3.13)`
- `Test (core install only)`
- `Build distribution`
- `Build site`

Deliberately **not** `Mutation testing (advisory)`. It is `continue-on-error`
and time-boxed; requiring it would make a 25-minute advisory job a merge
blocker.

### Two rules that cannot currently be satisfied

- **`require_code_owner_review: true`, and there is no `CODEOWNERS` file.**
  Either add one or drop the rule; a rule that names a file that does not
  exist is another guard that reads as protection and is not.
- **`required_approving_review_count: 1` on a single-maintainer repository.**
  GitHub will not let an author approve their own pull request, so this rule
  can only ever be satisfied by a second person or by an admin bypass — and a
  rule whose normal path is "bypass it" trains the reflex that defeats every
  other rule.

  The honest position for now: **required status checks are the rule that
  protects this repository; required reviews are theatre until there is a
  second maintainer.** Keep the `pull_request` rule for the merge-method and
  thread-resolution settings, set the approval count to 0, and raise it the
  day someone else has commit rights. Record the decision either way, because
  the alternative is discovering it at 11pm during a release.

### The tag ruleset is disabled, and would break tagging if enabled

`release tags` is `enforcement: disabled`, which is why `v0.5.3` and `v0.5.4`
pushed without complaint. Its rules are `deletion`, `non_fast_forward` and
`creation` against `~ALL` tags with no bypass actors — and `creation` on
`~ALL` forbids creating *any* tag, including the release tags the ruleset is
named for.

Fix before enabling: scope the condition to `refs/tags/v*`, keep `deletion`
and `non_fast_forward`, drop `creation`. That gives what the name suggests and
prevents a repeat of the withdrawn v0.3.0/v0.3.1 tags.

### Checklist

- [ ] `main-lock` condition -> `~DEFAULT_BRANCH`
- [ ] `main-lock` gains `required_status_checks` with the seven jobs above
- [ ] Decide `required_approving_review_count` (0 for now) and
      `require_code_owner_review` (drop, or add `CODEOWNERS`)
- [ ] `release tags`: scope to `refs/tags/v*`, drop `creation`, then enable
- [ ] Restore the plain "direct pushes are blocked" wording in
      `CONTRIBUTING.md` once the first two boxes are ticked — it has been
      softened to the truth in the meantime rather than left as a false claim

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

#### Decided: the interval both gates and reports, and the interval decides which

The earlier version of this entry left it open — "flag on the interval, or at
minimum report it". Both, and the choice is not a mode the user picks per run:
it falls out of where the interval sits.

| Interval vs threshold | Meaning | Verdict |
|---|---|---|
| entirely **above** | the finding is real | flag, with the check's normal `blocking` |
| **straddles** it | *you do not know* | `NEEDS_REVIEW`, non-blocking, interval in the detail |
| entirely **below** | clean | `OK` |

That three-way split is the machinery this library already has, and it maps
onto what the two outcomes are actually *for*: a report is something a
governance session discusses, a gate breaks a CI pipeline. "The disparity might
be 0.06 and might be 0.14" is a conversation, not a build failure.

#### Configurable, because a gate nobody can overrule gets switched off

The tool must not hold back a team that has looked at the uncertainty and
accepted the risk. New `UncertaintyConfig`:

| Field | Default | Effect |
|---|---|---|
| `compute_intervals` | `True` | off entirely — bootstrapping is not free |
| `confidence_level` | `0.95` | |
| `bootstrap_samples` | `1000` | the cost lever; recorded in metadata |
| `on_uncertain` | `"review"` | `"block"` \| `"review"` \| `"point"` |

`on_uncertain` is the escape hatch:

- **`"review"`** (default) — a straddling interval routes to a human.
- **`"block"`** — the precautionary posture. If it *might* breach, stop. Some
  regulated deployments will want this, and it is one branch in the same
  comparison.
- **`"point"`** — decide exactly as today, on the point estimate. **The
  interval is still computed and still printed in the detail string and the
  metadata**, so it reaches the governance pack and the conversation happens
  there. This is "we have seen the uncertainty and accepted it", and it must
  not require switching the check off to express.

Note what `"point"` deliberately does *not* do: suppress the number. Accepting
a risk and not being told about it are different things, and only the first is
a decision.

#### The constraint that will bite: an interval must not depend on row order

`tests/test_invariants.py::test_row_order_does_not_change_the_verdict` asserts
that permuting the validation set cannot move a verdict. A bootstrap that
resamples **positions** under a fixed seed breaks that: the same seed picks the
same indices, so a re-sorted frame yields a different resample and therefore a
different interval — and, near a threshold, a different verdict.

This is the `stable_sample` problem again, and the answer is the same one:
resample by row **content**, not position. Whatever the mechanism, the property
is non-negotiable and belongs in `test_invariants.py` beside the existing
permutation test. Sorting a CSV must not change whether a model ships.

#### Cost

Bootstrap the *statistic* over resampled indices in numpy. Never re-run a
check `bootstrap_samples` times — the fairness suite would take minutes.
Record `bootstrap_samples` in metadata so a reader knows how much evidence is
behind the interval.

#### Also in scope

- **A split-stability test**: halve the validation set at random and assert
  the verdict agrees. It would fail today, and it belongs beside the
  permutation-invariance test that found the sampling bug. Its failures are
  the map of which checks need intervals most.
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

- **A mutation kill-rate floor — but measure the decision surface, not the
  codebase.** The old framing of this item asked whether to floor the *rate* or
  the *absolute count*. Both are the wrong question, because the population
  being measured is mostly noise.

  Counted at 0.5.4 with mutmut's own operator table, across the modules the
  config actually mutates:

  | Operator | Mutants | Share |
  |---|---|---|
  | `arg_removal` | 6,034 | 48% |
  | `string` | 3,426 | 28% |
  | `assignment` | 1,114 | 9% |
  | **`swap_op`** | **706** | 6% |
  | **`number`** | **660** | 5% |
  | **`keywords`** | **253** | 2% |
  | **`remove_unary_ops`** | **81** | 1% |
  | everything else | 162 | 1% |
  | **total** | **12,436** | |

  **The 1,700 mutants that can produce a wrong verdict are 14% of the
  population.** The other 86% is inflating the score at one end and diluting it
  at the other:

  - `arg_removal` drops each call argument, and 2,837 of the 3,593 argument
    positions in those modules are *required positionals*. Dropping one raises
    `TypeError`, so any test touching the line kills it. Free kills that prove
    nothing.
  - `string` mutates prose detail strings. Killed or not depending on whether
    some test happens to assert on that substring, which says nothing about
    whether the verdict is right.

  That is why the trend was uninterpretable: 0.5.2 killed 286 *more* mutants
  than 0.5.1 and scored 0.7 points *lower*. It is a ratio over a population
  whose composition changes every release.

  And the run already tests only a fraction of it, chosen arbitrarily: 4,185 of
  7,004 at 0.5.2 (60%), so roughly a third of today's 12,436 — with *which*
  third decided by where the 25-minute box happens to stop. **Focusing is not a
  reduction in coverage; it replaces an arbitrary subset with a chosen one.**

  Proposal: mutate only `swap_op`, `number`, `keywords` and
  `remove_unary_ops`. About 1,700 mutants, roughly 10 minutes at the observed
  throughput — so the run **completes**, which is what makes a rate comparable
  release to release, every survivor actionable, and a floor safe to turn on.
  It also maps exactly onto the failure this project exists to prevent: a
  flipped comparison, a shifted threshold, an inverted boolean *is* the
  confident-wrong-green-number.

  Cost, stated honestly: we stop asking "would anything notice if `metadata=`
  were dropped?". A few of those are real findings — but they are a minority of
  a majority, and the current setup is not testing them either, because it
  times out first.

  Mechanism: mutmut 3.7 has **no operator filter**. `do_not_mutate_patterns` is
  line-regex based and reaches only 6% (measured — the noise is spread across
  every `CheckResult(...)` call, not concentrated on string-only lines). So
  this needs a small runner that prunes `mutmut.mutation.mutators.mutation_operators`
  before invoking the CLI, which means **pinning mutmut exactly** — currently
  unpinned, and it belongs with the tooling work above anyway.

  Decision still open: whether to take this or keep the whole-population
  number. Nothing else in this release depends on it.
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

## 0.6.1 — Release automation, in two stages

**Branch:** `feat/release-automation`

Publishing is manual. Two of the silent failures this project has shipped were
caught *only* by installing the published artifact, and v0.3.0 and v0.3.1 were
both tagged on commits with a red Python 3.9 job. Fitting, for a tool that
exists to gate deploys.

### The shape: publish a candidate, then promote it

Two workflows, two triggers, one artifact.

```
STAGE 1 — candidate            trigger: tag v* pushed
  guard        tag == pyproject version, and CI was green for this exact SHA
  build        sdist + wheel; suite run against the INSTALLED wheel
               -> SHA256SUMS recorded, dists uploaded as a run artifact
  testpypi     environment: testpypi, no approval          -> TestPyPI
  smoke        fresh venv, install FROM TestPyPI, run a real gate

        ... a human tests the candidate, for as long as that takes ...

STAGE 2 — promotion            trigger: GitHub Release published
  fetch        download stage 1's dists for this tag; RE-VERIFY SHA256
  pypi         environment: pypi, REQUIRED REVIEWER        -> PyPI
```

### Why two workflows rather than one with an approval gate

The obvious design is a single workflow whose last job waits on a required
reviewer. Three reasons not to.

**1. Least privilege by construction, not by approval.** Trusted Publishing
binds a publisher to a specific *workflow filename* plus environment. Register
PyPI's publisher against `release-promote.yml` **only**, and a tag push becomes
structurally incapable of reaching PyPI: no credential for it exists in that
workflow. In the single-workflow design the capability is present on every tag
push and held back by a person clicking the right button.

For a library whose entire purpose is stopping the wrong thing reaching
production, "it cannot" beats "someone must not".

**2. Promotion is a decision, and decisions need a record.** Publishing the
GitHub Release is the human act, and the release notes plus the environment
approval are the audit trail. A pending deployment inside a workflow run is
not a record anyone reads a year later.

**3. Testing a candidate takes days, and a pending run does not.** GitHub
expires pending deployments, and a run parked waiting for approval is awkward
to re-trigger. Decoupling means the candidate can sit on TestPyPI as long as it
needs to.

### The property separation puts at risk, and how stage 2 keeps it

> **PyPI must receive the exact bytes that TestPyPI was tested with.**

So **stage 2 must never rebuild.** Builds are not reproducible across runner
images and setuptools versions, and a rebuild would publish an artifact nobody
smoke-tested — which is precisely the failure mode this pipeline exists to
prevent, reintroduced by the pipeline itself.

Stage 1 records `SHA256SUMS` beside the dists. Stage 2 downloads that run's
artifact, re-verifies every digest, and **fails if the artifact is missing or a
digest differs**. No fallback to building; a missing candidate means re-running
stage 1, not improvising.

### One decision to take before writing any of it

**Does the candidate burn the real version number on TestPyPI?**

Version collisions are permanent on *both* indexes. Publish `0.6.0` to
TestPyPI, find a bug, and `0.6.0` can never be published to TestPyPI again.

The tempting escape is to publish `0.6.0.devN` to TestPyPI and `0.6.0` to PyPI
— but then the two artifacts are **not the same bytes**, and the property above
is gone. You cannot have both identical artifacts and a re-testable version
number.

**Take identical artifacts.** A failed candidate means bumping the patch
version and tagging again, which costs a version number and nothing else. Say
so in `CONTRIBUTING.md` so it is not rediscovered mid-release.

### Guards

- **Tag/version consistency** — fail if the tag does not match
  `pyproject.toml`. Cheap, and the whole release hangs off it.
- **Publish only on a green matrix for that commit** — query the CI conclusion
  for the tagged SHA rather than trusting that a tag implies a tested commit.
  This is the guard that v0.3.0 and v0.3.1 did not have.
- **Test the installed artifact, not the source tree.** The Python 3.9 import
  failure lived in the *published wheel*, and the shap/numpy clash only
  appeared on a clean install. Running the suite against the repo would have
  missed both.
- **`skip-existing: true` on TestPyPI, `false` on PyPI.** Re-running a
  candidate is normal; a duplicate on PyPI is a mistake and should fail loudly.
- **`--extra-index-url https://pypi.org/simple/` on the smoke test.** TestPyPI
  does not carry numpy or pandas, so `pip install --index-url` alone cannot
  resolve this package's dependencies. This trap has cost other projects an
  afternoon; it costs us one line.
- **Assert `__version__` from the installed wheel** in the smoke test, then run
  a real gate end to end and check the verdict — not just that the import
  succeeded.

### Manual prerequisites (repo admin, not code)

- [ ] Register **TestPyPI** trusted publisher: repo, `release-candidate.yml`,
      environment `testpypi`
- [ ] Register **PyPI** trusted publisher: repo, `release-promote.yml`,
      environment `pypi` — a *different* workflow filename, which is the point
- [ ] Create the `testpypi` environment (no protection)
- [ ] Create the `pypi` environment with a **required reviewer**
- [ ] Do the branch and tag ruleset work in
      [Before the next release](#before-the-next-release--the-repository-itself)
      first; only an Environment stands between a tag and PyPI, and only a
      ruleset stands between a bad commit and `main`

Two things to know: if either workflow **filename** changes, publishing breaks
until the trusted-publisher registration is updated; and **version collisions
are permanent**, per the decision above.

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
