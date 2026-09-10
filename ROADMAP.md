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

Current release: **0.6.0**. Next: **0.6.1**.

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
