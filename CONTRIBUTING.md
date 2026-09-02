# Contributing to bdp-model-gate

Thanks for looking. Issues, fixes, checks, docs and examples are all welcome,
and you do not need to be an ML specialist to help — a confusing error message
is a real bug in a governance tool.

One thing to read before anything else, because it shapes every convention
below.

## The failure mode this project exists to avoid

This library tells people whether a model may ship. A gate that crashes is a
nuisance. **A gate that returns a confident, wrong, green number is a
liability** — it launders a bad model through a process that was supposed to
catch it.

That has already happened here more than once. `DisparateImpactCheck` returned
a parity difference of `0.000` for a maximally discriminatory model, for a
whole release, and the test suite was green throughout: probabilities were
being compared for equality with `1`, so every selection rate was zero. The
test that should have caught it asserted `len(results) > 0`.

So the bar for a change is not "the tests pass". It is **"a test would have
noticed if this were wrong"**. Most of what follows is machinery for that one
question.

Practical consequences you will meet in review:

- A test that asserts a check *ran* is not a test. Assert the **value**.
- If a check cannot answer, it must return `NOT_APPLICABLE` **with a reason**,
  never a plausible number and never an exception.
- If a plot and a number could disagree, they must be computed by the same
  code — see [Adding a plot](#adding-a-plot).

## Getting set up

```bash
git clone https://github.com/vanjy-eng/model-gate.git
cd model-gate

python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,structured,plots,yaml,toml]"

pip install pre-commit && pre-commit install
```

Install all the extras for development even though they are optional at
runtime: `dev` alone will silently skip the plotting tests.

Python 3.9–3.13 are supported. Develop on whichever you have — the CI matrix
covers the range, and [one gotcha](#the-python-39-floor) is worth knowing
about up front.

## The local loop

```bash
ruff check .              # lint  (--fix to autofix)
ruff format .             # format
mypy bdp_model_gate       # type check
pytest -q                 # test — 85% coverage floor is enforced
```

`.pre-commit-config.yaml` runs ruff, mypy and basic hygiene on every commit and
mirrors CI, so most failures surface before you push.

Two notes on the tooling:

- **Ruff lints the notebooks too.** It has caught real portability bugs there,
  including a nested same-quote f-string that is valid on 3.12 and a syntax
  error on the 3.9 this package supports. It does not *format* them —
  `examples/*.ipynb` are excluded, because the formatter explodes compact
  keyword lists one-per-line and that reads worse in a walkthrough.
- **Ruff also formats Python inside Markdown code fences.** If `ruff format
  --check .` complains about a `.md` file, that is why.

## Testing standards

Coverage is a weak signal: it says a line executed, not that anything checked
the result. The suite is therefore organised by *what kind of wrongness each
file catches*, and a new check normally needs an entry in several.

| File | Catches |
|---|---|
| `test_known_answers.py` | a plausible number that is the wrong number — inputs whose correct answer is derivable on paper, asserted exactly |
| `test_invariants.py` | metamorphic breakage — permute the rows, rescale a feature, rename the groups; the verdict must stay put or move in a known way |
| `test_not_applicable.py` | a check that skips for the wrong reason, or skips when it should have run. Every skip path is asserted **on its reason string** |
| `test_model_matrix.py` | "it works on the one estimator the fixture uses". Every check against every model family, crossed with every task |
| `test_properties.py` | the inputs nobody imagined — `hypothesis` finds the empty group, the constant column, the value exactly on the threshold |
| `test_plots.py` | a chart that contradicts the number beside it |
| `test_reporting.py` | a report that leaks, fails to escape, or loses a finding |

### The suite-wide guard

`ModelGate` converts an exception inside a check into a blocking
`CHECK_ERROR` *result* rather than propagating it. That is right for
production — one broken check must not take down the gate — and it is why
`assert len(results) > 0` passes on a check that raised.

`tests/conftest.py` therefore wraps `ModelGate.run` with an autouse fixture and
**fails any test that produced a `CHECK_ERROR` it did not ask for**. When the
error is the thing under test, opt out explicitly:

```python
@pytest.mark.expect_check_error
def test_a_broken_check_does_not_crash_the_gate(...):
    ...
```

Reach for that marker only when you mean it. If you find yourself adding it to
make a failure go away, the failure is the point.

### Hostile fixtures

`conftest.py` ships the shapes that break naive implementations. Use them:

| Fixture | The bug it exists to catch |
|---|---|
| `wide_scale_frame` | feature magnitudes across seven orders of magnitude — a threshold derived from one global scale is dominated by the largest column |
| `tiny_group_protected` | a three-row group beside a 297-row one — small groups must be *reported, not scored*, or they produce wild ratios that read as findings |
| `severe_imbalance` | a 99.5 / 0.5 split, where accuracy flatters a model that never predicts the rare class |

### Mutation testing

At 0.5.2 the suite reports **91% line coverage** and a **42.0% mutation kill
rate** (1758 of 4185 mutants with a verdict). That gap is the honest measure
of how much of the suite executes code without asserting anything about it —
and the reason coverage alone is not the bar.

Read the rate as a trend, not a target. The run is time-boxed to 25 minutes,
so a release that adds code adds mutants faster than the box gets through
them: 0.5.2 killed 286 more mutants than 0.5.1 and still scored 0.7 points
lower.

```bash
mutmut run 2>&1 | tee mutation.log
python scripts/mutation_report.py mutation.log --min-tested 200
```

It is advisory in CI (`continue-on-error`), time-boxed to 25 minutes, and slow
locally — nobody expects you to run it on every change. It is the right tool
when you are hardening an area and want to know whether the tests would have
noticed.

Use `scripts/mutation_report.py` rather than reading mutmut's own output:
`mutmut results` lists **only survivors**, so counting statuses from it yields
a 0% kill rate no matter what actually happened. The report script parses the
run's own tally and refuses to report a score it cannot support.

## Adding a check

Decide first whether it belongs here at all. **If it is specific to your
organisation, ship it as a plugin** — the `bdp_model_gate.checks` entry-point
group is a supported, first-class route and needs no fork. See *Extending with
plugins* in the [README](README.md). A check belongs in core when it measures
something a wide range of regulated models need measured.

The interface:

```python
from bdp_model_gate import BaseCheck, CheckResult
from bdp_model_gate.task import CLASSIFICATION_TASKS


class MyCheck(BaseCheck):
    name = "my_check"  # snake_case; appears in the report
    category = "fairness"  # validation | fairness | performance | compliance | security
    blocking = True  # False routes to NEEDS_REVIEW instead
    supported_tasks = CLASSIFICATION_TASKS  # the gate reports NOT_APPLICABLE elsewhere

    def run(self, context) -> list[CheckResult]: ...
```

### Blocking is a judgement, not a default

`blocking=True` means a failure stops the deploy. `blocking=False` routes it to
a human as `NEEDS_REVIEW`. Fairness checks are non-blocking throughout this
library, deliberately: those findings need judgement, and a gate that hard-fails
on every one of them gets switched off, which protects nobody.

### Degrade, never guess

Every optional input has a skip path, and the skip has to say *why* in words
the reader can act on:

```python
return [
    CheckResult(
        self.name,
        self.category,
        "NOT_APPLICABLE",
        "no expected_loss supplied — margin parity needs a per-row expected loss "
        "(or technical premium) to compare the prediction against",
        self.blocking,
    )
]
```

Note what that message does not do: fall back to comparing raw prices. That
would answer a different question under the same name, which is the exact
failure this project is about.

The same applies to optional dependencies. `shap`, `fairlearn`, `scikit-learn`,
`matplotlib` and `seaborn` are all optional, and a check that needs one reports
`NOT_APPLICABLE` naming the extra rather than raising.

### Register it and cover it

1. Add it to `default_structured_checks()` in `bdp_model_gate/structured/__init__.py`.
2. Add a known-answer test with a hand-derivable expected value.
3. Add every skip path to `test_not_applicable.py`, asserting the reason.
4. Add its flag strings to the `VALID_FLAGS` set in `test_model_matrix.py`.
5. Document it in `web/docs/reference/checks.md` and any config fields in
   `web/docs/reference/configuration.md`.

## Adding a plot

Optional, and most checks should not have one. Draw only where a check
**collapses a distribution to a scalar and the shape is what the reader needs
to judge**. Latency and cost are genuinely scalars; a binary confusion matrix
is four numbers the detail line already carries. Charting those is decoration.

The contract is `plot(self, context, results=None, ax=None) -> Axes`: an `Axes`
in, the **same** `Axes` out. That is what lets a caller compose these into
their own figure. There is nothing to register — discovery is by override.

Four rules, and one that is not negotiable.

1. **Return `None` rather than an empty frame** when the inputs are missing.
2. **Never raise.** The report catches it and prints a note, but the reviewer
   still loses the chart.
3. **Recompute; do not store.** Small per-group dicts in `metadata` are
   *findings* and stay there. Anything array-sized — bin edges, curve points,
   per-row SHAP — is recomputed at plot time, so the archived JSON does not
   carry presentation data most consumers never read.
4. **Seaborn's axes-level functions only** (`barplot`, `lineplot`, `heatmap`,
   `scatterplot`, `histplot`). The figure-level ones (`relplot`, `catplot`,
   `displot`) build their own Figure and ignore `ax`.

And the one that matters most:

> **A chart may not contradict the number beside it.** A plot that disagrees
> with the finding printed next to it is worse than no plot: it is a second,
> more persuasive claim with nothing checking it.

Two checks score a *subsample* for speed, so a plot that re-sampled would
illustrate different rows than the verdict came from. The structural answers,
which you should copy rather than reinvent:

- **`stable_sample` is content-addressed.** It selects rows by their contents,
  not their position, so sorting a CSV cannot change which rows are scored.
- **One implementation, not two.** `AdversarialRobustnessCheck._measure(context,
  epsilon)` is called by both `run()` and `plot()`, so the epsilon sweep passes
  through its own reported point by construction.
  `ProxyCorrelationCheck._grid()` does the same for the heatmap.
- **Test by reading the value back off the `Axes`** and asserting it against
  `metadata`. `test_plots.py` checks bar heights against `group_tpr`, ray
  slopes against `group_loss_ratio`, and rebuilds the ECE from the plotted
  points and their marker areas.

On style, two rules exist for accessibility rather than taste. **Semantic
colours (pass / review / blocked) are never reused for a group** — a green bar
that happens to be group A, beside a green verdict pill, is a misread waiting
to happen. And **colour is never the only encoding**: pair it with marker
shapes or hatching, because these reports get printed in greyscale. Use the
helpers in `bdp_model_gate/plots/style.py` and you get both for free.

Full guidance: [`web/docs/reference/plots.md`](web/docs/reference/plots.md).

## Optional dependencies and the core install

The base install has no scikit-learn, fairlearn or shap, and no matplotlib or
seaborn. Everything must still work — degraded, and saying so.

CI has a dedicated **core-install job** that installs without the extras,
removes the plotting stack, runs the suite, and then asserts the HTML report
still renders every finding. That job is the only thing keeping the degradation
paths honest, so if you touch one, check it there.

In tests, guard an optional import at the top of the module rather than
importing it and marking `skipif` — on a core install the import itself is a
*collection* error that fails the run instead of skipping it:

```python
matplotlib = pytest.importorskip("matplotlib", reason="the [plots] extra is not installed")
```

## The Python 3.9 floor

Worth knowing because it is invisible locally and it once took down the whole
3.9 job.

`X | None` and `list[str]` in an annotation are **evaluated at runtime** on
Python 3.9. Without `from __future__ import annotations` at the top of the
module, that is an import-time `TypeError` on 3.9 while passing silently on
3.10+. In a dataclass field annotation it fires on import and takes every test
module down at collection.

`[tool.mypy] python_version` is pinned to 3.12 (numpy's bundled stubs need it),
so **the type checker cannot catch this**. Three things do: ruff's `FA` rules,
an AST test in `tests/test_package.py`, and the 3.9 job in the matrix. Note
that `target-version = "py39"` on its own does *not* imply those checks.

Practical rule: **every module in `bdp_model_gate/` starts with
`from __future__ import annotations`.**

## Notebooks

The six notebooks in `examples/` are committed **with outputs**, so those
outputs are a claim about how the library behaves.

```bash
./examples/run_all.sh          # all of them
./examples/run_all.sh 03 06    # only those prefixes
```

Notebooks 04 and 05 need `torch` and `xgboost` respectively. On macOS the two
link different OpenMP runtimes and **segfault in the same process** — a hard
crash, not an exception — which is why they are separate notebooks and not a
stylistic choice.

One rule, learned the hard way and violated repeatedly before it was:

> **When the prose and the output disagree, change the code or the data — not
> the prose.**

A notebook that says "the loss-ratio check is clean" above a table showing
1.42 against 1.12 is worse than no notebook, because the narrative is what
people read. If a cell does not demonstrate what you wanted, make it
demonstrate that, or describe what it actually shows. Re-read every output
after re-executing.

## Documentation and the website

Docstrings are the API reference — `web/docs/reference/api.md` is generated by
mkdocstrings, so it cannot drift from the code. Prose pages live under
`web/docs/`, and the landing page is hand-written HTML in `web/landing/`.

```bash
pip install -r web/requirements.txt
pip install -e .                  # mkdocstrings imports the package
./web/build.sh                    # output in web/_site (gitignored)
./web/build.sh serve              # live reload while writing
```

The build runs `mkdocs build --strict`, so a broken internal link fails it.
Notebooks are **copied** from `examples/` at build time rather than duplicated
in the repo, so there is one source of truth; the copies under
`web/docs/examples/` are gitignored.

### The site ships with the release

The website is part of a release, not a follow-up. It is the first thing an
evaluating bank or insurer sees, and a page advertising a version from two
releases ago undermines a governance tool more than a missing feature would.

Before a release is tagged:

| Update | Where |
|---|---|
| Version chip and colophon | `web/landing/index.html` |
| New or changed checks | `web/docs/reference/checks.md`, the checks grid in `web/landing/index.html`, and the table in `web/docs/index.md` |
| New config fields | `web/docs/reference/configuration.md` |
| A new **capability** | the landing page — "produces a report a reviewer can sign" is a claim, not a detail |
| Counts written in prose | notebooks, checks, plots — these go stale in silence |
| Notebook outputs | `examples/run_all.sh`, then `./web/build.sh` to re-copy them |

Two of those are enforced, because relying on memory is what let the landing
page advertise 0.4.1 through two releases while its "sixteen checks" grid
listed thirteen. `tests/test_package.py` now fails if the landing page's
version drifts from `pyproject.toml`, if any release-shaped version string is
left on the page, or if a check in the default suite is missing from
`reference/checks.md`.

The rest is judgement, and `--strict` cannot help: it catches a broken link,
never a sentence that is no longer true. Read the pages you changed.

## Versioning

The project is pre-1.0, and `pyproject.toml` is the single source of truth for
the number. Deciding which digit moves is a judgement, so here is the rule
this project actually follows, and the question that matters more than the
rule.

### The question that decides it

> **Can this change flip an existing verdict on unchanged inputs?**

That is the only question a downstream user genuinely needs answered, because
this library's output is a decision about whether a model ships. A pipeline
pinned to `~=0.5.2` and upgraded overnight must not start blocking a model it
passed yesterday without the release saying so.

Adding a **blocking** check does exactly that. So does changing a default
threshold, changing what a metric is compared against, or fixing a check that
was silently returning the wrong number. Every one of those has happened here.

Whatever digit moves, **a release that can flip a verdict says so in the first
two paragraphs of its changelog entry**, naming which checks are affected and
what a reader should do about it. That is not negotiable and it is not
enforceable by tooling; it is the reason the changelog is written in prose.

### Which digit

| Digit | Move it when | Examples from this project |
|---|---|---|
| **major** (`1.0.0`) | the public API is declared stable, or — after 1.0 — a documented name, signature or return shape changes incompatibly | reserved: 1.0.0 is the subclassable `ModelAdapter` and the API freeze |
| **minor** (`0.x.0`) | the suite can now grade something it structurally could not before — a new **task**, or a gap whose absence made the existing report *misleading* rather than merely incomplete | `0.2.0` a configurable metric (the score was hard-wired), `0.4.0` multiclass and ordinal, `0.5.0` separation and sufficiency — reporting demographic parity alone chose one of three incompatible notions silently |
| **patch** (`0.x.y`) | anything else: new checks within an existing shape, new inputs, new plots, fixes, hardening, docs | `0.5.1` plots and the HTML report, `0.5.2` the validation category, `0.5.3` exposure and the actuarial measures |

Two things that surprise people about that table.

**A substantial feature release can be a patch bump.** `0.5.2` added a whole
new category of five checks, four of them blocking, and `0.5.3` added four
more. Both were patch releases. That is deliberate: pre-1.0, the second digit is reserved for *the
shape of what the gate can grade* changing, and adding checks inside an
existing shape is not that, however much work it was.

**A patch release can be the most disruptive in the file.** `0.2.1` was three
bug fixes, and one of them — binarising `y_pred` before comparing selection
rates — turned a parity difference that had been *always exactly* `0.000` into
a real number. Every user following the documented quickstart went from a
clean `PASS` to a live `DISPARITY_RISK`, because the old behaviour was wrong.
Read the changelog, not the digit.

### The four places the version lives

`pyproject.toml`, `bdp_model_gate/__init__.py`, the `## [x.y.z]` heading in
`CHANGELOG.md`, and the website. `tests/test_package.py` fails if any of them
drifts, plus one more thing worth knowing: it also fails if **any**
release-shaped version string is left anywhere on the landing page, because
the page once advertised `0.4.1` through two releases while a stale figure sat
further down it.

```bash
pytest tests/test_package.py -q      # the fastest way to check a bump is complete
```

## CHANGELOG and ROADMAP

These two files have a strict division, and it is enforced by convention rather
than tooling, so please respect it:

- **`CHANGELOG.md` tracks everything that has happened.**
- **`ROADMAP.md` tracks everything that should happen.** Shipped entries are
  *removed* from it, not marked done.
- The only release appearing in both is the one currently being built.

`tests/test_package.py` does assert that the version in `pyproject.toml` has a
matching `## [x.y.z]` heading in the changelog, so a version bump without an
entry fails CI.

Write changelog entries so they are useful a year later. "Fixed a bug in
`DisparateImpactCheck`" says nothing; say what was wrong, what it produced, and
why nothing caught it.

### Cutting a release entry

The mechanics, in order, because the two files have to move together:

1. **Bump `pyproject.toml` and `__init__.py`.**
2. **Insert the entry under `## [Unreleased]`**, which stays at the top and
   stays empty. New entries go *below* it, never in place of it.
3. **Open with a one-line theme**, then a paragraph on *why the release
   exists* — what was wrong or missing before it. Look at `0.5.2` and `0.5.3`
   for the length and register. Then `### Added` / `### Changed` / `### Fixed`.
4. **Say what can flip a verdict**, per the rule above.
5. **Add the compare link at the foot of the file**, and repoint
   `[Unreleased]` at the new tag. Easy to forget, and it silently breaks every
   link below it.
6. **Delete the release's section from `ROADMAP.md`** and update its
   `Current release: … Next: …` line. Shipped entries are removed, not marked
   done — a roadmap of ticked boxes is a changelog with worse formatting.
7. **Work through the site checklist** in
   [The site ships with the release](#the-site-ships-with-the-release).

Two conventions that are easy to get wrong:

- **A roadmap entry is a plan, and plans change.** If the release ends up
  differing from what the roadmap said, the *changelog* records what shipped
  and the roadmap entry is deleted regardless. Do not retro-edit the roadmap
  to match; that loses the fact that a decision was revised.
- **Prose figures go stale in silence.** Check counts, plot counts, notebook
  counts, mutation figures. `tests/test_package.py` catches the check-coverage
  and version cases and nothing else, so grep for the old number before you
  are done.

Mutation figures are the one exception to "keep the numbers current". They are
labelled by the release that measured them (*"at 0.5.2 the suite reports …"*),
which stays true forever, and the run takes 25 minutes. Re-measure when you
are hardening an area, not on every release, and re-label rather than
overwrite.

## Branches

One branch per unit of work, named for the kind of change it is:

| Prefix | For |
|---|---|
| `feat/` | a new capability — a check, an input, a task |
| `fix/` | a bug, especially a wrong-but-plausible number |
| `docs/` | prose, the website, the notebooks, `CONTRIBUTING.md` |
| `chore/` | tooling, CI, dependencies, release plumbing |

Four rules, and the last one matters most.

1. **Branch from an up-to-date `main`.** `git fetch origin && git rebase
   origin/main` before you start — worth double-checking, since a stale local
   `main` produces a branch missing the previous release entirely, and the
   conflict surfaces in `CHANGELOG.md` at the worst moment.
2. **A release branch is named in its `ROADMAP.md` entry.** `0.6.0` says
   `**Branch:** feat/uncertainty`, so that is the branch. Naming it up front is
   what keeps a release from being assembled out of four unrelated branches
   whose merge order matters.
3. **Do not reuse a merged branch name.** Every branch in this repo is kept
   after merge rather than deleted, so `feat/robustness` and `docs/roadmap-0.5-0.6`
   still exist and still point at their old tips. Reusing one produces a
   diff against history nobody can read.
4. **A branch changes one kind of thing.** The version bump, the changelog
   entry and the site update belong to the release branch that earns them; a
   `docs/` branch never bumps a version. The temptation is always to fold a
   small unrelated fix into the release branch — resist it, because the release
   PR is the one a reviewer reads most carefully and every unrelated line
   spends that attention.

The one case where mixing is right: a **stale figure the release itself
created** — a check count, a plot count — is part of that release, not a
separate docs change. If the number was wrong *before* the release, it is a
`docs/` branch.

## Pull requests

The `main` branch requires a PR — direct pushes are blocked by a ruleset.

1. Branch from an up-to-date `main`, named per [Branches](#branches) above.
2. Make the change, with tests.
3. Run the local loop above, and `./web/build.sh` if you touched docs.
4. Open a PR against `main`. CI must be green, including the docs build.

In the description, say what the change does and **what would have gone wrong
without it**. A PR that fixes a silent-wrongness bug should say how the bug
stayed invisible — that is the part reviewers most need.

Commit messages: a short imperative subject line, then a body explaining *why*
in prose. Look at `git log` for the house style. Small, focused commits are
easier to review than one large one, but do not split a change from its tests.

## Reporting bugs

Open an issue with:

- what you expected and what happened
- a minimal reproduction — the smallest `StructuredGateContext` that shows it
- versions: `bdp-model-gate`, Python, and which extras you have installed

**A check that returned a wrong-but-plausible number is the highest-priority
kind of bug here**, more than a crash. If you think a verdict is wrong but you
are not sure, that is still worth an issue.

For anything with a security dimension, prefer a
[private security advisory](https://github.com/vanjy-eng/model-gate/security/advisories/new)
over a public issue.

## Scope

This library is a **pre-deployment gate**: it runs after training and before
promotion, and returns one status a pipeline can branch on. Things that are
deliberately out of scope, so nobody builds them by surprise:

- **Runtime monitoring and drift detection.** Different problem, different
  cadence, different data.
- **Mitigation.** The gate measures and reports; it does not reweight, calibrate
  or resample your model. Deciding what to do about a finding is the
  practitioner's job.
- **Being a plotting library.** The fourteen plots explain fourteen findings.
  Anything more general belongs in your own code.

`ci_examples/` is a separate thing again: those are pre-deployment gates for
models built *by consumers* of this library, not CI for the library itself.

If you are unsure whether something fits, open an issue before writing it — and
check [`ROADMAP.md`](ROADMAP.md) first, which records decisions already taken
so they do not get re-argued.

## Licence

Contributions are accepted under the [MIT Licence](LICENSE), the same terms as
the project.
