# Reports

A gate run produces three renderings of the same findings, for three different
readers.

| | Reader | Call |
|---|---|---|
| Summary | whoever is watching the pipeline | `report.summary()` |
| JSON | the system that files it | `report.to_json(path)` |
| HTML | the person who has to sign it | `report.to_html(path)` |

`BLOCKED` and `PASS` need no page — the pipeline acts on the exit code.
<span class="verdict-review">NEEDS_REVIEW</span> is the verdict that delegates
to a human, and that human should not be handed a JSON blob.

## One file, nothing fetched

```python
from bdp_model_gate import ModelGate

report = ModelGate().run(context)
report.to_html("gate-report.html")
```

That is the whole API. `ModelGate.run` attaches the checks and the context to
the report, so the charts are drawn without you re-supplying anything.

The page has no `<script>`, no stylesheet, no font and no image fetched from
anywhere. A governance record gets emailed, filed, and reopened years later,
and every external reference is a way for it to stop rendering. It opens from
a file:// URL on a laptop with no network.

Charts are inlined as **SVG**, not `<img src="data:...">`. Inline SVG
participates in the page's CSS, which is what lets one render read correctly
in light and dark, and it stays sharp when printed.

## Options

```python
report.to_html(
    path="gate-report.html",
    title="Retail credit scorecard v4",  # shown in the tab and the header
    include_plots=True,
)
```

Pass `checks=` and `context=` explicitly when rendering a report you rebuilt
from somewhere else:

```python
render_html(report, checks=gate.checks, context=context)
```

## It degrades; it does not fail

Three ways the page can lose its charts, and none of them loses a finding:

- **No `[plots]` extra installed** — text-only.
- **No checks or context** (a report reconstructed from JSON) — text-only.
- **A `plot()` raised** — that one chart is replaced in place by a note naming
  the exception. The findings around it are untouched.

That last one is deliberate. A chart is an aid; a renderer that throws must
never cost a reviewer the results it was illustrating.

## What is in the page

- The verdict, in plain words: *"A blocking check failed. This model must not
  be promoted as it stands."*
- The headline metric — whichever metric was configured, named, never assumed
  to be AUC.
- Categories in the order that matters: performance and compliance stop a
  deploy outright, security next, then fairness, which asks for a judgement.
- Every result, including `NOT_APPLICABLE` ones. What was *skipped and why* is
  part of the record — a report that silently omits them lets a reader assume
  coverage that never happened.
- Each result's `metadata` behind an **evidence** toggle, so the numbers
  behind a sentence are one click away and not in the way.
- The plot for each check that has one, beneath that check's findings.

## What is deliberately not in the page

The check objects and the validation set are attached to the `GateReport` for
rendering, and excluded from the constructor, the repr, equality and
`to_dict()`. A report is an archival record of *findings*. Neither your data
nor your model belongs in one.

```python
report.to_dict()  # findings only — safe to file, safe to ship
report.to_json("run.json")
```

## Printing and archiving

The page carries a print stylesheet: cards avoid breaking across pages and the
background drops out. `Ctrl/Cmd-P → Save as PDF` gives you the artefact to
attach to a change request.

## API

::: bdp_model_gate.reporting
    options:
      members: [render_html, CATEGORY_ORDER]
