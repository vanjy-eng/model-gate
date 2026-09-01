"""A gate report as a page a reviewer can read and sign.

`NEEDS_REVIEW` is a verdict that delegates to a human. Until now that human
received a JSON blob: correct, archival, and close to unreadable at the moment
a decision has to be made. This renders the same report as one self-contained
HTML file — no network, no JavaScript, nothing to install to open it — with
each check's plot inlined beside the number it explains.

Three properties are deliberate:

* **Self-contained.** No `<script>`, no external stylesheet, no remote font.
  A governance record is emailed, filed and reopened years later, and every
  external reference is a way for it to stop rendering.
* **Plots inlined as SVG**, not `<img src="data:...">`. Inline SVG inherits the
  page's CSS, which is what makes one render read correctly in light and dark.
  It also stays sharp when printed.
* **Degrades rather than fails.** Without the `[plots]` extra the page renders
  text-only. A plot that raises is reported in place as a note, because a
  broken chart must never cost a reviewer the findings around it.
"""

from __future__ import annotations

import html
import io
from datetime import datetime, timezone
from typing import Any

from ._logging import get_logger
from .core.base import BaseCheck

logger = get_logger("reporting")

#: Order categories are presented in. Validation first, because a finding
#: there says the evidence behind everything below it is unsound — a model
#: graded on its own training data reports a superb score and a clean
#: calibration curve. Then what stops a deploy outright, then what a human
#: has to weigh.
CATEGORY_ORDER = ("validation", "performance", "compliance", "security", "fairness")

_VERDICT_BLURB = {
    "PASS": "Every check passed. No action required before promotion.",
    "NEEDS_REVIEW": (
        "No blocking check failed, but findings below need a human judgement "
        "before this model is promoted."
    ),
    "BLOCKED": "A blocking check failed. This model must not be promoted as it stands.",
}

_STYLE = """
:root {
  --bg: #f7f9f8; --surface: #ffffff; --ink: #10221f; --muted: #55635f;
  --rule: #dde3e0; --accent: #0e5c55;
  --pass: #2e6b43; --review: #8a5a0b; --blocked: #9b2c2c;
  --plot-ink: var(--ink); --plot-muted: var(--muted);
  --plot-rule: var(--rule); --plot-surface: var(--surface);
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0c1513; --surface: #131f1d; --ink: #e8efec; --muted: #93a29d;
    --rule: #2a3a37; --accent: #5fb6ab;
    --pass: #6fbf87; --review: #d9a442; --blocked: #e2726e;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2.5rem 1.25rem 4rem; background: var(--bg); color: var(--ink);
  font-family: "IBM Plex Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 15px; line-height: 1.55;
}
main { max-width: 62rem; margin: 0 auto; display: flex; flex-direction: column; gap: 1.75rem; }
h1 { font-size: 1.6rem; margin: 0 0 .35rem; letter-spacing: -.01em; text-wrap: balance; }
h2 { font-size: 1.1rem; margin: 0; letter-spacing: -.005em; }
h3 { font-size: .95rem; margin: 0; font-weight: 600; }
p { margin: 0; }
.sub { color: var(--muted); font-size: .875rem; }
.card {
  background: var(--surface); border: 1px solid var(--rule); border-radius: 10px;
  padding: 1.25rem 1.4rem; display: flex; flex-direction: column; gap: 1rem;
}
.verdict { border-left: 5px solid var(--rule); }
.verdict.PASS { border-left-color: var(--pass); }
.verdict.NEEDS_REVIEW { border-left-color: var(--review); }
.verdict.BLOCKED { border-left-color: var(--blocked); }
.verdict-name { font-size: 1.05rem; font-weight: 700; letter-spacing: .06em; }
.verdict.PASS .verdict-name { color: var(--pass); }
.verdict.NEEDS_REVIEW .verdict-name { color: var(--review); }
.verdict.BLOCKED .verdict-name { color: var(--blocked); }
.facts { display: flex; flex-wrap: wrap; gap: 1.5rem 2.5rem; }
.fact { display: flex; flex-direction: column; gap: .15rem; }
.fact dt {
  font-size: .7rem; text-transform: uppercase; letter-spacing: .09em; color: var(--muted);
}
.fact dd { margin: 0; font-size: 1.05rem; font-variant-numeric: tabular-nums; }
.pill {
  display: inline-block; font-size: .68rem; font-weight: 700; letter-spacing: .07em;
  padding: .18rem .5rem; border-radius: 999px; border: 1px solid currentColor;
  white-space: nowrap;
}
.pill.ok { color: var(--pass); }
.pill.risk { color: var(--blocked); }
.pill.review { color: var(--review); }
.pill.na { color: var(--muted); }
.check { display: flex; flex-direction: column; gap: .7rem; }
.check + .check { border-top: 1px solid var(--rule); padding-top: 1.1rem; }
.finding { display: grid; grid-template-columns: 8.5rem 1fr; gap: .35rem .9rem; align-items: start; }
.finding .detail { font-size: .9rem; }
.blocking-note { color: var(--muted); font-size: .78rem; }
details { font-size: .82rem; color: var(--muted); }
summary { cursor: pointer; }
pre {
  overflow-x: auto; background: var(--bg); border: 1px solid var(--rule); border-radius: 6px;
  padding: .7rem .85rem; font-size: .78rem; line-height: 1.5;
  font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
}
figure { margin: 0; overflow-x: auto; }
figure svg { max-width: 100%; height: auto; }
.note { color: var(--muted); font-size: .82rem; font-style: italic; }
footer { color: var(--muted); font-size: .78rem; text-align: center; }
@media print {
  body { background: #fff; padding: 0; }
  .card { break-inside: avoid; border-color: #ccc; }
}
"""


def _pill_class(flag: str, blocking: bool) -> str:
    if flag in ("OK", "PASS"):
        return "ok"
    if flag == "NOT_APPLICABLE":
        return "na"
    return "risk" if blocking else "review"


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _render_metadata(metadata: dict) -> str:
    if not metadata:
        return ""
    rows = "\n".join(f"{_esc(k)}: {_esc(v)}" for k, v in metadata.items())
    return f"<details><summary>evidence</summary><pre>{rows}</pre></details>"


def _figure_svg(check: BaseCheck, context: Any, results: list) -> str | None:
    """Runs one check's `plot()` and returns it as an inlinable SVG.

    Returns None when the check draws nothing — which is the common case, and
    not an error: most checks produce a number whose shape adds nothing.
    """
    from .plots import require_plotting
    from .plots.style import themeable_svg

    plt, _ = require_plotting()
    # A deterministic salt per check keeps matplotlib's internal element ids
    # stable across renders *and* distinct between figures. Without it, several
    # inlined SVGs in one document share clip-path ids and the later ones are
    # clipped by the earlier ones' geometry.
    previous_salt = plt.rcParams.get("svg.hashsalt")
    plt.rcParams["svg.hashsalt"] = check.name
    ax = None
    try:
        ax = check.plot(context, results)
        if ax is None:
            return None
        figure = ax.get_figure()
        buffer = io.StringIO()
        figure.savefig(buffer, format="svg", transparent=True)
        raw = buffer.getvalue()
        # Drop the XML declaration and DOCTYPE: legal in a standalone file,
        # invalid partway through an HTML document.
        return themeable_svg(raw[raw.index("<svg") :])
    finally:
        plt.rcParams["svg.hashsalt"] = previous_salt
        if ax is not None:
            plt.close(ax.get_figure())


def _draws(check: BaseCheck) -> bool:
    """Whether this check overrides `plot`. Discovery is by override alone —
    there is no flag for a plugin author to remember to set."""
    return type(check).plot is not BaseCheck.plot


def render_html(
    report: Any,
    checks: Any = None,
    context: Any = None,
    title: str = "Model gate report",
    include_plots: bool = True,
    generated_at: str | None = None,
) -> str:
    """Renders a `GateReport` as one self-contained HTML document.

    `checks` and `context` are what plotting needs — a plot recomputes from
    the data rather than reading presentation arrays out of the archived
    JSON. `ModelGate.run` attaches both to the report it returns, so
    `report.to_html()` normally supplies them for you; pass them explicitly
    when rendering a report reconstructed from elsewhere.

    Without them, or without the `[plots]` extra, the page renders text-only.
    """
    from .plots import plotting_available

    stamp = generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    verdict = report.gate_status

    parts: list[str] = [
        "<main>",
        f'<header class="card verdict {verdict}">',
        f"<div><h1>{_esc(title)}</h1>"
        f'<p class="sub">Generated {_esc(stamp)} · bdp-model-gate</p></div>',
        f'<p class="verdict-name">{_esc(verdict)}</p>',
        f"<p>{_esc(_VERDICT_BLURB.get(verdict, ''))}</p>",
        '<dl class="facts">',
    ]

    facts: list[tuple[str, str]] = [("Findings", str(len(report.flags)))]
    if report.task:
        facts.append(("Task", report.task))
    if report.model_metric is not None and report.model_score is not None:
        facts.append((report.model_metric, f"{report.model_score:.4f}"))
    facts.append(("Checks run", str(len({r.check_name for r in report.results}))))
    facts.append(("Duration", f"{report.total_duration_ms:.0f} ms"))
    for name, value in facts:
        parts.append(f'<div class="fact"><dt>{_esc(name)}</dt><dd>{_esc(value)}</dd></div>')
    parts.append("</dl></header>")

    by_name = {getattr(c, "name", type(c).__name__): c for c in (checks or [])}
    draw = include_plots and bool(by_name) and context is not None and plotting_available()
    if include_plots and not draw:
        logger.debug(
            "rendering text-only: checks=%s context=%s plots_installed=%s",
            bool(by_name),
            context is not None,
            plotting_available(),
        )

    categories = list(CATEGORY_ORDER) + sorted(
        {r.category for r in report.results} - set(CATEGORY_ORDER)
    )
    for category in categories:
        rows = report.by_category(category)
        if not rows:
            continue
        flagged = sum(1 for r in rows if not r.is_ok)
        parts.append(
            f'<section class="card"><div><h2>{_esc(category.title())}</h2>'
            f'<p class="sub">{flagged} finding(s) across {len(rows)} result(s)</p></div>'
        )

        # Grouped by check so a plot sits with the numbers it illustrates,
        # preserving the order the checks ran in.
        seen: list[str] = []
        for r in rows:
            if r.check_name not in seen:
                seen.append(r.check_name)

        for check_name in seen:
            own = [r for r in rows if r.check_name == check_name]
            parts.append(f'<div class="check"><h3>{_esc(check_name)}</h3>')
            for r in own:
                pill = _pill_class(r.flag, r.blocking)
                blocking_note = (
                    "" if r.is_ok else ("blocks promotion" if r.blocking else "needs review")
                )
                parts.append(
                    f'<div class="finding">'
                    f'<span class="pill {pill}">{_esc(r.flag)}</span>'
                    f'<div><p class="detail">{_esc(r.detail)}</p>'
                    + (f'<p class="blocking-note">{blocking_note}</p>' if blocking_note else "")
                    + _render_metadata(r.metadata)
                    + "</div></div>"
                )
            check = by_name.get(check_name)
            if draw and check is not None and _draws(check):
                try:
                    svg = _figure_svg(check, context, own)
                except Exception as exc:
                    # A chart is an aid. Losing the findings around it because
                    # a renderer raised would be a worse outcome than no chart.
                    logger.warning("plot failed for check=%s: %r", check_name, exc)
                    parts.append(
                        f'<p class="note">chart unavailable — {_esc(type(exc).__name__)}: '
                        f"{_esc(exc)}</p>"
                    )
                else:
                    if svg:
                        parts.append(f"<figure>{svg}</figure>")
            parts.append("</div>")
        parts.append("</section>")

    parts.append(
        "<footer>Produced by bdp-model-gate. Findings are evidence for a human decision, "
        "not the decision itself.</footer></main>"
    )

    body = "\n".join(parts)
    return (
        "<!doctype html>\n"
        f'<html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_esc(title)} — {_esc(verdict)}</title>"
        f"<style>{_STYLE}</style></head><body>\n{body}\n</body></html>\n"
    )


__all__ = ["CATEGORY_ORDER", "render_html"]
