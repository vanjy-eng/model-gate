"""The HTML report is a governance record, so it has to behave like one.

Three properties are load-bearing and each has a test here: it renders with no
external reference of any kind, it escapes everything it is given, and it
degrades to text rather than failing when plotting is unavailable or a chart
raises. A report that silently loses findings because a renderer threw would
be worse than the JSON it replaces.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bdp_model_gate import ModelGate, StructuredGateContext
from bdp_model_gate.core.base import BaseCheck, CheckResult
from bdp_model_gate.core.report import GateReport
from bdp_model_gate.plots import plotting_available
from bdp_model_gate.reporting import render_html


@pytest.fixture
def context():
    rng = np.random.default_rng(4)
    n = 400
    X = pd.DataFrame(
        {
            "income": rng.lognormal(11.5, 0.4, n),
            "credit_score": rng.normal(650, 55, n),
        }
    )
    logit = -4 + 3e-6 * X["income"].to_numpy() + 0.005 * X["credit_score"].to_numpy()
    probability = 1 / (1 + np.exp(-logit))
    return StructuredGateContext(
        model=None,
        X=X,
        y_true=(rng.random(n) < probability).astype(int),
        y_pred=probability,
        protected_df=pd.DataFrame({"region": rng.choice(["Lagos", "Kano"], n)}),
        predict_fn=lambda frame: (probability[: len(frame)] > 0.5).astype(int),
        model_card={"use_case": "retail credit"},
    )


@pytest.fixture
def report(context):
    return ModelGate().run(context)


def test_the_gate_attaches_what_the_renderer_needs(report, context):
    """`report.to_html()` has to work without the caller re-supplying the
    checks and the data — otherwise nobody uses it."""
    assert report._context is context
    assert report._checks


def test_rendering_data_stays_out_of_the_archival_record(report):
    """The JSON is the record that gets filed. Neither the check objects nor
    the validation set belong in it, and neither may affect equality."""
    payload = report.to_dict()
    assert "_checks" not in payload and "_context" not in payload
    assert "_checks" not in repr(report)

    twin = GateReport(results=report.results, task=report.task)
    twin._checks = ["something", "else"]
    assert twin == GateReport(results=report.results, task=report.task)


def test_the_page_is_self_contained(report):
    """No script, no stylesheet, no font, no image fetched from anywhere. A
    record that stops rendering when a CDN moves is not a record."""
    page = report.to_html()
    assert page.startswith("<!doctype html>")
    for forbidden in ("<script", "<link", "@import", 'src="http', "href='http"):
        assert forbidden not in page
    assert 'href="http' not in page
    assert page.rstrip().endswith("</html>")


def test_the_verdict_is_stated_plainly(report):
    page = report.to_html()
    assert report.gate_status in page
    # And in the tab title, which is what a reviewer sees with ten open.
    assert f"<title>Model gate report — {report.gate_status}</title>" in page


def test_every_finding_survives_into_the_page(report):
    page = report.to_html()
    for result in report.results:
        assert result.check_name in page


def test_hostile_content_is_escaped():
    """Detail strings can carry a feature name, and a feature name can carry
    anything at all."""
    payload = "<script>alert('x')</script>"
    report = GateReport(
        results=[
            CheckResult(
                check_name=payload,
                category="fairness",
                flag="PROXY_RISK",
                detail=f"{payload} & more",
                blocking=False,
                metadata={payload: payload},
            )
        ],
        task="binary",
    )
    page = report.to_html()
    assert "<script>" not in page
    assert "&lt;script&gt;" in page
    assert "&amp; more" in page


def test_writing_to_a_path_returns_the_same_page(report, tmp_path):
    destination = tmp_path / "gate.html"
    returned = report.to_html(path=str(destination))
    assert destination.read_text(encoding="utf-8") == returned


def test_a_report_without_its_data_still_renders(report):
    """A report reconstructed from JSON has no context. It must still produce
    the findings, just without charts."""
    page = render_html(report, checks=None, context=None)
    assert report.gate_status in page
    assert "<svg" not in page
    for result in report.results:
        assert result.check_name in page


def test_plots_can_be_turned_off(report):
    assert "<svg" not in report.to_html(include_plots=False)


@pytest.mark.skipif(not plotting_available(), reason="the [plots] extra is not installed")
def test_a_broken_plot_costs_the_chart_and_nothing_else(context):
    """The failure mode that matters: a renderer raising must not take the
    findings around it down with the chart.

    Only meaningful where plotting is installed: without the extra the
    renderer never calls `plot()` at all, so there is no exception to survive
    — that path is covered by `test_a_report_without_its_data_still_renders`.
    """

    class Exploding(BaseCheck):
        name = "exploding_check"
        category = "security"
        blocking = False

        def run(self, ctx):
            return [CheckResult(self.name, self.category, "OK", "the finding survives", False)]

        def plot(self, ctx, results=None, ax=None):
            raise RuntimeError("no chart today")

    report = ModelGate(checks=[Exploding()]).run(context)
    page = report.to_html()
    assert "the finding survives" in page
    assert "chart unavailable" in page
    assert "no chart today" in page


def test_categories_lead_with_what_blocks(report):
    """Performance and compliance stop a deploy outright; fairness asks for a
    judgement. A reviewer should meet them in that order."""
    page = report.to_html()
    order = [page.find(f"<h2>{c.title()}</h2>") for c in ("Performance", "Compliance", "Fairness")]
    present = [i for i in order if i != -1]
    assert present == sorted(present)


@pytest.mark.skipif(not plotting_available(), reason="the [plots] extra is not installed")
def test_charts_are_inlined_as_svg_not_linked(report):
    page = report.to_html()
    assert page.count("<svg") >= 1
    assert "data:image" not in page
    # Inlined so the page's CSS reaches them — which is what the theme
    # variables in the rewritten SVG are for.
    assert "var(--plot-" in page
    assert ":root" in page and "--plot-ink" in page


@pytest.mark.skipif(not plotting_available(), reason="the [plots] extra is not installed")
def test_each_chart_gets_its_own_element_ids(report):
    """matplotlib reuses internal ids across figures. Two SVGs in one document
    sharing a clip-path id means the second is clipped by the first's
    geometry — a corruption that only appears once there is more than one
    chart, which is every real report."""
    import re

    page = report.to_html()
    clip_ids = re.findall(r'<clipPath id="([^"]+)"', page)
    assert len(clip_ids) > 1
    assert len(clip_ids) == len(set(clip_ids))
