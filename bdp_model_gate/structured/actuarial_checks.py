"""The four questions a pricing review asks that a regression suite does not.

An RMSE says a premium model is "wrong by 25,000 naira on average". No
pricing committee has ever been able to act on that sentence. What they ask
instead:

    ActualVsExpectedCheck    Did the book collect what it needed to — overall,
                             and in every band? The standard validation.
    RiskDiscriminationCheck  Does the rating structure *order* risk at all?
                             A separate question from whether its level is
                             right, and invisible to every error metric.
    MonotonicityCheck        Does premium move the way the regulator was told
                             it moves? More prior claims must not mean a
                             cheaper policy.
    DislocationCheck         Replacing an incumbent, how many policyholders
                             see a rise above 25%, and who are they?

The first two are `performance` findings: the model is not good enough. The
last two are `compliance` findings — a monotonicity violation is a broken
promise about the product's structure rather than a bad score, and a
dislocation profile is a conduct question that no threshold can settle for
you, which is why it is the one non-blocking check here.

All four read `context.exposure` when it is supplied, and none of them needs
scikit-learn. See `bdp_model_gate.actuarial` for the exposure convention and
the statistics themselves.
"""

from __future__ import annotations

import difflib

import numpy as np

from .._logging import get_logger
from ..actuarial import (
    DIRECTIONS,
    actual_over_expected,
    assign_bands,
    band_edges,
    exposure_array,
    lorenz_curve,
    lorenz_gini,
    monotonicity_breaks,
    partial_dependence,
    relative_change,
    weights_or_ones,
)
from ..config import ActuarialConfig
from ..core.base import BaseCheck, CheckResult
from ..exceptions import GateConfigurationError
from ..model import ModelAdapter
from ..task import BINARY, REGRESSION, resolve_task

logger = get_logger("actuarial_checks")


def _not_applicable(check: BaseCheck, reason: str) -> list[CheckResult]:
    return [CheckResult(check.name, check.category, "NOT_APPLICABLE", reason, check.blocking)]


def _exposure_note(context) -> str:
    return " [exposure-weighted]" if getattr(context, "exposure", None) is not None else ""


class ActualVsExpectedCheck(BaseCheck):
    """Actual over expected, for the whole book and band by band.

    Two findings, because they have different causes and different fixes.

    **The level.** `sum(actual) / sum(expected)` over everything. At 1.10 the
    book is under-priced by ten percent, which is one number a pricing
    committee can act on immediately, and which an RMSE cannot express
    because it is symmetric about zero — a model that over-charges half the
    book and under-charges the other half scores the same as one that is
    right everywhere.

    **The shape.** The same ratio within bands of the prediction, cut at
    equal *exposure*. An overall A/E of exactly 1.00 is routinely produced by
    a model that subsidises its worst risks out of its best — the level looks
    perfect and every individual price is wrong. This is the finding that
    reads as "under-priced in the top decile", which is a rating-structure
    problem rather than a rate-level one.

    A band holding fewer than `min_band_rows` rows is reported but not
    scored: three policies produce a ratio, and it means nothing.
    """

    name = "actual_vs_expected"
    category = "performance"
    blocking = True
    supported_tasks = (REGRESSION,)

    def __init__(self, config: ActuarialConfig | None = None):
        self.config = config or ActuarialConfig()

    def _bands(self, context) -> list[dict] | None:
        """Per-band A/E, or None when the prediction will not cut into bands.

        The single source for both `run()` and `plot()`: the bars a reviewer
        looks at are the numbers the verdict came from, not a re-derivation
        of them.
        """
        y_true = np.asarray(context.y_true, dtype=float)
        y_pred = np.asarray(context.y_pred, dtype=float)
        weights = weights_or_ones(exposure_array(context), len(y_pred))

        edges = band_edges(y_pred, self.config.n_bands, weights)
        if len(edges) < 3:
            return None
        band = assign_bands(y_pred, edges)

        bands = []
        for index in range(len(edges) - 1):
            cell = band == index
            n_rows = int(cell.sum())
            ratio = (
                actual_over_expected(y_true[cell], y_pred[cell], weights[cell])
                if n_rows
                else float("nan")
            )
            bands.append(
                {
                    "band": index + 1,
                    "label": f"{edges[index]:,.0f}–{edges[index + 1]:,.0f}",
                    "n_rows": n_rows,
                    "exposure": round(float(weights[cell].sum()), 4),
                    "ae": None if not np.isfinite(ratio) else round(float(ratio), 4),
                    "scored": n_rows >= self.config.min_band_rows and bool(np.isfinite(ratio)),
                }
            )
        return bands

    def run(self, context) -> list[CheckResult]:
        if context.y_true is None or context.y_pred is None:
            return _not_applicable(
                self,
                "no y_true/y_pred — actual-versus-expected compares realised outcomes "
                "against the prediction and needs both",
            )

        y_true = np.asarray(context.y_true, dtype=float)
        y_pred = np.asarray(context.y_pred, dtype=float)
        weights = weights_or_ones(exposure_array(context), len(y_pred))
        weighted = _exposure_note(context)

        overall = actual_over_expected(y_true, y_pred, weights)
        if not np.isfinite(overall):
            return _not_applicable(
                self,
                "the exposure-weighted total of y_pred is zero, so there is no expected "
                "amount for the actuals to be a ratio of",
            )

        deviation = abs(overall - 1.0)
        direction = "under-priced" if overall > 1 else "over-priced"
        results = [
            CheckResult(
                self.name,
                self.category,
                "AE_LEVEL_RISK" if deviation > self.config.max_overall_ae_deviation else "OK",
                detail=(
                    f"book A/E = {overall:.3f} ({deviation:.1%} from break-even, "
                    f"{direction}; tolerance {self.config.max_overall_ae_deviation:.1%})"
                    f"{weighted}"
                ),
                blocking=self.blocking,
                metadata={
                    "measure": "level",
                    "ae": round(float(overall), 4),
                    "deviation": round(float(deviation), 4),
                    "threshold": self.config.max_overall_ae_deviation,
                    "exposure_weighted": bool(weighted),
                },
            )
        ]

        bands = self._bands(context)
        if bands is None:
            logger.debug(
                "%s: y_pred does not cut into two distinct bands — reporting the level only",
                self.name,
            )
            return results

        scored = [b for b in bands if b["scored"]]
        skipped = [b for b in bands if not b["scored"]]
        if not scored:
            results.append(
                CheckResult(
                    self.name,
                    self.category,
                    "NOT_APPLICABLE",
                    f"no prediction band held at least min_band_rows="
                    f"{self.config.min_band_rows} rows, so the A/E curve would be noise",
                    self.blocking,
                    metadata={"measure": "bands", "bands": bands},
                )
            )
            return results

        worst = max(scored, key=lambda b: abs(b["ae"] - 1.0))
        worst_deviation = abs(worst["ae"] - 1.0)
        note = (
            f" [{len(skipped)} band(s) not scored, below min_band_rows={self.config.min_band_rows}]"
            if skipped
            else ""
        )
        results.append(
            CheckResult(
                self.name,
                self.category,
                "AE_BAND_RISK" if worst_deviation > self.config.max_band_ae_deviation else "OK",
                detail=(
                    f"worst band is {worst['band']} of {len(bands)} ({worst['label']}): "
                    f"A/E = {worst['ae']:.3f} over {worst['n_rows']} rows — "
                    f"{'under' if worst['ae'] > 1 else 'over'}-priced by "
                    f"{worst_deviation:.1%} (tolerance "
                    f"{self.config.max_band_ae_deviation:.1%}){weighted}{note}"
                ),
                blocking=self.blocking,
                metadata={
                    "measure": "bands",
                    "worst_band": worst["band"],
                    "worst_band_ae": worst["ae"],
                    "worst_deviation": round(float(worst_deviation), 4),
                    "threshold": self.config.max_band_ae_deviation,
                    "n_bands_scored": len(scored),
                    "exposure_weighted": bool(weighted),
                    "bands": bands,
                },
            )
        )
        return results

    def plot(self, context, results=None, ax=None):
        """A/E by prediction band, with the tolerance drawn on.

        The scalar names the worst band. It cannot say whether the rest of
        the book is flat around break-even with one bad decile — recalibrate
        that segment — or tilted end to end, which is a rating structure that
        does not hold. Those look nothing alike here and identical in a
        report that prints one number.

        Bar heights are read straight from the band table in `metadata`, so
        the chart is the finding rather than a second computation of it.
        """
        from ..plots import require_plotting
        from ..plots.style import RULE, caption, new_axes, verdict_colour

        require_plotting()
        if context.y_true is None or context.y_pred is None:
            return None
        results = self.run(context) if results is None else results
        finding = next(
            (
                r
                for r in results
                if r.metadata.get("measure") == "bands" and r.metadata.get("bands")
            ),
            None,
        )
        if finding is None:
            return None
        bands = [b for b in finding.metadata["bands"] if b["ae"] is not None]
        if not bands:
            return None

        tolerance = self.config.max_band_ae_deviation
        positions = np.arange(len(bands))
        heights = np.array([b["ae"] for b in bands], dtype=float)
        # Semantic colour, and hatching for the unscored bands, so the two
        # distinctions a reader needs survive a greyscale printout.
        colours = [
            verdict_colour("BLOCKED" if abs(b["ae"] - 1.0) > tolerance else "PASS") for b in bands
        ]
        hatches = ["" if b["scored"] else "///" for b in bands]

        ax = new_axes(ax, figsize=(6.8, 3.8))
        bars = ax.bar(positions, heights, color=colours, width=0.72, zorder=2)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
            bar.set_edgecolor("white")
        ax.axhline(1.0, color=RULE, linewidth=1.4, linestyle="--", zorder=3)
        for bound in (1.0 - tolerance, 1.0 + tolerance):
            ax.axhline(bound, color=RULE, linewidth=0.9, linestyle=":", zorder=3)

        ax.set_xticks(positions)
        ax.set_xticklabels([b["label"] for b in bands], fontsize=7.5, rotation=45, ha="right")
        ax.set_xlabel("predicted value, by band of equal exposure")
        ax.set_ylabel("actual ÷ expected")
        ax.set_title("Actual against expected, band by band")
        caption(
            ax,
            "the dashed line is break-even and the dotted lines are the tolerance.\n"
            "Hatched bars held too few rows to score. One bad band is a segment to "
            "recalibrate; a tilt across all of them is a rating structure that does not hold.",
        )
        return ax


class RiskDiscriminationCheck(BaseCheck):
    """Does the rating structure order risk — and by how much of what is there?

    Calibration and discrimination are independent. A model that charges
    every policy the book's average premium is perfectly calibrated overall
    and useless: it collects the right total and distributes it at random.
    Every error metric scores it respectably, because on a skewed book the
    mean is not a bad guess.

    The exposure-weighted Lorenz Gini answers the other question. Zero means
    the ordering carries nothing; **negative means it is inverted**, which is
    a finding rather than a poor score and is exactly what a sign error in a
    rating factor produces.

    Reported against the ceiling this book allows. No rating structure can
    predict which individual policy crashes, so the attainable maximum is
    well below 1.0 and varies by class of business — "0.28 of a possible
    0.52" is a judgement a reviewer can make, where "0.28" alone is not.
    """

    name = "risk_discrimination"
    category = "performance"
    blocking = True
    supported_tasks = (REGRESSION,)

    def __init__(self, config: ActuarialConfig | None = None):
        self.config = config or ActuarialConfig()

    def run(self, context) -> list[CheckResult]:
        if context.y_true is None or context.y_pred is None:
            return _not_applicable(
                self,
                "no y_true/y_pred — the Gini measures how realised losses concentrate "
                "in the dearest predictions and needs both",
            )

        exposure = exposure_array(context)
        weighted = _exposure_note(context)
        try:
            gini = lorenz_gini(context.y_true, context.y_pred, exposure)
            ceiling = lorenz_gini(context.y_true, context.y_true, exposure)
        except GateConfigurationError as exc:
            # A negative or all-zero realised outcome. The Lorenz curve is
            # genuinely undefined there, and saying so beats reporting a
            # number computed on shares that are not shares.
            return _not_applicable(self, f"the Lorenz Gini is not defined here — {exc}")

        share = gini / ceiling if ceiling > 0 else float("nan")
        has_share = bool(np.isfinite(share))
        share_note = f", {share:.0%} of the {ceiling:.3f} this book allows" if has_share else ""
        if gini <= self.config.min_gini:
            verdict = (
                "the ordering is inverted — the policies it prices highest carry the "
                "*lower* loss per unit of exposure, which is what a sign error in a "
                "rating factor looks like"
                if gini < 0
                else "the ordering carries no information: dear and cheap policies cost "
                "the same per unit of exposure"
            )
            flag, detail = (
                "DISCRIMINATION_RISK",
                f"Gini = {gini:.3f} (floor {self.config.min_gini:.3f}) — {verdict}{weighted}",
            )
        else:
            flag, detail = (
                "OK",
                f"Gini = {gini:.3f}{share_note} (floor {self.config.min_gini:.3f}){weighted}",
            )

        return [
            CheckResult(
                self.name,
                self.category,
                flag,
                detail=detail,
                blocking=self.blocking,
                metadata={
                    "gini": round(float(gini), 4),
                    "gini_ceiling": round(float(ceiling), 4),
                    "gini_share_of_ceiling": round(float(share), 4) if has_share else None,
                    "threshold": self.config.min_gini,
                    "exposure_weighted": bool(weighted),
                },
            )
        ]

    def plot(self, context, results=None, ax=None):
        """The Lorenz curve, against the diagonal and the attainable ceiling.

        A Gini of 0.30 earned by isolating one dreadful decile and a Gini of
        0.30 spread evenly across the book are different rating structures
        needing different work, and the index is identical for both. The
        curve separates them: where it pulls away from the diagonal is where
        the model is doing its discriminating.

        The ceiling curve — sorted by the realised outcome — is drawn behind
        it, because the honest question is not "is 0.30 good?" but "how much
        of what was available did the model capture?".
        """
        from ..plots import require_plotting
        from ..plots.style import ACCENT, MUTED, RULE, caption, new_axes

        require_plotting()
        if context.y_true is None or context.y_pred is None:
            return None
        exposure = exposure_array(context)
        try:
            model_x, model_y = lorenz_curve(context.y_true, context.y_pred, exposure)
            best_x, best_y = lorenz_curve(context.y_true, context.y_true, exposure)
        except GateConfigurationError:
            return None

        ax = new_axes(ax, figsize=(5.4, 5.0))
        ax.plot([0, 1], [0, 1], color=RULE, linewidth=1.4, linestyle="--", zorder=1)
        ax.plot(
            best_x,
            best_y,
            color=MUTED,
            linewidth=1.2,
            linestyle=":",
            label="ceiling — sorted by outcome",
            zorder=2,
        )
        ax.plot(model_x, model_y, color=ACCENT, linewidth=2.0, marker="", label="model", zorder=3)
        ax.fill_between(model_x, model_y, model_x, color=ACCENT, alpha=0.12, zorder=1)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("cumulative share of exposure, cheapest prediction first")
        ax.set_ylabel("cumulative share of realised outcome")
        ax.set_title("Lorenz curve — how losses concentrate in the dearest policies")
        ax.legend(loc="upper left")
        caption(
            ax,
            "the Gini is twice the shaded area. A curve that hugs the diagonal until "
            "the top\nend discriminates only among its worst risks; one that pulls away "
            "early separates the whole book.",
        )
        return ax


class MonotonicityCheck(BaseCheck):
    """Does the output move the way you told the regulator it moves?

    Filed rates carry structural claims — premium rises with prior claims,
    falls with a higher deductible, rises with sum insured. A gradient
    booster fitted on a thin cell will happily violate one, and nothing else
    in a validation report would notice: the model scores well, the book
    prices sensibly on average, and one segment of policyholders is charged
    *less* for being worse risks. That is a compliance exposure rather than
    a performance one, which is why this sits in `compliance` and blocks.

    Checked empirically, by partial dependence: each declared factor is swept
    across the quantiles of its own distribution while every other column
    keeps its real joint distribution, and the mean prediction is read off.
    No constraint on the model class, no assumption of linearity, and it
    works on a remote endpoint through `predict_fn`.

    Nothing is checked until you declare something. `monotonic_features` is
    empty by default — the constraint is a claim about your product, and a
    library cannot guess which factors it applies to or in which direction.

    A declared factor that could not be evaluated — misspelled, categorical,
    constant on the validation set — is reported as
    `MONOTONICITY_UNCHECKABLE` and **blocks**, rather than being skipped
    quietly. A typo in a rating-factor name would otherwise produce a green
    gate on an unverified regulatory constraint, which is precisely the
    confident-and-wrong outcome this library exists to prevent.
    """

    name = "monotonicity"
    category = "compliance"
    blocking = True
    supported_tasks = (BINARY, REGRESSION)

    def __init__(self, config: ActuarialConfig | None = None):
        self.config = config or ActuarialConfig()
        # Fail while the suite is being built, not six checks into a run.
        for feature, direction in self.config.monotonic_features.items():
            if direction not in DIRECTIONS:
                raise GateConfigurationError(
                    f"actuarial.monotonic_features[{feature!r}] = {direction!r} — must be "
                    f"one of {', '.join(DIRECTIONS)}"
                )

    def _predict(self, context):
        """The scoring function whose curve is drawn, or None with a reason.

        For a classifier the curve has to be over *probabilities*: a partial
        dependence of hard 0/1 labels is a staircase that can be monotone
        while the underlying score is not, and a rating factor's effect lives
        in the score.
        """
        adapter = ModelAdapter.from_context(context)
        if resolve_task(context) == BINARY:
            if not adapter.can_predict_proba:
                return None, (
                    "a monotonicity curve for a classifier needs probabilities, and this "
                    "model exposes neither .predict_proba() nor predict_proba_fn — hard "
                    "labels make a staircase that can look monotone while the score is not"
                )
            return adapter.predict_positive_proba, None
        return adapter.predict, None

    def _curve(self, context, feature: str) -> tuple[np.ndarray, np.ndarray] | None:
        """`(grid, partial dependence)` for one feature, or None.

        Shared by `run()` and `plot()` — `partial_dependence` samples through
        `stable_sample`, so both calls score identically the same rows and the
        drawn curve is the curve that was judged.
        """
        column = context.X[feature]
        if column.dtype.kind not in "if":
            return None
        values = column.to_numpy(dtype=float)
        quantiles = np.linspace(0.0, 1.0, self.config.monotonicity_grid_points)
        grid = np.unique(np.quantile(values, quantiles))
        if len(grid) < 2:
            return None
        predict, _ = self._predict(context)
        if predict is None:
            return None
        curve = partial_dependence(
            predict,
            context.X,
            feature,
            grid,
            max_rows=self.config.monotonicity_max_rows,
        )
        return grid, curve

    def run(self, context) -> list[CheckResult]:
        declared = dict(self.config.monotonic_features)
        if not declared:
            return _not_applicable(
                self,
                "no actuarial.monotonic_features declared — a monotonicity requirement "
                'is a claim about your product ({"prior_claims": "increasing"}), and no '
                "library can guess which factors carry one or in which direction",
            )

        predict, why_not = self._predict(context)
        if predict is None:
            return _not_applicable(self, why_not)

        results = []
        for feature, direction in declared.items():
            uncheckable = self._why_uncheckable(context, feature)
            if uncheckable is not None:
                results.append(
                    CheckResult(
                        self.name,
                        self.category,
                        "MONOTONICITY_UNCHECKABLE",
                        detail=(
                            f"{feature} was declared {direction} in premium, and the "
                            f"constraint could not be tested: {uncheckable}"
                        ),
                        blocking=self.blocking,
                        metadata={
                            "feature": feature,
                            "direction": direction,
                            "reason": uncheckable,
                        },
                    )
                )
                continue

            built = self._curve(context, feature)
            if built is None:  # pragma: no cover — _why_uncheckable covers every path
                results.append(
                    CheckResult(
                        self.name,
                        self.category,
                        "MONOTONICITY_UNCHECKABLE",
                        detail=(
                            f"{feature} was declared {direction} in premium, and no "
                            "partial-dependence curve could be built for it"
                        ),
                        blocking=self.blocking,
                        metadata={"feature": feature, "direction": direction},
                    )
                )
                continue
            grid, curve = built
            breaks = monotonicity_breaks(curve, direction, self.config.monotonicity_tolerance)
            base = {
                "feature": feature,
                "direction": direction,
                "tolerance": self.config.monotonicity_tolerance,
                "grid": [round(float(v), 6) for v in grid],
                "partial_dependence": [round(float(v), 6) for v in curve],
                "n_breaks": len(breaks),
                "rows_scored": min(self.config.monotonicity_max_rows, len(context.X)),
            }
            if not breaks:
                results.append(
                    CheckResult(
                        self.name,
                        self.category,
                        "OK",
                        detail=(
                            f"{feature}: partial dependence is {direction} across all "
                            f"{len(grid)} bands of the factor "
                            f"({curve[0]:,.2f} to {curve[-1]:,.2f})"
                        ),
                        blocking=self.blocking,
                        metadata=base,
                    )
                )
                continue

            index, drop = breaks[0]
            results.append(
                CheckResult(
                    self.name,
                    self.category,
                    "MONOTONICITY_RISK",
                    detail=(
                        f"{feature} was declared {direction} in premium, and is not: "
                        f"between {grid[index]:,.4g} and {grid[index + 1]:,.4g} the mean "
                        f"prediction moves {curve[index]:,.2f} → {curve[index + 1]:,.2f}, "
                        f"against the declared direction by {drop:.1%} of the curve's "
                        f"range. {len(breaks)} of {len(grid) - 1} steps break it"
                    ),
                    blocking=self.blocking,
                    metadata={
                        **base,
                        "worst_break_index": index,
                        "worst_break_at": round(float(grid[index]), 6),
                        "worst_break_size": round(float(drop), 4),
                        "breaks": [[int(i), round(float(d), 4)] for i, d in breaks],
                    },
                )
            )
        return results

    def _why_uncheckable(self, context, feature: str) -> str | None:
        """The reason this declared factor cannot be tested, or None.

        Kept separate from `run()` because each of these is a *finding*, not
        a skip: the constraint was asserted and has not been verified.
        """
        if feature not in context.X.columns:
            # Name the near miss. A declared constraint on a misspelled column
            # is the worst outcome available here — a green gate on a filed
            # rating rule nobody verified — so the message has to make the
            # cause obvious rather than merely reporting the absence.
            close = difflib.get_close_matches(feature, list(context.X.columns), n=1, cutoff=0.7)
            hint = f" (did you mean {close[0]!r}?)" if close else ""
            return f"it is not a column of X{hint}"
        column = context.X[feature]
        if column.dtype.kind not in "if":
            return (
                f"it is {column.dtype}, not numeric — a monotonicity direction needs an "
                "ordering on the factor's values, which a categorical column does not carry"
            )
        if column.nunique(dropna=True) < 2:
            return (
                "it takes a single value on the validation set, so sweeping it produces "
                "one point and the constraint is vacuous rather than satisfied"
            )
        return None

    def plot(self, context, results=None, ax=None):
        """The partial-dependence curve, with the offending steps marked.

        "Not monotone in prior_claims" is a verdict; it does not say whether
        the curve dips once in a thin cell — refit that segment — or sags
        across the middle of the book, which is a structural problem. The
        shape is the remedy.

        Drawn for the worst-breaking factor, and only when something broke:
        a compliant curve is a straight-ish line that the detail string
        already describes in full.
        """
        from ..plots import require_plotting, worst_result
        from ..plots.style import ACCENT, caption, new_axes, verdict_colour

        require_plotting()
        if not self.config.monotonic_features:
            return None
        results = self.run(context) if results is None else results
        finding = worst_result(results, "worst_break_size")
        if finding is None:
            return None

        grid = np.asarray(finding.metadata["grid"], dtype=float)
        curve = np.asarray(finding.metadata["partial_dependence"], dtype=float)
        feature = finding.metadata["feature"]
        direction = finding.metadata["direction"]
        broken = {int(i) for i, _ in finding.metadata["breaks"]}

        ax = new_axes(ax, figsize=(6.4, 3.8))
        ax.plot(grid, curve, color=ACCENT, marker="o", markersize=4, linewidth=1.6, zorder=2)
        for index in sorted(broken):
            ax.plot(
                grid[index : index + 2],
                curve[index : index + 2],
                color=verdict_colour("BLOCKED"),
                linewidth=3.4,
                marker="X",
                markersize=7,
                zorder=3,
            )

        ax.set_xlabel(f"{feature} (swept across its own quantiles)")
        ax.set_ylabel("mean prediction")
        ax.set_title(f"Partial dependence on {feature} — declared {direction}")
        caption(
            ax,
            "each point holds every other column at its real values and sweeps this one.\n"
            "The crossed segments move against the declared direction: there, a worse "
            "risk is charged less.",
        )
        return ax


class DislocationCheck(BaseCheck):
    """Replacing an incumbent, who moves and by how much?

    The question a pricing committee asks about a new model is not "is it
    more accurate?" — that is settled by the time it reaches a gate. It is
    "how many policyholders see a rise above 25%, and are they anyone in
    particular?". A model can be better on every statistical measure and
    still be undeployable because of who it re-prices.

    Needs `context.baseline_pred`: the incumbent's prediction for the same
    rows, or last quarter's version of this model. Without it there is
    nothing to be a change *from*, and the check reports NOT_APPLICABLE
    rather than treating the mean as a baseline.

    **Non-blocking, and deliberately.** A dislocated book may be entirely
    correct — that is often the point of a re-rate — and no threshold can
    decide whether this particular profile is acceptable. The check's job is
    to put the number and the affected group in front of a person, which is
    what `NEEDS_REVIEW` is for.

    Rows whose baseline is zero or negative are excluded and counted: a
    premium moving from 0 to 500 is not an increase of any percentage.
    """

    name = "prediction_dislocation"
    category = "compliance"
    blocking = False
    supported_tasks = (REGRESSION,)

    def __init__(self, config: ActuarialConfig | None = None):
        self.config = config or ActuarialConfig()

    def _change(self, context) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """`(relative change, defined mask, weights)`, or None if unusable."""
        if getattr(context, "baseline_pred", None) is None or context.y_pred is None:
            return None
        change, defined = relative_change(context.y_pred, context.baseline_pred)
        if not defined.any():
            return None
        weights = weights_or_ones(exposure_array(context), len(change))
        return change, defined, weights

    def run(self, context) -> list[CheckResult]:
        if context.y_pred is None:
            return _not_applicable(self, "no y_pred — there is no new price to compare")
        if getattr(context, "baseline_pred", None) is None:
            return _not_applicable(
                self,
                "no baseline_pred supplied — dislocation is measured against the "
                "incumbent's price for the same rows (the model being replaced, or the "
                "previous version of this one). There is no sensible stand-in: comparing "
                "against the book mean would answer a different question",
            )

        prepared = self._change(context)
        if prepared is None:
            return _not_applicable(
                self,
                "every baseline_pred value is zero or negative, so no relative change is "
                "defined — a price moving from zero is not an increase of any percentage",
            )
        change, defined, weights = prepared
        weighted = _exposure_note(context)

        threshold = self.config.dislocation_threshold
        exposure_total = float(weights[defined].sum())
        rising = defined & (change >= threshold)
        falling = defined & (change <= -threshold)
        rise_share = float(weights[rising].sum() / exposure_total)
        fall_share = float(weights[falling].sum() / exposure_total)
        n_excluded = int((~defined).sum())

        moves = change[defined]
        p95 = float(np.percentile(moves, 95))
        worst = float(moves.max())

        group_note, group_shares = self._by_group(context, change, defined, weights, threshold)
        excluded_note = (
            f" [{n_excluded} row(s) excluded: baseline is zero or negative]" if n_excluded else ""
        )

        flag = "DISLOCATION_RISK" if rise_share > self.config.max_dislocated_share else "OK"
        return [
            CheckResult(
                self.name,
                self.category,
                flag,
                detail=(
                    f"{rise_share:.1%} of exposure rises by {threshold:.0%} or more "
                    f"(tolerance {self.config.max_dislocated_share:.1%}); {fall_share:.1%} "
                    f"falls by as much. 95th percentile move {p95:+.1%}, largest rise "
                    f"{worst:+.1%}{group_note}{weighted}{excluded_note}"
                ),
                blocking=self.blocking,
                metadata={
                    "dislocation_threshold": threshold,
                    "threshold": self.config.max_dislocated_share,
                    "rise_share": round(rise_share, 4),
                    "fall_share": round(fall_share, 4),
                    "p95_change": round(p95, 4),
                    "largest_rise": round(worst, 4),
                    "median_change": round(float(np.median(moves)), 4),
                    "n_rows_compared": int(defined.sum()),
                    "n_rows_excluded": n_excluded,
                    "exposure_weighted": bool(weighted),
                    "rise_share_by_group": group_shares,
                },
            )
        ]

    @staticmethod
    def _by_group(context, change, defined, weights, threshold) -> tuple[str, dict]:
        """Rise share per protected group, and the sentence naming the worst.

        Dislocation concentrated in one group is a different conversation
        from dislocation spread evenly, and the overall percentage cannot
        tell them apart. Reported here rather than raised as a separate
        fairness flag: whether it is unfair depends on why those policies
        were mispriced before, which is a judgement for the reviewer.
        """
        protected_df = getattr(context, "protected_df", None)
        if protected_df is None or protected_df.empty:
            return "", {}

        shares: dict = {}
        for attr in protected_df.columns:
            column = protected_df[attr].astype(str)
            per_group = {}
            for value in column.unique():
                cell = defined & np.asarray(column == value)
                total = float(weights[cell].sum())
                if total <= 0:
                    continue
                rising = cell & (change >= threshold)
                per_group[str(value)] = round(float(weights[rising].sum() / total), 4)
            if per_group:
                shares[attr] = per_group

        flat = [
            (attr, group, share)
            for attr, groups in shares.items()
            for group, share in groups.items()
        ]
        if not flat:
            return "", shares
        attr, group, share = max(flat, key=lambda row: row[2])
        return f". Most affected: {attr}={group} at {share:.1%}", shares

    def plot(self, context, results=None, ax=None):
        """The distribution of relative price change, with the threshold drawn.

        "12% of exposure rises by more than a quarter" is one number over a
        shape that matters: a tight bump just past the threshold is a
        rounding decision, and a long tail reaching +200% is a conduct
        problem. Both report 12%.
        """
        from ..plots import require_plotting
        from ..plots.style import ACCENT, RULE, caption, new_axes, verdict_colour

        _, sns = require_plotting()
        prepared = self._change(context)
        if prepared is None:
            return None
        change, defined, weights = prepared
        threshold = self.config.dislocation_threshold

        ax = new_axes(ax, figsize=(6.4, 3.8))
        sns.histplot(
            x=change[defined],
            weights=weights[defined],
            bins=40,
            ax=ax,
            color=ACCENT,
            edgecolor="white",
            linewidth=0.5,
            stat="probability",
        )
        ax.axvline(0.0, color=RULE, linewidth=1.4, linestyle="--", zorder=3)
        for bound, style in ((threshold, "-"), (-threshold, ":")):
            ax.axvline(
                bound, color=verdict_colour("BLOCKED"), linewidth=1.3, linestyle=style, zorder=3
            )

        ax.set_xlabel("relative change against the incumbent")
        ax.set_ylabel("share of exposure")
        ax.set_title(f"Who moves, and by how much — dislocation at ±{threshold:.0%}")
        ax.xaxis.set_major_formatter(lambda value, _: f"{value:+.0%}")
        caption(
            ax,
            "bars are weighted by exposure. The solid line is the rise threshold, the "
            "dotted line\nthe fall. A tight bump past the line is a rounding decision; a "
            "long tail is a conduct problem.",
        )
        return ax


__all__ = [
    "ActualVsExpectedCheck",
    "DislocationCheck",
    "MonotonicityCheck",
    "RiskDiscriminationCheck",
]
