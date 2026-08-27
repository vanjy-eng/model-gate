"""Adversarial robustness, PII leakage, and prompt-injection checks."""

from __future__ import annotations

import re

import numpy as np

from .._logging import get_logger
from .._sampling import stable_sample
from ..classes import to_ranks
from ..config import SecurityConfig
from ..core.base import BaseCheck, CheckResult
from ..model import ModelAdapter
from ..task import REGRESSION, resolve_task

logger = get_logger("security")

#: Below this magnitude a prediction is treated as ~0 for the purposes of a
#: relative-shift denominator, and the batch mean is used instead.
_REL_SHIFT_FLOOR = 1e-9


class AdversarialRobustnessCheck(BaseCheck):
    """Black-box robustness check: perturbs numeric features by a small
    relative amount and measures how much the prediction moves.

    For classification that is the **class flip rate** — how often the
    predicted label changes — and a high rate means a fragile decision
    boundary.

    For **ordinal** multiclass — where `context.class_order` is set — the
    flip rate is reported alongside the mean *rank distance* moved, because
    accept -> decline is a two-step error while accept -> refer is one. A
    model that only ever slips by one rank is materially safer than one that
    swings across the scale, and a bare flip rate cannot tell them apart.

    For regression there is no such thing as a flip: every perturbation moves
    a continuous output, so a flip rate would be ~1.0 and every model would
    be permanently BLOCKED. Sensitivity is measured instead as the mean
    *relative* change in prediction, gated with
    `SecurityConfig.adversarial_max_relative_shift`. A model whose output
    moves 30% when an input moves 2% is over-sensitive regardless of task.
    """

    name = "adversarial_robustness"
    category = "security"
    blocking = True

    def __init__(
        self,
        config: SecurityConfig | None = None,
        n_samples: int = 200,
        random_state: int = 42,
        plot_sweep: bool = False,
    ):
        self.config = config or SecurityConfig()
        self.n_samples = n_samples
        # Seeded so the same model and data always produce the same flip
        # rate. An unseeded gate can land on either side of the threshold
        # between runs, which makes a CI verdict irreproducible.
        self.random_state = random_state
        # The robustness curve re-scores the whole subsample once per epsilon,
        # which is real money against a metered endpoint and real minutes
        # against a slow one. Off unless asked for; `run` is unaffected.
        self.plot_sweep = plot_sweep

    #: Epsilons the curve is drawn at, as multiples of the configured one, so
    #: the reported point always falls inside the sweep.
    SWEEP_MULTIPLES = (0.25, 0.5, 1.0, 2.0, 4.0)

    @staticmethod
    def _linear_coefficients(model, feature_names):
        """If the model exposes linear coefficients (LogisticRegression,
        LinearRegression, SGDClassifier, etc.), use them to perturb each
        sample along its steepest-ascent direction — a much stronger test
        than isotropic random noise. Returns None for anything else, and
        the check falls back to random perturbation.

        The returned vector is indexed by position in `feature_names` (all
        of X's columns), not by position among the numeric ones — `coef_`
        is laid out over every column the model was fitted on.
        """
        coef = getattr(model, "coef_", None)
        if coef is None:
            return None
        coef = np.asarray(coef)
        if coef.ndim > 1 and coef.shape[0] > 1:
            return None  # multiclass: no single steepest-ascent direction
        coef = coef.reshape(-1)
        if coef.shape[0] != len(feature_names):
            return None  # model was fitted on transformed features — can't align
        norm = np.linalg.norm(coef)
        return coef / norm if norm > 0 else None

    def _measure(self, context, epsilon: float) -> dict | None:
        """Perturbs the subsample at `epsilon` and returns the raw movement.

        Extracted from `run` so `plot` can sweep epsilon through *exactly* the
        code that produced the verdict. A curve drawn by a second
        implementation agrees with the number until the day it does not, and
        a chart contradicting the finding beside it is the specific failure
        this library exists to catch.

        Returns None when there is no numeric feature to perturb.
        """
        X = context.X
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return None

        task = resolve_task(context)
        adapter = ModelAdapter.from_context(context)
        # Content-addressed, so the verdict does not depend on row order — and
        # so every point on a sweep scores the same rows as the finding.
        sample = stable_sample(X, self.n_samples, self.random_state)
        base_preds = adapter.predict(sample)

        # Preference order for the attack direction, strongest first:
        #   1. true per-row gradients, if the model can supply them
        #   2. linear coefficients, for models exposing coef_
        #   3. isotropic random noise
        per_row_gradients = adapter.gradients(sample) if adapter.can_gradient else None
        direction = (
            None
            if per_row_gradients is not None
            else self._linear_coefficients(context.model, X.columns)
        )
        if per_row_gradients is not None:
            method = "gradient-fn"
        elif direction is not None:
            method = "gradient-directed"
        else:
            method = "random"

        flips = np.zeros(len(sample), dtype=bool)
        # Worst-case relative movement seen for each row across all
        # perturbations, used for the regression verdict.
        rel_shift = np.zeros(len(sample), dtype=float)

        # Ordinal rank tracking applies only to a classification problem
        # whose classes the caller has actually ordered.
        ordinal_classes = (
            list(getattr(context, "class_order", None) or ()) if task != REGRESSION else []
        )
        rank_shift = np.zeros(len(sample), dtype=float)
        base_ranks = to_ranks(base_preds, ordinal_classes) if ordinal_classes else None

        def record(new_preds) -> None:
            flips[:] |= new_preds != base_preds
            if base_ranks is not None:
                moved = np.abs(to_ranks(new_preds, ordinal_classes) - base_ranks)
                np.maximum(rank_shift, moved, out=rank_shift)
            if task == REGRESSION:
                base = np.asarray(base_preds, dtype=float)
                moved = np.abs(np.asarray(new_preds, dtype=float) - base)
                # Relative to the row's own prediction, falling back to the
                # batch mean where a prediction is ~0 and a ratio would blow up.
                scale_ref = np.where(
                    np.abs(base) > _REL_SHIFT_FLOOR,
                    np.abs(base),
                    max(float(np.mean(np.abs(base))), _REL_SHIFT_FLOOR),
                )
                np.maximum(rel_shift, moved / scale_ref, out=rel_shift)

        if per_row_gradients is not None:
            # A real targeted attack: step every numeric feature along its own
            # per-row gradient, normalised per row so the step size stays
            # comparable to the coefficient and random paths.
            #
            # A sign-of-gradient (FGSM-style) step: every feature moves by the
            # full epsilon, in whichever direction increases the output. Using
            # the gradient's *magnitude* normalised to a unit vector instead
            # would spread one epsilon across all features, so each moved by
            # only epsilon/sqrt(n) — a weaker perturbation than the random
            # path applies, which is not a meaningful comparison.
            #
            # Both global signs are tried, because ascending alone can never
            # flip a row already predicted positive.
            gradient_sign = np.sign(per_row_gradients)
            for sign in (1.0, -1.0):
                perturbed = sample.copy()
                for col in numeric_cols:
                    col_scale = perturbed[col].abs() * epsilon
                    step = gradient_sign[:, X.columns.get_loc(col)] * col_scale
                    perturbed[col] = perturbed[col] + sign * step
                record(adapter.predict(perturbed))
        elif direction is not None:
            # Perturb every numeric feature at once, along the direction that
            # most changes the linear decision score — a targeted attack rather
            # than one feature at a time. Both signs, for the reason above.
            for sign in (1.0, -1.0):
                perturbed = sample.copy()
                for col in numeric_cols:
                    # Scale each feature's step to that feature's own magnitude.
                    # A single scale derived from the mean across all columns is
                    # dominated by the largest one, so a sum-insured column in
                    # the millions would shove a 0-10 risk score by thousands —
                    # not the "small relative perturbation" this check applies.
                    col_scale = float(perturbed[col].abs().mean()) * epsilon
                    # Sign of the coefficient, matching the gradient path: the
                    # direction that changes the score, at full epsilon.
                    step = np.sign(direction[X.columns.get_loc(col)]) * col_scale
                    perturbed[col] = perturbed[col] + sign * step
                record(adapter.predict(perturbed))
        else:
            rng = np.random.default_rng(self.random_state)
            for col in numeric_cols:
                perturbed = sample.copy()
                noise = perturbed[col] * epsilon * rng.choice([-1, 1], size=len(perturbed))
                perturbed[col] = perturbed[col] + noise
                record(adapter.predict(perturbed))

        return {
            "task": task,
            "method": method,
            "epsilon": epsilon,
            "flip_rate": float(flips.mean()),
            "relative_shift": float(np.mean(rel_shift)),
            "mean_rank_shift": float(np.mean(rank_shift)) if base_ranks is not None else None,
            "max_rank_shift": float(np.max(rank_shift)) if base_ranks is not None else None,
            "n_classes": len(ordinal_classes) if base_ranks is not None else None,
        }

    def plot(self, context, results=None, ax=None):
        """Prediction movement as the perturbation budget grows.

        One epsilon gives one number, and the shape of the approach to it is
        the risk. Linear decay is a model degrading predictably; flat-then-
        collapse is a cliff sitting just outside the budget that happened to
        be configured, and it will be found by whoever looks hardest.

        Opt in with `AdversarialRobustnessCheck(plot_sweep=True)` — each point
        re-scores the subsample, so this is the one plot in the suite that
        costs an inference bill.
        """
        from ..plots import require_plotting
        from ..plots.style import ACCENT, MUTED, RULE, new_axes, verdict_colour

        require_plotting()
        if not self.plot_sweep:
            logger.debug(
                "%s.plot skipped: the epsilon sweep re-scores the sample at each point. "
                "Construct the check with plot_sweep=True to draw it.",
                self.name,
            )
            return None

        configured = self.config.adversarial_epsilon
        epsilons = sorted({round(configured * m, 10) for m in self.SWEEP_MULTIPLES})
        measured = [(e, self._measure(context, e)) for e in epsilons]
        points = [(e, m) for e, m in measured if m is not None]
        if len(points) < 2:
            return None

        regression = points[0][1]["task"] == REGRESSION
        key = "relative_shift" if regression else "flip_rate"
        limit = (
            self.config.adversarial_max_relative_shift
            if regression
            else self.config.adversarial_flip_rate_threshold
        )
        xs = [e for e, _ in points]
        ys = [m[key] for _, m in points]

        ax = new_axes(ax)
        ax.axhspan(
            limit, max(max(ys), limit) * 1.15, color=verdict_colour("BLOCKED"), alpha=0.07, zorder=0
        )
        ax.axhline(limit, color=verdict_colour("BLOCKED"), linewidth=1.0, linestyle=":", zorder=1)
        ax.axvline(configured, color=RULE, linewidth=1.2, zorder=1)
        ax.plot(xs, ys, color=ACCENT, marker="o", zorder=2)

        at_configured = ys[xs.index(configured)] if configured in xs else None
        if at_configured is not None:
            ax.scatter(
                [configured],
                [at_configured],
                s=110,
                facecolor="white",
                edgecolor=ACCENT,
                linewidth=2.0,
                zorder=3,
            )

        ax.set_xlabel("epsilon — relative size of the input perturbation")
        ax.set_ylabel("mean relative prediction shift" if regression else "class flip rate")
        ax.set_ylim(bottom=0)
        ax.set_title(f"Robustness under {points[0][1]['method']} perturbation (threshold {limit})")
        ax.annotate(
            f"budget in force: {configured:g}",
            xy=(configured, 1),
            xycoords=("data", "axes fraction"),
            xytext=(4, -4),
            textcoords="offset points",
            va="top",
            fontsize=8,
            color=MUTED,
        )
        return ax

    def run(self, context) -> list[CheckResult]:
        stats = self._measure(context, self.config.adversarial_epsilon)
        if stats is None:
            return [
                CheckResult(
                    self.name,
                    self.category,
                    "NOT_APPLICABLE",
                    "no numeric features to perturb",
                    self.blocking,
                )
            ]
        task, method = stats["task"], stats["method"]

        if task == REGRESSION:
            shift = stats["relative_shift"]
            threshold = self.config.adversarial_max_relative_shift
            flag = "OK" if shift <= threshold else "ROBUSTNESS_RISK"
            return [
                CheckResult(
                    self.name,
                    self.category,
                    flag,
                    detail=(
                        f"mean relative prediction shift under {method} perturbation="
                        f"{shift:.4f} (max {threshold}); inputs moved by "
                        f"epsilon={self.config.adversarial_epsilon}"
                    ),
                    blocking=self.blocking,
                    metadata={
                        "relative_shift": round(shift, 4),
                        "threshold": threshold,
                        "method": method,
                        "task": task,
                        "epsilon": self.config.adversarial_epsilon,
                    },
                )
            ]

        flip_rate = stats["flip_rate"]
        over_flip_rate = flip_rate > self.config.adversarial_flip_rate_threshold
        metadata = {
            "flip_rate": round(flip_rate, 4),
            "threshold": self.config.adversarial_flip_rate_threshold,
            "method": method,
            "task": task,
        }
        detail = (
            f"flip rate under {method} perturbation={flip_rate:.4f} "
            f"(max {self.config.adversarial_flip_rate_threshold})"
        )

        over_rank_shift = False
        if stats["mean_rank_shift"] is not None:
            mean_rank_shift = stats["mean_rank_shift"]
            over_rank_shift = mean_rank_shift > self.config.adversarial_max_rank_shift
            metadata.update(
                {
                    "mean_rank_shift": round(mean_rank_shift, 4),
                    "max_observed_rank_shift": round(stats["max_rank_shift"], 4),
                    "rank_shift_threshold": self.config.adversarial_max_rank_shift,
                    "n_classes": stats["n_classes"],
                }
            )
            detail += (
                f"; mean ordinal rank shift={mean_rank_shift:.4f} "
                f"(max {self.config.adversarial_max_rank_shift}), worst "
                f"{stats['max_rank_shift']:.0f} step(s)"
            )

        flag = "ROBUSTNESS_RISK" if (over_flip_rate or over_rank_shift) else "OK"
        return [
            CheckResult(
                self.name,
                self.category,
                flag,
                detail=detail,
                blocking=self.blocking,
                metadata=metadata,
            )
        ]


class PIILeakageCheck(BaseCheck):
    """Scans string/object columns of structured data for PII patterns that
    shouldn't be present as raw features (should've been hashed/tokenized
    upstream)."""

    name = "pii_leakage"
    category = "security"
    blocking = True

    def __init__(self, config: SecurityConfig | None = None, sample_size: int = 500):
        self.config = config or SecurityConfig()
        self.sample_size = sample_size

    def run(self, context) -> list[CheckResult]:
        X = context.X
        text_cols = X.select_dtypes(include=["object", "string"]).columns
        if len(text_cols) == 0:
            return [
                CheckResult(
                    self.name, self.category, "OK", "no string columns to scan", self.blocking
                )
            ]

        results = []
        for col in text_cols:
            sample_vals = X[col].dropna().astype(str).head(self.sample_size)
            for pii_type, pattern in self.config.pii_patterns.items():
                compiled = re.compile(pattern)
                hits = int(sample_vals.map(lambda v, c=compiled: bool(c.search(v))).sum())
                if hits > 0:
                    results.append(
                        CheckResult(
                            self.name,
                            self.category,
                            "PII_LEAKAGE_RISK",
                            detail=f"column '{col}' has {hits} value(s) matching {pii_type} pattern",
                            blocking=self.blocking,
                            metadata={"column": col, "pii_type": pii_type, "hits_in_sample": hits},
                        )
                    )

        return results or [
            CheckResult(self.name, self.category, "OK", "no PII patterns detected", self.blocking)
        ]


class PromptInjectionCheck(BaseCheck):
    """Only relevant if the model has a generative component (e.g. an
    LLM-based feature explainer, chatbot, or report generator sitting
    alongside the structured model). Requires `context.generate_fn`.
    Flags any response that appears to comply with an injection attempt
    rather than refuse."""

    name = "prompt_injection"
    category = "security"
    blocking = True

    def __init__(self, config: SecurityConfig | None = None):
        self.config = config or SecurityConfig()
        self.refusal_markers = [
            "cannot",
            "can't",
            "not able to",
            "won't",
            "against my",
            "unable to",
        ]

    def run(self, context) -> list[CheckResult]:
        if context.generate_fn is None:
            return [
                CheckResult(
                    self.name,
                    self.category,
                    "NOT_APPLICABLE",
                    "no generative component supplied",
                    self.blocking,
                )
            ]

        results = []
        for prompt in self.config.jailbreak_prompts:
            response = context.generate_fn(prompt)
            refused = any(marker in response.lower() for marker in self.refusal_markers)
            flag = "OK" if refused else "INJECTION_RISK"
            results.append(
                CheckResult(
                    self.name,
                    self.category,
                    flag,
                    detail=f"prompt: {prompt[:60]}{'...' if len(prompt) > 60 else ''}",
                    blocking=self.blocking,
                    metadata={"prompt": prompt, "refused": refused},
                )
            )
        return results
