"""Per-feature and outcome-level fairness checks for structured data models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._logging import get_logger
from .._sampling import stable_sample
from ..classes import favourable_mask, resolve_favourable
from ..config import FairnessConfig
from ..core.base import BaseCheck, CheckResult
from ..exceptions import GateConfigurationError
from ..metrics import to_class_labels, to_hard_labels
from ..model import ModelAdapter
from ..stats import correlation_ratio
from ..task import ALL_TASKS, CLASSIFICATION_TASKS, MULTICLASS, resolve_task

logger = get_logger("fairness")


class ProxyCorrelationCheck(BaseCheck):
    """Flags numeric input features that correlate strongly with a protected
    attribute — even when that attribute itself is excluded from the model."""

    name = "proxy_correlation"
    category = "fairness"
    blocking = False
    supported_tasks = ALL_TASKS  # compares features to attributes, not predictions

    def __init__(self, config: FairnessConfig | None = None):
        self.config = config or FairnessConfig()

    @staticmethod
    def _grid(X, protected_df) -> pd.DataFrame:
        """eta^2 for every (numeric feature, low-cardinality attribute) pair.

        Built once and shared by `run` and `plot`, so the heatmap and the
        findings can never disagree — the failure mode where a chart shows a
        cool cell beside a report line calling it a proxy.

        Attributes with ten or more distinct values are excluded: the
        correlation ratio treats each value as its own group, so a near-unique
        column drives eta^2 to 1 by arithmetic rather than by any real
        association.
        """
        features = [c for c in X.columns if X[c].dtype.kind in "if"]
        attributes = [c for c in protected_df.columns if protected_df[c].nunique() < 10]
        return pd.DataFrame(
            [[correlation_ratio(X[f], protected_df[a]) for a in attributes] for f in features],
            index=features,
            columns=attributes,
            dtype=float,
        )

    def plot(self, context, results=None, ax=None):
        """Heatmap of eta^2, feature by protected attribute.

        Replaces a table that runs to one row per pair — forty on a modest
        model. The eye finds the hot cell in a grid immediately and cannot
        scan forty rows for it, and the cool cells matter too: they are the
        evidence that the flagged feature is the exception rather than the
        whole feature set leaking.
        """
        from ..plots import require_plotting
        from ..plots.style import caption, new_axes, ring_cell, sharpen_colourbar, verdict_colour

        _, sns = require_plotting()
        if context.protected_df is None or context.protected_df.empty:
            return None
        grid = self._grid(context.X, context.protected_df)
        if grid.empty:
            return None

        # Height tracks the feature count: a fixed figure squeezes twenty
        # rows into unreadable slivers.
        ax = new_axes(ax, figsize=(1.6 + 1.3 * len(grid.columns), 1.2 + 0.34 * len(grid.index)))
        sns.heatmap(
            grid,
            ax=ax,
            annot=True,
            fmt=".2f",
            # Sequential, single-hue: eta^2 has a floor at zero and no
            # meaningful midpoint, so a diverging map would invent one.
            cmap="crest",
            vmin=0.0,
            vmax=1.0,
            linewidths=0.5,
            linecolor="white",
            # Short, because the bar is as tall as the grid and a three-feature
            # grid is two inches: a longer label runs off the top of the figure.
            cbar_kws={"label": "eta²"},
        )
        sharpen_colourbar(ax)

        # Ring what was actually reported, so the chart and the findings list
        # can be checked against each other at a glance.
        flagged = verdict_colour("NEEDS_REVIEW")
        for i, j in zip(*np.where(grid.to_numpy() > self.config.proxy_corr_threshold)):
            ring_cell(ax, int(j), int(i), flagged)

        ax.set_title(f"Proxy strength (ringed above {self.config.proxy_corr_threshold})")
        ax.set_xlabel(" ")  # a placeholder the caption can anchor beneath
        ax.set_ylabel("")
        ax.tick_params(labelrotation=0)
        caption(
            ax,
            "eta² is the share of the feature's variance explained by group membership.\n"
            "A hot cell means dropping the attribute from the model does not remove it.",
        )
        return ax

    def run(self, context) -> list[CheckResult]:
        X, protected_df = context.X, context.protected_df
        if protected_df is None or protected_df.empty:
            return [
                CheckResult(
                    self.name,
                    self.category,
                    "NOT_APPLICABLE",
                    "no protected_df supplied",
                    self.blocking,
                )
            ]

        grid = self._grid(X, protected_df)
        values = grid.to_numpy(dtype=float)
        results = []
        for i, feature in enumerate(grid.index):
            for j, attr in enumerate(grid.columns):
                eta_sq = float(values[i, j])
                if eta_sq > self.config.proxy_corr_threshold:
                    results.append(
                        CheckResult(
                            self.name,
                            self.category,
                            "PROXY_RISK",
                            detail=f"{feature} correlates with {attr} (eta^2={eta_sq:.3f})",
                            blocking=self.blocking,
                            metadata={
                                "feature": feature,
                                "protected_attr": attr,
                                "proxy_strength": round(eta_sq, 3),
                            },
                        )
                    )
        return results or [
            CheckResult(
                self.name,
                self.category,
                "OK",
                "no proxy correlations above threshold",
                self.blocking,
            )
        ]


class DisparateImpactCheck(BaseCheck):
    """Outcome-level disparity check per protected attribute (demographic parity).

    For multiclass, "predicted positive" means predicted into
    `context.favourable_classes` — for underwriting, typically `["accept"]`.
    That set defaults to the most favourable entry of `context.class_order`
    when one is given; with neither, the check reports NOT_APPLICABLE rather
    than picking a class arbitrarily, because which outcome counts as
    favourable is a judgement the data cannot supply.

    Demographic parity compares *selection rates* — the share of each group
    predicted positive — so it needs hard class labels. Continuous
    predictions are binarised at `config.decision_threshold` before being
    handed to fairlearn; predictions already in {0, 1} pass through
    untouched. Without that step a probability `y_pred` yields a selection
    rate of 0 in every group and a parity difference of exactly 0.0, which
    reads as "perfectly fair" no matter how skewed the model is.
    """

    name = "disparate_impact"
    category = "fairness"
    blocking = False
    # Demographic parity counts a selected class; there is none for a
    # continuous target. Regression uses the regression_fairness suite.
    supported_tasks = CLASSIFICATION_TASKS

    def __init__(self, config: FairnessConfig | None = None):
        self.config = config or FairnessConfig()

    def run(self, context) -> list[CheckResult]:
        if context.protected_df is None or context.protected_df.empty:
            return [
                CheckResult(
                    self.name,
                    self.category,
                    "NOT_APPLICABLE",
                    "no protected_df supplied",
                    self.blocking,
                )
            ]
        try:
            from fairlearn.metrics import demographic_parity_difference
        except ImportError:
            return [
                CheckResult(
                    self.name,
                    self.category,
                    "NOT_APPLICABLE",
                    "fairlearn not installed — pip install bdp-model-gate[structured]",
                    self.blocking,
                )
            ]

        task = resolve_task(context)
        class_order = getattr(context, "class_order", None)

        if task == MULTICLASS:
            favourable = resolve_favourable(
                getattr(context, "favourable_classes", None), class_order, task
            )
            if favourable is None:
                return [
                    CheckResult(
                        self.name,
                        self.category,
                        "NOT_APPLICABLE",
                        "multiclass parity needs to know which outcome counts as "
                        "favourable — set context.favourable_classes (e.g. ['accept']) "
                        "or context.class_order",
                        self.blocking,
                    )
                ]
            labels = to_class_labels(context.y_pred, class_order)
            # Collapse to a binary "got the good outcome" indicator, which is
            # what a selection rate means once there are more than two classes.
            y_pred = favourable_mask(labels, favourable).astype(int)
            y_true_eval = favourable_mask(
                to_class_labels(context.y_true, class_order), favourable
            ).astype(int)
            favourable_note = f" [favourable: {', '.join(map(str, favourable))}]"
        else:
            y_pred = to_hard_labels(context.y_pred, self.config.decision_threshold)
            y_true_eval = context.y_true
            favourable_note = ""
            if not np.array_equal(np.asarray(context.y_pred), y_pred):
                logger.debug(
                    "binarised continuous y_pred at decision_threshold=%s for demographic parity",
                    self.config.decision_threshold,
                )

        results = []
        for attr in context.protected_df.columns:
            dpd = demographic_parity_difference(
                y_true_eval,
                y_pred,
                sensitive_features=context.protected_df[attr],
            )
            flag = "DISPARITY_RISK" if abs(dpd) > self.config.disparity_threshold else "OK"
            results.append(
                CheckResult(
                    self.name,
                    self.category,
                    flag,
                    detail=f"{attr}: demographic parity diff={dpd:.3f}{favourable_note}",
                    blocking=self.blocking,
                    metadata={
                        "protected_attr": attr,
                        "demographic_parity_diff": round(dpd, 3),
                        "decision_threshold": self.config.decision_threshold,
                    },
                )
            )
        return results

    def plot(self, context, results=None, ax=None):
        """Parity difference swept across every decision threshold.

        A single cutoff produces a single number, and the number is a
        cliff-edge: 0.49 and 0.51 can sit on opposite sides of the verdict.
        The sweep answers the question a reviewer actually has — does this
        verdict survive a small change of cutoff, or was it an artefact of
        where the cutoff happened to land?

        Returns None for multiclass, where the prediction is a class rather
        than a score and there is no threshold to move.
        """
        from ..plots import require_plotting
        from ..plots.style import (
            MUTED,
            RULE,
            caption,
            categorical,
            markers,
            new_axes,
            verdict_colour,
        )

        require_plotting()
        if context.protected_df is None or context.protected_df.empty:
            return None
        if resolve_task(context) == MULTICLASS:
            return None
        try:
            from fairlearn.metrics import demographic_parity_difference
        except ImportError:
            return None

        scores = np.asarray(context.y_pred, dtype=float)
        if np.all(np.isin(scores, (0.0, 1.0))):
            return None  # already hard labels — every threshold gives the same split

        configured = self.config.decision_threshold
        limit = self.config.disparity_threshold
        # Include the configured cutoff explicitly rather than hoping the grid
        # lands on it, so the marked point is the verdict, not an interpolation.
        sweep = np.unique(np.concatenate([np.linspace(0.05, 0.95, 37), [configured]]))

        attributes = list(context.protected_df.columns)
        ax = new_axes(ax)
        ax.axhspan(limit, 1.0, color=verdict_colour("BLOCKED"), alpha=0.07, zorder=0)
        ax.axhline(limit, color=verdict_colour("BLOCKED"), linewidth=1.0, linestyle=":", zorder=1)
        ax.axvline(configured, color=RULE, linewidth=1.2, zorder=1)

        for colour, marker, attr in zip(
            categorical(len(attributes)), markers(len(attributes)), attributes
        ):
            sensitive = context.protected_df[attr]
            curve = [
                abs(
                    demographic_parity_difference(
                        context.y_true,
                        (scores >= t).astype(int),
                        sensitive_features=sensitive,
                    )
                )
                for t in sweep
            ]
            ax.plot(sweep, curve, color=colour, label=attr, zorder=2)
            at_configured = curve[int(np.argmin(np.abs(sweep - configured)))]
            ax.scatter(
                [configured],
                [at_configured],
                color=colour,
                marker=marker,
                s=70,
                edgecolor="white",
                linewidth=0.9,
                zorder=3,
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("decision threshold")
        ax.set_ylabel("|demographic parity difference|")
        ax.set_title("Does the parity verdict survive a change of cutoff?")
        ax.legend(loc="upper right")
        caption(
            ax,
            "marked points are the verdict as configured. A peak near the cutoff means the\n"
            "pass was luck: shading is the region that would be reported as a disparity.",
        )
        ax.annotate(
            f"cutoff in force: {configured:g}",
            xy=(configured, 1),
            xycoords=("data", "axes fraction"),
            xytext=(4, -4),
            textcoords="offset points",
            va="top",
            fontsize=8,
            color=MUTED,
        )
        return ax


class ShapSubgroupCheck(BaseCheck):
    """For each feature, checks whether its SHAP contribution differs
    meaningfully across protected-attribute groups — catches features that
    look fair on average but drive outcomes differently for a subgroup.

    The gap is measured **relative to the mean absolute SHAP contribution**,
    not in the raw units of the model output. SHAP values inherit the target's
    scale, so an absolute threshold that is sensible for a probability
    (contributions around 0.5) flags every feature on a premium model whose
    contributions run to thousands of naira. Relative, one threshold works
    on both: a value of 0.5 means "this feature's cross-group gap is worth
    half of a typical contribution".
    """

    name = "shap_subgroup_gap"
    category = "fairness"
    blocking = False
    supported_tasks = ALL_TASKS  # SHAP contributions are defined for any output

    def __init__(self, config: FairnessConfig | None = None):
        self.config = config or FairnessConfig()

    @staticmethod
    def _build_explainer(shap_module, model, X, adapter=None):
        """TreeExplainer is dramatically faster and exact for tree-based
        models; fall back to the generic (permutation/kernel) Explainer for
        everything else."""
        tree_module_markers = (
            "sklearn.ensemble",
            "sklearn.tree",
            "xgboost",
            "lightgbm",
            "catboost",
        )
        model_module = type(model).__module__
        is_tree_model = any(marker in model_module for marker in tree_module_markers)
        if is_tree_model:
            try:
                return shap_module.TreeExplainer(model)
            except Exception:
                pass  # fall through to generic explainer if TreeExplainer can't handle this model
        if model is not None:
            try:
                return shap_module.Explainer(model, X)
            except (TypeError, ValueError):
                pass  # not an estimator shap recognises — fall through
        # shap's generic Explainer wants a callable. Hand it the adapter's
        # predict — the documented black-box pattern — which works for a
        # predict_fn-only context where there is no model object at all.
        if adapter is None:
            adapter = ModelAdapter(model=model)
        return shap_module.Explainer(adapter.predict, X)

    @staticmethod
    def _positive_class_values(values, class_index=None):
        """Normalises SHAP output to one contribution per (row, feature).

        shap returns a 2-D array for regressors and some binary classifiers,
        but a 3-D (rows, features, classes) array for others —
        `RandomForestClassifier` among them, and which shape you get changed
        across shap versions.

        Binary reduces to the positive class. Multiclass reduces to
        `class_index`, the column of the favourable outcome, so the check
        answers "does this feature push some groups away from being
        accepted?" rather than averaging across unrelated classes. Without
        a class index there is no defensible reduction, so it returns None
        and the caller reports NOT_APPLICABLE.
        """
        arr = np.asarray(values)
        if arr.ndim == 2:
            return arr
        if arr.ndim != 3:
            return None
        if arr.shape[-1] == 2:
            return arr[:, :, 1]
        if class_index is not None and 0 <= class_index < arr.shape[-1]:
            return arr[:, :, class_index]
        return None

    @staticmethod
    def _favourable_class_index(context):
        """Column of the favourable class in a multiclass SHAP array.

        shap orders its class axis by the model's sorted class labels, which
        is what `class_order` is matched against here.
        """
        class_order = getattr(context, "class_order", None)
        if class_order is None:
            return None
        favourable = resolve_favourable(
            getattr(context, "favourable_classes", None), class_order, MULTICLASS
        )
        if not favourable:
            return None
        # shap's class axis follows the model's sorted classes, not the
        # favourability ordering the caller supplied.
        by_model_order = sorted(class_order, key=str)
        try:
            return by_model_order.index(favourable[0])
        except ValueError:  # pragma: no cover — resolve_favourable validates membership
            return None

    def run(self, context) -> list[CheckResult]:
        if context.protected_df is None or context.protected_df.empty:
            return [
                CheckResult(
                    self.name,
                    self.category,
                    "NOT_APPLICABLE",
                    "no protected_df supplied",
                    self.blocking,
                )
            ]
        try:
            import pandas as pd
            import shap
        except ImportError:
            return [
                CheckResult(
                    self.name,
                    self.category,
                    "NOT_APPLICABLE",
                    "shap not installed — pip install bdp-model-gate[structured]",
                    self.blocking,
                )
            ]

        try:
            explainer = self._build_explainer(
                shap, context.model, context.X, ModelAdapter.from_context(context)
            )
            shap_values = explainer(context.X)
        except Exception as exc:
            # A non-blocking fairness check must not block a deploy because
            # shap could not introspect the model. ModelGate would otherwise
            # convert the exception into a blocking CHECK_ERROR.
            logger.warning("shap could not explain this model: %r", exc)
            return [
                CheckResult(
                    self.name,
                    self.category,
                    "NOT_APPLICABLE",
                    f"shap could not explain this model ({type(exc).__name__}: {exc}) — "
                    "subgroup SHAP gaps were not evaluated",
                    self.blocking,
                )
            ]
        values = self._positive_class_values(
            shap_values.values, self._favourable_class_index(context)
        )
        if values is None:
            n_classes = np.asarray(shap_values.values).shape[-1]
            return [
                CheckResult(
                    self.name,
                    self.category,
                    "NOT_APPLICABLE",
                    f"multiclass SHAP output ({n_classes} classes) and no favourable "
                    "class to reduce to — set context.class_order or "
                    "context.favourable_classes so contributions can be compared for "
                    "one outcome",
                    self.blocking,
                )
            ]
        shap_df = pd.DataFrame(values, columns=context.X.columns)

        # Scale gaps by the typical contribution magnitude, so the threshold
        # is dimensionless and survives a change of target units.
        shap_scale = float(np.mean(np.abs(values)))
        if shap_scale <= 0:
            return [
                CheckResult(
                    self.name,
                    self.category,
                    "NOT_APPLICABLE",
                    "every SHAP contribution is zero — the model does not use its "
                    "features, so there are no subgroup gaps to compare",
                    self.blocking,
                )
            ]

        results = []
        for attr in context.protected_df.columns:
            for feature in context.X.columns:
                group_means = shap_df[feature].groupby(context.protected_df[attr].values).mean()
                gap = float(group_means.max() - group_means.min())
                relative_gap = abs(gap) / shap_scale
                if relative_gap > self.config.shap_gap_threshold:
                    results.append(
                        CheckResult(
                            self.name,
                            self.category,
                            "SUBGROUP_IMPACT_RISK",
                            detail=(
                                f"{feature} SHAP contribution gap across {attr}="
                                f"{gap:,.3f} — {relative_gap:.0%} of the mean absolute "
                                f"contribution {shap_scale:,.3f}"
                            ),
                            blocking=self.blocking,
                            metadata={
                                "feature": feature,
                                "protected_attr": attr,
                                "shap_gap": round(gap, 4),
                                "relative_gap": round(relative_gap, 4),
                                "shap_scale": round(shap_scale, 4),
                                "threshold": self.config.shap_gap_threshold,
                            },
                        )
                    )
        return results or [
            CheckResult(
                self.name,
                self.category,
                "OK",
                "no SHAP subgroup gaps above threshold",
                self.blocking,
            )
        ]


class CounterfactualFlipCheck(BaseCheck):
    """Flips protected-attribute values (when they're model inputs) and
    measures average prediction shift. Only meaningful if a protected
    attribute is actually included as a feature."""

    name = "counterfactual_flip"
    category = "fairness"
    blocking = False
    # Measures a shift in P(favourable outcome). Regression's analogue is
    # the mean prediction shift, which GroupMeanGapCheck already covers.
    supported_tasks = CLASSIFICATION_TASKS

    def __init__(self, config: FairnessConfig | None = None, n_samples: int = 200):
        self.config = config or FairnessConfig()
        self.n_samples = n_samples

    @staticmethod
    def _favourable_proba(adapter, frame, context):
        """Probability of the favourable outcome, for binary or multiclass."""
        if resolve_task(context) != MULTICLASS:
            return adapter.predict_positive_proba(frame)
        class_order = getattr(context, "class_order", None)
        favourable = resolve_favourable(
            getattr(context, "favourable_classes", None), class_order, MULTICLASS
        )
        if class_order is None or not favourable:
            # run() screens for this, but the helper must not depend on that
            # to stay correct if it is ever called from elsewhere.
            raise GateConfigurationError(
                "multiclass counterfactuals need context.class_order and a favourable "
                "class to measure the shift in"
            )
        matrix = adapter.predict_proba_matrix(frame)
        by_model_order = sorted(class_order, key=str)
        columns = [by_model_order.index(c) for c in favourable]
        return matrix[:, columns].sum(axis=1)

    def run(self, context) -> list[CheckResult]:
        if context.protected_df is None or context.protected_df.empty:
            return [
                CheckResult(
                    self.name,
                    self.category,
                    "NOT_APPLICABLE",
                    "no protected_df supplied",
                    self.blocking,
                )
            ]
        if resolve_task(context) == MULTICLASS and getattr(context, "class_order", None) is None:
            return [
                CheckResult(
                    self.name,
                    self.category,
                    "NOT_APPLICABLE",
                    "multiclass counterfactuals need context.class_order to identify "
                    "the favourable outcome to measure a shift in",
                    self.blocking,
                )
            ]
        adapter = ModelAdapter.from_context(context)
        if not adapter.can_predict_proba:
            return [
                CheckResult(
                    self.name,
                    self.category,
                    "NOT_APPLICABLE",
                    "no probability output available — this check needs either a model "
                    "with .predict_proba() or context.predict_proba_fn",
                    self.blocking,
                )
            ]

        X = context.X
        results = []
        for attr in context.protected_df.columns:
            if attr not in X.columns:
                continue  # attribute excluded from model inputs — nothing to flip
            sample = stable_sample(X, self.n_samples, 42)
            base_preds = self._favourable_proba(adapter, sample, context)
            for val in context.protected_df[attr].unique():
                flipped = sample.copy()
                flipped[attr] = val
                flipped_preds = self._favourable_proba(adapter, flipped, context)
                shift = float(np.mean(np.abs(flipped_preds - base_preds)))
                flag = (
                    "COUNTERFACTUAL_RISK"
                    if shift > self.config.counterfactual_shift_threshold
                    else "OK"
                )
                results.append(
                    CheckResult(
                        self.name,
                        self.category,
                        flag,
                        detail=f"flipping {attr} to {val!r} shifts predictions by {shift:.4f} on average",
                        blocking=self.blocking,
                        metadata={
                            "protected_attr": attr,
                            "flipped_to": str(val),
                            "avg_prediction_shift": round(shift, 4),
                        },
                    )
                )
        return results or [
            CheckResult(
                self.name,
                self.category,
                "NOT_APPLICABLE",
                "no protected attributes present as model inputs",
                self.blocking,
            )
        ]
