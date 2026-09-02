"""Metric resolution for the performance gate.

The performance gate scores a model with whichever metric the caller
configured (`PerformanceConfig.metric`). This module owns the mapping from
that config value to a callable, and — importantly — makes the choice
*explicit* in the report rather than silently depending on which optional
dependencies happen to be installed.

Three kinds of value are accepted:

    "auto"        try the metrics in AUTO_PREFERENCE in order, using the
                  first one whose dependencies are available. A fallback is
                  logged at WARNING level and named in the check's output,
                  so it never happens invisibly.
    "<name>"      any key of BUILTIN_METRICS. If its dependencies are
                  missing, that's a GateConfigurationError — an explicit
                  request is never silently substituted.
    callable      any `fn(y_true, y_pred) -> float`. Called with y_pred
                  exactly as supplied (no thresholding), since only the
                  caller knows what their metric expects.

Metrics differ in what they want from `y_pred`: ranking metrics like
`roc_auc` need continuous scores, while `accuracy`/`f1`/`precision`/
`recall` need hard class labels. `needs_hard_labels` records which, and
the check binarizes at `PerformanceConfig.decision_threshold` when needed.

They also differ in whether a per-row weight means anything to them.
`context.exposure` is bound in as `sample_weight` for the regression
family, where a twelve-month policy is twelve times the evidence about a
rate that a one-month policy is; it is *not* applied to a class-label
metric, and it is never applied to a callable of your own, whose signature
this module does not know. Whether it was applied is recorded on the
`ResolvedMetric` and printed in the check's detail, because "RMSE = 412"
means two different things depending on the answer.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Callable, Union  # Union: runtime alias, can't use PEP 604 on 3.9

import numpy as np

from ._logging import get_logger
from .actuarial import lorenz_gini
from .classes import to_ranks
from .exceptions import GateConfigurationError
from .task import BINARY, CLASSIFICATION_TASKS, MULTICLASS, REGRESSION

logger = get_logger("metrics")

MetricFn = Callable[[Any, Any], float]
MetricSetting = Union[str, MetricFn]


@dataclass(frozen=True)
class MetricSpec:
    """How to obtain one named metric, and what it expects from y_pred."""

    name: str
    sklearn_fn: str
    needs_hard_labels: bool
    #: Pure-numpy implementation. For a metric with a `sklearn_fn` this is a
    #: *fallback*, used only when scikit-learn is absent. For one with an
    #: empty `sklearn_fn` — rmse, mape, poisson_deviance, lorenz_gini —
    #: scikit-learn has no equivalent and this is the only implementation
    #: there is, which is why `used_fallback_impl` distinguishes the two.
    #: None means the metric genuinely requires scikit-learn.
    fallback: MetricFn | None = None
    #: False for error metrics (RMSE, MAE, MAPE, deviance), where a *lower*
    #: value is better. These are gated with `max_error`, not `min_score`.
    greater_is_better: bool = True
    #: Which prediction tasks this metric can score.
    tasks: tuple[str, ...] = CLASSIFICATION_TASKS
    #: True for metrics whose scikit-learn form needs an `average=` strategy
    #: once there are more than two classes (f1, precision, recall).
    needs_average: bool = False
    #: True for metrics defined only on an ordinal scale, which therefore
    #: require `context.class_order`.
    needs_class_order: bool = False
    #: True for metrics that accept a per-row weight, which is how
    #: `context.exposure` reaches them. Set on the regression metrics only:
    #: exposure is a statement about how much of a period a row represents,
    #: and that is a claim about a rate, not about a class label. A metric
    #: without it is scored unweighted and the report says so.
    supports_weight: bool = False


def _accuracy_numpy(y_true: Any, y_pred: Any) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def _as_floats(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)


def _mean(values: np.ndarray, sample_weight: Any = None) -> float:
    """Mean, exposure-weighted when a weight is supplied.

    Every numpy metric below routes its averaging through here, so the
    weighted and unweighted forms cannot drift apart — on a weight of None
    it is exactly `np.mean`.
    """
    if sample_weight is None:
        return float(np.mean(values))
    weights = np.asarray(sample_weight, dtype=float)
    total = float(weights.sum())
    if total <= 0:
        raise GateConfigurationError(
            "the rows this metric could score carry no exposure between them, so there "
            "is no weighted average to take"
        )
    return float(np.dot(values, weights) / total)


# The regression metrics are short enough to implement directly, which keeps
# them available on a core install and sidesteps scikit-learn's churn around
# `mean_squared_error(squared=False)` / `root_mean_squared_error`.


def _rmse_numpy(y_true: Any, y_pred: Any, sample_weight: Any = None) -> float:
    t, p = _as_floats(y_true, y_pred)
    return float(np.sqrt(_mean((t - p) ** 2, sample_weight)))


def _mae_numpy(y_true: Any, y_pred: Any, sample_weight: Any = None) -> float:
    t, p = _as_floats(y_true, y_pred)
    return _mean(np.abs(t - p), sample_weight)


def _r2_numpy(y_true: Any, y_pred: Any, sample_weight: Any = None) -> float:
    t, p = _as_floats(y_true, y_pred)
    w = np.ones_like(t) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    # Both sums carry the same weights, and the baseline is the *weighted*
    # mean: comparing a weighted residual against an unweighted baseline
    # would make r2 depend on the exposure profile of the book rather than
    # on the model.
    ss_res = float(np.dot(w, (t - p) ** 2))
    ss_tot = float(np.dot(w, (t - _mean(t, sample_weight)) ** 2))
    if ss_tot == 0.0:
        # A constant target has no variance to explain. Perfect prediction is
        # 1.0; anything else is undefined rather than arbitrarily bad.
        return 1.0 if ss_res == 0.0 else float("-inf")
    return 1.0 - ss_res / ss_tot


def _mape_numpy(y_true: Any, y_pred: Any, sample_weight: Any = None) -> float:
    """Mean absolute percentage error, skipping zero actuals.

    MAPE is the natural metric for skewed money targets like claims severity,
    but it is undefined where the actual is 0. Those rows are excluded and the
    exclusion is logged, rather than returning inf for the whole batch.
    """
    t, p = _as_floats(y_true, y_pred)
    nonzero = t != 0
    n_skipped = int((~nonzero).sum())
    if n_skipped:
        logger.warning(
            "mape: skipped %d row(s) with a zero actual — MAPE is undefined there. "
            "Consider 'mae' or 'poisson_deviance' for targets with true zeros.",
            n_skipped,
        )
    if not nonzero.any():
        raise GateConfigurationError(
            "every y_true value is zero, so MAPE is undefined for this dataset — "
            "use 'mae', 'rmse' or 'poisson_deviance' instead"
        )
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)[nonzero]
    return _mean(np.abs((t[nonzero] - p[nonzero]) / t[nonzero]), weights)


def ordinal_mae(y_true: Any, y_pred: Any, class_order: Any) -> float:
    """Mean absolute error in *rank* space.

    The point of an ordinal metric: for accept/refer/decline, predicting
    "decline" on an "accept" case is two steps wrong while "refer" is one.
    Plain accuracy scores both as a single mistake, which is exactly the
    distinction an underwriting gate needs to make.
    """
    t = to_ranks(y_true, class_order)
    p = to_ranks(y_pred, class_order)
    return float(np.mean(np.abs(t - p)))


def quadratic_kappa(y_true: Any, y_pred: Any, class_order: Any) -> float:
    """Cohen's kappa with quadratic weights — the standard ordinal agreement
    measure. 1.0 is perfect, 0.0 is chance, and negative is worse than
    chance. Disagreements are penalised by the *square* of their rank
    distance, so a two-step error costs four times a one-step one.
    """
    t = to_ranks(y_true, class_order).astype(int)
    p = to_ranks(y_pred, class_order).astype(int)
    n_classes = len(list(class_order))

    observed = np.zeros((n_classes, n_classes), dtype=float)
    for actual, predicted in zip(t, p):
        observed[actual, predicted] += 1

    # Expected counts under independence of the two marginals.
    actual_hist = np.bincount(t, minlength=n_classes).astype(float)
    pred_hist = np.bincount(p, minlength=n_classes).astype(float)
    expected = np.outer(actual_hist, pred_hist) / max(len(t), 1)

    indices = np.arange(n_classes)
    weights = (indices[:, None] - indices[None, :]) ** 2 / max((n_classes - 1) ** 2, 1)

    denominator = float(np.sum(weights * expected))
    if denominator == 0.0:
        # Everything landed in one cell: perfect agreement, or no signal to
        # disagree about. Either way there is no chance-correction to make.
        return 1.0 if float(np.sum(weights * observed)) == 0.0 else 0.0
    return 1.0 - float(np.sum(weights * observed)) / denominator


def _poisson_deviance_numpy(y_true: Any, y_pred: Any, sample_weight: Any = None) -> float:
    """Mean Poisson deviance — the right error measure for count targets such
    as claims frequency, where RMSE understates the cost of over-dispersion."""
    t, p = _as_floats(y_true, y_pred)
    if np.any(p <= 0):
        raise GateConfigurationError(
            "poisson_deviance requires strictly positive predictions "
            "(it takes their log); got a prediction <= 0"
        )
    if np.any(t < 0):
        raise GateConfigurationError("poisson_deviance requires non-negative y_true")
    # x*log(x/mu) -> 0 as x -> 0, so the zero-actual rows contribute only the
    # (mu - x) term. np.where alone would still evaluate log(0), hence the mask.
    safe_t = np.where(t > 0, t, 1.0)
    term = np.where(t > 0, t * np.log(safe_t / p), 0.0)
    return _mean(2.0 * (term - (t - p)), sample_weight)


def _lorenz_gini_metric(y_true: Any, y_pred: Any, sample_weight: Any = None) -> float:
    """The Lorenz Gini index, as a gateable metric.

    A thin adapter onto `bdp_model_gate.actuarial.lorenz_gini` so the number
    the performance gate scores and the number `ActualVsExpectedCheck`
    reports come from one implementation. Two would eventually disagree, and
    a report carrying two different Ginis is worse than one carrying none.
    """
    return lorenz_gini(y_true, y_pred, exposure=sample_weight)


BUILTIN_METRICS: dict[str, MetricSpec] = {
    # Ranking metrics stay binary-only: their multiclass forms need a full
    # (n, n_classes) probability matrix, which the y_pred contract does not
    # carry. Use a label-based metric, or ordinal_mae / quadratic_kappa.
    "roc_auc": MetricSpec("roc_auc", "roc_auc_score", needs_hard_labels=False, tasks=(BINARY,)),
    "average_precision": MetricSpec(
        "average_precision", "average_precision_score", needs_hard_labels=False, tasks=(BINARY,)
    ),
    "accuracy": MetricSpec(
        "accuracy", "accuracy_score", needs_hard_labels=True, fallback=_accuracy_numpy
    ),
    "balanced_accuracy": MetricSpec(
        "balanced_accuracy", "balanced_accuracy_score", needs_hard_labels=True
    ),
    "f1": MetricSpec("f1", "f1_score", needs_hard_labels=True, needs_average=True),
    "precision": MetricSpec(
        "precision", "precision_score", needs_hard_labels=True, needs_average=True
    ),
    "recall": MetricSpec("recall", "recall_score", needs_hard_labels=True, needs_average=True),
    # Ordinal. Both need context.class_order and are numpy-native.
    "ordinal_mae": MetricSpec(
        "ordinal_mae",
        "",
        needs_hard_labels=True,
        greater_is_better=False,
        tasks=(MULTICLASS,),
        needs_class_order=True,
    ),
    "quadratic_kappa": MetricSpec(
        "quadratic_kappa",
        "",
        needs_hard_labels=True,
        greater_is_better=True,
        tasks=(MULTICLASS,),
        needs_class_order=True,
    ),
    # Regression. All have numpy implementations, so they work on a core
    # install; scikit-learn is used when present for the ones it defines.
    "rmse": MetricSpec(
        "rmse",
        "",
        needs_hard_labels=False,
        fallback=_rmse_numpy,
        greater_is_better=False,
        tasks=(REGRESSION,),
        supports_weight=True,
    ),
    "mae": MetricSpec(
        "mae",
        "mean_absolute_error",
        needs_hard_labels=False,
        fallback=_mae_numpy,
        greater_is_better=False,
        tasks=(REGRESSION,),
        supports_weight=True,
    ),
    "mape": MetricSpec(
        "mape",
        "",
        needs_hard_labels=False,
        fallback=_mape_numpy,
        greater_is_better=False,
        tasks=(REGRESSION,),
        supports_weight=True,
    ),
    "poisson_deviance": MetricSpec(
        "poisson_deviance",
        "",
        needs_hard_labels=False,
        fallback=_poisson_deviance_numpy,
        greater_is_better=False,
        tasks=(REGRESSION,),
        supports_weight=True,
    ),
    "r2": MetricSpec(
        "r2",
        "r2_score",
        needs_hard_labels=False,
        fallback=_r2_numpy,
        greater_is_better=True,
        tasks=(REGRESSION,),
        supports_weight=True,
    ),
    # The pricing convention for discrimination. Numpy-native and never
    # taken from scikit-learn, which has no equivalent: `gini` there is a
    # tree-splitting impurity, an unrelated quantity with the same name.
    "lorenz_gini": MetricSpec(
        "lorenz_gini",
        "",
        needs_hard_labels=False,
        fallback=_lorenz_gini_metric,
        greater_is_better=True,
        tasks=(REGRESSION,),
        supports_weight=True,
    ),
}

#: Order tried by metric="auto", per task. For classification, roc_auc is
#: threshold-independent and the better default for the imbalanced problems
#: this gate typically sees, with accuracy as the base-install fallback. For
#: regression, r2 is scale-free — an RMSE default would mean nothing without
#: knowing whether the target is premiums in naira or claim counts.
AUTO_PREFERENCE_BY_TASK: dict[str, tuple[str, ...]] = {
    BINARY: ("roc_auc", "accuracy"),
    # balanced_accuracy, not plain accuracy: underwriting and fraud classes
    # are usually skewed, and accuracy flatters a model that never predicts
    # the rare class. Falls back to accuracy on a core install.
    MULTICLASS: ("balanced_accuracy", "accuracy"),
    REGRESSION: ("r2",),
}

#: Backwards-compatible alias for the binary preference order.
AUTO_PREFERENCE = AUTO_PREFERENCE_BY_TASK[BINARY]

AUTO = "auto"


@dataclass(frozen=True)
class ResolvedMetric:
    """A metric ready to call, plus how it was arrived at."""

    name: str
    fn: MetricFn
    needs_hard_labels: bool
    #: False for error metrics — gated with `max_error` rather than `min_score`.
    greater_is_better: bool = True
    #: True when metric="auto" could not use its first preference. The check
    #: surfaces this in its detail string so a reader of the report knows
    #: the score isn't the metric they'd expect by default.
    is_fallback: bool = False
    #: Set when scikit-learn *has* a form of this metric and it could not be
    #: loaded, so the numpy implementation stood in. Not set for a metric
    #: scikit-learn does not define at all: reporting "computed without
    #: scikit-learn" beside an RMSE on a machine where scikit-learn is
    #: installed is a false statement in a governance record.
    used_fallback_impl: bool = False
    #: True when `context.exposure` was bound in as a per-row weight. Named
    #: in the check's detail string either way, because "RMSE = 412" means
    #: two different things depending on this flag.
    exposure_weighted: bool = False


def validate_metric(metric: MetricSetting, task: str | None = None) -> None:
    """Cheap, import-free check that `metric` is a usable setting.

    Called when the check is constructed so a typo'd metric name fails at
    configuration time rather than midway through a gate run. Whether the
    metric's dependencies are actually installed is deliberately *not*
    checked here — see `resolve_metric`.
    """
    if callable(metric):
        return
    if not isinstance(metric, str):
        raise GateConfigurationError(
            f"performance.metric must be a metric name or a callable, got {type(metric).__name__}"
        )
    if metric == AUTO:
        return
    if metric not in BUILTIN_METRICS:
        valid = ", ".join([AUTO, *sorted(BUILTIN_METRICS)])
        raise GateConfigurationError(
            f"unknown performance.metric {metric!r} — valid options: {valid}"
        )
    if task is not None and task not in BUILTIN_METRICS[metric].tasks:
        applicable = ", ".join(sorted(m for m, sp in BUILTIN_METRICS.items() if task in sp.tasks))
        raise GateConfigurationError(
            f"performance.metric={metric!r} does not apply to a {task} task — "
            f"metrics available for {task}: {applicable}"
        )


#: Numpy-native metrics that take extra bound arguments rather than being
#: resolved from scikit-learn.
_ORDINAL_IMPLS: dict[str, Callable[..., float]] = {}


def _load_sklearn_metric(spec: MetricSpec) -> Callable[..., float] | None:
    # Returns Callable[..., float] rather than MetricFn: scikit-learn's
    # metrics accept extra keyword arguments such as `average`, which the
    # two-positional-argument MetricFn alias cannot express.
    try:
        from sklearn import metrics as sk_metrics
    except ImportError:
        return None
    return getattr(sk_metrics, spec.sklearn_fn, None)


def _weighted(fn: Callable[..., float], spec: MetricSpec, exposure: Any) -> tuple[Any, bool]:
    """Binds `exposure` in as a per-row weight, where the metric takes one.

    Returns the callable and whether the weight was actually applied. A
    metric that cannot take a weight is left alone and the caller reports
    that it was not weighted — silently dropping the exposure would put an
    unweighted number in a report the reader believes is weighted, which is
    the failure this library exists to avoid.
    """
    if exposure is None:
        return fn, False
    if not spec.supports_weight:
        logger.warning(
            "context.exposure was supplied but metric %r takes no per-row weight — "
            "scoring it unweighted. The report says so; pick a regression metric if "
            "you need the weighting.",
            spec.name,
        )
        return fn, False
    return functools.partial(fn, sample_weight=np.asarray(exposure, dtype=float)), True


def resolve_metric(
    metric: MetricSetting,
    task: str = BINARY,
    average: str = "macro",
    class_order: Any = None,
    exposure: Any = None,
) -> ResolvedMetric:
    """Turns a config value into a callable metric.

    Raises GateConfigurationError if an explicitly named metric can't be
    satisfied — the gate reports that as a blocking CHECK_ERROR rather than
    scoring the model with something the caller didn't ask for.

    `exposure` is bound in as a per-row weight for the metrics that accept
    one (the regression family). A custom callable never receives it: its
    signature is unknown, and passing an unexpected keyword would turn a
    working metric into a CHECK_ERROR.
    """
    validate_metric(metric, task)

    if callable(metric):
        name = getattr(metric, "__name__", None) or type(metric).__name__
        if exposure is not None:
            logger.warning(
                "context.exposure was supplied but performance.metric is your own "
                "callable %r, whose signature this library does not know — it is called "
                "unweighted. Apply the weighting inside your function if you need it.",
                name,
            )
        # A custom callable's direction is unknowable, so it is treated as
        # greater-is-better and gated with min_score. Negate inside your own
        # function, or name a built-in error metric, if that is wrong.
        return ResolvedMetric(name=name, fn=metric, needs_hard_labels=False)

    if metric == AUTO:
        return _resolve_auto(task, exposure)

    spec = BUILTIN_METRICS[metric]

    if spec.needs_class_order:
        if class_order is None:
            raise GateConfigurationError(
                f"performance.metric={metric!r} is an ordinal metric and needs "
                "context.class_order — the ordered class labels, least to most "
                'favourable, e.g. ["decline", "refer", "accept"]. Without an ordering '
                "there is no notion of how wrong a prediction is."
            )
        return ResolvedMetric(
            spec.name,
            functools.partial(_ORDINAL_IMPLS[spec.name], class_order=class_order),
            spec.needs_hard_labels,
            spec.greater_is_better,
        )

    fn = _load_sklearn_metric(spec)
    if fn is not None and spec.needs_average and task == MULTICLASS:
        # f1/precision/recall default to average="binary", which raises on a
        # multiclass target. macro weights every class equally, so a rarely
        # predicted "decline" counts as much as a common "accept".
        fn = functools.partial(fn, average=average)
    if fn is not None:
        weighted_fn, was_weighted = _weighted(fn, spec, exposure)
        return ResolvedMetric(
            spec.name,
            weighted_fn,
            spec.needs_hard_labels,
            spec.greater_is_better,
            exposure_weighted=was_weighted,
        )
    if spec.fallback is not None:
        stood_in = bool(spec.sklearn_fn)
        logger.debug(
            "scoring %r with the built-in numpy implementation%s",
            spec.name,
            " because scikit-learn is not installed" if stood_in else " (its only one)",
        )
        weighted_fn, was_weighted = _weighted(spec.fallback, spec, exposure)
        return ResolvedMetric(
            spec.name,
            weighted_fn,
            spec.needs_hard_labels,
            spec.greater_is_better,
            used_fallback_impl=stood_in,
            exposure_weighted=was_weighted,
        )
    raise GateConfigurationError(
        f"performance.metric={metric!r} requires scikit-learn — install it with "
        "`pip install bdp-model-gate[structured]`, or set performance.metric to "
        f"one of: {', '.join(sorted(m for m, s in BUILTIN_METRICS.items() if s.fallback))}"
    )


def _resolve_auto(task: str = BINARY, exposure: Any = None) -> ResolvedMetric:
    preference = AUTO_PREFERENCE_BY_TASK.get(task)
    if not preference:
        raise GateConfigurationError(
            f'performance.metric="auto" has no default for a {task} task — name a metric explicitly'
        )
    preferred = preference[0]
    for position, name in enumerate(preference):
        spec = BUILTIN_METRICS[name]
        fn = _load_sklearn_metric(spec)
        if fn is not None:
            weighted_fn, was_weighted = _weighted(fn, spec, exposure)
            return ResolvedMetric(
                spec.name,
                weighted_fn,
                spec.needs_hard_labels,
                spec.greater_is_better,
                is_fallback=position > 0,
                exposure_weighted=was_weighted,
            )
        if spec.fallback is not None:
            logger.warning(
                "performance.metric='auto': %r is unavailable (scikit-learn not installed) — "
                "scoring with %r instead. Set performance.metric explicitly to silence this, "
                "and remember min_score is interpreted against %r, not %r.",
                preferred,
                spec.name,
                spec.name,
                preferred,
            )
            weighted_fn, was_weighted = _weighted(spec.fallback, spec, exposure)
            return ResolvedMetric(
                spec.name,
                weighted_fn,
                spec.needs_hard_labels,
                spec.greater_is_better,
                is_fallback=position > 0,
                used_fallback_impl=bool(spec.sklearn_fn),
                exposure_weighted=was_weighted,
            )
    raise GateConfigurationError(  # pragma: no cover — accuracy always has a numpy fallback
        f"no metric in the {task} preference order could be resolved"
    )


def to_class_labels(y_pred: Any, class_order: Any = None) -> Any:
    """Reduces multiclass predictions to one label per row.

    Accepts predicted labels as-is. An (n, n_classes) probability matrix is
    reduced by argmax, mapped back through `class_order` when it is known so
    the result is comparable with `y_true` rather than a bare column index.
    """
    arr = np.asarray(y_pred)
    if arr.ndim == 1:
        return arr
    if arr.ndim != 2:
        raise GateConfigurationError(
            f"y_pred has {arr.ndim} dimensions; expected labels or an "
            "(n_rows, n_classes) probability matrix"
        )
    indices = np.argmax(arr, axis=1)
    if class_order is None:
        logger.warning(
            "y_pred looks like a probability matrix but context.class_order is unset — "
            "using column indices as class labels, which will not match y_true unless "
            "your classes are 0..k-1"
        )
        return indices
    ordered = list(class_order)
    if arr.shape[1] != len(ordered):
        raise GateConfigurationError(
            f"y_pred has {arr.shape[1]} columns but context.class_order lists "
            f"{len(ordered)} classes"
        )
    return np.array([ordered[i] for i in indices])


def to_hard_labels(y_pred: Any, threshold: float) -> Any:
    """Binarizes continuous scores for a metric that needs class labels.

    Values already restricted to {0, 1} are passed through untouched, so
    callers who supply hard labels aren't affected by `decision_threshold`.
    """
    arr = np.asarray(y_pred)
    if arr.dtype.kind not in "fc":
        return arr
    if np.all(np.isin(arr, (0, 1))):
        return arr.astype(int)
    logger.debug("binarizing continuous y_pred at decision_threshold=%s", threshold)
    return (arr >= threshold).astype(int)


_ORDINAL_IMPLS.update({"ordinal_mae": ordinal_mae, "quadratic_kappa": quadratic_kappa})


__all__ = [
    "AUTO",
    "AUTO_PREFERENCE",
    "AUTO_PREFERENCE_BY_TASK",
    "BUILTIN_METRICS",
    "MetricFn",
    "MetricSetting",
    "MetricSpec",
    "ResolvedMetric",
    "resolve_metric",
    "ordinal_mae",
    "quadratic_kappa",
    "to_class_labels",
    "to_hard_labels",
    "validate_metric",
]
