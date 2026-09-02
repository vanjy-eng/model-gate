"""Eager input validation for StructuredGateContext.

Runs once, before any check executes. A check-writer can assume the
context it receives is well-formed: X is a non-empty DataFrame, y_true/
y_pred/X are aligned in length, the model exposes .predict(), and any
optional inputs that are present are internally consistent. If something's
wrong, GateValidationError is raised with a message that names the field
and the problem, rather than letting a check fail with a confusing
downstream exception (e.g. a shape mismatch surfacing inside SHAP).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .._logging import get_logger
from ..classes import resolve_favourable, validate_class_order
from ..exceptions import GateConfigurationError, GateValidationError
from ..injection import CORPUS
from ..model import ModelAdapter
from ..task import REGRESSION, resolve_task, validate_task

if TYPE_CHECKING:
    from .context import StructuredGateContext

logger = get_logger("validation")

#: Shorter than this and a canary matches by accident. Four characters of
#: anything appear somewhere in ordinary prose, and a false leak report is
#: exactly the confident-wrong-verdict this library exists to avoid.
MIN_CANARY_LENGTH = 8


def validate_structured_context(context: StructuredGateContext) -> None:
    _validate_task(context)
    _validate_classes(context)
    _validate_model(context)
    _validate_features(context)
    _validate_labels(context)
    _validate_expected_loss(context)
    _validate_exposure(context)
    _validate_baseline_pred(context)
    _validate_protected_df(context)
    _validate_model_card(context)
    _validate_performance_inputs(context)
    _validate_generate_fn(context)
    _validate_canaries(context)


def _validate_model(context: StructuredGateContext) -> None:
    for name in ("predict_fn", "predict_proba_fn", "gradient_fn"):
        fn = getattr(context, name, None)
        if fn is not None and not callable(fn):
            raise GateValidationError(f"context.{name} must be callable, got {type(fn).__name__}")

    if ModelAdapter.from_context(context).can_predict:
        return

    if context.model is None:
        raise GateValidationError(
            "no model supplied: pass either context.model (anything with .predict(), "
            "or a callable) or context.predict_fn=lambda df: ... for a model this "
            "library cannot call directly, such as a PyTorch module or a remote endpoint"
        )
    raise GateValidationError(
        f"context.model is a {type(context.model).__name__}, which has no .predict() "
        "method and is not callable — supply context.predict_fn instead"
    )


def _validate_features(context: StructuredGateContext) -> None:
    if context.X is None:
        raise GateValidationError("context.X is required — no feature data to evaluate")
    if not isinstance(context.X, pd.DataFrame):
        raise GateValidationError(
            f"context.X must be a pandas DataFrame, got {type(context.X).__name__}"
        )
    if context.X.empty:
        raise GateValidationError("context.X is empty — no rows to evaluate")
    if context.X.shape[1] == 0:
        raise GateValidationError("context.X has no columns")


def _validate_labels(context: StructuredGateContext) -> None:
    n_rows = len(context.X)
    for label_name, label_value in (("y_true", context.y_true), ("y_pred", context.y_pred)):
        if label_value is None:
            continue
        length = len(label_value)
        if length != n_rows:
            raise GateValidationError(
                f"context.{label_name} has length {length}, but context.X has "
                f"{n_rows} rows — they must be aligned"
            )

    if context.y_true is None:
        return

    task = resolve_task(context)
    unique_labels = pd.unique(np.asarray(context.y_true))

    if len(unique_labels) < 2:
        detail = (
            "a constant target has no variance to explain, so every regression metric is degenerate"
            if task == REGRESSION
            else "most checks (AUC, disparate impact) need at least two classes present"
        )
        raise GateValidationError(
            f"context.y_true has only one unique value ({unique_labels!r}) — {detail}"
        )

    if task == REGRESSION:
        for name, values in (("y_true", context.y_true), ("y_pred", context.y_pred)):
            if values is None:
                continue
            arr = np.asarray(values)
            if arr.dtype.kind not in "iuf":
                raise GateValidationError(
                    f"context.{name} must be numeric for a regression task, got dtype "
                    f"{arr.dtype} — set context.task explicitly if this is really "
                    "a classification problem"
                )
            if not np.all(np.isfinite(arr.astype(float))):
                raise GateValidationError(
                    f"context.{name} contains NaN or infinite values, which every "
                    "regression metric would propagate"
                )


def _validate_task(context: StructuredGateContext) -> None:
    validate_task(getattr(context, "task", "auto"))


def _validate_classes(context: StructuredGateContext) -> None:
    class_order = getattr(context, "class_order", None)
    try:
        validate_class_order(class_order, getattr(context, "y_true", None))
        resolve_favourable(
            getattr(context, "favourable_classes", None),
            class_order,
            resolve_task(context),
        )
    except GateConfigurationError as exc:
        # Surface class-structure problems the same way as any other bad
        # input: eagerly, before a check trips over them mid-run.
        raise GateValidationError(str(exc)) from exc


def _validate_expected_loss(context: StructuredGateContext) -> None:
    expected_loss = getattr(context, "expected_loss", None)
    if expected_loss is None:
        return
    arr = np.asarray(expected_loss)
    if arr.dtype.kind not in "iuf":
        raise GateValidationError(f"context.expected_loss must be numeric, got dtype {arr.dtype}")
    if len(arr) != len(context.X):
        raise GateValidationError(
            f"context.expected_loss has length {len(arr)}, but context.X has "
            f"{len(context.X)} rows — they must be row-aligned"
        )
    if np.any(np.asarray(arr, dtype=float) < 0):
        raise GateValidationError(
            "context.expected_loss contains negative values — an expected loss cannot be below zero"
        )


def _validate_exposure(context: StructuredGateContext) -> None:
    """Exposure is a weight, so the ways it can be wrong are specific.

    A negative exposure is meaningless; an all-zero column would silently
    turn every weighted mean into NaN, which would read in the report as
    "could not be measured" rather than "you passed zeros"; and a NaN weight
    poisons every total it enters. All three are refused here rather than
    surfacing as an unexplained skip six checks later.
    """
    exposure = getattr(context, "exposure", None)
    if exposure is None:
        return
    arr = np.asarray(exposure)
    if arr.dtype.kind not in "iuf":
        raise GateValidationError(f"context.exposure must be numeric, got dtype {arr.dtype}")
    values = arr.astype(float)
    if len(values) != len(context.X):
        raise GateValidationError(
            f"context.exposure has length {len(values)}, but context.X has "
            f"{len(context.X)} rows — they must be row-aligned"
        )
    if not np.all(np.isfinite(values)):
        raise GateValidationError(
            "context.exposure contains NaN or infinite values — a weight that is not a "
            "number propagates into every exposure-weighted total"
        )
    if np.any(values < 0):
        raise GateValidationError(
            "context.exposure contains negative values — exposure is a measure of time "
            "or amount at risk and cannot be below zero"
        )
    if float(values.sum()) <= 0:
        raise GateValidationError(
            "every context.exposure value is zero, so nothing carries any weight — omit "
            "exposure entirely if the target is a per-policy total rather than a rate"
        )

    task = resolve_task(context)
    if task != REGRESSION:
        logger.warning(
            "context.exposure was supplied but the task resolved to %r. Exposure weighting "
            "applies to the regression metrics and the actuarial checks; the "
            "classification checks ignore it, and the report says so.",
            task,
        )


def _validate_baseline_pred(context: StructuredGateContext) -> None:
    baseline = getattr(context, "baseline_pred", None)
    if baseline is None:
        return
    arr = np.asarray(baseline)
    if arr.dtype.kind not in "iuf":
        raise GateValidationError(
            f"context.baseline_pred must be numeric, got dtype {arr.dtype} — it is the "
            "incumbent model's prediction for the same rows"
        )
    if len(arr) != len(context.X):
        raise GateValidationError(
            f"context.baseline_pred has length {len(arr)}, but context.X has "
            f"{len(context.X)} rows — they must be row-aligned"
        )
    if not np.all(np.isfinite(arr.astype(float))):
        raise GateValidationError(
            "context.baseline_pred contains NaN or infinite values, so the relative "
            "change against it is undefined"
        )


def _validate_protected_df(context: StructuredGateContext) -> None:
    if context.protected_df is None:
        return
    if not isinstance(context.protected_df, pd.DataFrame):
        raise GateValidationError(
            f"context.protected_df must be a pandas DataFrame, got "
            f"{type(context.protected_df).__name__}"
        )
    if len(context.protected_df) != len(context.X):
        raise GateValidationError(
            f"context.protected_df has {len(context.protected_df)} rows, but "
            f"context.X has {len(context.X)} rows — they must be row-aligned"
        )
    all_nan_cols = [c for c in context.protected_df.columns if context.protected_df[c].isna().all()]
    if all_nan_cols:
        raise GateValidationError(
            f"context.protected_df column(s) {all_nan_cols} are entirely NaN — "
            "fairness checks cannot group on an empty attribute"
        )


def _validate_model_card(context: StructuredGateContext) -> None:
    if context.model_card is None:
        return
    if not isinstance(context.model_card, dict):
        raise GateValidationError(
            f"context.model_card must be a dict, got {type(context.model_card).__name__}"
        )


def _validate_performance_inputs(context: StructuredGateContext) -> None:
    if context.latencies_ms is None:
        return
    if len(context.latencies_ms) == 0:
        raise GateValidationError("context.latencies_ms is an empty sequence")
    if any(v < 0 for v in context.latencies_ms):
        raise GateValidationError("context.latencies_ms contains negative values")


def _validate_generate_fn(context: StructuredGateContext) -> None:
    for name in ("generate_fn", "inject_fn", "judge_fn"):
        fn = getattr(context, name, None)
        if fn is not None and not callable(fn):
            raise GateValidationError(f"context.{name} must be callable, got {type(fn).__name__}")


def _validate_canaries(context: StructuredGateContext) -> None:
    """Canaries are the one input whose *contents* decide whether a check works.

    Three ways to get them wrong, all of which produce a confidently wrong
    verdict rather than an error, which is why they are refused here:

    A **short** canary matches by accident. `"NIN"` appears in ordinary prose
    and would report a leak on every response.

    A canary that appears in the **corpus** cannot distinguish a leak from the
    model quoting the attack back at you. The corpus is fixed and shipped, so
    this is checkable rather than a matter of care.

    A **blank** one matches everything.
    """
    canaries = getattr(context, "canaries", None)
    if canaries is None:
        return
    if isinstance(canaries, (str, bytes)):
        raise GateValidationError(
            "context.canaries must be a sequence of strings, not a single string — "
            "a bare string would be iterated character by character, and every "
            "response contains the letter 'e'"
        )

    values = list(canaries)
    if not values:
        raise GateValidationError(
            "context.canaries is empty — omit it entirely rather than passing an "
            "empty sequence, so the report says the leak checks could not be judged"
        )

    for canary in values:
        if not isinstance(canary, str):
            raise GateValidationError(
                f"every context.canaries entry must be a string, got {type(canary).__name__}"
            )
        stripped = canary.strip()
        if not stripped:
            raise GateValidationError(
                "context.canaries contains a blank entry, which matches everything"
            )
        if len(stripped) < MIN_CANARY_LENGTH:
            raise GateValidationError(
                f"context.canaries entry {canary!r} is shorter than "
                f"{MIN_CANARY_LENGTH} characters. A short canary matches by accident "
                "and would report a leak on an innocent response — plant something "
                "distinctive, such as a fake policy number or a sentence from the "
                "system prompt"
            )

    corpus_text = "\n".join(attack.payload for attack in CORPUS).lower()
    for canary in values:
        if canary.strip().lower() in corpus_text:
            raise GateValidationError(
                f"context.canaries entry {canary!r} appears in the built-in injection "
                "corpus, so a response quoting the attack back would be indistinguishable "
                "from a real leak. Plant a canary of your own instead"
            )
