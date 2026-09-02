"""Execution context objects — the bundle of data/model a gate run needs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd


@dataclass
class StructuredGateContext:
    """Everything a structured-data governance check needs to run.

    Only `model`, `X`, `y_true`, and `y_pred` are required. Every other
    field is optional — omitting one simply causes the checks that depend
    on it to report NOT_APPLICABLE rather than raise or fail the gate.
    All inputs are validated eagerly by `ModelGate.run()` before any check
    executes; see `bdp_model_gate.core.validation`.

    Attributes:
        model: A fitted model exposing `.predict()` — scikit-learn, Keras,
            LightGBM, XGBoost's sklearn API, or your own class — or any
            plain callable. Optional if `predict_fn` is supplied instead,
            which is the route for a PyTorch module, a raw Booster, or a
            remote scoring endpoint where there is no model object at all.
        X: Feature dataframe used for validation/inference.
        y_true: Ground-truth labels for the validation set.
        y_pred: Model predictions on X (probabilities or hard labels, per your metric).
        protected_df: Dataframe of protected attributes (gender, region, etc.),
            row-aligned to X. Needed for the fairness checks.
        latencies_ms: Per-request inference latencies from a benchmark run, for
            the performance gate.
        cost_per_inference: Estimated or measured cost per inference, for the
            performance gate.
        model_card: Dict describing the model (legal_basis, use_case, etc.),
            for the compliance gate.
        generate_fn: `fn(str) -> str`, the entry point of any generative
            component sitting alongside the structured model — an explanation
            writer, a chatbot, a report generator. This is the **direct**
            injection surface: the payload arrives as the user turn.
        inject_fn: `fn(payload: str) -> str`, where the payload is placed
            wherever your pipeline puts *retrieved* content — a claim
            description, a customer email, a broker note, an uploaded
            document. This is the **indirect** surface, and it is the one
            that matters for a bank or an insurer: the realistic attack is
            not a user typing "ignore previous instructions", it is untrusted
            text arriving as data. The payload signature is what lets one
            corpus be fired at both surfaces, and the two are reported
            separately — a model hardened against the first and open to the
            second is the common case.
        canaries: Strings that must never appear in generated output: the
            system prompt itself, a planted fake PII record, an internal URL,
            a pricing rule. **This is what makes the injection check
            gateable.** "Did the model refuse?" is not decidable from a
            string; "did the secret come back out?" is, so a canary in a
            response is an unambiguous leak and blocks. Without canaries the
            leak attacks can only be routed to a human for judgement.
        judge_fn: `fn(prompt: str, response: str) -> bool`, returning True if
            the injection succeeded. Optional, for teams who want a model in
            the loop on the cases a canary cannot decide. Never required,
            never the default, and never a network call this library makes on
            its own initiative — it is called only for responses that are
            otherwise unjudgeable, and an exception from it degrades to
            "needs review" rather than failing the gate.
        expected_loss: Per-row expected loss (or technical/pure premium),
            row-aligned to X. Enables `LossRatioParityCheck`, which asks
            whether one group is charged a higher margin over its own
            expected cost — the actuarially meaningful fairness question for
            a pricing model, since risk-based premium differences are not by
            themselves discriminatory.
        exposure: Per-row exposure — earned vehicle-years, sum-insured-years,
            months on risk — row-aligned to X. Supply it when `y_true` and
            `y_pred` are *rates*: a policy written for one month and one
            written for twelve are not equal evidence about a claims rate,
            and an unweighted RMSE says they are. Omit it when the target is
            a per-policy total, where the exposure is already inside the
            value. It weights the regression metrics, the four regression
            fairness checks, and the whole actuarial suite; see
            `bdp_model_gate.actuarial` for the convention.
        baseline_pred: The incumbent's prediction for the same rows — the
            model being replaced, or last quarter's version of this one.
            Enables `DislocationCheck`, which answers the question a conduct
            review actually asks: not "is the new model better?" but "how
            many policyholders see a rise above 25%, and which ones?".
        predict_fn: `fn(DataFrame) -> array` returning point predictions.
            Takes precedence over `model`. The boundary is deliberately
            "DataFrame in, array out": your function owns tensor conversion,
            device placement and batching, so this library never imports a
            deep-learning framework.
        predict_proba_fn: `fn(DataFrame) -> array` returning class
            probabilities. `(n,)`, `(n, 1)` (Keras sigmoid) and `(n, 2)`
            (scikit-learn) are all accepted. Enables `CounterfactualFlipCheck`
            for models with no `.predict_proba()`.
        gradient_fn: `fn(DataFrame) -> array` of shape `(n_rows, n_features)`,
            aligned to `X.columns`. Lets a differentiable model drive a real
            targeted attack in `AdversarialRobustnessCheck` instead of the
            random-noise fallback.
        class_order: For multiclass problems, the class labels in ascending
            order of favourability, e.g. ["decline", "refer", "accept"].
            Supplying it marks the problem as **ordinal**, unlocking metrics
            that know a decline-vs-accept error costs more than a
            refer-vs-accept one. Omit it for a nominal problem where no
            ordering exists.
        favourable_classes: Which outcomes count as a positive result for
            demographic parity. Defaults to the most favourable class when
            `class_order` is given (and says so in the log); without either,
            the multiclass parity check reports NOT_APPLICABLE rather than
            picking one arbitrarily.
        X_train: The feature frame the model was **trained** on, if you have
            it. Unlocks the validation-methodology checks: rows shared
            between the two frames, and train-serve skew. It does not need to
            be row-aligned to anything — only its columns and distributions
            are read, never its labels.
        expected_features: The columns the model expects, in the order it
            expects them. Usually unnecessary — the list is read from
            `model.feature_names_in_`, a booster's own feature names, or
            `X_train.columns`. Supply it for a remote endpoint, where the
            schema is documented somewhere this library cannot reach.
        task: "auto" (default), "binary", "multiclass" or "regression".
            "auto" infers from y_true and logs what it inferred. Set it
            explicitly for anything you gate on: a claims-frequency target of
            0/1/2/3 is indistinguishable from a four-class problem by shape.
            See `bdp_model_gate.task`.
    """

    # All four carry defaults purely so `model` can be optional when
    # `predict_fn` is used. Positional order is unchanged, and X remains
    # required in practice — validation rejects a missing one.
    model: Any = None
    X: pd.DataFrame = None  # type: ignore[assignment]
    y_true: Sequence[Any] | None = None
    y_pred: Sequence[Any] | None = None
    protected_df: pd.DataFrame | None = None
    X_train: pd.DataFrame | None = None
    expected_features: Sequence[str] | None = None
    latencies_ms: Sequence[float] | None = None
    cost_per_inference: float | None = None
    model_card: dict | None = None
    generate_fn: Callable[[str], str] | None = None
    inject_fn: Callable[[str], str] | None = None
    canaries: Sequence[str] | None = None
    judge_fn: Callable[[str, str], bool] | None = None
    expected_loss: Sequence[float] | None = None
    exposure: Sequence[float] | None = None
    baseline_pred: Sequence[float] | None = None
    predict_fn: Callable[[pd.DataFrame], Any] | None = None
    predict_proba_fn: Callable[[pd.DataFrame], Any] | None = None
    gradient_fn: Callable[[pd.DataFrame], Any] | None = None
    class_order: Sequence[Any] | None = None
    favourable_classes: Sequence[Any] | None = None
    modality: str = "structured"
    task: str = "auto"
