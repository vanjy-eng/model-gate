"""Command-line entry point for running the gate in a CI/CD pipeline.

Installed as the `bdp-model-gate` console script. Exit codes are chosen so
a pipeline can distinguish "safe to proceed", "needs a human", and "hard
stop":

    0 -> PASS           safe to proceed automatically
    2 -> NEEDS_REVIEW    route to a manual approval step, don't auto-deploy
    1 -> BLOCKED         hard fail the pipeline

Example (Azure Pipelines / GitHub Actions):

    bdp-model-gate \
        --model model.joblib \
        --data validation.csv \
        --target-col label \
        --protected protected.csv \
        --model-card model_card.json \
        --cost-per-inference 0.0008 \
        --output gate_report.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from ._logging import configure_logging, get_logger
from .exceptions import BDPModelGateError
from .metrics import AUTO, BUILTIN_METRICS
from .model import ModelAdapter
from .task import ALL_TASKS
from .task import AUTO as TASK_AUTO

logger = get_logger("cli")

#: Config-file keys that have been renamed. Still applied (via the property
#: alias on the config dataclass), but called out in the log — a silently
#: honoured deprecated key is how a stale threshold survives a rename.
DEPRECATED_CONFIG_KEYS = {
    ("performance", "min_accuracy"): "min_score",
}


def _load_model(path: str):
    import joblib

    return joblib.load(path)


def _split_labels(value: str | None) -> list[str] | None:
    """Parses a comma-separated class-label list from the command line."""
    if not value:
        return None
    labels = [part.strip() for part in value.split(",") if part.strip()]
    if not labels:
        raise BDPModelGateError(f"no class labels parsed from {value!r}")
    return labels


def _load_via_loader(spec: str):
    """Imports and calls a `"package.module:factory"` loader.

    joblib can only read pickles, which rules out `.pt` checkpoints, Keras
    SavedModel directories, ONNX graphs and remote endpoints. Rather than
    take a dependency on every framework, the CLI lets you name a function
    that returns something callable — your loader does the importing.

        # mypkg/serving.py
        def load_scorer():
            net = torch.load("model.pt"); net.eval()
            return lambda df: net(torch.tensor(df.values).float()).detach().numpy()

        bdp-model-gate --model-loader "mypkg.serving:load_scorer" ...
    """
    if ":" not in spec:
        raise BDPModelGateError(f"--model-loader must be 'package.module:factory', got {spec!r}")
    module_name, _, attr = spec.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise BDPModelGateError(
            f"could not import {module_name!r} for --model-loader — is it on PYTHONPATH? ({exc})"
        ) from exc
    try:
        factory = getattr(module, attr)
    except AttributeError as exc:
        raise BDPModelGateError(f"module {module_name!r} has no attribute {attr!r}") from exc
    if not callable(factory):
        raise BDPModelGateError(f"{spec!r} is not callable")

    loaded = factory()
    if not (callable(loaded) or hasattr(loaded, "predict")):
        raise BDPModelGateError(
            f"{spec!r} returned a {type(loaded).__name__}, which is neither callable nor "
            "has a .predict() method — return a model or a scoring function"
        )
    logger.info("loaded model via %s -> %s", spec, type(loaded).__name__)
    return loaded


def _predict(model, X: pd.DataFrame, task: str):
    """Produces y_pred appropriate to the task.

    Only *binary* classification wants `predict_proba(X)[:, 1]`. Taking
    column 1 of a multiclass model's probabilities silently yields P(class 1)
    — a real number that scores as though it were the positive class, which
    is how a gate reports a confident, meaningless verdict. Regression has no
    predict_proba at all.
    """
    adapter = ModelAdapter(model=model)
    wants_proba = task in (TASK_AUTO, "binary")
    if wants_proba and adapter.can_predict_proba:
        try:
            return adapter.predict_positive_proba(X)
        except BDPModelGateError as exc:
            # Multi-column output means it is not a binary classifier.
            logger.info("%s — falling back to .predict(). Pass --task to be explicit.", exc)
    return adapter.predict(X)


def _load_structured_config_file(path: str) -> dict[str, Any]:
    """Loads threshold overrides from JSON, YAML, or TOML, based on extension."""
    suffix = Path(path).suffix.lower()
    text = Path(path).read_text()

    if suffix == ".json":
        return json.loads(text)

    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise BDPModelGateError(
                "reading a YAML config requires PyYAML — install with `pip install pyyaml`"
            ) from exc
        return yaml.safe_load(text) or {}

    if suffix == ".toml":
        try:
            import tomllib  # Python 3.11+
        except ModuleNotFoundError:
            try:
                import tomli as tomllib  # type: ignore[no-redef]  # backport for < 3.11
            except ImportError as exc:
                raise BDPModelGateError(
                    "reading a TOML config on Python < 3.11 requires tomli — "
                    "install with `pip install tomli`"
                ) from exc
        return tomllib.loads(text)

    raise BDPModelGateError(
        f"unrecognized config file extension '{suffix}' — use .json, .yaml/.yml, or .toml"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bdp-model-gate",
        description="Run the BDP Model Gate pre-deployment governance gate against a trained model.",
    )
    model_source = parser.add_mutually_exclusive_group(required=True)
    model_source.add_argument("--model", help="Path to a joblib-serialized model")
    model_source.add_argument(
        "--model-loader",
        help=(
            "A 'package.module:factory' function returning a model or a "
            "fn(DataFrame) -> array. Use this for anything joblib cannot unpickle — "
            "PyTorch checkpoints, Keras SavedModel, ONNX, or a remote scoring endpoint. "
            "Your loader does the framework import, so this package needs no "
            "deep-learning dependency."
        ),
    )
    parser.add_argument("--data", required=True, help="Path to a CSV of validation data")
    parser.add_argument("--target-col", required=True, help="Column name of the ground-truth label")
    parser.add_argument(
        "--protected", help="Path to a CSV of protected attributes, row-aligned to --data"
    )
    parser.add_argument(
        "--train-data",
        help=(
            "Path to a CSV of the TRAINING features. Unlocks the validation checks "
            "that need both frames: rows shared between the two splits, and "
            "train-serve skew. Only its columns and distributions are read — the "
            "target column is dropped if present, and no labels are used"
        ),
    )
    parser.add_argument("--model-card", help="Path to a JSON model card")
    parser.add_argument(
        "--latencies", help="Path to a text/CSV file of per-request latencies in ms, one per line"
    )
    parser.add_argument("--cost-per-inference", type=float, help="Estimated cost per inference")
    parser.add_argument(
        "--task",
        choices=[TASK_AUTO, *ALL_TASKS],
        default=TASK_AUTO,
        help=(
            "Prediction task (default: auto, which infers from the target column and "
            "logs what it inferred). Set explicitly for anything you gate on — a count "
            "target such as claims frequency looks identical to a multiclass one."
        ),
    )
    parser.add_argument(
        "--class-order",
        help=(
            "Comma-separated class labels in ascending order of favourability, e.g. "
            "'decline,refer,accept'. Marks a multiclass problem as ordinal, which "
            "unlocks the ordinal_mae and quadratic_kappa metrics and rank-aware "
            "robustness — a decline-vs-accept error then counts as worse than "
            "refer-vs-accept."
        ),
    )
    parser.add_argument(
        "--favourable-classes",
        help=(
            "Comma-separated labels counting as a positive outcome for demographic "
            "parity, e.g. 'accept'. Defaults to the last entry of --class-order."
        ),
    )
    parser.add_argument(
        "--average",
        choices=["macro", "micro", "weighted"],
        help=(
            "Multiclass averaging for f1/precision/recall (default: macro, which "
            "weights every class equally)"
        ),
    )
    parser.add_argument(
        "--expected-loss-col",
        help=(
            "Column in --data holding a per-row expected loss or technical premium. "
            "Enables the loss-ratio parity fairness check for regression models."
        ),
    )
    parser.add_argument(
        "--exposure-col",
        help=(
            "Column in --data holding a per-row exposure — earned vehicle-years, "
            "months on risk, sum-insured-years. Weights the regression metrics and the "
            "actuarial checks, so a one-month policy stops counting as much as a "
            "twelve-month one. Supply it when the target is a rate; leave it out when "
            "the target is a per-policy total"
        ),
    )
    parser.add_argument(
        "--baseline-col",
        help=(
            "Column in --data holding the incumbent model's prediction for the same "
            "rows. Enables the dislocation check: how much of the book moves by more "
            "than the tolerance, and which group carries it"
        ),
    )
    parser.add_argument(
        "--metric",
        choices=[AUTO, *sorted(BUILTIN_METRICS)],
        help=(
            "Metric the model is scored on for the performance gate "
            f"(default: {AUTO}, which prefers roc_auc and falls back to accuracy "
            "with a warning if scikit-learn is unavailable)"
        ),
    )
    parser.add_argument(
        "--min-score",
        type=float,
        help=(
            "Minimum acceptable value of --metric, for higher-is-better metrics "
            "(roc_auc, f1, r2, ...); below this the gate blocks"
        ),
    )
    parser.add_argument(
        "--max-error",
        type=float,
        help=(
            "Maximum acceptable value of --metric, for error metrics where lower is "
            "better (rmse, mae, mape, poisson_deviance); above this the gate blocks. "
            "Required when one of those metrics is selected — there is no default, "
            "since a sensible ceiling depends on the scale of your target."
        ),
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        help=(
            "Probability cutoff used to turn continuous predictions into class "
            "labels for metrics that need them (accuracy, f1, precision, recall). "
            "Ignored by ranking metrics like roc_auc."
        ),
    )
    parser.add_argument(
        "--config", help="Path to a JSON, YAML, or TOML file of threshold overrides"
    )
    parser.add_argument(
        "--output", default="gate_report.json", help="Where to write the JSON report"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug-level logging")
    return parser


def _apply_config_overrides(gate_config, overrides: dict[str, Any]):
    for section, values in overrides.items():
        sub_config = getattr(gate_config, section, None)
        if sub_config is None:
            logger.warning("config file references unknown section '%s' — ignoring", section)
            continue
        for key, value in values.items():
            replacement = DEPRECATED_CONFIG_KEYS.get((section, key))
            if replacement is not None:
                logger.warning(
                    "config key '%s.%s' is deprecated — rename it to '%s.%s'. Applying it for now.",
                    section,
                    key,
                    section,
                    replacement,
                )
            elif not hasattr(sub_config, key):
                logger.warning("config section '%s' has no field '%s' — ignoring", section, key)
                continue
            setattr(sub_config, key, value)
    return gate_config


def _apply_cli_overrides(gate_config, args):
    """CLI flags win over the --config file, so a pipeline can pin a
    threshold inline without maintaining a separate config file."""
    for flag_name, config_field in (
        ("metric", "metric"),
        ("min_score", "min_score"),
        ("max_error", "max_error"),
        ("average", "average"),
        ("decision_threshold", "decision_threshold"),
    ):
        value = getattr(args, flag_name, None)
        if value is not None:
            setattr(gate_config.performance, config_field, value)
            logger.debug("performance.%s set to %r from --%s", config_field, value, flag_name)
    return gate_config


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging(verbose=args.verbose)

    try:
        from bdp_model_gate import GateConfig, ModelGate, StructuredGateContext
        from bdp_model_gate.exceptions import GateValidationError
        from bdp_model_gate.structured import default_structured_checks

        model = _load_model(args.model) if args.model else _load_via_loader(args.model_loader)
        df = pd.read_csv(args.data)
        y_true = df[args.target_col].values
        drop_cols = [args.target_col]

        # Every one of these is a column of --data that is *not* a feature, so
        # each is read out and then dropped: leaving an exposure or a baseline
        # premium in X would hand the model its own answer at scoring time,
        # which is the leak `LeakageCheck` exists to find.
        def _side_column(flag_name: str, column: str | None):
            if not column:
                return None
            if column not in df.columns:
                raise BDPModelGateError(f"--{flag_name} {column!r} is not a column in {args.data}")
            drop_cols.append(column)
            return df[column].to_numpy()

        expected_loss = _side_column("expected-loss-col", args.expected_loss_col)
        exposure = _side_column("exposure-col", args.exposure_col)
        baseline_pred = _side_column("baseline-col", args.baseline_col)

        X = df.drop(columns=drop_cols)
        y_pred = _predict(model, X, args.task)

        protected_df = pd.read_csv(args.protected) if args.protected else None

        X_train = None
        if args.train_data:
            X_train = pd.read_csv(args.train_data)
            # Drop whatever the validation frame dropped, so the two are
            # compared on features alone. Overlap detection would otherwise
            # miss a shared row whose label column happened to differ.
            X_train = X_train.drop(columns=[c for c in drop_cols if c in X_train.columns])
        model_card = json.load(open(args.model_card)) if args.model_card else None

        latencies_ms = None
        if args.latencies:
            with open(args.latencies) as f:
                latencies_ms = [float(line.strip()) for line in f if line.strip()]

        gate_config = GateConfig()
        if args.config:
            overrides = _load_structured_config_file(args.config)
            gate_config = _apply_config_overrides(gate_config, overrides)
        gate_config = _apply_cli_overrides(gate_config, args)

        context = StructuredGateContext(
            model=model,
            X=X,
            y_true=y_true,
            y_pred=y_pred,
            protected_df=protected_df,
            X_train=X_train,
            latencies_ms=latencies_ms,
            cost_per_inference=args.cost_per_inference,
            model_card=model_card,
            expected_loss=expected_loss,
            exposure=exposure,
            baseline_pred=baseline_pred,
            task=args.task,
            class_order=_split_labels(args.class_order),
            favourable_classes=_split_labels(args.favourable_classes),
        )

        report = ModelGate(checks=default_structured_checks(gate_config)).run(context)

    except GateValidationError as exc:
        logger.error("invalid gate input: %s", exc)
        print(f"Invalid input: {exc}", file=sys.stderr)
        return 1
    except BDPModelGateError as exc:
        logger.error("configuration error: %s", exc)
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        logger.error("file not found: %s", exc)
        print(f"File not found: {exc}", file=sys.stderr)
        return 1

    report.to_json(args.output)
    print(report.summary())
    print(f"Full report written to {args.output}")

    if report.gate_status == "BLOCKED":
        return 1
    if report.gate_status == "NEEDS_REVIEW":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
