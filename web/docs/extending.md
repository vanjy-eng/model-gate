# Extending

## Writing a check

Subclass `BaseCheck`, set the class attributes, implement `run(context)`,
return a list of `CheckResult`.

```python
from bdp_model_gate import BaseCheck, CheckResult


class ConstantFeatureCheck(BaseCheck):
    """Flags features that never vary in the validation set.

    A column that is constant here contributes nothing to any prediction, and
    usually means a join went wrong or a default is being filled in upstream.
    """

    name = "constant_features"
    category = "validation"  # validation | fairness | performance | compliance | security
    blocking = False  # worth a look, not a hard stop
    supported_tasks = ("binary", "multiclass", "regression")

    def __init__(self, max_constant_fraction: float = 0.0):
        self.max_constant_fraction = max_constant_fraction

    def run(self, context):
        constant = [c for c in context.X.columns if context.X[c].nunique(dropna=False) <= 1]
        fraction = len(constant) / context.X.shape[1]
        if fraction <= self.max_constant_fraction:
            return [
                CheckResult(
                    self.name,
                    self.category,
                    "OK",
                    f"all {context.X.shape[1]} features vary",
                    self.blocking,
                )
            ]
        return [
            CheckResult(
                self.name,
                self.category,
                "CONSTANT_FEATURE_RISK",
                detail=(
                    f"{feature} takes one value across the whole validation set — "
                    "it cannot be influencing any prediction"
                ),
                blocking=self.blocking,
                metadata={"feature": feature, "n_constant": len(constant)},
            )
            for feature in constant
        ]
```

Note the shape of the two returns. A check that finds nothing emits an
explicit `OK` rather than an empty list, so the report records that it ran —
a check that vanishes when it passes is indistinguishable from one that was
never registered.

Run it alongside the standard suite:

```python
from bdp_model_gate.structured import default_structured_checks

checks = default_structured_checks(config) + [ConstantFeatureCheck()]
report = ModelGate(checks=checks).run(context)
```

### Two attributes worth thinking about

**`blocking`** is the important one. `True` fails the build; `False` routes to
human review. Ask whether a false positive on your check should stop a
deployment at 2am. If not, it is non-blocking.

**`supported_tasks`** declares what your check can answer. It defaults to
every task, so a check written before 0.3.0 keeps working — but if yours only
makes sense for one, say so and the gate will report `NOT_APPLICABLE`
elsewhere rather than letting it produce a meaningless number.

### An optional third attribute: `plot`

Override `plot(self, context, results=None, ax=None)` and the report renders a
chart beneath your check's findings. Discovery is by override alone, so there
is nothing to register.

Draw only where your check collapses a distribution to a scalar **and the
shape is what a reader needs to judge** — a scalar that is genuinely a scalar
should be left alone. Take an optional `Axes` and return it, return `None`
when there is nothing to draw, and never raise. Full guidance, including why a
chart must never contradict the number beside it, is in
[Plots](reference/plots.md).

### A broken check is contained

`ModelGate` catches exceptions per check and converts them to a blocking
`CHECK_ERROR`, so one bad check cannot take down the suite — the others still
run and the pipeline still stops.

That means a `CHECK_ERROR` in a report is *always* a bug or an explicit
expectation, never noise.

## Plugins

A separate package can register checks without forking, via the
`bdp_model_gate.checks` entry-point group:

```toml
# in your plugin package's pyproject.toml
[project.entry-points."bdp_model_gate.checks"]
my_check = "my_package.checks:MyCustomCheck"
```

Once installed alongside `bdp-model-gate`, `default_structured_checks()` picks
it up automatically. Pass `include_plugins=False` to opt out.

Each entry point must resolve to a `BaseCheck` **subclass**, not an instance,
and is constructed with no arguments — so a plugin needing configuration
should read it from its own defaults or environment. A plugin that fails to
import, or does not resolve to a `BaseCheck`, is logged and skipped rather
than crashing the gate.

```python
from bdp_model_gate.registry import discover_plugin_checks

print(discover_plugin_checks())
```

## Contributing to the library itself

Everything above is about extending the gate from *your* code. To change the
library, see
[`CONTRIBUTING.md`](https://github.com/vanjy-eng/model-gate/blob/main/CONTRIBUTING.md)
— development setup, the testing standards, and how a check or a plot gets
reviewed.

```bash
pip install -e ".[dev,structured,plots,yaml,toml]"

ruff check .        # lint
ruff format .       # format
mypy bdp_model_gate # types
pytest -q           # tests, 85% coverage floor
```

`.pre-commit-config.yaml` runs the same hygiene on every commit. CI runs lint,
types and the test suite across Python 3.9–3.13, plus a core-install job with
no `structured` extra — that job is what keeps the graceful-degradation paths
honest.
