"""The code in the prose, checked against the code in the package.

`mkdocs build --strict` catches a broken link and nothing else. The generated
API reference cannot drift, because mkdocstrings reads the docstrings; the
notebooks cannot drift, because `run_all.sh` executes them. The **prose
snippets** under `web/docs/` were the remaining gap: an API change could leave
forty-odd examples wrong and every build would stay green.

## What this checks, and what it deliberately does not

Most of those snippets are *fragments* — `config.actuarial.monotonic_features
= {...}` on its own, with `config` defined three pages earlier. Executing them
would mean inventing a fixture per snippet, and forty bespoke fixtures rot
faster than the thing they guard.

So this checks the part that actually drifts:

1. **Every block parses.** A snippet with a syntax error is already wrong.
2. **Every name imported from `bdp_model_gate` exists.** This is the one that
   catches a rename.
3. **Every keyword passed to a `*Config(...)` exists on that dataclass**, and
   every `config.<section>.<field>` assignment names a real field. Two
   releases added about twenty config fields between them, which makes this
   the likeliest thing in the docs to be stale.

It does not check that a snippet *does what the prose says*. Nothing does, and
the honest place to say so is here rather than in a docstring implying more.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

#: Prose that carries runnable examples. The notebooks are covered by
#: `examples/run_all.sh` and the API reference is generated, so neither is here.
PROSE = sorted(
    [*(REPO / "web" / "docs").rglob("*.md"), REPO / "README.md", REPO / "CONTRIBUTING.md"]
)

FENCE = re.compile(r"^```python\b[^\n]*\n(.*?)^```", re.M | re.S)

#: An HTML comment immediately above a fence exempts it. Invisible in the
#: rendered page, explicit in the source.
#:
#: A few snippets are deliberately pseudo-code — `def test_thing(...)` in the
#: contributing guide, `...` standing in for "and more model-card fields" in
#: concepts. Spelling those out to satisfy a parser would make the prose worse
#: for every reader in order to please one test, so they opt out and
#: `test_the_escape_hatch_stays_exceptional` keeps the exemption from
#: spreading.
PSEUDO_CODE = "<!-- pseudo-code: illustrative, not runnable -->"


def _blocks(path: Path) -> list[tuple[int, str]]:
    """Every runnable ```python block in `path`, with the line it starts on."""
    text = path.read_text()
    found = []
    for match in FENCE.finditer(text):
        preceding = text[: match.start()].rstrip().rsplit("\n", 1)[-1]
        if preceding.strip() == PSEUDO_CODE:
            continue
        found.append((text[: match.start()].count("\n") + 1, match.group(1)))
    return found


def _exempt_count() -> int:
    return sum(path.read_text().count(PSEUDO_CODE) for path in PROSE)


ALL_BLOCKS = [
    pytest.param(path, line, source, id=f"{path.relative_to(REPO)}:{line}")
    for path in PROSE
    for line, source in _blocks(path)
]


def test_the_escape_hatch_stays_exceptional():
    """The marker exists for genuine pseudo-code, not as a way to silence this
    file. If it starts spreading, the checks below stop meaning anything."""
    exempt = _exempt_count()
    assert exempt <= 4, (
        f"{exempt} snippet(s) are exempted as pseudo-code. Either make them run "
        "or make the case for raising this bound."
    )


def test_the_prose_carries_examples_worth_checking():
    """A guard on the guard: if the fence pattern stops matching, every test
    below passes by finding nothing."""
    assert len(ALL_BLOCKS) >= 30, f"only found {len(ALL_BLOCKS)} python block(s)"


@pytest.mark.parametrize("path,line,source", ALL_BLOCKS)
def test_every_documented_snippet_parses(path, line, source):
    try:
        ast.parse(source)
    except SyntaxError as exc:
        pytest.fail(
            f"{path.relative_to(REPO)}:{line} does not parse — {exc.msg} "
            f"on snippet line {exc.lineno}"
        )


def _imported_names(source: str):
    """`(module, name)` for everything imported from this package."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("bdp_model_gate"):
            for alias in node.names:
                yield node.module, alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("bdp_model_gate"):
                    yield alias.name, None


@pytest.mark.parametrize("path,line,source", ALL_BLOCKS)
def test_every_documented_import_exists(path, line, source):
    """The check that catches a rename.

    A snippet importing `bdp_model_gate.metrics.roc_auc` after it moved is
    wrong in the most expensive way: a reader copies it, it fails, and the
    documentation is what they stop trusting.
    """
    for module_name, symbol in _imported_names(source):
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            pytest.fail(f"{path.relative_to(REPO)}:{line} imports {module_name!r} — {exc}")
        if symbol is not None and not hasattr(module, symbol):
            pytest.fail(
                f"{path.relative_to(REPO)}:{line} imports {symbol!r} from "
                f"{module_name!r}, which no longer has it"
            )


def _config_classes() -> dict[str, type]:
    """Every config dataclass, by class name and by `GateConfig` field name."""
    from bdp_model_gate import config as config_module
    from bdp_model_gate.config import GateConfig

    classes = {
        name: obj
        for name, obj in vars(config_module).items()
        if isinstance(obj, type) and dataclasses.is_dataclass(obj)
    }
    classes.update(
        {field.name: field.type for field in dataclasses.fields(GateConfig)}  # type: ignore[misc]
    )
    # `field.type` is a string under `from __future__ import annotations`.
    resolved = {}
    for key, value in classes.items():
        resolved[key] = value if isinstance(value, type) else classes.get(str(value))
    return {k: v for k, v in resolved.items() if isinstance(v, type)}


def _field_names(cls: type) -> set[str]:
    names = {f.name for f in dataclasses.fields(cls)}
    # Deprecated aliases are properties rather than fields, and the docs
    # deliberately mention them in the deprecation table.
    names |= {n for n, v in vars(cls).items() if isinstance(v, property)}
    return names


@pytest.mark.parametrize("path,line,source", ALL_BLOCKS)
def test_every_documented_config_field_exists(path, line, source):
    """Two releases added about twenty config fields between them, which makes
    this the likeliest thing in the prose to be stale.

    Covers both shapes the docs use: a keyword to a constructor,
    `PerformanceConfig(metric="roc_auc")`, and an attribute assignment,
    `config.actuarial.monotonic_features = {...}`.
    """
    classes = _config_classes()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        # PerformanceConfig(min_score=0.85)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            cls = classes.get(node.func.id)
            if cls is not None and node.func.id.endswith("Config"):
                for keyword in node.keywords:
                    if keyword.arg and keyword.arg not in _field_names(cls):
                        pytest.fail(
                            f"{path.relative_to(REPO)}:{line} passes "
                            f"{keyword.arg!r} to {node.func.id}, which has no such field"
                        )

        # config.performance.min_score = 0.85
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if not isinstance(target, ast.Attribute):
                    continue
                parent = target.value
                if not (isinstance(parent, ast.Attribute) and isinstance(parent.value, ast.Name)):
                    continue
                cls = classes.get(parent.attr)
                if cls is not None and target.attr not in _field_names(cls):
                    pytest.fail(
                        f"{path.relative_to(REPO)}:{line} sets "
                        f"{parent.attr}.{target.attr}, which {cls.__name__} has no field for"
                    )
