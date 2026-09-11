"""Package-level invariants: the things that go stale silently.

The version is declared in four places — `pyproject.toml`, `__init__.py`, the
changelog heading and the website — and a release tag is cut against them, so
they must not drift apart. The website ones are here because they already had:
the landing page advertised 0.4.1 for two releases, and its "sixteen checks"
grid listed thirteen.

A release convention that depends on someone remembering is not a convention.
"""

import re
from pathlib import Path

import bdp_model_gate

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _pyproject_version() -> str:
    """Read the version straight out of pyproject.toml rather than via
    importlib.metadata — an editable install caches its metadata at install
    time, so a fresh bump would compare against a stale value and pass."""
    match = re.search(r'^version = "([^"]+)"', PYPROJECT.read_text(), re.M)
    assert match, "no version declared in pyproject.toml"
    return match.group(1)


def test_dunder_version_matches_pyproject():
    assert bdp_model_gate.__version__ == _pyproject_version()


def test_changelog_documents_the_current_version():
    changelog = (PYPROJECT.parent / "CHANGELOG.md").read_text()
    assert f"## [{_pyproject_version()}]" in changelog


#: Files carrying a hand-written version stamp, and the template each uses.
#: `web/README.md` is on this list because it was missed: the first version of
#: this guard covered only the landing page, and `web/README.md` then sat at
#: 0.5.1 through the whole of 0.5.2's development. A guard with a hole in it is
#: worse than none, because it is trusted.
VERSION_STAMPS = (
    ("web/landing/index.html", '<span class="version">{version}</span>'),
    ("web/landing/index.html", "<span>BDP Model Gate {version}</span>"),
    ("web/README.md", "Documents **bdp-model-gate {version}**."),
)


def test_the_website_advertises_the_current_version():
    """The most visible number on the site, and the easiest to forget."""
    version = _pyproject_version()
    for relative, template in VERSION_STAMPS:
        expected = template.format(version=version)
        text = (PYPROJECT.parent / relative).read_text()
        assert expected in text, f"{relative} does not advertise {version} — expected {expected!r}"


def test_no_stale_version_is_left_on_the_landing_page():
    """A leftover elsewhere on the page misleads as much as a stale masthead."""
    landing = (PYPROJECT.parent / "web" / "landing" / "index.html").read_text()
    version = _pyproject_version()
    stale = {v for v in re.findall(r"\b\d+\.\d+\.\d+\b", landing) if v != version}
    assert not stale, f"stale version(s) left on the landing page: {sorted(stale)}"


def test_every_default_check_is_documented():
    """A check nobody can find is a check nobody runs.

    The reference page is the contract: if a check ships in the default suite,
    its name appears there. This is deliberately name-level rather than
    prose-level — it cannot judge whether the description is any good, only
    that the check was not added and then forgotten.
    """
    from bdp_model_gate.structured import default_structured_checks

    page = (PYPROJECT.parent / "web" / "docs" / "reference" / "checks.md").read_text()
    missing = [
        check.name
        for check in default_structured_checks(include_plugins=False)
        if check.name not in page
    ]
    assert not missing, (
        f"undocumented check(s): {missing} — add them to "
        "web/docs/reference/checks.md and the grid in web/landing/index.html"
    )


def test_no_runtime_pep604_without_future_import():
    """Guards the bug that took down the whole Python 3.9 CI job.

    `X | None` in an annotation is evaluated at runtime unless the module has
    `from __future__ import annotations`. On 3.10+ that is fine, so the
    problem is invisible on a modern interpreter — but on 3.9 it is an
    import-time TypeError, and a dataclass field annotation makes it fire on
    import, taking every test module down at collection.

    Ruff's FA102 also catches this; this test means the guard survives even
    if the lint config changes.
    """
    import ast

    offenders = []
    for path in sorted((PYPROJECT.parent / "bdp_model_gate").rglob("*.py")):
        tree = ast.parse(path.read_text())
        has_future = any(
            isinstance(node, ast.ImportFrom)
            and node.module == "__future__"
            and any(alias.name == "annotations" for alias in node.names)
            for node in tree.body
        )
        if has_future:
            continue

        annotations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign):
                annotations.append(node.annotation)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                annotations.extend(a.annotation for a in node.args.args if a.annotation)
                if node.returns:
                    annotations.append(node.returns)

        if any(
            isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.BitOr)
            for annotation in annotations
            for inner in ast.walk(annotation)
        ):
            offenders.append(path.name)

    assert not offenders, (
        f"{offenders} use PEP 604 unions without `from __future__ import annotations`, "
        "which is an import-time TypeError on Python 3.9"
    )


def test_the_bootstrap_default_is_not_the_one_the_suite_runs_with():
    """`tests/conftest.py` lowers `bootstrap_samples` to keep the suite quick.
    That is a test-speed decision and must not leak into the library: a
    governance tool should not ship a 150-draw percentile interval because it
    made someone's test run faster.
    """
    import inspect

    from bdp_model_gate.config import UncertaintyConfig

    declared = inspect.signature(UncertaintyConfig).parameters["bootstrap_samples"].default
    assert declared == 1000, "the shipped default changed — was that deliberate?"


def test_pre_commit_pins_match_the_lint_extra():
    """A developer running pre-commit and the same developer reading a CI
    failure must be looking at the same linter.

    Before 0.6.0 they were not: `.pre-commit-config.yaml` pinned ruff v0.13.2
    while CI installed the latest, which was v0.16.4 — two linters that can
    disagree about the same file, with no way to tell which one you were
    arguing with. Both are now exact, and this asserts they stay equal.
    """
    pyproject = PYPROJECT.read_text()
    pre_commit = (PYPROJECT.parent / ".pre-commit-config.yaml").read_text()

    for tool in ("ruff", "mypy"):
        pinned = re.search(rf'^\s*"{tool}==([\d.]+)"', pyproject, re.M)
        assert pinned, f"the lint extra no longer pins {tool} exactly"
        version = pinned.group(1)
        assert f"rev: v{version}" in pre_commit, (
            f"pre-commit runs a different {tool} than the lint extra pins "
            f"({version}) — they will disagree about the same file"
        )


def _extra(name: str) -> list[str]:
    """Requirement strings for one optional-dependency extra.

    Regex rather than `tomllib`, which is stdlib only from Python 3.11 — and
    this package supports 3.9. The first version of the tests below imported
    it and failed on 3.9 and 3.10, which is a fitting way for a guard against
    3.9 breakage to break. `_pyproject_version` above reads the file the same
    way, for the same reason.
    """
    section = re.search(
        r"^\[project\.optional-dependencies\]$(.*?)^\[", PYPROJECT.read_text(), re.M | re.S
    )
    assert section, "no [project.optional-dependencies] in pyproject.toml"
    block = re.search(rf"^{name} = \[(.*?)^\]", section.group(1), re.M | re.S)
    assert block, f"no {name!r} extra in pyproject.toml"
    return re.findall(r'"([^"]+)"', block.group(1))


def test_the_extras_parser_reads_what_is_there():
    """A guard on the guard: a regex that stops matching would make both tests
    below pass by finding nothing."""
    assert any("pytest" in spec for spec in _extra("dev"))
    assert any("ruff==" in spec for spec in _extra("lint"))


def test_no_exact_pin_lives_in_the_dev_extra():
    """The invariant that would have caught a real 3.9 breakage.

    `dev` is installed on every interpreter in the matrix, so an exact pin
    there has to be installable on the oldest one — and a tool's newest
    release usually is not. `mutmut==3.7.0` went into `dev` and broke
    `pip install -e ".[dev]"` on Python 3.9 immediately, because mutmut 3.7
    requires 3.10. The range it replaced had resolved to 3.3.1 there and
    installed fine, so the problem arrived with the pin rather than with the
    dependency.

    Exact pins belong in a single-purpose extra that runs on one interpreter:
    `lint` and `mutation`. `dev` carries ranges, and pip picks whatever the
    running interpreter supports.
    """
    pinned = [spec for spec in _extra("dev") if "==" in spec]
    assert not pinned, (
        f"{pinned} pins an exact version in `dev`, which is installed on every "
        "interpreter in the matrix. Move it to its own extra — see `lint` and "
        "`mutation` — or use a range."
    )


def test_the_mutation_extra_is_not_part_of_dev():
    """`scripts/mutmut_decision_surface.py` reaches into mutmut's internals, so
    the pin must be exact; and mutmut 3.7 needs Python 3.10 while this package
    supports 3.9. Those two facts together mean it cannot live in `dev`."""
    assert any("mutmut" in spec for spec in _extra("mutation"))
    assert not any("mutmut" in spec for spec in _extra("dev"))


#: Standard-library modules newer than this package's floor. `cli.py` imports
#: `tomllib` behind a `try`/`except` with a `tomli` fallback, which is the
#: correct pattern for shipped code — but a *test* has no such fallback and
#: does not need one, so an unguarded import there is simply a 3.9 failure
#: waiting for CI to find it.
TOO_NEW_FOR_THE_FLOOR = {"tomllib": "3.11"}


def test_no_test_module_imports_stdlib_newer_than_the_floor():
    """The mistake this exists for, in full.

    `test_no_exact_pin_lives_in_the_dev_extra` was written to catch a pin that
    broke Python 3.9 — and imported `tomllib` to do it, which is stdlib only
    from 3.11. It failed on 3.9 and 3.10: a guard against 3.9 breakage, broken
    on 3.9.

    `_pyproject_version` in this very file already reads `pyproject.toml` with
    a regex rather than a TOML parser, for exactly this reason. The precedent
    was two functions away.
    """
    import ast

    offenders = []
    for path in sorted((PYPROJECT.parent / "tests").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            for name in names:
                if name in TOO_NEW_FOR_THE_FLOOR:
                    offenders.append(
                        f"{path.name}:{node.lineno} imports {name} "
                        f"(stdlib from {TOO_NEW_FOR_THE_FLOOR[name]})"
                    )

    assert not offenders, (
        "these imports fail on the oldest supported interpreter:\n  " + "\n  ".join(offenders)
    )
