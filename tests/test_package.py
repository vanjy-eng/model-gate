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
