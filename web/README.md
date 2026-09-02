# web/

The project website: a hand-built landing page with the documentation beneath
it, mirroring how `pandas.pydata.org` is put together.

```
web/
  landing/index.html   landing page, deployed at /
  mkdocs.yml           MkDocs Material config, deployed at /docs/
  docs/                the guide, reference and rendered notebooks
  requirements.txt     pinned toolchain
  build.sh             builds both into web/_site
```

Documents **bdp-model-gate 0.5.3**. The site carries no version of its own:
a second version number is a second thing to forget at release time, and it
was already the stale one.

## Build

```bash
pip install -r web/requirements.txt
./web/build.sh
```

Output lands in `web/_site` (gitignored):

- `_site/index.html` — the landing page
- `_site/docs/` — the documentation

For live reload while writing:

```bash
./web/build.sh serve
```

That serves only the docs; the landing page is a single static file you can
open directly.

## How the pieces stay honest

- **The API reference is generated** from docstrings via `mkdocstrings`, so it
  cannot drift from the source.
- **Notebooks are copied from `examples/`** at build time, never edited here.
  `examples/run_all.sh` re-executes them and is the thing that keeps their
  outputs true. `docs/examples/*.ipynb` is gitignored for that reason.
- **`mkdocs build --strict`** fails on a broken internal link or a page
  missing from the nav, so a rename cannot quietly orphan a page.

- **The version stamp is asserted.** `tests/test_package.py` fails if the
  landing page's version chip drifts from `pyproject.toml`, and if a check in
  the default suite is missing from `docs/reference/checks.md`. Both had
  already drifted once before the test existed.

Prose code blocks are **not** currently executed, so an API change can leave a
snippet wrong while `--strict` still passes. That is the remaining gap, and it
is tracked in [`ROADMAP.md`](../ROADMAP.md) under 0.6.0.

## Deploy

`.github/workflows/docs.yml` builds on every push to `main` and publishes to
GitHub Pages. Enable Pages with source **GitHub Actions** in repository
settings; the first run creates the site.

## Editing

| Change | File |
|---|---|
| Landing page | `landing/index.html` — self-contained, no build step |
| A guide page | `docs/*.md`, `docs/tasks/*.md` |
| Reference prose | `docs/reference/*.md` |
| API reference | the docstrings in `bdp_model_gate/` |
| Navigation | the `nav:` block in `mkdocs.yml` |
| Colours and type | `docs/stylesheets/brand.css`, and the tokens in `landing/index.html` |

The two builds share a palette — deep teal `#0e5c55` on cool green-grey paper,
IBM Plex Sans and Mono. Keep them in step when changing either.
`bdp_model_gate/plots/style.py` carries the same palette, so a chart pasted
into either page does not look like a foreign object.

## At every release

The site is part of the release, not a follow-up. Before tagging:

| Update | Where |
|---|---|
| Version chip and colophon | `landing/index.html` — asserted by `tests/test_package.py` |
| New or changed checks | `docs/reference/checks.md`, the checks grid in `landing/index.html`, and the table in `docs/index.md` |
| New config fields | `docs/reference/configuration.md` |
| A new capability | the landing page — "produces a report a reviewer can sign" is a claim, not a detail |
| Counts in prose | notebooks, checks, plots — the numbers that go stale silently |
| Notebook outputs | `examples/run_all.sh`, then `./web/build.sh` to re-copy |

Then run `./web/build.sh` and read the pages you changed. `--strict` catches a
broken link; it cannot catch a sentence that is no longer true.
