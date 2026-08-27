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

Status: **0.4.1-alpha** — the site is new and its structure may move. The
library it documents is 0.5.1.

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

Prose code blocks are **not** currently executed. That gap is tracked in
[`ROADMAP.md`](../ROADMAP.md) against the 0.4.2 robustness work.

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
