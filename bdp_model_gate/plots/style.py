"""House style, shared with the documentation site.

Two colour systems, kept strictly apart:

* **Semantic** — pass / review / blocked. These *mean* something, and a group
  must never borrow them. A green bar that happens to be group A, next to a
  green verdict pill, is a misread waiting to happen.
* **Categorical** — for groups. Colour-blind safe, because roughly 8% of men
  have some colour vision deficiency and a gate report is a document a
  regulator may read. Accessibility is not optional here.

Colour is also never the *only* encoding: helpers pair it with marker shape or
hatching, since these reports get printed in greyscale.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

# Matches the landing page and the docs theme, so a plot pasted into either
# does not look like a foreign object.
INK = "#10221f"
MUTED = "#55635f"
RULE = "#dde3e0"
ACCENT = "#0e5c55"

#: Verdict colours. Semantic — never use these for a group.
VERDICT_COLOURS = {
    "PASS": "#2e6b43",
    "OK": "#2e6b43",
    "NEEDS_REVIEW": "#8a5a0b",
    "BLOCKED": "#9b2c2c",
    "NOT_APPLICABLE": MUTED,
}

#: Okabe–Ito, the standard colour-blind-safe qualitative palette. Deliberately
#: not derived from the accent: a group is not a verdict.
CATEGORICAL = [
    "#0072b2",  # blue
    "#e69f00",  # orange
    "#009e73",  # bluish green
    "#cc79a7",  # reddish purple
    "#56b4e9",  # sky blue
    "#d55e00",  # vermillion
    "#f0e442",  # yellow
    "#000000",  # black
]

#: Paired with CATEGORICAL so a series is identifiable in greyscale.
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
HATCHES = ["", "///", "...", "xxx", "\\\\\\", "|||", "---", "+++"]

_RC = {
    "figure.dpi": 110,
    "savefig.dpi": 110,
    "savefig.bbox": "tight",
    # Overridden per session by `apply_style` with only the faces actually
    # installed — see `_font_stack`.
    "font.family": ["sans-serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    # "bold", not a numeric weight: DejaVu Sans — the face almost every
    # machine falls back to — ships regular and bold only, and asking for 600
    # logs a substitution warning for every title drawn.
    "axes.titleweight": "bold",
    "axes.titlepad": 10,
    "axes.labelsize": 10,
    "axes.labelcolor": MUTED,
    "axes.edgecolor": RULE,
    "axes.facecolor": "none",  # inherit the page, so the SVG is theme-aware
    "figure.facecolor": "none",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",  # horizontal rules only; verticals add noise
    "grid.color": RULE,
    "grid.linewidth": 0.6,
    "grid.alpha": 0.9,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "legend.fontsize": 9,
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
    "text.color": INK,
}


#: Preferred faces, best first. Plex matches the documentation site; the rest
#: are the usual near-misses before matplotlib's own bundled face.
_FONT_PREFERENCE = ("IBM Plex Sans", "Inter", "Helvetica Neue", "Arial", "DejaVu Sans")


@lru_cache(maxsize=1)
def _font_stack() -> list[str]:
    """The preferred faces that are actually installed.

    Naming a missing family costs matplotlib a `findfont` warning for *every
    text element it draws* — several hundred lines of stderr for a ten-panel
    report, describing a substitution that was always going to happen and is
    entirely fine. Filtering first keeps the fallback silent.
    """
    from matplotlib import font_manager

    available = {face.name for face in font_manager.fontManager.ttflist}
    return [name for name in _FONT_PREFERENCE if name in available] or ["sans-serif"]


def apply_style() -> None:
    """Applies the house style to the current matplotlib session.

    Called by every `plot()`, so a user who only wants one chart gets the
    styling without setting anything up. It sets rcParams rather than using a
    style context, so a caller composing several plots into one figure gets a
    consistent result — and can override afterwards, since their call comes
    last.
    """
    plt, _ = _import()
    plt.rcParams.update({**_RC, "font.family": _font_stack()})


def new_axes(ax: Any = None, figsize: tuple[float, float] = (6.4, 3.8)) -> Any:
    """Returns the caller's Axes, or a new one.

    Accepting an Axes is the whole composition contract: a caller can lay out
    small multiples and pass each cell in.
    """
    plt, _ = _import()
    apply_style()
    if ax is not None:
        return ax
    _, ax = plt.subplots(figsize=figsize)
    return ax


def categorical(n: int) -> list[str]:
    """`n` colour-blind-safe colours, cycling if more are asked for."""
    return [CATEGORICAL[i % len(CATEGORICAL)] for i in range(n)]


def markers(n: int) -> list[str]:
    """Marker shapes to pair with `categorical`, so colour is never the only
    encoding."""
    return [MARKERS[i % len(MARKERS)] for i in range(n)]


def hatches(n: int) -> list[str]:
    """Bar hatchings to pair with `categorical`, for the same reason `markers`
    exists: these reports get printed in greyscale, so colour is never allowed
    to be the only thing distinguishing two series."""
    return [HATCHES[i % len(HATCHES)] for i in range(n)]


def caption(ax: Any, text: str) -> None:
    """A note below the axes saying how to read the plot.

    The vertical position is anchored to the x-axis *label artist* rather than
    to a fixed offset from the axes: tick labels vary in height — one line on
    a sweep, three on a banded heatmap — and any constant offset that clears
    the tallest overlaps on the shortest. The horizontal position stays in
    axes coordinates, because the x label is centred and hanging the caption
    off its left edge would indent it to the middle of the plot.

    Worth the trouble because these charts are read by people who did not
    build the model. "Above 1 means under-predicted" is the difference
    between a plot that informs a decision and one that decorates a page.
    """
    ax.annotate(
        text,
        xy=(0, 0),
        xycoords=("axes fraction", ax.xaxis.label),
        xytext=(0, -12),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=8,
        color=MUTED,
    )


def ring_cell(ax: Any, x: int, y: int, colour: str) -> None:
    """Outlines one heatmap cell, legibly on any fill underneath it.

    Two strokes: a wider one in the surface colour, then the real one on top.
    A single stroke has to be readable against both the palest and the
    darkest cell in the map, and no colour is.

    Inset slightly, because a cell on the edge of the grid has half its
    outline clipped by the axes and reads as absent — which on a confusion
    matrix silently drops the two corners of the diagonal.
    """
    from matplotlib.patches import Rectangle

    for width, edge in ((3.4, "#ffffff"), (1.6, colour)):
        ax.add_patch(
            Rectangle(
                (x + 0.025, y + 0.025),
                0.95,
                0.95,
                fill=False,
                edgecolor=edge,
                linewidth=width,
                zorder=4,
            )
        )


def sharpen_colourbar(ax: Any) -> None:
    """Keeps a heatmap's colour bar as vector geometry.

    matplotlib rasterises colour bar solids by default — a workaround for
    hairline seams between cells in some PDF viewers — which embeds a base64
    PNG inside what is otherwise a vector figure. That PNG is then the one
    soft edge on a printed report, and it inflates a self-contained page by
    the size of a bitmap per chart. The seams are the lesser problem.
    """
    bar = getattr(ax.collections[0], "colorbar", None) if len(ax.collections) else None
    if bar is not None and getattr(bar, "solids", None) is not None:
        bar.solids.set_rasterized(False)


def verdict_colour(flag: str) -> str:
    """Semantic colour for a `CheckResult.flag`. Anything unrecognised is a
    risk flag, so it reads as blocked rather than silently neutral."""
    return VERDICT_COLOURS.get(flag, VERDICT_COLOURS["BLOCKED"])


#: Structural colours the page is allowed to restyle, mapped to a CSS custom
#: property with the original as its fallback. Applied to an inlined SVG by
#: `themeable_svg`.
#:
#: Only the *furniture* appears here — text, axes, gridlines, the surface
#: behind a bar's edge. Data colours are deliberately absent: a group's hue
#: and a verdict's hue carry meaning, and a page that could repaint them could
#: turn a blocked finding green.
_THEMEABLE = {
    INK: "var(--plot-ink, {0})",
    MUTED: "var(--plot-muted, {0})",
    RULE: "var(--plot-rule, {0})",
    "#ffffff": "var(--plot-surface, {0})",
}


def themeable_svg(svg: str) -> str:
    """Rewrites an SVG's structural colours as CSS custom properties.

    An SVG inlined into HTML — rather than referenced as `<img src="data:">` —
    participates in the page's cascade, so this is what lets one render read
    correctly in both light and dark. Every property carries the original hex
    as its fallback, so the SVG still stands alone in a viewer that has never
    heard of the variables.
    """
    for literal, template in _THEMEABLE.items():
        replacement = template.format(literal)
        svg = svg.replace(literal, replacement).replace(literal.upper(), replacement)
    return svg


def _import():
    from . import require_plotting

    return require_plotting()


__all__ = [
    "CATEGORICAL",
    "caption",
    "ring_cell",
    "sharpen_colourbar",
    "HATCHES",
    "MARKERS",
    "VERDICT_COLOURS",
    "apply_style",
    "categorical",
    "hatches",
    "markers",
    "new_axes",
    "themeable_svg",
    "verdict_colour",
]
