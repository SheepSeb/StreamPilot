"""Drawn diagrams. algos: iStream-AC (a critic and TD error per drone) vs cStream-AC (one joint critic).
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .common import save

OUTPUTS = ["algos"]

# Box colours, shared with the schematic of scenes.py
INK = "#333333"
ENV_FACE, ENV_EDGE = "#e8eef5", "#4a6c8f"
ACTOR_FACE, ACTOR_EDGE = "#e3f4ec", "#1baf7a"
CRITIC_FACE, CRITIC_EDGE = "#fdeee6", "#eb6834"
UPDATE_FACE, UPDATE_EDGE = "#f1f1f1", "#666666"


def add_args(parser) -> None:
    pass


def run(args, outputs: list[str]) -> None:
    if "algos" in outputs:
        plot_algos(args.out_dir / "algos.pdf")


def box(ax, center, width, height, text, face, edge, fontsize=7.5) -> None:
    x, y = center
    ax.add_patch(FancyBboxPatch((x - width / 2, y - height / 2), width, height,
                                boxstyle="round,pad=0.02,rounding_size=0.12", facecolor=face, edgecolor=edge, lw=1.1,
                                zorder=3))
    ax.text(x, y, text, fontsize=fontsize, ha="center", va="center", zorder=4, linespacing=1.4)


def arrow(ax, start, end, dashed=False) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9, color=INK, lw=1.0, shrinkA=0, shrinkB=0,
                                 ls=(0, (4, 2)) if dashed else "-", zorder=5))


# --- algos -------------------------------------------------------------------------------------------------------

COLUMNS = [(1.4, "1"), (4.8, "2"), (8.2, "N")]
BOX_W, BOX_H = 2.5, 0.9
ACTOR_Y, X_Y, CRITIC_Y = 4.9, 3.15, 1.4
ROUTE = 1.7  # offset of the TD-error line from a column's centre
FONT = 7


def label(ax, xy, text, ha="left") -> None:
    ax.text(*xy, text, fontsize=FONT, ha=ha, va="center", color=INK, zorder=6)


def td_route(ax, c: float, start_x: float, start_y: float, text: str) -> None:
    """TD error from the critic, up the right of the column, into the actor (dashed)."""
    right = c + ROUTE
    ax.plot([start_x, right, right], [start_y, start_y, ACTOR_Y], color=INK, lw=1.0, ls=(0, (4, 2)), zorder=5)
    arrow(ax, (right, ACTOR_Y), (c + BOX_W / 2, ACTOR_Y))
    label(ax, (right + 0.1, (X_Y + ACTOR_Y) / 2 + 0.15), text)


def columns(ax, title: str) -> None:
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 5.5)
    ax.set_aspect("equal")
    ax.axis("off")
    half = BOX_H / 2
    for c, i in COLUMNS:
        box(ax, (c, ACTOR_Y), BOX_W, BOX_H, f"actor $\\pi_{i}$", ACTOR_FACE, ACTOR_EDGE, FONT)
        box(ax, (c, X_Y), BOX_W, BOX_H, f"$x_{i}$", "white", "#888888", FONT)
        arrow(ax, (c, X_Y + half), (c, ACTOR_Y - half))
        arrow(ax, (c, X_Y - half), (c, CRITIC_Y + half))
    ax.set_title(title, fontsize=8, pad=2)


def plot_algos(out: Path) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.4, 1.9), gridspec_kw={"wspace": 0.08})
    columns(left, "(a) iStream-AC")
    for c, i in COLUMNS:
        box(left, (c, CRITIC_Y), BOX_W, BOX_H, f"critic $V_{i}(x_{i})$", CRITIC_FACE, CRITIC_EDGE, FONT)
        td_route(left, c, c + BOX_W / 2, CRITIC_Y, f"$\\delta_{i}$")
        arrow(left, (c, 0.15), (c, CRITIC_Y - BOX_H / 2))
        label(left, (c - 0.12, 0.5), "$\\tilde R_t$", ha="right")

    columns(right, "(b) cStream-AC")
    lo, hi = COLUMNS[0][0] - BOX_W / 2, COLUMNS[-1][0] + ROUTE + 0.25
    centre = (lo + hi) / 2
    box(right, (centre, CRITIC_Y), hi - lo, BOX_H, "critic $V(s)$,  $s = [x_1, x_2, \\dots, x_N]$", CRITIC_FACE,
        CRITIC_EDGE, FONT)
    for c, _ in COLUMNS:
        td_route(right, c, c + ROUTE, CRITIC_Y + BOX_H / 2, "$\\delta$")
    arrow(right, (centre, 0.15), (centre, CRITIC_Y - BOX_H / 2))
    label(right, (centre - 0.12, 0.5), "$\\tilde R_t$", ha="right")
    save(fig, out, dpi=250, pad_inches=0.02)
