"""Shared by the figures and tables: learner names and colours, run loading, statistics, saving, tables.
"""

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# Learners, in the order (and colours) of every plot.
ALGOS = ["cstream_ac", "ippo", "istream_ac", "mappo"]
ORDER = ["istream_ac", "cstream_ac", "ippo", "mappo"]  # table rows: the two Stream learners first
NAMES = {"cstream_ac": "cStream-AC", "ippo": "IPPO", "istream_ac": "iStream-AC", "mappo": "MAPPO"}
MARKERS = {"cstream_ac": "o", "ippo": "s", "istream_ac": "^", "mappo": "D"}
COLORS = dict(zip(ALGOS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))

MAX_RETURN = 500.0  # formation tracking: 500 steps, at most 1 per step
DRONES = [1, 2, 3, 5]
SEEDS = [1, 2, 3, 4, 5]
REPS = 2000  # bootstrap resamples

# Run dirs look like "<algorithm>_<task>_<n>d_seed<seed>", e.g. "cstream_ac_formation-tracking_3d_seed2"
RUN_RE = re.compile(r"^(?P<algo>.+)_(?P<task>[^_]+)_(?P<n>\d+)d_seed(?P<seed>\d+)$")


def load_runs(root: Path, task: str = "formation-tracking") -> dict[tuple[str, int], pd.DataFrame]:
    """(algorithm, n_drones) -> metrics of all seeds of ``task`` concatenated, with a 'seed' column."""
    runs = defaultdict(list)
    for run_dir in sorted(Path(root).iterdir()):
        m = RUN_RE.match(run_dir.name)
        if m is None or m["task"] != task or not (run_dir / "metrics.parquet").exists():
            continue
        df = pd.read_parquet(run_dir / "metrics.parquet")
        df["seed"] = int(m["seed"])
        runs[(m["algo"], int(m["n"]))].append(df)
    return {key: pd.concat(dfs, ignore_index=True) for key, dfs in runs.items()}


# --- statistics (Agarwal et al. 2021, "Deep RL at the edge of the statistical precipice") --------------------------

def iqm(x: np.ndarray, axis=None) -> np.ndarray:
    """Interquartile mean: of all values (axis None), or per row of x flattened to (rows, -1) (axis 1)."""
    x = np.sort(x.reshape(x.shape[0], -1) if axis == 1 else np.ravel(x), axis=-1)
    k = x.shape[-1]
    return x[..., k // 4 : k - k // 4].mean(-1)


AGGREGATES = {
    "Median": lambda s: np.median(s.reshape(s.shape[0], -1), -1),
    "IQM": lambda s: iqm(s, axis=1),
    "Mean": lambda s: s.reshape(s.shape[0], -1).mean(-1),
    "Optimality gap": lambda s: np.clip(1 - s.reshape(s.shape[0], -1), 0, None).mean(-1),
}


def stratified_bootstrap(scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """scores (seeds, tasks) -> (REPS, seeds, tasks): seeds resampled with replacement within each task."""
    idx = rng.integers(0, scores.shape[0], size=(REPS, *scores.shape))
    return np.take_along_axis(np.broadcast_to(scores, (REPS, *scores.shape)), idx, axis=1)


def ci(samples: np.ndarray) -> tuple[float, float]:
    """95 % percentile interval over the first axis."""
    return tuple(np.percentile(samples, [2.5, 97.5], axis=0))


def probability_of_improvement(x: np.ndarray, y: np.ndarray) -> float:
    """x, y (seeds, tasks): mean over tasks of P(a run of x beats a run of y), ties counting half."""
    per_task = [((xi[:, None] > yi[None]) + 0.5 * (xi[:, None] == yi[None])).mean() for xi, yi in zip(x.T, y.T)]
    return float(np.mean(per_task))


def improvement_ci(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    bx, by = stratified_bootstrap(x, rng), stratified_bootstrap(y, rng)
    return ci(np.array([probability_of_improvement(bx[r], by[r]) for r in range(0, REPS, 4)]))


# --- figures -----------------------------------------------------------------------------------------------------

def save(fig, out: Path, dpi: int = 200, **kwargs) -> None:
    """``out`` as given (usually .pdf) and as .png."""
    out.parent.mkdir(parents=True, exist_ok=True)
    kwargs.setdefault("bbox_inches", "tight")
    fig.savefig(out, dpi=dpi if out.suffix == ".png" else "figure", **kwargs)
    if out.suffix != ".png":
        fig.savefig(out.with_suffix(".png"), dpi=dpi, **kwargs)
    print(f"wrote {out}" + (f" and {out.with_suffix('.png')}" if out.suffix != ".png" else ""))


def style(ax, axis: str = "both") -> None:
    ax.grid(axis=axis, color="0.9", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def learner_legend(fig, y: float, algos=ALGOS, markersize: float = 4, fontsize: float = 9) -> None:
    handles = [Line2D([], [], color=COLORS[a], marker=MARKERS[a], markersize=markersize, label=NAMES[a]) for a in algos]
    fig.legend(handles=handles, loc="upper center", ncol=len(algos), frameon=False, fontsize=fontsize,
               bbox_to_anchor=(0.5, y))


def millions(ax) -> None:
    ax.xaxis.set_major_formatter(lambda x, _: f"{x / 1e6:g}M")


# --- markdown tables ---------------------------------------------------------------------------------------------

def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def pm(v, digits: int = 2) -> str:
    """mean ± std (a single value as is), '–' if there is none."""
    v = np.asarray(v, float)
    if np.isnan(v).all():
        return "–"
    if v.size == 1:
        text = f"{float(v.ravel()[0]):.{digits}f}"
    else:
        text = f"{np.nanmean(v):.{digits}f} ± {np.nanstd(v):.{digits}f}"
    return text.replace("-0." + "0" * digits, "0." + "0" * digits)


def bold_best(cells: list[str], values: list[float], higher: bool = True) -> list[str]:
    """Bold the best cell, unless its displayed value is shared (a tie)."""
    best = np.nanargmax(values) if higher else np.nanargmin(values)
    shown = [c.split(" ±")[0] for c in cells]
    if shown.count(shown[best]) > 1:
        return cells
    return [f"**{c}**" if k == best else c for k, c in enumerate(cells)]


def write(out: Path, lines: list[str]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"wrote {out}")
