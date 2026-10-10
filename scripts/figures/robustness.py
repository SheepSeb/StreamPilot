"""Detection dropout x noise (runs_dropout_noise/): the shift results of compare_dropout_noise.py.

robustness_curves     return vs dropout, noise and both, trained clean / with the curriculum
robustness_heatmaps   the whole grid per learner
robustness_tables     robustness_tables.md
curriculum            curriculum training curves and ramp ablation
"""

import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import (
    ALGOS,
    COLORS,
    MARKERS,
    MAX_RETURN,
    NAMES,
    ORDER,
    bold_best,
    learner_legend,
    millions,
    pm,
    save,
    style,
    table,
    write,
)

OUTPUTS = ["robustness_curves", "robustness_heatmaps", "robustness_tables", "curriculum"]
TRAINING = {"clean": "shift_results.csv", "curriculum": "shift_curriculum_results.csv"}
PANELS = [
    ("(a) Dropout only", "dropout", lambda d: d[d["noise"] == 0], "detection dropout $p$"),
    ("(b) Noise only", "noise", lambda d: d[d["dropout"] == 0], r"detection noise $\sigma$"),
    ("(c) Both", "noise", lambda d: d[d["dropout"] == d["noise"]], r"$p = \sigma$"),
]


def add_args(parser) -> None:
    parser.add_argument("--root", type=Path, default=Path("runs_dropout_noise"))
    parser.add_argument("--metric", default="return", help="figures: a column of the shift results (return, survival..)")
    parser.add_argument("--train-level", type=float, default=0.3, help="the curriculum's final level, marked")


def run(args, outputs: list[str]) -> None:
    if {"robustness_curves", "robustness_heatmaps", "robustness_tables"} & set(outputs):
        data = {name: pd.read_csv(args.root / csv) for name, csv in TRAINING.items()}
        for d in data.values():  # as a fraction of the maximum: 500 steps, at most 1 per step
            d["return"] /= MAX_RETURN
        label = {"return": "normalised return", "survival": "survival rate"}.get(args.metric, args.metric)
    if "robustness_curves" in outputs:
        plot_curves(data, args.metric, label, args.train_level, args.out_dir / "robustness_curves.pdf")
    if "robustness_heatmaps" in outputs:
        plot_heatmaps(data, args.metric, label, args.out_dir / "robustness_heatmaps.pdf")
    if "robustness_tables" in outputs:
        robustness_tables(data, args.out_dir / "robustness_tables.md")
    if "curriculum" in outputs:
        runs = load_curriculum(args.root / "curriculum")
        plot_curriculum_curves(runs, args.out_dir / "curriculum_curves.png")
        plot_curriculum_heatmap(runs, args.out_dir / "curriculum_heatmap.png")


def plot_curves(data: dict, metric: str, label: str, train_level: float, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10, 4.8), sharex=True, sharey=True)
    for row, name in zip(axes, TRAINING):
        for ax, (title, x, select, xlabel) in zip(row, PANELS):
            if name == "curriculum":  # the range the curriculum trained on
                ax.axvspan(0, train_level, color="0.965", linewidth=0, zorder=0)
            for algo in ALGOS:
                d = select(data[name][data[name]["method"] == algo])
                stats = d.groupby(x)[metric].agg(["mean", "std"]).sort_index()
                ax.plot(stats.index, stats["mean"], color=COLORS[algo], marker=MARKERS[algo], markersize=3.5,
                        linewidth=1.5)
                ax.fill_between(stats.index, stats["mean"] - stats["std"], stats["mean"] + stats["std"],
                                color=COLORS[algo], alpha=0.1, linewidth=0)
            style(ax, "y")
            if name == "clean":
                ax.set_title(title, fontsize=10)
            else:
                ax.set_xlabel(xlabel)
        row[0].set_ylabel(f"trained {name}\n{label}")
    axes[1, 0].text(train_level / 2, 0.04, "curriculum range", transform=axes[1, 0].get_xaxis_transform(),
                    ha="center", fontsize=7, color="0.45")
    learner_legend(fig, 1.04, markersize=3.5)
    fig.tight_layout()
    save(fig, out)


def plot_heatmaps(data: dict, metric: str, label: str, out: Path) -> None:
    grids = {(name, algo): data[name][data[name]["method"] == algo].groupby(["dropout", "noise"])[metric].mean()
             .unstack("noise").sort_index() for name in TRAINING for algo in ALGOS}
    vmax = 1.0 if metric in ("return", "survival") else max(g.to_numpy().max() for g in grids.values())
    fig, axes = plt.subplots(2, 4, figsize=(10, 4.6), sharex=True, sharey=True, constrained_layout=True)
    for row, name in zip(axes, TRAINING):
        for ax, algo in zip(row, ALGOS):
            g = grids[name, algo]
            im = ax.imshow(g.to_numpy(), origin="lower", cmap="Blues", vmin=0, vmax=vmax)
            for (i, j), v in np.ndenumerate(g.to_numpy()):
                text = f"{v:.2f}".replace("-0.00", "0.00").lstrip("0") if vmax <= 1 else f"{round(v) or 0:d}"
                ax.text(j, i, text, ha="center", va="center", fontsize=6, color="white" if v > 0.55 * vmax else "0.15")
            ax.set_xticks(range(g.shape[1]), [f"{v:g}" for v in g.columns], fontsize=7)
            ax.set_yticks(range(g.shape[0]), [f"{v:g}" for v in g.index], fontsize=7)
            if name == "clean":
                ax.set_title(NAMES[algo], fontsize=10)
            else:
                ax.set_xlabel(r"noise $\sigma$", fontsize=8)
        row[0].set_ylabel(f"trained {name}\n" + r"dropout $p$", fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.8, label=label)
    save(fig, out)


# --- tables ------------------------------------------------------------------------------------------------------

CONDITIONS = [  # (dropout, noise), header
    ((0.0, 0.0), "Clean"),
    ((0.1, 0.1), "0.1 / 0.1"),
    ((0.3, 0.3), "0.3 / 0.3"),
    ((0.5, 0.5), "0.5 / 0.5"),
    ((0.5, 0.0), "Dropout 0.5 only"),
    ((0.0, 0.5), "Noise 0.5 only"),
]
HEADER = ["Learner", *(h for _, h in CONDITIONS), "Grid mean"]


def per_seed(d: pd.DataFrame, algo: str, metric: str, cond=None) -> np.ndarray:
    """One value per seed: at ``cond`` (dropout, noise), or the mean over the whole grid."""
    d = d[d["method"] == algo]
    if cond is not None:
        d = d[(d["dropout"] == cond[0]) & (d["noise"] == cond[1])]
    return d.groupby("seed")[metric].mean().to_numpy()


def condition_table(data: dict, metric: str) -> list[str]:
    """One table per training (learners as rows), the best learner of each column in bold."""
    out = []
    for t in TRAINING:
        rows = [[NAMES[a]] for a in ORDER]
        for cond in [c for c, _ in CONDITIONS] + [None]:
            values = [per_seed(data[t], a, metric, cond) for a in ORDER]
            for row, c in zip(rows, bold_best([pm(v) for v in values], [v.mean() for v in values])):
                row.append(c)
        out += [f"**Trained {t}**", "", table(HEADER, rows), ""]
    return out


def robustness_tables(data: dict, out_path: Path) -> None:
    seeds = data["clean"]["seed"].unique()
    out = [
        "# Detection dropout × noise: tables",
        "",
        (f"Formation tracking, N = 3, {len(seeds)} seeds per learner. *Trained clean*: the policies of `runs/`, trained "
        "without corruption. *Trained curriculum*: trained with the detection corruption rising linearly from none "
        "to dropout 0.3 / noise 0.3 over the first 1.75M of 2M steps. Every policy is evaluated under each condition "
        "on the same 100 episodes; mean ± std over seeds. Columns: dropout *p* / noise σ. *Grid mean*: per seed, the "
        "mean over all 49 conditions (*p*, σ ∈ {0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5}). **Bold**: best learner in the "
        "column (none on a tie). Generated by `scripts/make_figures.py robustness`."),
        "",
        "## 1. Normalised return",
        "",
        *condition_table(data, "return"),
        "## 2. Change from clean to curriculum training",
        "",
        "Curriculum minus clean, normalised return, mean over seeds. Positive: the curriculum helps.",
        "",
    ]
    body = []
    for a in ORDER:
        row = [NAMES[a]]
        for cond in [c for c, _ in CONDITIONS] + [None]:
            delta = per_seed(data["curriculum"], a, "return", cond).mean() - per_seed(data["clean"], a, "return", cond).mean()
            row.append(f"{delta:+.2f}")
        body.append(row)
    out += [table(HEADER, body), ""]
    out += [
        "## 3. Fraction of steps on target",
        "",
        ("Within the tracking tolerance (distance and heading). Return rewards closeness continuously; this counts "
        "only the steps inside the tolerance."),
        "",
        *condition_table(data, "on target"),
        "## 4. Survival",
        "",
        "Fraction of episodes reaching the time limit without a collision or leaving the arena.",
        "",
        *condition_table(data, "survival"),
    ]
    write(out_path, out)


# --- curriculum training runs ------------------------------------------------------------------------------------

CURRICULUM_RE = re.compile(r"^(?P<algo>.+)_p(?P<p>[\d.]+)_n(?P<n>[\d.]+)_ramp(?P<ramp>[\d.]+)_seed(?P<seed>\d+)$")
CURVES = {
    "eval/mean_on_target": ("On target", 1.0),
    "eval/mean_heading_error": ("Heading err. [deg]", np.rad2deg(1.0)),
    "eval/mean_formation_error": ("Formation err. [m]", 1.0),
}
TOTAL_STEPS = 2_000_000


def load_curriculum(root: Path) -> dict[tuple, pd.DataFrame]:
    """(algo, p, n, ramp) -> metrics of all seeds concatenated, with a 'seed' column."""
    runs = defaultdict(list)
    for run_dir in sorted(root.iterdir()):
        m = CURRICULUM_RE.match(run_dir.name)
        if m is None or not (run_dir / "metrics.parquet").exists():
            continue
        df = pd.read_parquet(run_dir / "metrics.parquet")
        df["seed"] = int(m["seed"])
        runs[(m["algo"], float(m["p"]), float(m["n"]), float(m["ramp"]))].append(df)
    return {k: pd.concat(v, ignore_index=True) for k, v in runs.items()}


def plot_curriculum_curves(runs: dict, out: Path) -> None:
    configs = sorted({k[1:] for k in runs})
    fig, axes = plt.subplots(1 + len(CURVES), len(configs), figsize=(max(4 * len(configs), 8), 8.5), sharex=True,
                             sharey="row", squeeze=False, gridspec_kw={"height_ratios": [0.45] + [1] * len(CURVES)})
    for col, (p, n, ramp) in enumerate(configs):
        ramp_end = ramp * TOTAL_STEPS
        ax = axes[0, col]  # schedule strip: the training detection level as a fraction of the final one
        steps = np.linspace(0, TOTAL_STEPS, 200)
        ax.fill_between(steps, np.minimum(steps / ramp_end, 1.0), color="#9ea3ab", alpha=0.5, lw=0)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 1])
        ax.set_title(f"p = {p:g}, noise = {n:g}, ramp = {ramp:g}", fontsize=10)
        if col == 0:
            ax.set_ylabel("Curriculum\nscale")

        for row, (metric, (label, scale)) in enumerate(CURVES.items(), start=1):
            ax = axes[row, col]
            for algo in ALGOS:
                if (algo, p, n, ramp) not in runs:
                    continue
                # mean / min / max over seeds per eval step; eval steps differ slightly between learners, not seeds
                e = runs[algo, p, n, ramp].dropna(subset=[metric]).groupby(["seed", "step"])[metric].last().mul(scale)
                c = e.groupby("step").agg(["mean", "min", "max"])
                ax.plot(c.index, c["mean"], color=COLORS[algo], lw=2, label=NAMES[algo])
                ax.fill_between(c.index, c["min"], c["max"], color=COLORS[algo], alpha=0.12, lw=0)
            if col == 0:
                ax.set_ylabel(label)
            if metric == "eval/mean_on_target":
                ax.set_ylim(0, 1)

        for ax in axes[:, col]:
            ax.axvspan(0, ramp_end, color="#9ea3ab", alpha=0.08, lw=0)  # ramp period behind every panel
            ax.axvline(ramp_end, color="#6b7079", lw=1, ls="--")
            ax.grid(alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)
        axes[-1, col].set_xlabel("Environment steps")
        millions(axes[-1, col])

    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(ALGOS), frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Noise curriculum, N = 3: eval at the full level (line = seed mean, band = min–max; "
                 "shaded / dashed = ramp)", fontsize=10, y=0.965)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, out, dpi=150)


def plot_curriculum_heatmap(runs: dict, out: Path) -> None:
    rows = []
    for (algo, p, n, ramp), df in runs.items():
        last = df.dropna(subset=["eval/mean_on_target"]).groupby("seed")["eval/mean_on_target"].last()
        rows.append({"algo": algo, "level": f"p={p:g}, n={n:g}", "ramp": ramp, "mean": last.mean(), "std": last.std()})
    final = pd.DataFrame(rows).set_index(["algo", "level", "ramp"])
    levels = sorted(final.index.unique("level"))
    ramps = sorted(final.index.unique("ramp"))
    fig, axes = plt.subplots(1, len(ALGOS), figsize=(3.2 * len(ALGOS), 3.2), sharey=True)
    for ax, algo in zip(axes, ALGOS):
        for i, level in enumerate(levels):
            for j, ramp in enumerate(ramps):
                if (algo, level, ramp) not in final.index:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///", ec="#c3c2b7", lw=0))
                    ax.text(j, i, "not run", ha="center", va="center", fontsize=8, color="#6b7079")
                else:
                    m, s = final.loc[(algo, level, ramp)]
                    ax.text(j, i, f"{m:.2f}\n±{s:.2f}", ha="center", va="center", fontsize=9,
                            color="white" if m > 0.55 else "#1a1a19")
        grid = final["mean"].reindex(pd.MultiIndex.from_product([[algo], levels, ramps])).to_numpy()
        im = ax.imshow(grid.reshape(len(levels), len(ramps)), cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(ramps)), [f"{r:g}" for r in ramps])
        ax.set_yticks(range(len(levels)), levels)
        ax.set_xlabel("Ramp length (fraction of steps)")
        ax.set_title(NAMES[algo], fontsize=10)
        ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="Final on target")
    fig.suptitle("Curriculum ablation, N = 3: final eval on target (mean ± std over seeds)", fontsize=10, y=1.04)
    save(fig, out, dpi=150)
    print(final.round(3).to_string())
