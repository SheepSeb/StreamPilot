"""Continual task sequence (runs_continual/): tracking -> waypoint -> landing -> tracking, 500k steps each.

continual_sequence   eval return (or --metric score) over the sequence, normalised per task
continual_matrix     end-of-phase probe score of every task, per learner
continual_tables     continual_tables.md
continual_summary    final score, plasticity, forgetting, relearning per learner
continual_phases     train and eval return within each phase
continual_run        every metric of one run (--run DIR)
"""

import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from .common import ALGOS, COLORS, MARKERS, NAMES, pm, save, table, write

OUTPUTS = ["continual_sequence", "continual_matrix", "continual_tables", "continual_summary", "continual_phases",
           "continual_run"]
TASKS = ["tracking", "waypoint", "landing"]  # task_id order
SEQUENCE = [0, 1, 2, 0]  # task id of each phase
PHASE_LEN = 5e5
PHASES = [f"{k + 1}. {TASKS[t]}" + (" (again)" if k == 3 else "") for k, t in enumerate(SEQUENCE)]


def add_args(parser) -> None:
    parser.add_argument("--root", type=Path, default=Path("runs_continual"))
    parser.add_argument("--metric", choices=["return", "score"], default="return", help="continual_sequence's y axis")
    parser.add_argument("--fraction", type=float, default=0.8, help="tables, adaptation speed: fraction of the gain")
    parser.add_argument("--run", type=Path, help="continual_run: the run directory")


def run(args, outputs: list[str]) -> None:
    out = args.out_dir
    data = {a: r for a in ALGOS if (r := load(args.root, a))}
    if not data:
        raise SystemExit(f"no finished runs in {args.root}")
    print({a: len(r) for a, r in data.items()}, "seeds")
    if "continual_sequence" in outputs:
        plot_sequence(data, args.metric, out / "continual_sequence.pdf")
    if "continual_matrix" in outputs:
        plot_matrix(data, out / "continual_matrix.pdf")
    if "continual_tables" in outputs:
        continual_tables(data, args.fraction, out / "continual_tables.md")
    if "continual_summary" in outputs:
        plot_summary(data, out / "continual_summary.png")
    if "continual_phases" in outputs:
        plot_phases({a: load(args.root, a, finished=False) for a in ALGOS}, out / "continual_phases.png")
    if "continual_run" in outputs and args.run:
        plot_run(args.run, out / f"continual_run_{args.run.name}.png")


def load(root: Path, algo: str, finished: bool = True) -> list[pd.DataFrame]:
    """The runs of ``algo``; ``finished``: only those with all 4 x 3 probes logged."""
    runs = []
    for f in sorted(glob.glob(str(root / f"{algo.removesuffix('_ac')}-continue_seed*/metrics.parquet"))):
        d = pd.read_parquet(f)
        if not finished or ("probe/score" in d and d["probe/score"].notna().sum() == 12):
            runs.append(d)
    return runs


def matrix(d: pd.DataFrame) -> np.ndarray:
    """[phase, task] -> probe score."""
    m = np.full((4, 3), np.nan)
    for _, r in d.dropna(subset=["probe/score"]).iterrows():
        m[int(r["phase"]), int(r["probe_task"])] = r["probe/score"]
    return m


def phase_labels(ax) -> None:
    for p in range(4):
        if p % 2:
            ax.axvspan(p * PHASE_LEN, (p + 1) * PHASE_LEN, color="0.965", linewidth=0, zorder=0)
        if p:
            ax.axvline(p * PHASE_LEN, color="0.25", linewidth=1.0, zorder=1)
        ax.text((p + 0.5) * PHASE_LEN, 1.0, PHASES[p][3:], transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=9.5)


def plot_sequence(data: dict, metric: str, out: Path) -> None:
    curves = {}  # (algo, phase) -> (x, mean, std)
    for algo, runs in data.items():
        for p in range(4):
            rows = [d[(d["phase"] == p) & d[f"eval/{metric}"].notna()].groupby("phase_step").last() for d in runs]
            y = np.array([r[f"eval/{metric}"].to_numpy() for r in rows])
            curves[algo, p] = (p * PHASE_LEN + rows[0].index.to_numpy(), y.mean(0), y.std(0))
    lo, hi = np.zeros(3), np.ones(3)
    if metric == "return":  # per task, over the mean curves of all learners, so the tasks share the axis
        for t in range(3):
            values = [mu for (_, p), (_, mu, _) in curves.items() if SEQUENCE[p] == t]
            lo[t], hi[t] = min(v.min() for v in values), max(v.max() for v in values)

    fig, ax = plt.subplots(figsize=(10, 3.2))
    phase_labels(ax)
    for (algo, p), (x, mu, sd) in curves.items():
        t = SEQUENCE[p]
        mu, sd = (mu - lo[t]) / (hi[t] - lo[t]), sd / (hi[t] - lo[t])
        ax.plot(x, mu, color=COLORS[algo], linewidth=1.5, marker=MARKERS[algo], markersize=2.5, zorder=3,
                label=NAMES[algo] if p == 0 else None)
        ax.fill_between(x, mu - sd, mu + sd, color=COLORS[algo], alpha=0.08, linewidth=0, zorder=2)
    ax.set_ylim(-0.08, 1.08)
    ax.set_xlim(0, 4 * PHASE_LEN)
    ax.set_xticks(np.arange(5) * PHASE_LEN, ["0", "0.5M", "1M", "1.5M", "2M"])
    ax.set_xlabel("team steps")
    ax.set_ylabel("normalised eval return" if metric == "return" else "eval score\n(success / on target)")
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    handles, _ = ax.get_legend_handles_labels()
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    save(fig, out)


def plot_matrix(data: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, len(data), figsize=(2.3 * len(data) + 0.8, 2.9), sharey=True, constrained_layout=True)
    rows = [f"after {TASKS[t]}" + (" (2)" if p == 3 else "") for p, t in enumerate(SEQUENCE)]
    for ax, (algo, runs) in zip(axes, data.items()):
        mat = np.nanmean([matrix(d) for d in runs], 0)
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        for (p, t), v in np.ndenumerate(mat):
            ax.text(t, p, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.55 else "0.15")
        for p, t in enumerate(SEQUENCE):  # the phase's own task
            ax.add_patch(Rectangle((t - 0.5, p - 0.5), 1, 1, fill=False, edgecolor="0.1", linewidth=1.2))
        ax.set_xticks(range(3), TASKS, fontsize=8)
        ax.set_yticks(range(4), rows, fontsize=8)
        ax.tick_params(length=0)
        ax.set_title(NAMES[algo], fontsize=10, color=COLORS[algo], fontweight="bold")
    fig.colorbar(im, ax=axes, shrink=0.85, label="probe score")
    save(fig, out)


def summary(m: np.ndarray) -> dict:
    """m (seeds, phase, task) -> metric -> per-seed values."""
    own = np.stack([m[:, p, SEQUENCE[p]] for p in range(4)], 1)
    return {
        "final avg score": m[:, 3].mean(1),
        "plasticity\n(own task after phase)": own.mean(1),
        "tracking forgotten\n(phase 0 - phase 2)": m[:, 0, 0] - m[:, 2, 0],
        "forgetting\n(waypoint, landing)": ((m[:, 1, 1] - m[:, 3, 1]) + (m[:, 2, 2] - m[:, 3, 2])) / 2,
        "tracking relearn\n(phase 3 - phase 0)": m[:, 3, 0] - m[:, 0, 0],
    }


def plot_summary(data: dict, out: Path) -> None:
    sums = {a: summary(np.array([matrix(d) for d in runs])) for a, runs in data.items()}
    names = list(next(iter(sums.values())))
    fig, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names), 3.6))
    for ax, name in zip(axes, names):
        for i, a in enumerate(data):
            v = sums[a][name]
            ax.bar(i, v.mean(), color=COLORS[a], alpha=0.7)
            ax.scatter([i] * len(v), v, color="k", s=12, zorder=3)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(len(data)), [NAMES[a] for a in data], rotation=30)
        ax.set_title(name, fontsize=10)
    fig.tight_layout()
    save(fig, out, dpi=110)
    for a in data:
        print(f"{NAMES[a]:11s}", {k.replace("\n", " "): round(float(v.mean()), 3) for k, v in sums[a].items()})


def plot_phases(data: dict, out: Path) -> None:
    grid = np.linspace(0, PHASE_LEN, 251)

    def curve(d, phase, key, smooth):
        e = d[(d["phase"] == phase) & d[key].notna() & d["phase_step"].notna()]
        x, y = e["phase_step"].to_numpy(), e[key].to_numpy()
        if smooth and len(y) > smooth:
            x, y = x[smooth - 1 :], np.convolve(y, np.ones(smooth) / smooth, "valid")
        return np.interp(grid, x, y, left=np.nan, right=np.nan) if len(y) else np.full(len(grid), np.nan)

    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharex=True)
    for row, (key, smooth, label) in enumerate([("episode/return", 100, "train return (moving avg)"),
                                                ("eval/return", 0, "eval return")]):
        for p in range(4):
            ax = axes[row, p]
            for algo, runs in data.items():
                if not runs:
                    continue
                c = np.array([curve(d, p, key, smooth) for d in runs])
                mu, sd = np.nanmean(c, 0), np.nanstd(c, 0)
                ax.plot(grid, mu, color=COLORS[algo], label=f"{NAMES[algo]} (n={len(runs)})")
                ax.fill_between(grid, mu - sd, mu + sd, color=COLORS[algo], alpha=0.2)
            if row == 0:
                ax.set_title(f"phase {PHASES[p]}")
            if p == 0:
                ax.set_ylabel(label)
            if row == 1:
                ax.set_xlabel("team steps in phase")
    axes[0, 0].legend()
    fig.tight_layout()
    save(fig, out, dpi=110, bbox_inches=None)


RUN_PANELS = [  # title, train key, eval key
    ("team return", "episode/return", "eval/return"),
    ("collision rate", "episode/collision", "eval/collision"),
    ("out of bounds", "episode/out_of_bounds", "eval/out_of_bounds"),
    ("episode length", "episode/length", "eval/length"),
    ("mean standoff error [m]", "episode/mean_standoff_error", "eval/mean_standoff_error"),
    ("mean heading error [rad]", "episode/mean_heading_error", "eval/mean_heading_error"),
    ("mean separation [m]", "episode/mean_separation", "eval/mean_separation"),
    ("target in view", "episode/mean_target_in_view", "eval/mean_target_in_view"),
    ("on target / success", "episode/mean_on_target", "eval/score"),
]


def plot_run(run: Path, out: Path, smooth: int = 200) -> None:
    d = pd.read_parquet(run / "metrics.parquet")
    starts = d.groupby("phase")["step"].min().iloc[1:]
    fig, axes = plt.subplots(3, 3, figsize=(18, 11), sharex=True)
    for ax, (title, train, evaluation) in zip(axes.flat, RUN_PANELS):
        if train in d:
            e = d.dropna(subset=[train])
            y = e[train].rolling(smooth).mean() if len(e) >= smooth else e[train]
            ax.plot(e["step"], y, color=COLORS["cstream_ac"], label="train (moving avg)")
        if evaluation in d:
            e = d.dropna(subset=[evaluation])
            ax.plot(e["step"], e[evaluation], color=COLORS["ippo"], marker=".", ms=3, lw=0.8, label="eval")
        for b in starts:
            ax.axvline(b, color="k", lw=0.6)
        ax.set_title(title)
    axes[0, 0].legend()
    for ax in axes[-1]:
        ax.set_xlabel("team steps (phases: " + " | ".join(TASKS[t] for t in SEQUENCE) + ")")
    fig.tight_layout()
    save(fig, out, dpi=110, bbox_inches=None)


# --- tables ------------------------------------------------------------------------------------------------------

def adaptation_steps(d: pd.DataFrame, phase: int, fraction: float) -> float:
    """Team steps from the start of ``phase`` until the eval return first covers ``fraction`` of the way from its
    value at the start of the phase to its best in the phase (nan if it never rises)."""
    e = d[(d["phase"] == phase) & d["eval/return"].notna()].groupby("phase_step")["eval/return"].last()
    start, best = e.iloc[0], e.max()
    if best <= start:
        return np.nan
    return float(e.index[np.argmax(e.to_numpy() >= start + fraction * (best - start))])


def continual_tables(data: dict, fraction: float, out_path: Path) -> None:
    seeds = sorted({len(r) for r in data.values()})
    mats = {a: np.array([matrix(d) for d in runs]) for a, runs in data.items()}  # (seed, phase, task)

    def probe(d, phase: int, task: int, key: str) -> float:
        r = d[(d["phase"] == phase) & (d["probe_task"] == task) & d[key].notna()]
        return float(r[key].iloc[-1])

    out = [
        "# Continual task sequence: tables",
        "",
        (f"Sequence tracking → waypoint → landing → tracking, 500k team steps per phase, N = 3 drones, "
        f"{'/'.join(map(str, seeds))} seeds per learner, mean ± std over seeds. Generated by "
        "`scripts/make_figures.py continual` from `runs_continual/`."),
        "",
        ("Score: the task's success rate (waypoint, landing) or the fraction of steps on target (tracking), in [0, 1]. "
        "Probe values are measured at the end of each phase on all three tasks (50 episodes). The runs use "
        "`--restore-best`, so a probe scores the phase's best checkpoint."),
        "",
        "## 1. Summary",
        "",
    ]
    rows = [[NAMES[a], *(pm(v) for v in summary(m).values())] for a, m in mats.items()]
    out += [
        table(["Learner", "Final avg. score ↑", "Plasticity ↑", "Tracking forgotten ↓", "Forgetting (waypoint, landing) ↓",
               "Tracking relearn ↑"], rows),
        "",
        "- **Final avg. score**: mean probe score over the three tasks after the last phase.",
        "- **Plasticity**: mean probe score on each phase's own task at the end of that phase.",
        "- **Tracking forgotten**: tracking score after phase 1 minus after phase 3 (before it is trained again).",
        ("- **Forgetting (waypoint, landing)**: score right after the task's own phase minus at the end of the sequence, "
        "averaged over the two tasks."),
        "- **Tracking relearn**: tracking score after phase 4 minus after phase 1 (> 0: better the second time).",
        "",
        "## 2. Performance matrices (probe score)",
        "",
        "Rows: the end of each phase; columns: the task probed. **Bold**: the phase's own task.",
        "",
    ]
    for a, m in mats.items():
        rows = [[f"after {PHASES[p]}"] + [(f"**{pm(m[:, p, t])}**" if t == SEQUENCE[p] else pm(m[:, p, t]))
                                          for t in range(3)] for p in range(4)]
        out += [f"### {NAMES[a]}", "", table(["", *TASKS], rows), ""]

    out += [
        "## 3. Return on the phase's own task (probe, end of phase)",
        "",
        ("Raw team return (not normalised; scales differ between tasks). *Frozen*: the policy at the end of phase 1, "
        "never trained again, on tracking at the end of phase 4."),
        "",
    ]
    rows = []
    for a, runs in data.items():
        own = [pm([probe(d, p, SEQUENCE[p], "probe/return") for d in runs], 1) for p in range(4)]
        frozen = [d[(d["phase"] == 3) & d["frozen/return"].notna()]["frozen/return"].iloc[-1] for d in runs]
        rows.append([NAMES[a], *own, pm(frozen, 1)])
    out += [table(["Learner", *PHASES, "Frozen (tracking)"], rows), ""]

    pct = round(100 * fraction)
    out += [
        f"## 4. Adaptation speed (team steps to {pct} % of the phase's gain)",
        "",
        (f"Steps from the task switch until the evaluation return first covers {pct} % of the way from its value at the "
        "start of the phase to its best in the phase. Evaluations every 50k steps (Stream-AC) or ~49k steps, the "
        "first at ~98k (IPPO, MAPPO, whole rollout batches), which bounds the resolution. –: no gain in the phase."),
        "",
    ]
    rows = [[NAMES[a]] + [pm([adaptation_steps(d, p, fraction) / 1e3 for d in runs], 0) + "k" for p in range(4)]
            for a, runs in data.items()]
    out += [table(["Learner", *PHASES], rows), ""]
    write(out_path, out)
