"""Formation tracking (runs/): final scores from runs/compare_formation_tracking.csv, normalised by 500.

rl_curves        eval return vs team steps per N, 95 % bootstrap CI
rl_aggregate     median, IQM, mean, optimality gap with stratified bootstrap CIs
rl_profiles      performance profile, sample efficiency, probability of improvement
rl_tables        rl_tables.md
failure_curves   training collision / out-of-bounds rates (per N, and per learner)
scaling          final on-target and formation error vs N (broken runs dropped unless --keep-outliers)
results_table    results_table.tex, from runs/_eval
"""

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from .common import (
    AGGREGATES,
    ALGOS,
    COLORS,
    DRONES,
    MARKERS,
    MAX_RETURN,
    NAMES,
    ORDER,
    REPS,
    SEEDS,
    bold_best,
    ci,
    improvement_ci,
    iqm,
    learner_legend,
    load_runs,
    millions,
    pm,
    probability_of_improvement,
    save,
    stratified_bootstrap,
    style,
    table,
    write,
)

OUTPUTS = ["rl_curves", "rl_aggregate", "rl_profiles", "rl_tables", "failure_curves", "scaling", "results_table"]
GRID = np.linspace(1e5, 2e6, 20)  # team steps, for curves shared across learners


def add_args(parser) -> None:
    parser.add_argument("--root", type=Path, default=Path("runs"))
    parser.add_argument("--keep-outliers", action="store_true", help="scaling: keep the broken runs")


def run(args, outputs: list[str]) -> None:
    root, out = args.root, args.out_dir
    if {"rl_curves", "rl_profiles", "rl_tables", "failure_curves"} & set(outputs):
        runs = load_runs(root)
        returns = eval_returns(runs)
    if {"rl_aggregate", "rl_profiles", "rl_tables"} & set(outputs):
        scores = final_scores(root)
        for a in ALGOS:
            print(f"{NAMES[a]:11s} final score per N {np.round(scores[a].mean(0), 3)}")
    if "rl_curves" in outputs:
        plot_curves(returns, out / "rl_curves.pdf")
    if "rl_aggregate" in outputs:
        plot_aggregate(scores, out / "rl_aggregate.pdf")
    if "rl_profiles" in outputs:
        plot_profiles(returns, scores, out / "rl_profiles.pdf")
    if "rl_tables" in outputs:
        rl_tables(root, returns, scores, out / "rl_tables.md")
    if "failure_curves" in outputs:
        plot_failures(runs, out / "failure_curves.png", by_n=False)
        plot_failures(runs, out / "failure_curves_by_n.png", by_n=True)
    if "scaling" in outputs:
        plot_scaling(root / "compare_formation_tracking.csv", out / "scaling.pdf", args.keep_outliers)
    if "results_table" in outputs:
        results_table(root / "_eval", out / "results_table.tex")


def eval_returns(runs: dict) -> dict:
    """(algo, n) -> list over SEEDS of the evaluation return by step (last row per step)."""
    out = {}
    for (algo, n), df in runs.items():
        out[algo, n] = [df[df["seed"] == s].dropna(subset=["eval/return"]).groupby("step")["eval/return"].last()
                        for s in SEEDS]
    return out


def final_scores(root: Path) -> dict:
    """algo -> (seeds, tasks) final normalised score."""
    pivot = pd.read_csv(root / "compare_formation_tracking.csv").pivot_table(
        index=["method", "seed"], columns="drones", values="return")
    return {a: pivot.loc[a].loc[SEEDS, DRONES].to_numpy() / MAX_RETURN for a in ALGOS}


# --- rliable figures ---------------------------------------------------------------------------------------------

def plot_curves(returns: dict, out: Path) -> None:
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(DRONES), figsize=(10, 2.6), sharey=True)
    for ax, n in zip(axes, DRONES):
        for algo in ALGOS:
            seeds = returns[algo, n]
            y = np.array([e.to_numpy() for e in seeds])
            lo, hi = ci(y[rng.integers(0, len(y), size=(REPS, len(y)))].mean(1))
            ax.plot(seeds[0].index, y.mean(0), color=COLORS[algo], linewidth=1.4, marker=MARKERS[algo], markersize=2.5,
                    markevery=3)
            ax.fill_between(seeds[0].index, lo, hi, color=COLORS[algo], alpha=0.15, linewidth=0)
        ax.set_title(f"N = {n}", fontsize=10)
        ax.set_xticks([0, 1e6, 2e6], ["0", "1M", "2M"])
        ax.set_xlabel("team steps", fontsize=8.5)
        style(ax)
    axes[0].set_ylabel("evaluation return")
    axes[0].set_ylim(0, MAX_RETURN)
    learner_legend(fig, 1.12)
    fig.tight_layout()
    save(fig, out)


def interval_bar(ax, k: int, point: float, lo: float, hi: float, color: str) -> None:
    ax.barh(k, hi - lo, left=lo, height=0.6, color=color, alpha=0.75)
    ax.plot([point, point], [k - 0.3, k + 0.3], color="0.1", linewidth=1.5)


def bar_axis(ax) -> None:
    ax.grid(axis="x", color="0.9", linewidth=0.6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)


def plot_aggregate(scores: dict, out: Path) -> None:
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(AGGREGATES), figsize=(10, 2.2), sharey=True)
    for ax, (name, fn) in zip(axes, AGGREGATES.items()):
        for k, algo in enumerate(ALGOS):
            interval_bar(ax, k, fn(scores[algo][None])[0], *ci(fn(stratified_bootstrap(scores[algo], rng))),
                         COLORS[algo])
        ax.set_title(name + (" ↓" if name == "Optimality gap" else " ↑"), fontsize=10)
        ax.set_yticks(range(len(ALGOS)), [NAMES[a] for a in ALGOS])
        bar_axis(ax)
    axes[0].invert_yaxis()  # shared: once for all
    fig.supxlabel("normalised return", fontsize=9, y=0.02)
    fig.tight_layout()
    save(fig, out)


def plot_profiles(returns: dict, scores: dict, out: Path) -> None:
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    taus = np.linspace(0, 1, 101)
    ax = axes[0]
    for algo in ALGOS:
        s = scores[algo]
        boot = stratified_bootstrap(s, rng).reshape(REPS, -1)
        lo, hi = ci((boot[..., None] > taus).mean(1))
        ax.plot(taus, (s.ravel()[:, None] > taus).mean(0), color=COLORS[algo], linewidth=1.4)
        ax.fill_between(taus, lo, hi, color=COLORS[algo], alpha=0.15, linewidth=0)
    ax.set_xlim(0.4, 1.0)  # every run scores above 0.4
    ax.set_xlabel(r"normalised score $\tau$")
    ax.set_ylabel(r"fraction of runs with score $> \tau$")
    ax.set_title("(a) Performance profile", fontsize=10)
    style(ax)

    ax = axes[1]
    for algo in ALGOS:
        curves = np.array([[np.interp(GRID, returns[algo, n][s].index, returns[algo, n][s]) / MAX_RETURN
                            for n in DRONES] for s in range(len(SEEDS))])  # (seeds, tasks, steps)
        point = iqm(curves.reshape(-1, len(GRID)).T, axis=1)
        idx = rng.integers(0, len(SEEDS), size=(REPS, len(SEEDS), len(DRONES)))
        boot = np.stack([iqm(np.take_along_axis(curves[..., t][None].repeat(REPS, 0), idx, axis=1), axis=1)
                         for t in range(len(GRID))], -1)
        lo, hi = ci(boot)
        ax.plot(GRID, point, color=COLORS[algo], linewidth=1.4, marker=MARKERS[algo], markersize=2.5, markevery=3)
        ax.fill_between(GRID, lo, hi, color=COLORS[algo], alpha=0.15, linewidth=0)
    ax.set_xticks([0, 1e6, 2e6], ["0", "1M", "2M"])
    ax.set_xlabel("team steps")
    ax.set_ylabel("IQM normalised score")
    ax.set_title("(b) Sample efficiency", fontsize=10)
    style(ax)

    ax = axes[2]
    pairs = list(combinations(ORDER, 2))
    for k, (x, y) in enumerate(pairs):
        interval_bar(ax, k, probability_of_improvement(scores[x], scores[y]), *improvement_ci(scores[x], scores[y], rng),
                     COLORS[x])
    ax.axvline(0.5, color="0.5", linestyle="--", linewidth=0.8)
    ax.set_yticks(range(len(pairs)), [f"{NAMES[x]} > {NAMES[y]}" for x, y in pairs], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("P(X > Y)")
    ax.set_title("(c) Probability of improvement", fontsize=10)
    bar_axis(ax)
    learner_legend(fig, 1.1)
    fig.tight_layout()
    save(fig, out)


# --- tables ------------------------------------------------------------------------------------------------------

METRICS = [  # column of the eval csv, header, digits
    ("on target", "On target ↑", 2),
    ("survival", "Survival ↑", 2),
    ("collision", "Collision ↓", 2),
    ("distance [m]", "Standoff err. [m] ↓", 3),
    ("heading [deg]", "Heading err. [°] ↓", 1),
    ("formation [m]", "Formation err. [m] ↓", 3),
]
THRESHOLDS = [0.7, 0.8, 0.9]


def rl_tables(root: Path, returns: dict, scores: dict, out_path: Path) -> None:
    rng = np.random.default_rng(0)
    evals = pd.read_csv(root / "compare_formation_tracking.csv")
    evals = evals[evals["seed"].isin(SEEDS) | (evals["method"] == "pid")]
    learners = ORDER

    out = [
        "# Formation tracking: tables",
        "",
        (f"4 learners × team size N ∈ {{{', '.join(map(str, DRONES))}}} × {len(SEEDS)} seeds, 2M team steps each. "
        "Final values: every final checkpoint re-evaluated deterministically on the same 100 episodes "
        "(`compare_formation_tracking.py eval`); mean ± std over seeds. **Normalised return** = return / 500, "
        "the maximum (500 steps, at most 1 per step); even a perfect policy stays below 1, since the team first has "
        "to fly into formation. **Bold**: best learner in the column (none on a tie). At N = 1, IPPO and MAPPO are "
        "the same algorithm (a centralised critic of one drone) with the same seeds, so their rows are identical. "
        "Generated by `scripts/make_figures.py tracking`."),
        "",
        "## 1. Final normalised return per team size",
        "",
    ]
    body = [[NAMES[a]] for a in learners]
    for n in DRONES:
        col = [scores[a][:, DRONES.index(n)] for a in learners]
        for row, c in zip(body, bold_best([pm(v) for v in col], [np.mean(v) for v in col])):
            row.append(c)
    allruns = [pm(scores[a].mean(1)) for a in learners]  # mean over N, then mean +- std over seeds
    for row, c in zip(body, bold_best(allruns, [scores[a].mean() for a in learners])):
        row.append(c)
    pid = evals[evals["method"] == "pid"].set_index("drones")["return"] / MAX_RETURN
    body.append(["PID (no learning)"] + [pm(pid.get(n, np.nan)) for n in DRONES] + [pm(pid.reindex(DRONES).mean())])
    out += [table(["Learner", *(f"N = {n}" for n in DRONES), "Mean over N"], body), "",
            "*Mean over N*: per seed, the mean over the team sizes; then mean ± std over seeds.", ""]

    out += ["## 2. Task metrics of the final policies", "",
            ("Per episode, averaged over drones and steps, then over the 100 episodes; mean ± std over seeds. "
            "*On target*: fraction of steps within the tracking tolerance; *survival*: episodes reaching the time "
            "limit without a collision or leaving the arena."), ""]
    for n in DRONES:
        sub = evals[evals["drones"] == n]
        body = [[NAMES[a]] for a in learners] + [["PID"]]
        for col, header, digits in METRICS:
            values = [sub[sub["method"] == a][col].to_numpy() for a in learners]
            for row, c in zip(body, bold_best([pm(v, digits) for v in values], [np.nanmean(v) for v in values],
                                              "↑" in header)):
                row.append(c)
            body[-1].append(pm(sub[sub["method"] == "pid"][col].to_numpy(), digits))
        out += [f"### N = {n}", "", table(["Learner", *(h for _, h, _ in METRICS)], body), ""]

    out += ["## 3. Aggregate final performance (all team sizes)", "",
            (f"Over all {len(SEEDS) * len(DRONES)} runs per learner (rliable-style); point estimate with a 95 % "
            f"stratified bootstrap CI ({REPS} resamples, seeds resampled within each N). *IQM*: interquartile mean; "
            "*optimality gap*: mean shortfall from 1."), ""]
    body = []
    for a in learners:
        boot = stratified_bootstrap(scores[a], rng)
        row = [NAMES[a]]
        for fn in AGGREGATES.values():
            lo, hi = ci(fn(boot))
            row.append(f"{fn(scores[a][None])[0]:.3f} [{lo:.3f}, {hi:.3f}]")
        body.append(row)
    out += [table(["Learner", *(n + (" ↓" if n == "Optimality gap" else " ↑") for n in AGGREGATES)], body), ""]

    out += ["## 4. Learning speed", "",
            ("Team steps until the evaluation return during training first reaches a fraction of the maximum "
            "(evaluations every ~100k steps, which bounds the resolution); mean ± std over seeds, in thousands. "
            "> 2M: not reached by some seed (the mean is over the seeds that reached it, their count in brackets)."), ""]
    for n in DRONES:
        body = []
        for a in learners:
            row = [NAMES[a]]
            for th in THRESHOLDS:
                hits = []
                for e in returns[a, n]:
                    reached = np.flatnonzero(e.to_numpy() / MAX_RETURN >= th)
                    hits.append(e.index[reached[0]] / 1e3 if reached.size else np.nan)
                hits = np.array(hits)
                ok = ~np.isnan(hits)
                cell = pm(hits[ok], 0) + "k" if ok.any() else "> 2M"
                if not ok.all() and ok.any():
                    cell = f"> 2M ({ok.sum()}/{len(hits)}: {cell})"
                row.append(cell)
            body.append(row)
        out += [f"### N = {n}", "", table(["Learner", *(f"to {th:g}" for th in THRESHOLDS)], body), ""]

    out += ["### IQM normalised return during training (all team sizes)", ""]
    checkpoints = [2.5e5, 5e5, 1e6, 2e6]
    body = []
    for a in learners:
        row = [NAMES[a]]
        for c in checkpoints:
            v = np.array([[np.interp(c, returns[a, n][s].index, returns[a, n][s]) / MAX_RETURN for n in DRONES]
                          for s in range(len(SEEDS))])
            lo, hi = ci(iqm(stratified_bootstrap(v, rng), axis=1))
            row.append(f"{iqm(v):.3f} [{lo:.3f}, {hi:.3f}]")
        body.append(row)
    out += [table(["Learner", *(f"{c / 1e6:g}M steps" if c >= 1e6 else f"{c / 1e3:g}k steps" for c in checkpoints)],
                  body), "", "Interpolated between evaluations; 95 % stratified bootstrap CI.", ""]

    out += ["## 5. Probability of improvement", "",
            ("P(X > Y): the probability that a run of X ends with a higher final return than a run of Y on the same "
            "team size, averaged over the team sizes (ties count half); 95 % stratified bootstrap CI. Above 0.5: X "
            "tends to beat Y. With 5 seeds per N, a value of 1.00 means every X run beat every Y run."), ""]
    body = []
    for x, y in combinations(learners, 2):
        lo, hi = improvement_ci(scores[x], scores[y], rng)
        body.append([f"{NAMES[x]} > {NAMES[y]}", f"{probability_of_improvement(scores[x], scores[y]):.2f}",
                     f"[{lo:.2f}, {hi:.2f}]"])
    out += [table(["X > Y", "P(X > Y)", "95 % CI"], body), ""]
    write(out_path, out)


# Columns of results_table: header -> (csv column, scale, decimals, LaTeX header); env reward is the mean over drones
COLUMNS = {
    "Return/drone": ("return", 1, 1, "Return/drone"),
    "Formation err [m]": ("formation [m]", 1, 3, "Formation err.\\ [m]"),
    "Heading err [deg]": ("heading [deg]", 1, 1, "Heading err.\\ [$^\\circ$]"),
    "Standoff err [m]": ("distance [m]", 1, 3, "Standoff err.\\ [m]"),
    "Collision %": ("collision", 100, 1, "Collision [\\%]"),
    "OOB %": ("oob", 100, 1, "OOB [\\%]"),
    "Survival %": ("survival", 100, 1, "Survival [\\%]"),
}
# (family, critic) -> method; with N = 1 both critics see the same input, so only the independent run is shown
FAMILIES = {"Stream AC": {"Indep.": "istream_ac", "Central.": "cstream_ac"}, "PPO": {"Indep.": "ippo", "Central.": "mappo"}}
# Meaningless with a single drone: its slot error is its standoff error, and there is no one to hit
SINGLE_DRONE_NA = {"Formation err [m]", "Collision %"}


def results_table(eval_dir: Path, out: Path) -> None:
    """One booktabs table, a block per algorithm family (needs \\usepackage{booktabs, graphicx}); also printed as
    markdown."""
    df = pd.concat([pd.read_csv(f) for f in sorted(eval_dir.glob("*.csv"))], ignore_index=True)
    df["oob"] = 1 - df["survival"] - df["collision"]  # only flagged when the step didn't already end in a collision

    def rows(family: str) -> list[list[str]]:
        rows = []
        for n in sorted(df["drones"].unique()):
            for critic, method in FAMILIES[family].items():
                if n == 1 and critic != "Indep.":
                    continue
                runs = df[(df["drones"] == n) & (df["method"] == method)]
                row = [str(n), "—" if n == 1 else critic]
                for label, (col, scale, dec, _) in COLUMNS.items():
                    v = runs[col] * scale
                    row.append("—" if n == 1 and label in SINGLE_DRONE_NA else f"{v.mean():.{dec}f} ± {v.std():.{dec}f}")
                rows.append(row)
        return rows

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        ("\\caption{Final evaluation on formation tracking: mean $\\pm$ std over 5 seeds, 100 episodes per seed. "
        "With $N=1$ both critics see the same input, so one row is shown; formation error and collisions are undefined.}"),
        "\\label{tab:formation-tracking-results}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l l " + "c " * len(COLUMNS) + "}",
        "\\toprule",
        "$N$ & Critic & " + " & ".join(c[3] for c in COLUMNS.values()) + " \\\\",
    ]
    for family in FAMILIES:
        print(f"\n### {family} (mean ± std over seeds, 100 eval episodes per seed)\n")
        print(table(["N", "Critic", *COLUMNS], rows(family)))
        lines += ["\\midrule", f"\\multicolumn{{{len(COLUMNS) + 2}}}{{l}}{{\\textit{{{family}}}}} \\\\", "\\midrule"]
        for row in rows(family):
            cells = row[:2] + [v if v == "—" else "$" + v.replace("±", "\\pm") + "$" for v in row[2:]]
            lines.append(" & ".join(c.replace("—", "---") for c in cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}", ""]
    write(out, lines)


# --- failures and scaling ----------------------------------------------------------------------------------------

FAILURES = {"episode/collision": "Collision rate", "episode/out_of_bounds": "Out-of-bounds rate"}
N_COLORS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # ordinal blue ramp, light -> dark with team size


def binned(df: pd.DataFrame, metric: str, bin_steps: int) -> pd.DataFrame:
    """Episode-weighted mean of ``metric`` per (seed, step bin). Stream runs log one row per episode; PPO runs log one
    row per update with ``episode/count`` episodes behind it."""
    df = df.dropna(subset=[metric])
    weight = df["episode/count"] if "episode/count" in df else pd.Series(1.0, index=df.index)
    out = pd.DataFrame({"seed": df["seed"], "bin": (df["step"] // bin_steps + 0.5) * bin_steps,
                        "wx": df[metric] * weight, "w": weight})
    out = out.groupby(["seed", "bin"])[["wx", "w"]].sum()
    return (out["wx"] / out["w"]).rename(metric).reset_index()


def plot_failures(runs: dict, out: Path, by_n: bool, bin_steps: int = 50_000) -> None:
    """Panels per N, a line per learner; or (by_n) panels per learner, a line per N."""
    algos = [a for a in ALGOS if any(k[0] == a for k in runs)]
    drones = sorted({n for _, n in runs})
    panels, lines = (algos, list(zip(drones, N_COLORS))) if by_n else (drones, [(a, COLORS[a]) for a in algos])
    fig, axes = plt.subplots(len(FAILURES), len(panels), figsize=(3.2 * len(panels), 5.5), sharex=True, sharey="row")
    for row, (metric, label) in enumerate(FAILURES.items()):
        for col, panel in enumerate(panels):
            ax = axes[row, col]
            for line, color in lines:
                key = (panel, line) if by_n else (line, panel)
                if key not in runs:
                    continue
                curve = binned(runs[key], metric, bin_steps).groupby("bin")[metric].agg(["mean", "min", "max"])
                ax.plot(curve.index, curve["mean"], color=color, lw=2, label=f"N = {line}" if by_n else NAMES[line])
                ax.fill_between(curve.index, curve["min"], curve["max"], color=color, alpha=0.12, lw=0)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(NAMES[panel] if by_n else f"{panel} drone{'s' if panel > 1 else ''}")
            if col == 0:
                ax.set_ylabel(label)
            if row == len(FAILURES) - 1:
                ax.set_xlabel("Environment steps")
                millions(ax)
    axes[0, 0].legend(frameon=False, fontsize=8)
    what = "Failure rate vs team size N" if by_n else "Training episodes ending in failure"
    fig.suptitle(f"{what} ({bin_steps // 1000}k-step bins; line = seed mean, band = seed min–max)", fontsize=10)
    fig.tight_layout()
    save(fig, out, dpi=150, bbox_inches=None)

    if not by_n:  # final-bin numbers for a quick read
        for metric in FAILURES:
            rows = []
            for (algo, n), df in sorted(runs.items()):
                per_seed = binned(df, metric, bin_steps)
                last = per_seed[per_seed["bin"] == per_seed["bin"].max()][metric]
                rows.append({"algo": algo, "n_drones": n, "last_bin_mean": last.mean()})
            print(f"\n{metric} (last {bin_steps // 1000}k steps)")
            print(pd.DataFrame(rows).pivot(index="algo", columns="n_drones", values="last_bin_mean").round(3).to_string())


SCALING = [  # csv column, y label, min N, log scale
    ("on target", "on-target fraction ↑", 1, False),
    ("formation [m]", "formation error [m] ↓", 2, True),
]


def plot_scaling(csv: Path, out: Path, keep_outliers: bool) -> None:
    """Lines: IQM over seeds; every seed a dot instead of a CI band, since outcomes are bimodal (a seed either solves
    the task or collapses). Formation error starts at N = 2. PID: a single deterministic reference run."""
    df = pd.read_csv(csv)
    broken = (df["formation [m]"] > 0.3) | (df["in view"] < 0.9)  # healthy runs: <= 0.09 m and 100 % in view
    if not keep_outliers:
        for (m, n), g in df[broken].groupby(["method", "drones"]):
            print(f"excluded {m} N={n}: seed {', '.join(map(str, g['seed']))}")
        df = df[~broken]
    ns = sorted(df["drones"].unique())

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, len(SCALING), figsize=(7.0, 2.8))
    for ax, (key, label, n_min, log) in zip(axes, SCALING):
        xs = [n for n in ns if n >= n_min]
        for i, m in enumerate(ORDER):
            d = df[(df["method"] == m) & (df["drones"] >= n_min)]
            if d.empty:
                continue
            per_n = d.groupby("drones")[key]
            ax.plot(per_n.apply(lambda v: iqm(v.to_numpy())), color=COLORS[m], marker=MARKERS[m], ms=5, lw=1.8,
                    label=NAMES[m], zorder=3)
            off = (i - (len(ORDER) - 1) / 2) * 0.07
            ax.scatter(d["drones"] + off, d[key], color=COLORS[m], s=9, alpha=0.45, lw=0, zorder=2)
        pid = df[(df["method"] == "pid") & (df["drones"] >= n_min)].groupby("drones")[key].first()
        if not pid.empty:
            ax.plot(pid, color="0.35", ls="--", lw=1.2, label="PID", zorder=1)
        ax.set_xticks(xs)
        ax.set_xlim(xs[0] - 0.4, xs[-1] + 0.4)
        ax.set_xlabel("number of drones $N$")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25, lw=0.6)
        if log:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
        else:
            ax.set_ylim(-0.04, 1.04)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, out)
    plt.rcParams.update(plt.rcParamsDefault)
