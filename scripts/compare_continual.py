"""E2.1 task sequence (tracking -> waypoint -> landing -> tracking): run the conditions, report, plot.

    uv run python scripts/compare_continual.py train --steps 1000000 --seeds 1 2 3
    uv run python scripts/compare_continual.py report
    uv run python scripts/compare_continual.py plot

Conditions (``streampilot-continual --method M --mode X``), runs/continual_M-X_seedS/:
  stream-continue  StreamX keeps learning          stream-scratch   StreamX, a fresh learner on every task
  mappo-continue   MAPPO fine-tuned                mappo-scratch    MAPPO, a fresh learner on every task (the reference for its transfer)
  stream-frozen    StreamX frozen after the first phase: scored inside the stream-continue runs (``frozen/`` keys)
Add ``--methods ippo`` for IPPO. ``--swap-phase P`` (stream-continue only) also swaps a drone mid-phase.

``report`` prints, as mean +- std over seeds, and writes runs/compare_continual.csv:
  per phase and condition (the end-of-phase probe, 50 episodes): success rate, time to complete [steps, successful
    episodes only], accuracy [m] (waypoint: distance to the ball, landing: slot error, tracking: standoff error)
  forward transfer (FT): per task, the area under the evaluation-score curve over the phase (zero-shot start
    included) minus that of the same task trained from scratch with the same method, in score units
  backward transfer (BWT): tracking score after the other tasks (end of the landing phase, before relearning) minus
    at the end of the first phase. Negative: forgetting
  recovery: steps from the start of a phase (the task change) until the score is >= --recover of the scratch
    learner's final score on that task (inf: never); after a swap, until it is >= --recover of the pre-swap score
"""

import argparse
import csv
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

TASKS = ["tracking", "waypoint", "landing"]
SEQUENCE = [0, 1, 2, 0]  # task id per phase
CONDITIONS = {  # name: (method, mode)
    "stream-continue": ("stream", "continue"),
    "stream-scratch": ("stream", "scratch"),
    "mappo-continue": ("mappo", "continue"),
    "mappo-scratch": ("mappo", "scratch"),
    "ippo-continue": ("ippo", "continue"),
    "ippo-scratch": ("ippo", "scratch"),
}
DEFAULT_CONDITIONS = ["stream-continue", "stream-scratch", "mappo-continue", "mappo-scratch"]
SINGLE_THREADED = ("stream-continue", "stream-scratch")


def run_dir(condition: str, drones: int, seed: int) -> Path:
    method, mode = CONDITIONS[condition]
    return Path("runs") / f"continual_{method}-{mode}{'' if drones == 3 else f'_{drones}d'}_seed{seed}"


def train(args: argparse.Namespace) -> int:
    def launch(condition: str, seed: int) -> subprocess.Popen:
        method, mode = CONDITIONS[condition]
        out = run_dir(condition, args.drones, seed)
        out.mkdir(parents=True, exist_ok=True)
        command = ["uv", "run", "streampilot-continual", "--method", method, "--mode", mode, "--drones", str(args.drones),
                   "--seed", str(seed), "--steps-per-task", str(args.steps), "--eval-every", str(args.eval_every),
                   "--out", str(out), *args.extra[args.extra[:1] == ["--"]:]]  # fmt: skip
        if args.swap_phase is not None and condition == "stream-continue":
            command += ["--swap-phase", str(args.swap_phase)]
        print(f"starting {out}", flush=True)
        return subprocess.Popen(command, stdout=(out / "train.log").open("w"), stderr=subprocess.STDOUT)

    failed, running = [], []

    def finish(proc, name):
        if proc.wait() != 0:
            failed.append(name)

    single = [(c, s) for c in args.conditions if c in SINGLE_THREADED for s in args.seeds]
    for condition, seed in single:
        if len(running) >= args.jobs:
            finish(*running.pop(0))
        running.append((launch(condition, seed), f"{condition} seed{seed}"))
    for item in running:
        finish(*item)
    for condition in args.conditions:  # MAPPO/IPPO already use every core for their environments
        if condition not in SINGLE_THREADED:
            for seed in args.seeds:
                finish(launch(condition, seed), f"{condition} seed{seed}")
    if failed:
        print(f"failed: {', '.join(failed)}; see runs/*/train.log", file=sys.stderr)
        return 1
    return 0


def load(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(path).to_pydict()
    rows = [{k: v[i] for k, v in table.items() if v[i] is not None} for i in range(len(table["step"]))]
    for r in rows:  # the parquet has only floats
        for key in ("phase", "task_id", "probe_task", "event_swap"):
            if key in r:
                r[key] = int(r[key])
    return rows


def probes(rows, phase: int, task: int) -> dict:
    return next(r for r in rows if r.get("probe_task") == task and r["phase"] == phase)


def curve(rows, phase: int, prefix: str = "eval") -> list[tuple[float, float]]:
    """``(phase_step, score)``, in order, of the periodic evaluations of a phase (swap rows excluded)."""
    return [(r["phase_step"], r[f"{prefix}/score"]) for r in rows
            if r["phase"] == phase and f"{prefix}/score" in r and "event_swap" not in r]  # fmt: skip


def auc(points: list[tuple[float, float]]) -> float:
    """Mean score over the phase (trapezoid rule, normalised by its length)."""
    import numpy as np

    x, y = np.array(points).T
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def first_reach(points, target: float) -> float:
    return next((x for x, y in points if y >= target), float("inf"))


def metrics_of(condition: str, rows: list[dict], scratch_rows: list[dict] | None, args) -> list[dict]:
    """One row per (phase, quantity) for a run."""
    out = []
    for phase, task in enumerate(SEQUENCE):
        p = probes(rows, phase, task)
        row = {"condition": condition, "phase": phase, "task": TASKS[task],
               "success": p["probe/success"], "time": p.get("probe/time_to_complete", float("nan")),
               "accuracy": p["probe/accuracy"], "score": p["probe/score"]}  # fmt: skip
        if phase > 0 and scratch_rows is not None and condition != "stream-scratch":
            ours, ref = curve(rows, phase), curve(scratch_rows, phase)
            row["forward_transfer"] = auc(ours) - auc(ref)
            row["recovery_steps"] = first_reach(ours, args.recover * ref[-1][1])
        out.append(row)
    return out


def swap_recovery(rows, args) -> dict | None:
    swap = [r for r in rows if r.get("event_swap") == 0]
    if not swap:
        return None
    pre, phase = swap[0], swap[0]["phase"]
    after = [(r["phase_step"] - pre["phase_step"], r["eval/score"]) for r in rows
             if r["phase"] == phase and r["phase_step"] >= pre["phase_step"] and "eval/score" in r and r.get("event_swap") != 0]  # fmt: skip
    drop = after[0][1]  # event_swap == 1: right after the swap
    return {"pre_swap": pre["eval/score"], "after_swap": drop,
            "recovery_steps": first_reach(after, args.recover * pre["eval/score"])}  # fmt: skip


def frozen_rows(rows: list[dict]) -> list[dict]:
    out = []
    for phase, task in enumerate(SEQUENCE):
        ends = [r for r in rows if r["phase"] == phase and "frozen/score" in r]
        if ends:
            e = ends[-1]
            out.append({"condition": "stream-frozen", "phase": phase, "task": TASKS[task], "success": e["frozen/success"],
                        "time": e.get("frozen/time_to_complete", float("nan")), "accuracy": e["frozen/accuracy"],
                        "score": e["frozen/score"]})  # fmt: skip
    return out


def report(args: argparse.Namespace) -> int:
    import numpy as np

    runs: dict[str, dict[int, list[dict]]] = defaultdict(dict)
    for condition in CONDITIONS:
        for seed in args.seeds:
            path = run_dir(condition, args.drones, seed) / "metrics.parquet"
            if path.exists():
                runs[condition][seed] = load(path)
    if not runs:
        print("no runs found", file=sys.stderr)
        return 1

    table: list[dict] = []
    for condition, by_seed in runs.items():
        scratch = runs.get(condition.replace("continue", "scratch"), {})
        for seed, rows in by_seed.items():
            for r in metrics_of(condition, rows, scratch.get(seed), args) + (frozen_rows(rows) if condition == "stream-continue" else []):
                table.append({"seed": seed, **r})
            if condition.endswith("continue"):  # backward transfer: tracking before vs after the other tasks
                before, after = probes(rows, 0, 0)["probe/score"], probes(rows, 2, 0)["probe/score"]
                table.append({"seed": seed, "condition": condition, "phase": 3, "task": "tracking",
                              "backward_transfer": after - before})  # fmt: skip
                frozen_after = rows[-1].get("frozen/score")
                if condition == "stream-continue" and frozen_after is not None:
                    table.append({"seed": seed, "condition": "stream-frozen", "phase": 3, "task": "tracking",
                                  "backward_transfer": 0.0})  # fmt: skip
            swap = swap_recovery(rows, args)
            if swap:
                table.append({"seed": seed, "condition": condition, "phase": next(r["phase"] for r in rows if r.get("event_swap") == 0),
                              "task": "swap", **{f"swap_{k}": v for k, v in swap.items()}})  # fmt: skip

    def cell(rows, key, fmt):
        vals = [r[key] for r in rows if key in r and not np.isnan(r[key])]
        if not vals:
            return "-"
        if any(np.isinf(vals)):
            return f"inf ({sum(np.isinf(vals))}/{len(vals)} never)" if len(vals) > 1 else "inf"
        return f"{np.mean(vals):{fmt}}" + (f" ± {np.std(vals):{fmt}}" if len(vals) > 1 else "")

    keys = [("success", ".2f"), ("time", ".0f"), ("accuracy", ".3f"), ("score", ".2f"), ("forward_transfer", "+.3f"),
            ("recovery_steps", ".0f"), ("backward_transfer", "+.3f"), ("swap_pre_swap", ".2f"), ("swap_after_swap", ".2f"),
            ("swap_recovery_steps", ".0f")]  # fmt: skip
    groups = sorted({(r["phase"], r["task"], r["condition"]) for r in table})
    print(f"{'phase':<6}{'task':<10}{'condition':<18}" + "".join(f"{k:<22}" for k, _ in keys))
    for phase, task, condition in groups:
        rows = [r for r in table if (r["phase"], r["task"], r["condition"]) == (phase, task, condition)]
        print(f"{phase:<6}{task:<10}{condition:<18}" + "".join(f"{cell(rows, k, f):<22}" for k, f in keys))
    out = Path("runs/compare_continual.csv")
    fields = ["seed", "condition", "phase", "task", *[k for k, _ in keys]]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fields, restval="")
        writer.writeheader()
        writer.writerows(table)
    print(f"wrote {out}")
    return 0


def plot(args: argparse.Namespace) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    panels = [("score", "score (success / on target)"), ("success", "success rate (tracking: survival)"),
              ("time_to_complete", "time to complete [steps]"), ("accuracy", "accuracy: final error [m]")]  # fmt: skip
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 3 * len(panels)), sharex=True)
    first = None
    for condition in CONDITIONS:
        by_seed = [load(p) for s in args.seeds if (p := run_dir(condition, args.drones, s) / "metrics.parquet").exists()]
        if not by_seed:
            continue
        first = first or by_seed[0]
        for ax, (key, label) in zip(axes, panels):
            series = [[(r["step"], r[f"eval/{key}"]) for r in rows if f"eval/{key}" in r and "event_swap" not in r] for rows in by_seed]
            x = np.array([p[0] for p in series[0]])
            y = np.array([[p[1] for p in s] for s in series if len(s) == len(x)])
            ax.plot(x, np.nanmean(y, 0), label=condition, linestyle="--" if "scratch" in condition else "-")
            ax.fill_between(x, np.nanmin(y, 0), np.nanmax(y, 0), alpha=0.15)
            ax.set_ylabel(label)
            if condition == "stream-continue":
                frozen = [[(r["step"], r[f"frozen/{key}"]) for r in rows if f"frozen/{key}" in r] for rows in by_seed]
                fx = np.array([p[0] for p in frozen[0]])
                fy = np.array([[p[1] for p in s] for s in frozen if len(s) == len(fx)])
                ax.plot(fx, np.nanmean(fy, 0), color="gray", linestyle=":", label="stream-frozen")
    if first is None:
        print("no runs found", file=sys.stderr)
        return 1
    steps = max(r["step"] for r in first) / len(SEQUENCE)
    for ax in axes:
        for i in range(1, len(SEQUENCE)):
            ax.axvline(i * steps, color="k", linewidth=0.5)
    for i, task in enumerate(SEQUENCE):
        axes[0].text((i + 0.5) * steps, 1.02, TASKS[task], ha="center", transform=axes[0].get_xaxis_transform())
    axes[0].legend(ncols=3, loc="lower left", fontsize=8)
    axes[-1].set_xlabel("team steps")
    out = Path("runs/compare_continual_curves.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "report", "plot"):
        p = sub.add_parser(name)
        p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
        p.add_argument("--drones", type=int, default=3)
        if name == "train":
            p.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=DEFAULT_CONDITIONS)
            p.add_argument("--methods", nargs="+", choices=["ippo"], default=[], help="add IPPO's conditions")
            p.add_argument("--steps", type=int, default=1_000_000, help="K, team steps per task")
            p.add_argument("--eval-every", type=int, default=50_000)
            p.add_argument("--jobs", type=int, default=4, help="Stream AC runs at a time")
            p.add_argument("--swap-phase", type=int, default=None)
            p.add_argument("extra", nargs=argparse.REMAINDER, help="passed to streampilot-continual")
        if name == "report":
            p.add_argument("--recover", type=float, default=0.9, help="fraction of the reference score that counts as recovered")
    args = parser.parse_args()
    if args.command == "train":
        args.conditions += [c for c in CONDITIONS if CONDITIONS[c][0] in args.methods and c not in args.conditions]
    return {"train": train, "report": report, "plot": plot}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
