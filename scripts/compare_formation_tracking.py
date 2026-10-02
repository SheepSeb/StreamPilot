"""Formation tracking: independent Stream AC, CTDE Stream AC, IPPO, MAPPO and PID, over team sizes and seeds.

    uv run python scripts/compare_formation_tracking.py train      # 4 learners x N in 1 2 3 5 x seeds 1-3, 2M steps
    uv run python scripts/compare_formation_tracking.py eval       # final table (learners from final.pt, plus PID)
    uv run python scripts/compare_formation_tracking.py plot       # learning curves, one panel per N
    uv run python scripts/compare_formation_tracking.py train --drones 2 --methods mappo --seeds 1

``train`` runs the Stream AC learners (single-threaded, one process each) ``--jobs`` at a time, then
IPPO and MAPPO one run at a time (they already parallelise their environments over all cores). Runs go
to runs/METHOD_formation-tracking_Nd_seedSEED/ (train.log, metrics.parquet, final.pt). PID needs no training.

``eval`` re-evaluates every final.pt, and PID, deterministically on the same episodes (seeds 10000...),
so the methods see identical target trajectories, and prints mean +- std over seeds for each N. It also
writes runs/compare_formation_tracking.csv (one row per run).

``plot`` draws the evaluation return (mean and min-max band over seeds) from the metrics.parquet of each
run, with PID as a dashed line. Writes runs/compare_formation_tracking_curves[_X].png. ``--x`` picks the axis:
  steps    team steps (default): environment interactions, the same budget for every method
  updates  network updates: one per actor and critic update. Independent Stream AC does 2N per team step,
           CTDE Stream AC N + 1, IPPO/MAPPO 2 per minibatch step (epochs x minibatches per rollout)
  wall     training seconds, evaluation excluded (PPO runs its environments on all cores, Stream AC on one)

Metrics (per episode, averaged over the drones and steps, then over episodes):
  distance   leader standoff error to the target [m]      heading    heading error to the target [deg]
  formation  slot error [m]                               in view    fraction of steps the target is seen
  collision  fraction of episodes with a collision        on target  fraction of steps within the tracking tolerance
  survival   fraction of episodes that reach the time limit without a collision or leaving
  drone dist mean pairwise distance between drones [m]    min dist   mean closest-pair distance [m]
  return     team return (mean of the drones' returns)
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

TASK = "formation-tracking"
# method -> training command after `uv run` (None: nothing to train)
METHODS = {
    "istream_ac": ["streampilot-train-formation-stream", "--critic", "independent"],
    "cstream_ac": ["streampilot-train-formation-stream", "--critic", "centralized"],
    "ippo": ["streampilot-train-formation", "--algo", "ippo"],
    "mappo": ["streampilot-train-formation", "--algo", "mappo"],
    "pid": None,
}
LEARNERS = [m for m, cmd in METHODS.items() if cmd]
SINGLE_THREADED = ("istream_ac", "cstream_ac")
EVAL_SEED = 10_000
# label, summary key, scale
METRICS = [
    ("distance [m]", "eval/mean_standoff_error", 1.0),
    ("heading [deg]", "eval/mean_heading_error", 180 / 3.141592653589793),
    ("formation [m]", "eval/mean_formation_error", 1.0),
    ("in view", "eval/mean_target_in_view", 1.0),
    ("collision", "eval/collision", 1.0),
    ("survival", "eval/survived", 1.0),
    ("on target", "eval/mean_on_target", 1.0),
    ("drone dist [m]", "eval/mean_inter_drone_distance", 1.0),
    ("min dist [m]", "eval/mean_separation", 1.0),
    ("return", "eval/return", 1.0),
]


def run_dir(method: str, drones: int, seed: int) -> Path:
    return Path("runs") / f"{method}_{TASK}_{drones}d_seed{seed}"


def train(args: argparse.Namespace) -> int:
    def launch(method: str, drones: int, seed: int) -> subprocess.Popen:
        out = run_dir(method, drones, seed)
        out.mkdir(parents=True, exist_ok=True)
        command = ["uv", "run", *METHODS[method], TASK, "--drones", str(drones), "--seed", str(seed),
                   "--steps", str(args.steps), "--eval-every", str(args.eval_every), "--out", str(out), *args.extra]
        if method in SINGLE_THREADED:
            command.append("--compile")
        print(f"starting {out}", flush=True)
        return subprocess.Popen(command, stdout=(out / "train.log").open("w"), stderr=subprocess.STDOUT)

    failed = []

    def finish(proc: subprocess.Popen, name: str) -> None:
        if proc.wait() != 0:
            failed.append(name)

    def grid(methods):
        return [(m, n, s) for m in methods for n in args.drones for s in args.seeds]

    running: list[tuple[subprocess.Popen, str]] = []
    for method, n, seed in grid(m for m in args.methods if m in SINGLE_THREADED):
        if len(running) >= args.jobs:
            finish(*running.pop(0))
        running.append((launch(method, n, seed), f"{method} {n}d seed{seed}"))
    for item in running:
        finish(*item)
    for method, n, seed in grid(m for m in args.methods if m in ("ippo", "mappo")):
        finish(launch(method, n, seed), f"{method} {n}d seed{seed}")

    if failed:
        print(f"failed: {', '.join(failed)}; see runs/*/train.log", file=sys.stderr)
        return 1
    print("all runs finished; now: compare_formation_tracking.py eval, then plot")
    return 0


def evaluate_pid(drones: int, episodes: int) -> dict:
    """The PID velocity tracker on privileged state, on the same episodes as the learners."""
    import gymnasium as gym
    import numpy as np

    import streampilot.env  # noqa: F401  (registers the environments)
    from streampilot.control import ControllerPolicy
    from streampilot.train_formation import TASKS, summarize
    from streampilot.vec_env import episode_summary

    env = gym.make(TASKS[TASK], obs_mode="state", num_drones=drones)
    base, policy, results = env.unwrapped, ControllerPolicy("pid"), []
    for episode in range(episodes):
        env.reset(seed=EVAL_SEED + episode)
        policy.reset(base)
        total, length, done = 0.0, 0, False
        while not done:
            _, reward, terminated, truncated, info = env.step(policy(base))
            total += reward
            length += 1
            done = terminated or truncated
        results.append(episode_summary(info, total, length))
    env.close()
    return summarize(results, "eval")


def evaluate_all(args: argparse.Namespace) -> int:
    import numpy as np
    import torch

    from streampilot.train_formation_stream import evaluate

    rows = []
    for n in args.drones:
        for method in args.methods:
            if method == "pid":
                results = [(0, evaluate_pid(n, args.episodes))]
                print(f"evaluated pid {n}d", flush=True)
            else:
                results = []
                for seed in args.seeds:
                    checkpoint = run_dir(method, n, seed) / "final.pt"
                    if not checkpoint.exists():
                        print(f"missing {checkpoint}", file=sys.stderr)
                        continue
                    config = torch.load(checkpoint, map_location="cpu", weights_only=False)["config"]
                    results.append((seed, evaluate(checkpoint, config, args.episodes, seed=EVAL_SEED)))
                    print(f"evaluated {checkpoint}", flush=True)
            for seed, result in results:
                values = {label: result[key] * scale for label, key, scale in METRICS}
                values = {k: v if np.isfinite(v) else np.nan for k, v in values.items()}  # no pairs when N = 1
                rows.append({"drones": n, "method": method, "seed": seed, **values})

    if not rows:
        print("no checkpoints found", file=sys.stderr)
        return 1
    labels = [label for label, *_ in METRICS]
    csv_path = Path("runs") / "compare_formation_tracking.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, ["drones", "method", "seed", *labels])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{TASK}, {args.episodes} episodes per run, mean +- std over seeds (PID: one deterministic run)")
    for n in args.drones:
        print(f"\nN = {n}\n{'method':<12}{'seeds':>6}" + "".join(f"{label:>18}" for label in labels))
        for method in args.methods:
            mine = [r for r in rows if r["drones"] == n and r["method"] == method]
            if mine:
                cells = "".join(f"{np.nanmean(v):>11.3f} ±{np.nanstd(v):5.3f}" for v in ([r[l] for r in mine] for l in labels))
                print(f"{method:<12}{len(mine):>6}{cells}")
    print(f"\nper-run rows: {csv_path}")
    return 0


def plot(args: argparse.Namespace) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pyarrow.parquet as pq

    column = {"steps": None, "updates": "train/grad_updates", "wall": "train/wall_seconds"}[args.x]
    fig, axes = plt.subplots(1, len(args.drones), figsize=(4.5 * len(args.drones), 3.6), squeeze=False, sharex=True)
    for ax, n in zip(axes[0], args.drones):
        for method in (m for m in args.methods if m != "pid"):
            curves = []
            for seed in args.seeds:
                path = run_dir(method, n, seed) / "metrics.parquet"
                if path.exists():
                    table = pq.read_table(path, columns=["step", "eval/return", *([column] if column else [])]).to_pydict()
                    xs = table[column] if column else table["step"]
                    curves.append({s: (r, x) for s, r, x in zip(table["step"], table["eval/return"], xs) if r is not None})
            if not curves:
                continue
            steps = sorted(set.intersection(*(set(c) for c in curves)))
            values = np.array([[c[s][0] for s in steps] for c in curves])
            x = np.array([[c[s][1] for s in steps] for c in curves]).mean(0)  # seeds differ only in wall time
            ax.plot(x, values.mean(0), label=f"{method} ({len(curves)})")
            ax.fill_between(x, values.min(0), values.max(0), alpha=0.2)
        if "pid" in args.methods:
            ax.axhline(evaluate_pid(n, args.episodes)["eval/return"], color="k", ls="--", label="pid")
        ax.set_title(f"N = {n}")
        ax.set_xlabel({"steps": "team steps", "updates": "network updates", "wall": "training seconds"}[args.x])
        if args.x != "steps":
            ax.set_xscale("log")
    axes[0][0].set_ylabel("evaluation return")
    axes[0][-1].legend(fontsize=8)
    fig.tight_layout()
    out = Path("runs") / f"compare_formation_tracking_curves{'' if args.x == 'steps' else '_' + args.x}.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=["train", "eval", "plot"])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--drones", nargs="+", type=int, choices=[1, 2, 3, 4, 5], default=[1, 2, 3, 5], help="team sizes N")
    parser.add_argument("--steps", type=int, default=2_000_000, help="train: team steps (same budget for every method)")
    parser.add_argument("--eval-every", type=int, default=100_000, help="train: steps between evaluations (the curve points)")
    parser.add_argument("--jobs", type=int, default=os.cpu_count(), help="train: parallel Stream AC runs")
    parser.add_argument("--x", choices=["steps", "updates", "wall"], default="steps", help="plot: x axis")
    parser.add_argument("--episodes", type=int, default=100, help="eval: episodes per run")
    args, args.extra = parser.parse_known_args()  # unknown arguments go to every trainer
    return {"train": train, "eval": evaluate_all, "plot": plot}[args.phase](args)


if __name__ == "__main__":
    sys.exit(main())
