"""Formation tracking, N = 3: detection dropout x noise.

    uv run python scripts/compare_dropout_noise.py train      # -> runs_dropout_noise/
    uv run python scripts/compare_dropout_noise.py eval       # tables + runs_dropout_noise/results.csv
    uv run python scripts/compare_dropout_noise.py plot       # one heatmap per method
    uv run python scripts/compare_dropout_noise.py train --methods ippo --seeds 1
    uv run python scripts/compare_dropout_noise.py shift      # no training: runs/ N = 3 checkpoints under all conditions
    uv run python scripts/compare_dropout_noise.py curriculum # train with a noise curriculum -> runs_dropout_noise/curriculum/
    uv run python scripts/compare_dropout_noise.py shift --source curriculum   # ... and evaluate those under all conditions

Detection dropout p in {0, .05, .1, .2, .3, .4, .5} x detection noise in {0, .05, .1, .2, .3, .4, .5}: 49 conditions, N = 3.
Runs that already have a final.pt are skipped, so an interrupted ``train`` resumes.

Evaluation is on the same deterministic episodes (seeds 10000...), under the conditions the run was trained with.

Curriculum: the detection starts clean and degrades linearly to --level (dropout, noise) over the first
--curriculum-fraction of the steps, then stays there. Each learner trains once per seed.
"""

import argparse
import csv
import itertools
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from compare_formation_tracking import (
    EVAL_SEED,
    METHODS,
    METRICS,
    SINGLE_THREADED,
    TASK,
)  # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path("runs_dropout_noise")
DRONES = 3
DROPOUTS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
NOISES = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
LEARNERS = ["istream_ac", "cstream_ac", "ippo", "mappo"]


def run_dir(method: str, cond: tuple[float, float], seed: int) -> Path:
    p, s = cond
    return ROOT / f"{method}_p{p:g}_n{s:g}_seed{seed}"


def curriculum_dir(method: str, args, seed: int) -> Path:
    p, s = args.level
    return ROOT / "curriculum" / f"{method}_p{p:g}_n{s:g}_ramp{args.curriculum_fraction:g}_seed{seed}"


def grid(args) -> list[tuple[str, tuple, int]]:
    return [
        (m, c, s)
        for m in args.methods
        for c in itertools.product(DROPOUTS, NOISES)
        for s in args.seeds
    ]


def train(args: argparse.Namespace) -> int:
    while args.wait_for and Path(f"/proc/{args.wait_for}").exists():
        time.sleep(60)

    curriculum = args.phase == "curriculum"

    def dirfn(method, cond, seed):
        return curriculum_dir(method, args, seed) if curriculum else run_dir(method, cond, seed)

    def launch(method, cond, seed):
        out, (p, s) = dirfn(method, cond, seed), cond
        out.mkdir(parents=True, exist_ok=True)
        command = [
            "uv",
            "run",
            *METHODS[method],
            TASK,
            "--drones",
            str(DRONES),
            "--seed",
            str(seed),
            "--steps",
            str(args.steps),
            "--eval-every",
            str(args.eval_every),
            "--out",
            str(out),
            "--detection-dropout",
            str(p),
            "--detection-noise",
            str(s),
            *args.extra,
        ]
        if curriculum:
            command += ["--curriculum-fraction", str(args.curriculum_fraction)]
        if method in SINGLE_THREADED:
            command.append("--compile")
        print(f"starting {out}", flush=True)
        return subprocess.Popen(
            command, stdout=(out / "train.log").open("w"), stderr=subprocess.STDOUT
        )

    failed, running = [], []

    def finish(proc, name):
        if proc.wait() != 0:
            failed.append(name)

    runs = [(m, tuple(args.level), s) for m in args.methods for s in args.seeds] if curriculum else grid(args)
    todo = [(m, c, s) for m, c, s in runs if not (dirfn(m, c, s) / "final.pt").exists()]
    for method, cond, seed in (t for t in todo if t[0] in SINGLE_THREADED):
        if len(running) >= args.jobs:
            finish(*running.pop(0))
        running.append((launch(method, cond, seed), str(dirfn(method, cond, seed))))
    for item in running:
        finish(*item)
    for method, cond, seed in (
        t for t in todo if t[0] not in SINGLE_THREADED
    ):  # PPO: parallel inside
        finish(launch(method, cond, seed), str(dirfn(method, cond, seed)))

    if failed:
        print(f"failed: {', '.join(failed)}; see train.log in each", file=sys.stderr)
        return 1
    print("all runs finished; now: compare_dropout_noise.py eval, then plot")
    return 0


def _shift_job(job):
    import torch

    from streampilot.train_formation_stream import evaluate

    method, (p, s), seed, episodes, checkpoint = job
    torch.set_num_threads(1)
    config = torch.load(checkpoint, map_location="cpu", weights_only=False)["config"]
    config["env_kwargs"] = {**config["env_kwargs"], "detection_dropout": p, "detection_noise": s}
    result = evaluate(checkpoint, config, episodes, seed=EVAL_SEED)
    values = {label: result[key] * scale for label, key, scale in METRICS}
    return {"method": method, "dropout": p, "noise": s, "seed": seed, **values}


def shift(args: argparse.Namespace) -> int:
    """Evaluate trained checkpoints (``--source``: runs/, trained without dropout or noise, or the curriculum
    runs) under every condition."""
    from concurrent.futures import ProcessPoolExecutor

    jobs = []
    for method, cond, seed in grid(args):
        if args.source == "curriculum":
            checkpoint = curriculum_dir(method, args, seed) / "final.pt"
        else:
            checkpoint = Path("runs") / f"{method}_{TASK}_{DRONES}d_seed{seed}" / "final.pt"
        if checkpoint.exists():
            jobs.append((method, cond, seed, args.episodes, checkpoint))
        else:
            print(f"missing {checkpoint}", file=sys.stderr)
    with ProcessPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(_shift_job, jobs))
    tag = "" if args.source == "runs" else f"_{args.source}"
    return report(args, rows, f"shift{tag}_results.csv", f"shift{tag}_heatmap.png")


def collect(args) -> list[dict]:
    import numpy as np
    import torch

    from streampilot.train_formation_stream import evaluate

    rows = []
    for method, cond, seed in grid(args):
        checkpoint = run_dir(method, cond, seed) / "final.pt"
        if not checkpoint.exists():
            print(f"missing {checkpoint}", file=sys.stderr)
            continue
        config = torch.load(checkpoint, map_location="cpu", weights_only=False)[
            "config"
        ]
        result = evaluate(checkpoint, config, args.episodes, seed=EVAL_SEED)
        values = {label: result[key] * scale for label, key, scale in METRICS}
        rows.append(
            {
                "method": method,
                "dropout": cond[0],
                "noise": cond[1],
                "seed": seed,
                **values,
            }
        )
        print(f"evaluated {checkpoint}", flush=True)
    return rows


def evaluate_all(args: argparse.Namespace) -> int:
    return report(args, collect(args), "results.csv", "dropout_noise_heatmap.png")


def report(args, rows, csv_name, png_name) -> int:
    import numpy as np

    if not rows:
        print("no checkpoints found", file=sys.stderr)
        return 1
    labels = [label for label, *_ in METRICS]
    keys = ["method", "dropout", "noise", "seed"]
    with (ROOT / csv_name).open("w", newline="") as f:
        writer = csv.DictWriter(f, [*keys, *labels])
        writer.writeheader()
        writer.writerows(rows)

    def mean_std(rs, label):
        v = [r[label] for r in rs]
        return f"{np.nanmean(v):>8.2f} ±{np.nanstd(v):5.2f}"

    print(f"\nN = {DRONES}, {args.episodes} episodes per run, mean ± std over seeds")
    for method in args.methods:
        mine = [r for r in rows if r["method"] == method]
        for label in ("return", "survival"):
            print(f"\n{method}: {label}   (rows: dropout, columns: noise)")
            print(f"{'':>8}" + "".join(f"{s:>17g}" for s in NOISES))
            for p in DROPOUTS:
                cells = [
                    [r for r in mine if r["dropout"] == p and r["noise"] == s]
                    for s in NOISES
                ]
                print(
                    f"{p:>8g}"
                    + "".join(f"{mean_std(c, label) if c else '-':>17}" for c in cells)
                )
    print(f"\nper-run rows: {ROOT / csv_name}")
    plot_heatmaps(args, rows, png_name)
    return 0


def plot(args: argparse.Namespace) -> int:
    import csv as _csv

    path = ROOT / "results.csv"
    if not path.exists():
        print("run eval first", file=sys.stderr)
        return 1
    rows = [{k: (v if k == "method" else float(v)) for k, v in r.items()} for r in _csv.DictReader(path.open())]
    plot_heatmaps(args, rows, "dropout_noise_heatmap.png")
    return 0


def plot_heatmaps(args, rows, png_name) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    noise_methods = [m for m in args.methods if any(r["method"] == m for r in rows)]
    if noise_methods:
        fig, axes = plt.subplots(
            1,
            len(noise_methods),
            figsize=(4.2 * len(noise_methods), 3.6),
            squeeze=False,
        )
        for ax, method in zip(axes[0], noise_methods):
            z = np.full((len(DROPOUTS), len(NOISES)), np.nan)
            for i, p in enumerate(DROPOUTS):
                for j, s in enumerate(NOISES):
                    v = [
                        r["return"]
                        for r in rows
                        if r["method"] == method
                        and r["dropout"] == p
                        and r["noise"] == s
                    ]
                    if v:
                        z[i, j] = np.mean(v)
            im = ax.imshow(z, origin="lower", cmap="viridis")
            ax.set_xticks(range(len(NOISES)), [f"{s:g}" for s in NOISES])
            ax.set_yticks(range(len(DROPOUTS)), [f"{p:g}" for p in DROPOUTS])
            ax.set_xlabel("detection noise")
            ax.set_ylabel("detection dropout")
            ax.set_title(method)
            for (i, j), v in np.ndenumerate(z):
                if np.isfinite(v):
                    ax.text(
                        j,
                        i,
                        f"{v:.0f}",
                        ha="center",
                        va="center",
                        color="w",
                        fontsize=8,
                    )
            fig.colorbar(im, ax=ax, label="team return")
        fig.tight_layout()
        fig.savefig(ROOT / png_name, dpi=150)
        print(f"wrote {ROOT / png_name}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("phase", choices=["train", "eval", "plot", "shift", "curriculum"])
    parser.add_argument("--methods", nargs="+", choices=LEARNERS, default=LEARNERS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument(
        "--steps", type=int, default=2_000_000, help="train: team steps"
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=100_000,
        help="train: steps between evaluations",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count(),
        help="train: parallel Stream AC runs",
    )
    parser.add_argument(
        "--wait-for", type=int, default=0, help="train: start once this PID has exited"
    )
    parser.add_argument(
        "--episodes", type=int, default=100, help="eval: episodes per run"
    )
    parser.add_argument("--level", nargs=2, type=float, default=[0.3, 0.3], metavar=("DROPOUT", "NOISE"), help="curriculum: the final detection levels")
    parser.add_argument("--curriculum-fraction", type=float, default=0.875, help="curriculum: fraction of the steps spent ramping up (0.875 of 2M: 1.75M, then 250k at the full level)")
    parser.add_argument("--source", choices=["runs", "curriculum"], default="runs", help="shift: which checkpoints")
    args, args.extra = parser.parse_known_args()
    ROOT.mkdir(exist_ok=True)
    return {"train": train, "curriculum": train, "eval": evaluate_all, "plot": plot, "shift": shift}[args.phase](args)


if __name__ == "__main__":
    sys.exit(main())
