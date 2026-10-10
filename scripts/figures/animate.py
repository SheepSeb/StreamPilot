"""Animated versions of two figures for the project page: the curves draw in over training (mp4).

rl_curves            evaluation return against team steps, one panel per N, mean over seeds with a 95 % CI
continual_sequence   normalised eval return over the task sequence, a playhead sweeping through the phases
"""

import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .common import ALGOS, COLORS, DRONES, MARKERS, MAX_RETURN, NAMES, REPS, ci, learner_legend, style
from .continual import PHASE_LEN, PHASES, SEQUENCE, load
from .tracking import eval_returns, load_runs

OUTPUTS = ["rl_curves", "continual_sequence"]
FPS, DRAW, HOLD = 30, 5.0, 2.0  # seconds drawing, then holding the full figure
DPI = 150


def add_args(parser) -> None:
    parser.add_argument("--root", type=Path, default=Path("runs"))
    parser.add_argument("--continual-root", type=Path, default=Path("runs_continual"))


def run(args, outputs: list[str]) -> None:
    if "rl_curves" in outputs:
        animate_curves(args.root, args.out_dir / "rl_curves.mp4")
    if "continual_sequence" in outputs:
        animate_sequence(args.continual_root, args.out_dir / "continual_sequence.mp4")


def film(fig, update, out: Path) -> None:
    """Calls ``update(progress)`` for progress 0 -> 1 over DRAW seconds, holds HOLD seconds, encodes H.264."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    out.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}", "-r", str(FPS),
         "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "20", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    n = int(DRAW * FPS)
    for k in range(n + int(HOLD * FPS)):
        if k <= n:
            x = k / n
            update(1 - (1 - x) ** 2)  # ease out
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba()).tobytes()
        ffmpeg.stdin.write(frame)
    ffmpeg.stdin.close()
    ffmpeg.wait()
    plt.close(fig)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


def animate_curves(root: Path, out: Path) -> None:
    returns = eval_returns(load_runs(root))
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(DRONES), figsize=(10, 3.0), dpi=DPI, sharey=True)
    series = []  # (ax, algo, x, mean, lo, hi, line)
    for ax, n in zip(axes, DRONES):
        for algo in ALGOS:
            seeds = returns[algo, n]
            y = np.array([e.to_numpy() for e in seeds])
            lo, hi = ci(y[rng.integers(0, len(y), size=(REPS, len(y)))].mean(1))
            x = seeds[0].index.to_numpy()
            (line,) = ax.plot([], [], color=COLORS[algo], linewidth=1.4, marker=MARKERS[algo], markersize=2.5,
                              markevery=3)
            series.append([ax, algo, x, y.mean(0), lo, hi, line, None])
        ax.set_title(f"N = {n}", fontsize=10)
        ax.set_xlim(0, 2.05e6)
        ax.set_xticks([0, 1e6, 2e6], ["0", "1M", "2M"])
        ax.set_xlabel("team steps", fontsize=8.5)
        style(ax)
    axes[0].set_ylabel("evaluation return")
    axes[0].set_ylim(0, MAX_RETURN)
    learner_legend(fig, 0.99)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.17, top=0.78, wspace=0.08)
    cursors = [ax.axvline(0, color="0.6", lw=0.8, ls="--") for ax in axes]

    def update(p: float) -> None:
        c = p * 2e6
        for s in series:
            ax, algo, x, mu, lo, hi, line, band = s
            keep = x <= c
            line.set_data(x[keep], mu[keep])
            if band is not None:
                band.remove()
            s[7] = ax.fill_between(x[keep], lo[keep], hi[keep], color=COLORS[algo], alpha=0.15, linewidth=0)
        for cursor in cursors:
            cursor.set_xdata([c, c])
            cursor.set_visible(p < 1)

    film(fig, update, out)


def animate_sequence(root: Path, out: Path) -> None:
    data = {a: r for a in ALGOS if (r := load(root, a))}
    curves = {}
    for algo, runs in data.items():
        for p in range(4):
            rows = [d[(d["phase"] == p) & d["eval/return"].notna()].groupby("phase_step").last() for d in runs]
            y = np.array([r["eval/return"].to_numpy() for r in rows])
            curves[algo, p] = (p * PHASE_LEN + rows[0].index.to_numpy(), y.mean(0), y.std(0))
    lo, hi = np.zeros(3), np.ones(3)
    for t in range(3):  # per task, over the mean curves of all learners, as continual_sequence
        values = [mu for (_, p), (_, mu, _) in curves.items() if SEQUENCE[p] == t]
        lo[t], hi[t] = min(v.min() for v in values), max(v.max() for v in values)

    fig, ax = plt.subplots(figsize=(10, 3.4), dpi=DPI)
    for p in range(4):
        if p % 2:
            ax.axvspan(p * PHASE_LEN, (p + 1) * PHASE_LEN, color="0.965", linewidth=0, zorder=0)
        if p:
            ax.axvline(p * PHASE_LEN, color="0.25", linewidth=1.0, zorder=1)
        ax.text((p + 0.5) * PHASE_LEN, 1.0, PHASES[p][3:], transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=9.5)
    series = []
    for (algo, p), (x, mu, sd) in curves.items():
        t = SEQUENCE[p]
        scale = hi[t] - lo[t]
        (line,) = ax.plot([], [], color=COLORS[algo], linewidth=1.5, marker=MARKERS[algo], markersize=2.5, zorder=3,
                          label=NAMES[algo] if p == 0 else None)
        series.append([algo, x, (mu - lo[t]) / scale, sd / scale, line, None])
    ax.set_ylim(-0.08, 1.08)
    ax.set_xlim(0, 4 * PHASE_LEN)
    ax.set_xticks(np.arange(5) * PHASE_LEN, ["0", "0.5M", "1M", "1.5M", "2M"])
    ax.set_xlabel("team steps")
    ax.set_ylabel("normalised eval return")
    style(ax, "y")
    fig.legend(loc="upper center", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.995))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.8)
    cursor = ax.axvline(0, color="0.4", lw=1.0, ls="--", zorder=4)

    def update(progress: float) -> None:
        c = progress * 4 * PHASE_LEN
        for s in series:
            algo, x, mu, sd, line, band = s
            keep = x <= c
            line.set_data(x[keep], mu[keep])
            if band is not None:
                band.remove()
            s[5] = ax.fill_between(x[keep], mu[keep] - sd[keep], mu[keep] + sd[keep], color=COLORS[algo], alpha=0.08,
                                   linewidth=0, zorder=2)
        cursor.set_xdata([c, c])
        cursor.set_visible(progress < 1)

    film(fig, update, out)
