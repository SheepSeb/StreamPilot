"""Videos, GIFs and images of the project page, all in the warehouse scenes, into --out (static/).

    uv run python scripts/record_web_media.py                  # everything
    uv run python scripts/record_web_media.py --only teaser onboard

teaser     videos/teaser.mp4: a short film, one act per task (tracking, waypoint, landing), N = 3
onboard    images/onboard_{clean,noise,dropout}.png: the leader's onboard view and its detection
videos     videos/{tracking,waypoint,landing}_{method}.mp4: one learner, one task, a whole episode
continual  images/continual_{tracking,waypoint,landing}.gif (or .mp4): the four learners of the task sequence, 2 x 2

Tracking policies are the runs/ checkpoints (trained on tracking only); waypoint and landing come from the task
sequence (runs_continual/, after the task's own phase). For each learner and task, the seed with the best
end-of-phase probe score is shown. Every learner flies the same episode (--seed).
"""

import argparse
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

from streampilot.policy import TeamPolicy
from streampilot.train_formation_stream import make_env
from streampilot.visualize import draw_detection

METHODS = {"istream_ac": "iStream-AC", "cstream_ac": "cStream-AC", "ippo": "IPPO", "mappo": "MAPPO"}
TASKS = {"tracking": 0, "waypoint": 1, "landing": 2}  # task id = the phase that trains it
CLEAN, NOISY = (27, 175, 122), (235, 104, 52)


def checkpoint(method: str, task: str) -> tuple[Path, str]:
    """The checkpoint shown for ``method`` on ``task``, and a note on where it comes from."""
    if task == "tracking":
        return Path(f"runs/{method}_formation-tracking_3d_seed1/final.pt"), "seed 1"
    best, best_seed = -1.0, None
    for seed in (1, 2, 3):
        d = pd.read_parquet(f"runs_continual/{method.removesuffix('_ac')}-continue_seed{seed}/metrics.parquet")
        probe = d[(d["phase"] == TASKS[task]) & (d["probe_task"] == TASKS[task])]["probe/score"].dropna()
        if probe.iloc[-1] > best:
            best, best_seed = probe.iloc[-1], seed
    path = Path(f"runs_continual/{method.removesuffix('_ac')}-continue_seed{best_seed}/after_phase{TASKS[task]}.pt")
    return path, f"seed {best_seed}, probe score {best:.2f}"


def team(path: Path, task: str, **env_kwargs):
    config = torch.load(path, map_location="cpu", weights_only=False)["config"]
    config["task"] = f"formation-warehouse-{task}"
    config["env_kwargs"] = {**config["env_kwargs"], "detection_dropout": 0.0, "detection_noise": 0.0, **env_kwargs}
    return make_env(config), TeamPolicy.load(path)


def rollout(path: Path, task: str, seed: int, size: tuple[int, int], every: int = 1, start: int = 0,
            steps: int | None = None, inset: int = 0, title: str | None = None, camera: str = "hall"):
    """Frames of ``camera``, one every ``every`` steps from ``start``, with the leader's onboard view
    inset and a title; the episode runs until it ends or ``steps``."""
    env, policy = team(path, task)
    obs, _ = env.reset(seed=seed)
    policy.reset()
    base, frames, ret, step = env.unwrapped, [], 0.0, 0
    font = ImageFont.load_default(size=max(12, size[1] // 22))
    while steps is None or step < start + steps:
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        ret, step = ret + reward, step + 1
        if step > start and (step - start) % every == 0 or terminated or truncated:
            frame = base.camera_image(camera, width=size[0], height=size[1])
            if inset:
                onboard = draw_detection(base.onboard_image(0, inset, inset), base.target_detections()[0])
                frame[-inset - 6 : -6, -inset - 6 : -6] = onboard
            if title:
                frame = caption(frame, title, font)
            frames.append(frame)
        if terminated or truncated:
            break
    env.close()
    end = "collision" if info.get("collision") else "out of bounds" if info.get("out_of_bounds") else (
        "success" if info.get("is_success") else "time limit" if truncated else "clip end")
    print(f"  {path}  {task}: {step} steps, return {ret:.1f}, {end}")
    return frames


def caption(frame: np.ndarray, text: str, font) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    box = draw.textbbox((10, 8), text, font=font)
    draw.rectangle((box[0] - 5, box[1] - 4, box[2] + 5, box[3] + 4), fill=(0, 0, 0))
    draw.text((10, 8), text, fill=(255, 255, 255), font=font)
    return np.asarray(image)


GIF_LIMIT = 3e6  # bytes; a larger clip ships as an autoplaying, looping mp4 instead


def clip(frames: list[np.ndarray], out: Path, fps: int) -> None:
    """``out``.gif if it stays under GIF_LIMIT, otherwise ``out``.mp4 (the other one is removed)."""
    gif, mp4 = out.with_suffix(".gif"), out.with_suffix(".mp4")
    encode(frames, gif, fps)
    if gif.stat().st_size > GIF_LIMIT:
        gif.unlink()
        encode(frames, mp4, fps)
        print(f"  {gif.name} over {GIF_LIMIT / 1e6:g} MB: kept {mp4.name}")
    else:
        mp4.unlink(missing_ok=True)


def encode(frames: list[np.ndarray], out: Path, fps: int) -> None:
    """mp4 (H.264) or gif (one optimised palette), from the file name."""
    out.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    if out.suffix == ".gif":
        codec = ["-filter_complex", "split[a][b];[a]palettegen=max_colors=160:stats_mode=diff[p];"
                 "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle", "-loop", "0"]
    else:
        codec = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart"]
    ffmpeg = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                               "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", *codec, str(out)], stdin=subprocess.PIPE)
    for f in frames:
        ffmpeg.stdin.write(np.ascontiguousarray(f).tobytes())
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FPS = 20  # one frame per control step: real time


def ease(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def fade(t: float, start: float, end: float, ramp: float = 0.5) -> float:
    """1 inside [start, end], ramping from / to 0 over ``ramp`` seconds."""
    return min(ease((t - start) / ramp), ease((end - t) / ramp))


class Director:
    """Moves the world-fixed ``overview`` camera: placed at ``pos``, looking at ``target``."""

    def __init__(self, base):
        import mujoco

        self.mujoco, self.base, self.cam = mujoco, base, base.model.camera("overview").id

    def look(self, pos, target, fovy: float = 45.0) -> None:
        pos = np.array([np.clip(pos[0], -4.6, 4.6), np.clip(pos[1], -4.6, 4.6), np.clip(pos[2], 0.5, 5.0)])  # inside the racks
        forward = (target - pos) / np.linalg.norm(target - pos)
        right = np.cross(forward, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        quat = np.zeros(4)
        self.mujoco.mju_mat2Quat(quat, np.column_stack([right, up, -forward]).ravel())  # cameras look along -z
        m = self.base.model
        m.cam_pos[self.cam], m.cam_quat[self.cam], m.cam_fovy[self.cam] = pos, quat, fovy
        self.mujoco.mj_camlight(m, self.base.data)


def text(image: Image.Image, xy, words: str, size: int, alpha: float, anchor: str = "la", bold: bool = False,
         color=(255, 255, 255)) -> None:
    if alpha <= 0:
        return
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    font = ImageFont.truetype(FONT_BOLD if bold else FONT, size)
    ImageDraw.Draw(layer).text(xy, words, font=font, anchor=anchor, fill=(*color, int(255 * alpha)))
    image.alpha_composite(layer)


def focus(base, task: str) -> np.ndarray:
    """Where the camera looks: between the team and what it flies to."""
    team = base.drone_pos[:, :2].mean(0)
    if task == "tracking":
        goal = base._target_pos[:2]
    elif task == "landing":
        goal = base._pad_top[:2]
    else:
        ids = [base.model.body(f"waypoint{i}").mocapid[0] for i in range(base.num_drones)]
        goal = base.data.mocap_pos[ids, :2].mean(0)
    return np.array([*(0.5 * team + 0.5 * goal), 1.0])


ACTS = [  # task, learner, steps (None: the whole episode), caption
    ("tracking", "istream_ac", 260, "Formation tracking: three drones escort a mobile robot"),
    ("waypoint", "istream_ac", None, "Waypoints: each drone flies to its own scan point, stage after stage"),
    ("landing", "cstream_ac", None, "Landing: the team lines up around the dock, then hands off"),
]


class Writer:
    """Frames straight into ffmpeg (H.264), so a long film never sits in memory."""

    def __init__(self, out: Path, w: int, h: int, fps: int):
        out.parent.mkdir(parents=True, exist_ok=True)
        self.out = out
        self.ffmpeg = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r",
             str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21", "-preset", "slow",
             "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        self.ffmpeg.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        self.ffmpeg.stdin.close()
        self.ffmpeg.wait()
        print(f"wrote {self.out} ({self.out.stat().st_size / 1e6:.1f} MB)")


def teaser(args, size: tuple[int, int] = (1280, 720), ratio: float = 2.39) -> None:
    """A short film, one act per task (all in the warehouse, N = 3): each act one episode, with camera moves,
    dissolves between its shots, captions, and a fade through black between acts. Letterboxed to ``ratio``."""
    w, h = size
    ch = int(round(w / ratio / 2)) * 2  # the picture between the bars, even for H.264
    bar = (h - ch) // 2
    writer = Writer(args.out / "videos" / "teaser.mp4", w, h, FPS)
    for number, (task, method, steps, caption) in enumerate(ACTS, start=1):
        path, note = checkpoint(method, task)
        env, policy = team(path, task)
        obs, _ = env.reset(seed=args.seed)
        policy.reset()
        base = env.unwrapped
        base.model.vis.global_.offwidth, base.model.vis.global_.offheight = max(w, 1024), max(h, 1024)
        director = Director(base)
        state = {"look": None}

        def render(camera: str, width: int = w) -> np.ndarray:
            return base.camera_image(camera, width=width, height=ch)

        def crane(t: float, length: float = 5.0) -> np.ndarray:  # high in the corner, down onto the team
            k, look = ease(t / length), state["look"]
            director.look(look + (1 - k) * np.array([-5.5, -5.5, 4.0]) + k * np.array([-3.0, -3.4, 0.9]), look,
                          55 - 10 * k)
            return render("overview")

        def orbit(t: float, start: float = -130, radius: float = 3.8, height: float = 0.8) -> np.ndarray:
            a, look = np.deg2rad(start + 24 * t), state["look"]
            director.look(look + [radius * np.cos(a), radius * np.sin(a), height], look + [0, 0, -0.2], 50)
            return render("overview")

        def high(t: float) -> np.ndarray:  # a slow turn high above the team
            a, look = np.deg2rad(-100 + 12 * t), state["look"]
            director.look(look + [4.2 * np.cos(a), 4.2 * np.sin(a), 3.2], look, 55)
            return render("overview")

        def split(t: float) -> np.ndarray:  # the leader from the side, and what it sees
            leader = base.drone_pos[0]
            onboard = base.data.cam_xmat[base.model.camera("d0_onboard").id].reshape(3, 3)
            forward = -onboard[:2, 2] / np.linalg.norm(onboard[:2, 2])
            normal = np.array([-forward[1], forward[0]])
            director.look(leader + [*(1.6 * normal - 0.6 * forward), 0.15 + 0.02 * t], leader + [*(0.8 * forward), -0.1],
                          50)
            left = render("overview", w - ch)
            right = draw_detection(base.onboard_image(0, ch, ch), base.target_detections()[0])
            frame = np.concatenate([left, right], axis=1)
            frame[:, w - ch - 2 : w - ch + 2] = 0
            return frame

        shots = {  # start [s], render(time in shot)
            "tracking": [(0.0, crane), (5.0, orbit), (9.0, split)],
            "waypoint": [(0.0, high), (4.0, lambda t: orbit(t, -60, 4.2, 1.4)), (8.0, lambda t: render("d0_chase"))],
            "landing": [(0.0, lambda t: orbit(t, -150, 3.2, 1.3)), (3.0, split)],
        }[task]
        dissolve, act = 0.6, []
        for step in range(steps or 10_000):
            obs, _, terminated, truncated, info = env.step(policy(obs))
            t = (step + 1) / FPS
            point = focus(base, task)
            state["look"] = point if state["look"] is None else state["look"] + 0.08 * (point - state["look"])
            k = max(i for i, s in enumerate(shots) if s[0] <= t)
            picture = shots[k][1](t - shots[k][0]).astype(float)
            if k + 1 < len(shots) and t > shots[k + 1][0] - dissolve:
                a = ease((t - shots[k + 1][0] + dissolve) / dissolve)
                picture = (1 - a) * picture + a * shots[k + 1][1](t - shots[k + 1][0])
            act.append((picture.astype(np.uint8), shots[k][1] is split))
            if terminated or truncated:
                break
        env.close()
        result = "success" if info.get("is_success") else "collision" if info.get("collision") else "clip end"
        print(f"  act {number} {task}: {path} ({note}), {len(act)} steps, {result}")
        act += [act[-1]] * int(0.8 * FPS)  # hold the last frame
        name = METHODS[method]
        length = len(act) / FPS
        for i, (picture, is_split) in enumerate(act):
            t = i / FPS
            picture = picture.astype(float) * fade(t, 0.0, length, 0.5)  # through black between acts
            if number == 1:  # dim the picture under the title
                picture *= 1 - 0.55 * fade(t, 0.6, 4.6, 0.7)
            frame = Image.new("RGBA", (w, h), (0, 0, 0, 255))
            frame.paste(Image.fromarray(picture.astype(np.uint8)), (0, bar))
            on = fade(t, 0.6, length - 0.6, 0.5)
            text(frame, (48, bar - 30), f"{number} / {len(ACTS)}   {task.upper()}", 22, on, "ls", bold=True)
            text(frame, (w - 48, bar - 30), name, 22, on, "rs", color=(170, 175, 185))
            text(frame, (48, h - bar + 44), caption, 24, on, "ls")
            if is_split:
                text(frame, (w - ch + 16, bar + 34), "LEADER CAMERA", 20, on, bold=True)
            if number == 1:
                title = fade(t, 0.8, 4.4, 0.7)
                text(frame, (w / 2, bar + ch / 2 - 18), "SwarmStream", 88, title, "ms", bold=True)
                text(frame, (w / 2, bar + ch / 2 + 30), "Extending Streaming Reinforcement Learning to Multi-Agent UAVs",
                     28, title, "mm")
            writer.write(np.asarray(frame.convert("RGB")))
    writer.close()


def onboard(args, step: int = 150, size: int = 480, noise: float = 0.3) -> None:
    """The setup figure's onboard panels: clean, noisy (true box dotted) and dropped detections."""
    env, policy = team(checkpoint("istream_ac", "tracking")[0], "tracking")
    obs, _ = env.reset(seed=args.seed)
    policy.reset()
    for _ in range(step):
        obs, *_ = env.step(policy(obs))
    base = env.unwrapped
    view = base.onboard_image(0, size, size)
    exact = base.target_detections()[0]
    if not exact[0]:
        raise SystemExit(f"the target is not in the leader's view at step {step}")
    base.detection_noise = noise
    base.np_random = np.random.default_rng(0)
    noisy = base.detection_obs(base.target_detections())[0]
    env.close()

    def rect(box):
        _, cx, cy, w, h = box
        return [(cx - w / 2) * size, (cy - h / 2) * size, (cx + w / 2) * size, (cy + h / 2) * size]

    def dotted(draw, box, color, dash=8):
        x0, y0, x1, y1 = rect(box)
        for a, b in [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]:
            length = np.hypot(b[0] - a[0], b[1] - a[1])
            for t in np.arange(0, length, 2 * dash):
                p, q = t / length, min(t + dash, length) / length
                draw.line([a[0] + p * (b[0] - a[0]), a[1] + p * (b[1] - a[1]),
                           a[0] + q * (b[0] - a[0]), a[1] + q * (b[1] - a[1])], fill=color, width=3)

    font = ImageFont.load_default(size=22)
    for name in ("clean", "noise", "dropout"):
        image = Image.fromarray(view)
        draw = ImageDraw.Draw(image)
        if name == "clean":
            draw.rectangle(rect(exact), outline=CLEAN, width=4)
        elif name == "noise":
            dotted(draw, exact, CLEAN)
            draw.rectangle(rect(noisy), outline=NOISY, width=4)
        else:
            text = "no detection (observation = 0)"
            box = draw.textbbox((0, 0), text, font=font)
            x, y = (size - box[2]) / 2, size - box[3] - 24
            draw.rounded_rectangle((x - 10, y - 6, x + box[2] + 10, y + box[3] + 8), 8, fill=(192, 57, 43))
            draw.text((x, y), text, fill="white", font=font)
        out = args.out / "images" / f"onboard_{name}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        image.save(out, optimize=True)
        print(f"wrote {out}")


def videos(args) -> None:
    for task in TASKS:
        for method, name in METHODS.items():
            path, note = checkpoint(method, task)
            frames = rollout(path, task, args.seed, (960, 540), inset=200, title=f"{name}, warehouse {task}")
            encode(frames, args.out / "videos" / f"{task}_{method}.mp4", fps=20)
            print(f"  ({note})")


def continual(args, panel: tuple[int, int] = (400, 225)) -> None:
    """The four learners side by side on one task, every 3rd step at 10 fps (1.5x real time)."""
    w, h = panel
    for task in TASKS:
        clips = []
        for method, name in METHODS.items():
            path, _ = continual_checkpoint(method, task)
            clips.append(rollout(path, task, args.seed, panel, every=3, steps=450, inset=90, title=name))
        n = max(len(c) for c in clips)
        frames = []
        for k in range(n):
            canvas = np.zeros((2 * h, 2 * w, 3), dtype=np.uint8)
            for i, clip in enumerate(clips):  # a finished episode holds its last frame
                r, c = divmod(i, 2)
                canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = clip[min(k, len(clip) - 1)]
            frames.append(canvas)
        clip(frames, args.out / "images" / f"continual_{task}", fps=10)


def continual_checkpoint(method: str, task: str) -> tuple[Path, str]:
    """As ``checkpoint``, but tracking too comes from the task sequence (its first phase)."""
    if task != "tracking":
        return checkpoint(method, task)
    best, best_seed = -1.0, None
    for seed in (1, 2, 3):
        d = pd.read_parquet(f"runs_continual/{method.removesuffix('_ac')}-continue_seed{seed}/metrics.parquet")
        probe = d[(d["phase"] == 0) & (d["probe_task"] == 0)]["probe/score"].dropna()
        if probe.iloc[-1] > best:
            best, best_seed = probe.iloc[-1], seed
    return Path(f"runs_continual/{method.removesuffix('_ac')}-continue_seed{best_seed}/after_phase0.pt"), ""


STEPS = {"teaser": teaser, "onboard": onboard, "videos": videos, "continual": continual}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=STEPS, default=list(STEPS))
    parser.add_argument("--seed", type=int, default=10_000, help="episode seed, the same for every learner")
    parser.add_argument("--out", type=Path, default=Path("static"))
    args = parser.parse_args()
    for name in args.only:
        print(f"=== {name}")
        STEPS[name](args)


if __name__ == "__main__":
    main()
