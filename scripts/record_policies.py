"""Side-by-side video of trained formation policies flying the same episode, with the leader's onboard view.

    uv run python scripts/record_policies.py    # -> runs/policies_formation-warehouse-tracking_3d.mp4
    uv run python scripts/record_policies.py --methods istream_ac mappo --camera overview
"""

import argparse
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from streampilot.policy import TeamPolicy
from streampilot.train_formation_stream import make_env
from streampilot.visualize import draw_detection

LABELS = {"istream_ac": "Independent Stream AC", "cstream_ac": "CTDE Stream AC", "ippo": "IPPO", "mappo": "MAPPO"}


def label(frame: np.ndarray, lines: list[str], font) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    y = 8
    for text in lines:
        box = draw.textbbox((10, y), text, font=font)
        draw.rectangle((box[0] - 4, box[1] - 3, box[2] + 4, box[3] + 3), fill=(0, 0, 0))
        draw.text((10, y), text, fill=(255, 255, 255), font=font)
        y = box[3] + 8
    return np.asarray(image)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="+", default=list(LABELS), choices=LABELS)
    parser.add_argument("--task", default="formation-warehouse-tracking", help="the scene flown")
    parser.add_argument("--trained-on", default="formation-tracking", help="task in the checkpoints' run names")
    parser.add_argument("--drones", type=int, default=3)
    parser.add_argument("--run-seed", type=int, default=1, help="training seed of the checkpoints")
    parser.add_argument("--seed", type=int, default=10_000, help="episode seed")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--camera", default="hall", help="hall (warehouse overview), overview, or d0_chase")
    parser.add_argument("--size", nargs=2, type=int, default=[640, 480], metavar=("W", "H"), help="per panel")
    parser.add_argument("--inset", type=int, default=176, help="onboard inset size, 0 for none")
    parser.add_argument("--fps", type=int, default=20, help="20: real time")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or Path("runs") / f"policies_{args.task}_{args.drones}d.mp4"

    teams = []
    for method in args.methods:
        checkpoint = Path("runs") / f"{method}_{args.trained_on}_{args.drones}d_seed{args.run_seed}" / "final.pt"
        config = torch.load(checkpoint, map_location="cpu", weights_only=False)["config"]
        config["task"] = args.task
        env = make_env(config)
        policy = TeamPolicy.load(checkpoint)
        obs, _ = env.reset(seed=args.seed)
        policy.reset()
        teams.append({"name": LABELS[method], "env": env, "policy": policy, "obs": obs, "done": False, "ret": 0.0})

    w, h = args.size
    cols = 2 if len(teams) > 1 else 1
    rows = -(-len(teams) // cols)
    font = ImageFont.load_default(size=18)
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{cols * w}x{rows * h}",
         "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)],
        stdin=subprocess.PIPE,
    )
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for step in range(args.steps):
        for k, team in enumerate(teams):
            if team["done"]:
                continue
            obs, reward, terminated, truncated, info = team["env"].step(team["policy"](team["obs"]))
            team["obs"], team["ret"] = obs, team["ret"] + reward
            base = team["env"].unwrapped
            frame = base.camera_image(args.camera, width=w, height=h)
            if args.inset:
                onboard = draw_detection(base.onboard_image(0, args.inset, args.inset), base.target_detections()[0])
                frame[h - args.inset - 8 : h - 8, w - args.inset - 8 : w - 8] = onboard
            status = ""
            if terminated or truncated:
                team["done"] = True
                status = "  [collision]" if info.get("collision") else "  [out of bounds]" if info.get("out_of_bounds") else ""
            frame = label(frame, [team["name"], f"step {step + 1}  return {team['ret']:.1f}{status}"], font)
            r, c = divmod(k, cols)
            canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = frame
        ffmpeg.stdin.write(canvas.tobytes())
        if all(t["done"] for t in teams):
            break
    ffmpeg.stdin.close()
    ffmpeg.wait()
    for team in teams:
        team["env"].close()
        print(f"{team['name']:<24} return {team['ret']:.1f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
