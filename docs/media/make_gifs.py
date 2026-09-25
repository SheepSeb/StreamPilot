"""Render scripted-controller episodes for each task to GIFs and stills for the README.

    uv run python docs/media/make_gifs.py
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path

import gymnasium as gym
from PIL import Image

import streampilot.env  # noqa: F401
from streampilot.visualize import FORMATION_POLICIES, SCRIPTED_POLICIES, draw_detection

OUT = Path(__file__).resolve().parent
TASKS = {"waypoint": "DroneWaypoint-v0", "landing": "DroneLanding-v0", "tracking": "DroneTracking-v0"}
FORMATION_TASKS = {
    "formation-waypoint": "DroneFormationWaypoint-v0",
    "formation-landing": "DroneFormationLanding-v0",
    "formation-tracking": "DroneFormationTracking-v0",
}
INSET = 110
FORMATION_INSET = 90
RENDER_SIZE = 320
ONBOARD_STILL_SIZE = 320
FRAME_STRIDE = 2  # env steps at 20 Hz; keep every other frame -> 10 fps, smaller file


def render_task(name: str, env_id: str, seed: int = 0, max_steps: int = 200):
    env = gym.make(
        env_id,
        obs_mode="detection",
        render_mode="rgb_array",
        camera="chase",
        width=RENDER_SIZE,
        height=RENDER_SIZE,
    )
    base = env.unwrapped
    obs, _ = env.reset(seed=seed)
    frames = []
    best_still, best_score = None, float("inf")
    target_area = 0.12  # a well-framed shot, not a close-up
    done, steps = False, 0
    while not done and steps < max_steps:
        action = SCRIPTED_POLICIES[name](base, base.state_obs())
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

        frame = env.render().copy()
        inset = draw_detection(base.camera_image(width=INSET, height=INSET), obs)
        pad = 6
        frame[pad : pad + INSET, -pad - INSET : -pad if pad else None] = inset
        frames.append(Image.fromarray(frame))

        # Track the best-framed onboard view (detection box closest to target_area) as a still,
        # so the README can show exactly what the detector sees.
        if obs[0]:
            score = abs(obs[3] * obs[4] - target_area)
            if score < best_score:
                onboard = draw_detection(base.camera_image(width=ONBOARD_STILL_SIZE, height=ONBOARD_STILL_SIZE), obs)
                best_still, best_score = Image.fromarray(onboard), score
    env.close()
    print(f"{name}: {len(frames)} frames, success={info.get('is_success')}")
    return frames, best_still


def render_formation_task(name: str, env_id: str, seed: int = 0, max_steps: int = 300, num_drones: int = 3):
    env = gym.make(
        env_id,
        obs_mode="detection",
        render_mode="rgb_array",
        camera="overview",
        width=RENDER_SIZE,
        height=RENDER_SIZE,
        num_drones=num_drones,
    )
    base = env.unwrapped
    obs, _ = env.reset(seed=seed)
    frames = []
    done, steps = False, 0
    while not done and steps < max_steps:
        action = FORMATION_POLICIES[name](base)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

        frame = env.render().copy()
        pad = 6
        # Onboard insets along the top-right edge, leader on the left, each teammate boxed in magenta.
        for k in range(num_drones):
            inset = draw_detection(base.onboard_image(k, FORMATION_INSET, FORMATION_INSET), obs[k, :5], base.detect_teammates(k))
            x1 = frame.shape[1] - pad - (FORMATION_INSET + pad) * (num_drones - 1 - k)
            frame[pad : pad + FORMATION_INSET, x1 - FORMATION_INSET : x1] = inset
        frames.append(Image.fromarray(frame))
    env.close()
    print(f"{name}: {len(frames)} frames, success={info.get('is_success')}")
    return frames


def save_gif(frames: list[Image.Image], path: Path, fps: int = 10) -> None:
    frames = frames[::FRAME_STRIDE]
    quantized = [f.quantize(colors=128, method=Image.MEDIANCUT) for f in frames]
    quantized[0].save(
        path,
        save_all=True,
        append_images=quantized[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    for name, env_id in TASKS.items():
        frames, onboard_still = render_task(name, env_id)
        save_gif(frames, OUT / f"{name}.gif")
        onboard_still.save(OUT / f"{name}_onboard.png")
        print(f"wrote {OUT / f'{name}_onboard.png'}")

    for name, env_id in FORMATION_TASKS.items():
        frames = render_formation_task(name, env_id)
        save_gif(frames, OUT / f"{name.replace('-', '_')}.gif")
