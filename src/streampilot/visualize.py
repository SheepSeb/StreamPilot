"""Watch the drone tasks in the MuJoCo viewer.

    uv run streampilot waypoint                  # scripted controller
    uv run streampilot landing --policy random
    uv run streampilot tracking --camera overview --detection-noise 0.02
    uv run streampilot waypoint --policy runs/waypoint_seed0/final.pt   # trained policy

The top-right inset is the onboard camera, with the detection the policy observes drawn in
green (red frame: target not detected). Close the viewer window to stop; press Esc in the
viewer to switch to the free camera.
"""

import argparse
import time

import gymnasium as gym
import mujoco
import numpy as np

import streampilot.env  # noqa: F401  (registers the environments)
from streampilot.env.base import wrap_angle

TASKS = {"waypoint": "DroneWaypoint-v0", "landing": "DroneLanding-v0", "tracking": "DroneTracking-v0"}
INSET_SIZE = 240

# Scripted controllers. They cheat: they read the privileged state from ``env.state_obs()``
# (x, y 0:2, vx, vy 2:4, cos/sin yaw 4:6, yaw rate 6, task obs 7:), not the detection.


def body_action(env, state, world_vel, desired_yaw) -> np.ndarray:
    """Normalized ``[vx, vy, yaw_rate]`` body-frame action for a horizontal world-frame velocity."""
    yaw = np.arctan2(state[5], state[4])
    c, s = np.cos(yaw), np.sin(yaw)
    vx, vy = world_vel
    yaw_rate = 2.0 * wrap_angle(desired_yaw - yaw)
    return np.clip(np.array([c * vx + s * vy, -s * vx + c * vy, yaw_rate]) / env.action_scale, -1.0, 1.0)


def waypoint_policy(env, state) -> np.ndarray:
    rel = state[7:9]
    heading = np.arctan2(rel[1], rel[0])
    # Turn towards the waypoint first, then fly at it.
    facing = max(0.0, np.cos(wrap_angle(heading - np.arctan2(state[5], state[4]))))
    return body_action(env, state, np.clip(2.0 * rel, -1.5, 1.5) * facing, heading)


def landing_policy(env, state) -> np.ndarray:
    pos, rel = state[0:2], state[7:9]
    pad = pos + rel
    # Hand-off point: handoff_distance from the pad on its arena-centre side, so it stays in bounds.
    side = -pad / np.linalg.norm(pad) if np.linalg.norm(pad) > 0.3 else -rel / max(np.linalg.norm(rel), 1e-6)
    goal = pad + side * env.handoff_distance
    return body_action(env, state, np.clip(1.5 * (goal - pos), -0.8, 0.8), np.arctan2(rel[1], rel[0]))


def tracking_policy(env, state) -> np.ndarray:
    pos, rel, target_vel = state[0:2], state[7:9], state[9:11]
    away = -rel / max(np.linalg.norm(rel), 1e-6)
    limit = env.arena_half_extent - 0.2
    goal = np.clip(pos + rel + away * env.follow_distance, -limit, limit)
    return body_action(env, state, target_vel + 2.0 * (goal - pos), np.arctan2(rel[1], rel[0]))


SCRIPTED_POLICIES = {"waypoint": waypoint_policy, "landing": landing_policy, "tracking": tracking_policy}


def draw_detection(image: np.ndarray, detection: np.ndarray) -> np.ndarray:
    """Draw a ``[visible, cx, cy, w, h]`` (YOLO xywhn) box on ``image``."""
    image = image.copy()
    size = image.shape[0]
    if detection[0] == 0.0:
        image[:3], image[-3:], image[:, :3], image[:, -3:] = [220, 30, 30], [220, 30, 30], [220, 30, 30], [220, 30, 30]
        return image
    _, cx, cy, w, h = detection
    x0, x1 = (np.clip([cx - w / 2, cx + w / 2], 0, 1) * (size - 1)).astype(int)
    y0, y1 = (np.clip([cy - h / 2, cy + h / 2], 0, 1) * (size - 1)).astype(int)
    green = [40, 255, 40]
    image[y0 : y0 + 2, x0 : x1 + 1] = green
    image[y1 - 1 : y1 + 1, x0 : x1 + 1] = green
    image[y0 : y1 + 1, x0 : x0 + 2] = green
    image[y0 : y1 + 1, x1 - 1 : x1 + 1] = green
    return image


def show_onboard(env, detection: np.ndarray, reward: float, total: float) -> None:
    viewer = env.unwrapped._viewer
    if viewer is None or viewer.viewport is None:
        return
    image = draw_detection(env.unwrapped.camera_image(width=INSET_SIZE, height=INSET_SIZE), detection)
    vp = viewer.viewport
    rect = mujoco.MjrRect(vp.width - INSET_SIZE - 10, vp.height - INSET_SIZE - 10, INSET_SIZE, INSET_SIZE)
    viewer.set_images((rect, image))
    labels = "detection [vis cx cy w h]\nreward / return\nstep"
    values = f"{np.array2string(detection, precision=2)}\n{reward:+.3f} / {total:.2f}\n{env.unwrapped.step_count}"
    viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, labels, values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch the drone tasks in the MuJoCo viewer.")
    parser.add_argument("task", choices=TASKS)
    parser.add_argument(
        "--policy", default="scripted", help="scripted, random, zero, or a training checkpoint (.pt)"
    )
    parser.add_argument("--camera", choices=["chase", "overview", "onboard", "free"], default="chase")
    parser.add_argument("--detection-noise", type=float, default=0.0)
    parser.add_argument("--detection-dropout", type=float, default=0.0)
    parser.add_argument("--episodes", type=int, default=None, help="default: until the viewer is closed")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    trained = None
    if args.policy.endswith(".pt"):
        from streampilot.policy import Policy

        trained = Policy.load(args.policy)
        assert trained.config["task"] == args.task, f"checkpoint is for {trained.config['task']}"
    elif args.policy not in ("scripted", "random", "zero"):
        parser.error(f"unknown policy {args.policy!r}")
    obs_mode = trained.config["obs_mode"] if trained else "detection"

    env = gym.make(
        TASKS[args.task],
        obs_mode=obs_mode,
        render_mode="human",
        camera=None if args.camera == "free" else args.camera,
        detection_noise=args.detection_noise,
        detection_dropout=args.detection_dropout,
    )
    base = env.unwrapped
    policies = {
        "scripted": lambda obs: SCRIPTED_POLICIES[args.task](base, base.state_obs()),
        "random": lambda obs: env.action_space.sample(),
        "zero": lambda obs: np.zeros(3),
    }
    policy = trained or policies[args.policy]
    env.action_space.seed(args.seed)

    episode = 0
    while args.episodes is None or episode < args.episodes:
        obs, _ = env.reset(seed=args.seed + episode)
        if trained:
            trained.reset()
        total, done = 0.0, False
        while not done and base.viewer_running:
            obs, reward, terminated, truncated, info = env.step(policy(obs))
            total += reward
            done = terminated or truncated
            detection = obs if obs_mode == "detection" else base.detect(base._task_target_geom())
            show_onboard(env, detection, reward, total)
        if not base.viewer_running:
            break
        info = {k: round(v, 3) if isinstance(v, float) else v for k, v in info.items()}
        print(f"episode {episode}: return {total:7.2f}  steps {base.step_count}  {info}", flush=True)
        time.sleep(1.0)  # linger on the final state
        episode += 1
    env.close()


if __name__ == "__main__":
    main()
