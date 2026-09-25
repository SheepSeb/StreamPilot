"""Watch the drone tasks in the MuJoCo viewer.

    uv run streampilot waypoint                  # scripted controller
    uv run streampilot landing --policy random
    uv run streampilot tracking --camera overview --detection-noise 0.02
    uv run streampilot waypoint --policy runs/waypoint_seed0/final.pt   # trained policy
    uv run streampilot formation-landing --drones 2       # multi-drone formation tasks

The top-right inset is the onboard camera, with the detection the policy observes drawn in
green (red frame: target not detected); formation tasks show one inset per drone, leader on the
left, with the teammates it sees boxed in magenta.
Close the viewer window to stop; press Esc in the viewer to switch to the free camera.
"""

import argparse
import time

import gymnasium as gym
import mujoco
import numpy as np

import streampilot.env  # noqa: F401  (registers the environments)
from streampilot.env.base import wrap_angle
from streampilot.env.formation.base import unit

TASKS = {"waypoint": "DroneWaypoint-v0", "landing": "DroneLanding-v0", "tracking": "DroneTracking-v0"}
FORMATION_TASKS = {
    "formation-waypoint": "DroneFormationWaypoint-v0",
    "formation-landing": "DroneFormationLanding-v0",
    "formation-tracking": "DroneFormationTracking-v0",
}
INSET_SIZE = 240
FORMATION_INSET_SIZE = 180

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


def formation_policy(env, max_speed: float = 1.0, safe_distance: float = 0.95) -> np.ndarray:
    """``(num_drones, 3)`` action for the formation tasks: each drone flies to its goal from
    ``env.goals()`` (plus the target's velocity, when it moves) and faces the task target.

    Collision avoidance: within ``safe_distance`` of a teammate, a drone increasingly cancels its
    velocity towards it (fully at ``min_separation``), is pushed away and slides sideways, so two
    drones meeting head-on pass each other instead of stalling."""
    goals, look_at = env.goals()
    pos, state = env.drone_pos[:, :2], env.state_obs()
    target_vel = getattr(env, "_target_vel", np.zeros(3))[:2]
    ahead = pos + 0.3 * env.drone_vel[:, :2]  # where the drones will be, given the ~0.2 s velocity lag
    actions = []
    for i in range(env.num_drones):
        vel = 2.0 * (goals[i] - pos[i])
        speed = np.linalg.norm(vel)
        if speed > max_speed:
            vel *= max_speed / speed
        vel = vel + target_vel
        for j in range(env.num_drones):
            dist = np.linalg.norm(ahead[i] - ahead[j])
            if j == i or dist >= safe_distance:
                continue
            away = (ahead[i] - ahead[j]) / max(dist, 1e-6)
            weight = np.clip((safe_distance - dist) / (safe_distance - env.min_separation - 0.1), 0.0, 1.0)
            vel = vel + weight * (max(0.0, -vel @ away) + 0.5) * away + weight * 0.5 * np.array([-away[1], away[0]])
        rel = (goals[i] if look_at is None else look_at) - pos[i]
        yaw = np.arctan2(state[i, 5], state[i, 4])
        if look_at is None and np.linalg.norm(rel) < 0.3:
            heading = yaw  # at the waypoint: any heading
        else:
            heading = np.arctan2(rel[1], rel[0])
        actions.append(body_action(env, state[i], vel, heading))
    return np.array(actions)


def formation_landing_policy(env) -> np.ndarray:
    """``formation_policy``, with the leader approaching the pad from the arena-centre side (as in
    ``landing_policy``) so the formation behind it stays in bounds."""
    pad = env._pad_top[:2]
    side = -pad / np.linalg.norm(pad) if np.linalg.norm(pad) > 0.3 else unit(env.drone_pos[0, :2] - pad)
    goals, look_at = env.goals()
    goals[0] = pad + side * env.handoff_distance
    return formation_policy(_Goals(env, goals, look_at))


class _Goals:
    """``env`` with ``goals()`` replaced."""

    def __init__(self, env, goals, look_at):
        self._env, self._goals = env, (goals, look_at)

    def goals(self):
        return self._goals

    def __getattr__(self, name):
        return getattr(self._env, name)


SCRIPTED_POLICIES = {"waypoint": waypoint_policy, "landing": landing_policy, "tracking": tracking_policy}
FORMATION_POLICIES = {
    "formation-waypoint": formation_policy,
    "formation-landing": formation_landing_policy,
    "formation-tracking": formation_policy,
}


TEAMMATE_BOX_COLOR = [255, 40, 255]


def draw_detection(image: np.ndarray, detection: np.ndarray, teammates=()) -> np.ndarray:
    """Draw a ``[visible, cx, cy, w, h]`` (YOLO xywhn) box on ``image`` in green (a red frame when
    not detected), and the visible ``teammates`` boxes in magenta."""
    image = image.copy()
    for box in teammates:
        if box[0]:
            draw_box(image, box, TEAMMATE_BOX_COLOR)
    if detection[0] == 0.0:
        image[:3], image[-3:], image[:, :3], image[:, -3:] = [220, 30, 30], [220, 30, 30], [220, 30, 30], [220, 30, 30]
        return image
    draw_box(image, detection, [40, 255, 40])
    return image


def draw_box(image: np.ndarray, box: np.ndarray, color) -> None:
    """Draw the outline of a ``[visible, cx, cy, w, h]`` box on ``image`` in place."""
    size = image.shape[0]
    _, cx, cy, w, h = box
    x0, x1 = (np.clip([cx - w / 2, cx + w / 2], 0, 1) * (size - 1)).astype(int)
    y0, y1 = (np.clip([cy - h / 2, cy + h / 2], 0, 1) * (size - 1)).astype(int)
    image[y0 : y0 + 2, x0 : x1 + 1] = color
    image[y1 - 1 : y1 + 1, x0 : x1 + 1] = color
    image[y0 : y1 + 1, x0 : x0 + 2] = color
    image[y0 : y1 + 1, x1 - 1 : x1 + 1] = color


def show_onboard(env, detections: np.ndarray, reward: float, total: float) -> None:
    """Onboard insets (one per drone for the formation tasks) and the episode stats."""
    viewer = env.unwrapped._viewer
    if viewer is None or viewer.viewport is None:
        return
    base, vp = env.unwrapped, viewer.viewport
    if detections.ndim == 1:
        images = [base.camera_image(width=INSET_SIZE, height=INSET_SIZE)]
        detections = detections[None]
    else:
        size = FORMATION_INSET_SIZE
        images = [base.onboard_image(i, size, size) for i in range(base.num_drones)]
    insets = []
    for k, (image, detection) in enumerate(zip(images, detections)):
        size = image.shape[0]
        # Along the top edge, right-aligned, leader on the left.
        rect = mujoco.MjrRect(vp.width - (size + 10) * (len(images) - k), vp.height - size - 10, size, size)
        teammates = base.detect_teammates(k) if detections.shape[0] > 1 else ()
        insets.append((rect, draw_detection(image, detection, teammates)))
    viewer.set_images(insets)
    labels = "detection [vis cx cy w h]\nreward / return\nstep"
    values = f"{np.array2string(detections[0], precision=2)}\n{reward:+.3f} / {total:.2f}\n{base.step_count}"
    viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, labels, values))


def readable(value):
    """Info value as plain Python for printing, floats rounded to 3 decimals."""
    if isinstance(value, np.ndarray | np.generic):
        value = np.round(value, 3) if np.issubdtype(value.dtype, np.floating) else value
        return value.tolist()
    return round(value, 3) if isinstance(value, float) else value


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch the drone tasks in the MuJoCo viewer.")
    parser.add_argument("task", choices=[*TASKS, *FORMATION_TASKS])
    parser.add_argument(
        "--policy", default="scripted", help="scripted, random, zero, or a training checkpoint (.pt)"
    )
    parser.add_argument(
        "--camera", choices=["chase", "overview", "onboard", "free"], help="default: chase, overview for formations"
    )
    parser.add_argument("--detection-noise", type=float, default=0.0)
    parser.add_argument("--detection-dropout", type=float, default=0.0)
    parser.add_argument("--episodes", type=int, default=None, help="default: until the viewer is closed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--drones", type=int, default=3, help="formation tasks: 2 or 3 drones")
    args = parser.parse_args()
    formation = args.task in FORMATION_TASKS

    trained = None
    if formation and args.policy.endswith(".pt"):
        parser.error("trained policies are not supported for the formation tasks yet")
    if args.policy.endswith(".pt"):
        from streampilot.policy import Policy

        trained = Policy.load(args.policy)
        assert trained.config["task"] == args.task, f"checkpoint is for {trained.config['task']}"
    elif args.policy not in ("scripted", "random", "zero"):
        parser.error(f"unknown policy {args.policy!r}")
    obs_mode = trained.config["obs_mode"] if trained else "detection"

    camera = args.camera or ("overview" if formation else "chase")
    camera = None if camera == "free" else camera
    if formation and camera in ("chase", "onboard"):
        camera = f"d0_{camera}"  # the leader's
    env = gym.make(
        FORMATION_TASKS[args.task] if formation else TASKS[args.task],
        obs_mode=obs_mode,
        render_mode="human",
        camera=camera,
        detection_noise=args.detection_noise,
        detection_dropout=args.detection_dropout,
        **({"num_drones": args.drones} if formation else {}),
    )
    base = env.unwrapped
    policies = {
        "scripted": lambda obs: (
            FORMATION_POLICIES[args.task](base) if formation else SCRIPTED_POLICIES[args.task](base, base.state_obs())
        ),
        "random": lambda obs: env.action_space.sample(),
        "zero": lambda obs: np.zeros(env.action_space.shape),
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
            if formation:
                detection = obs[:, :5]
            else:
                detection = obs if obs_mode == "detection" else base.detect(base._task_target_geom())
            show_onboard(env, detection, reward, total)
        if not base.viewer_running:
            break
        info = {k: readable(v) for k, v in info.items()}
        print(f"episode {episode}: return {total:7.2f}  steps {base.step_count}  {info}", flush=True)
        time.sleep(1.0)  # linger on the final state
        episode += 1
    env.close()


if __name__ == "__main__":
    main()
