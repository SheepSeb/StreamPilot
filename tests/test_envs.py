import dataclasses

import gymnasium as gym
import mujoco
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import streampilot.env  # noqa: F401  (registers the environments)
from streampilot.visualize import landing_policy, tracking_policy, waypoint_policy

ENV_IDS = ["DroneWaypoint-v0", "DroneLanding-v0", "DroneTracking-v0"]
OBS_MODES = ["detection", "pixels", "state"]


def rollout(env, policy, seed):
    """Run one episode with a scripted ``policy(env, state)``; return (return, last info, infos)."""
    base = env.unwrapped
    env.reset(seed=seed)
    total, infos = 0.0, []
    while True:
        _, reward, terminated, truncated, info = env.step(policy(base, base.state_obs()))
        total += reward
        infos.append(info)
        if terminated or truncated:
            return total, info, infos


@pytest.mark.parametrize("obs_mode", OBS_MODES)
@pytest.mark.parametrize("env_id", ENV_IDS)
def test_env_checker(env_id, obs_mode):
    env = gym.make(env_id, obs_mode=obs_mode)
    if obs_mode == "pixels":
        # Rendering is only reproducible up to a few intensity levels (see test below), so skip
        # check_env's exact-equality reset check.
        env.unwrapped.spec = dataclasses.replace(env.spec, nondeterministic=True)
    check_env(env.unwrapped, skip_render_check=True)
    env.close()


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_pixel_reset_is_seeded_up_to_gpu_noise(env_id):
    # The GPU shades the floor slightly differently depending on what it rendered before
    # (observed: <= 4 of 255 levels, NVIDIA/EGL), so compare with a tolerance.
    env = gym.make(env_id, obs_mode="pixels")
    obs_a, _ = env.reset(seed=3)
    env.reset(seed=4)
    obs_b, _ = env.reset(seed=3)
    assert np.abs(obs_a.astype(int) - obs_b).max() <= 8
    env.close()


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_reset_is_seeded(env_id):
    env = gym.make(env_id, obs_mode="state")
    obs_a, _ = env.reset(seed=3)
    obs_b, _ = env.reset(seed=3)
    np.testing.assert_array_equal(obs_a, obs_b)
    env.close()


def test_action_is_body_frame():
    env = gym.make("DroneWaypoint-v0")
    base = env.unwrapped
    env.reset(seed=0)
    base._set_drone_state([0.0, 0.0], yaw=np.pi / 2)  # nose along +y
    for _ in range(20):
        env.step(np.array([0.5, 0.0, 0.0]))  # forward at 1 m/s
    np.testing.assert_allclose(base.drone_vel, [0.0, 1.0, 0.0], atol=0.02)
    env.close()


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_altitude_is_held(env_id):
    env = gym.make(env_id)
    base = env.unwrapped
    env.reset(seed=0)
    env.action_space.seed(0)
    for _ in range(200):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert abs(base.drone_pos[2] - base.flight_altitude) < 1e-3
        if terminated or truncated:
            env.reset()
    env.close()


def place_waypoint_on_optical_axis(env, distance=2.0):
    base = env.unwrapped
    env.reset(seed=0)
    base._set_drone_state([0.0, 0.0], yaw=0.0)
    mujoco.mj_forward(base.model, base.data)  # update the camera pose
    # The camera is level and looks along the nose.
    base.data.mocap_pos[base._marker] = base.data.cam_xpos[base._onboard_cam_id] + [distance, 0.0, 0.0]
    mujoco.mj_forward(base.model, base.data)
    return base


def test_detection_geometry():
    env = gym.make("DroneWaypoint-v0")
    base = place_waypoint_on_optical_axis(env)
    visible, cx, cy, w, h = base.detect(base._marker_geom)
    # Silhouette width of the ball; the box projects its bounding cube, so it is a bit larger.
    silhouette = 2 * base.reach_radius / 2.0 / (2 * np.tan(base._half_fov))
    assert visible == 1.0
    np.testing.assert_allclose([cx, cy], [0.5, 0.5], atol=5e-3)
    assert silhouette <= w < 1.2 * silhouette and silhouette <= h < 1.5 * silhouette

    base._set_drone_state([0.0, 0.0], yaw=np.pi)  # facing away
    mujoco.mj_forward(base.model, base.data)
    assert base.detect(base._marker_geom)[0] == 0.0
    env.close()


def test_detection_matches_rendered_pixels():
    env = gym.make("DroneWaypoint-v0", obs_mode="pixels", image_size=128)
    base = place_waypoint_on_optical_axis(env)
    base.data.mocap_pos[base._marker] += [0.0, 0.4, 0.2]  # off-centre
    mujoco.mj_forward(base.model, base.data)
    image = base.camera_image().astype(int)
    green = (image[..., 1] > 150) & (image[..., 0] < 100) & (image[..., 2] < 100)
    ys, xs = np.nonzero(green)
    assert len(xs) > 20
    pixel_box = np.array([(xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2]) / 128
    _, cx, cy, w, h = base.detect(base._marker_geom)
    np.testing.assert_allclose([cx, cy], pixel_box, atol=2 / 128)
    env.close()


def test_detection_noise_and_dropout():
    env = gym.make("DroneWaypoint-v0", detection_dropout=1.0)
    place_waypoint_on_optical_axis(env)
    assert not env.unwrapped.detection_obs().any()
    env.close()

    env = gym.make("DroneWaypoint-v0", detection_noise=0.05)
    base = place_waypoint_on_optical_axis(env)
    noisy = base.detection_obs()
    assert noisy[0] == 1.0 and not np.allclose(noisy[1:3], [0.5, 0.5], atol=1e-3)
    env.close()


def test_waypoint_scripted_controller_succeeds():
    env = gym.make("DroneWaypoint-v0")
    for seed in range(5):
        total, info, infos = rollout(env, waypoint_policy, seed)
        assert info["is_success"], info
        assert total > 0
        assert any(i["target_in_view"] for i in infos)
    env.close()


def test_landing_scripted_controller_hands_off():
    env = gym.make("DroneLanding-v0")
    base = env.unwrapped
    for seed in range(10):
        _, info, infos = rollout(env, landing_policy, seed)
        assert info["is_success"], (seed, info)
        assert infos[-1]["target_in_view"]
        # Hand-off from straight in front of the pad, at the hand-off distance.
        forward, lateral = info["handoff_offset"]
        assert abs(forward - base.handoff_distance) <= base.distance_tolerance and abs(lateral) < 0.25
    env.close()


def test_landing_hover_above_pad_is_not_a_handoff():
    # Straight above the pad, the level camera cannot see it: no hand-off however long it hovers.
    env = gym.make("DroneLanding-v0", max_episode_steps=100)
    env.reset(seed=0)
    base = env.unwrapped
    base._set_drone_state(base._pad_top[:2] - [0.1, 0.0], yaw=0.0)
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(np.zeros(3))
    assert truncated and not info["is_success"] and not info["target_in_view"]
    env.close()


def test_tracking_scripted_controller_keeps_target_in_view():
    env = gym.make("DroneTracking-v0")
    for seed in range(5):
        followed, info, infos = rollout(env, tracking_policy, seed)
        hovered, _, _ = rollout(env, lambda base, state: np.zeros(3), seed)
        assert len(infos) == 500 and "out_of_bounds" not in info, (seed, len(infos), info)
        assert info["standoff_error"] < 0.3
        assert np.mean([i["target_in_view"] for i in infos]) > 0.9
        assert followed > 2 * hovered
    env.close()


def test_rgb_render():
    env = gym.make("DroneWaypoint-v0", render_mode="rgb_array", width=64, height=48)
    env.reset(seed=0)
    assert env.render().shape == (48, 64, 3)
    env.close()
