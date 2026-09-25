import dataclasses
import itertools

import gymnasium as gym
import mujoco
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import streampilot.env  # noqa: F401  (registers the environments)
from streampilot.env.formation import formation_offsets
from streampilot.visualize import formation_landing_policy, formation_policy

ENV_IDS = ["DroneFormationWaypoint-v0", "DroneFormationLanding-v0", "DroneFormationTracking-v0"]


def rollout(env, policy, seed):
    """Run one episode with a scripted ``policy(env)``; return (return, last info, infos)."""
    base = env.unwrapped
    env.reset(seed=seed)
    total, infos = 0.0, []
    while True:
        _, reward, terminated, truncated, info = env.step(policy(base))
        total += reward
        infos.append(info)
        if terminated or truncated:
            return total, info, infos


def pairwise_distances(xy):
    return sorted(np.linalg.norm(a - b) for a, b in itertools.combinations(xy, 2))


@pytest.mark.parametrize("obs_mode", ["detection", "pixels", "state"])
@pytest.mark.parametrize("env_id", ENV_IDS)
def test_env_checker(env_id, obs_mode):
    env = gym.make(env_id, obs_mode=obs_mode)
    if obs_mode == "pixels":
        env.unwrapped.spec = dataclasses.replace(env.spec, nondeterministic=True)  # see test_envs.py
    check_env(env.unwrapped, skip_render_check=True)
    env.close()


@pytest.mark.parametrize("num_drones", [2, 3])
@pytest.mark.parametrize("env_id", ENV_IDS)
def test_spaces_and_seeded_reset(env_id, num_drones):
    env = gym.make(env_id, num_drones=num_drones, obs_mode="state")
    obs_a, _ = env.reset(seed=3)
    obs_b, _ = env.reset(seed=3)
    np.testing.assert_array_equal(obs_a, obs_b)
    assert env.action_space.shape == (num_drones, 3)
    assert env.unwrapped.min_pairwise_distance() >= env.unwrapped.min_separation
    env.close()

    env = gym.make(env_id, num_drones=num_drones)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (num_drones, 5 + num_drones + 2 * (num_drones - 1))
    np.testing.assert_array_equal(obs[:, 5 : 5 + num_drones], np.eye(num_drones))  # slot one-hot
    env.close()


def test_formation_shapes():
    column = formation_offsets(2, 1.0)
    np.testing.assert_allclose(column, [[0.0, 0.0], [-1.0, 0.0]])  # one behind the other
    triangle = formation_offsets(3, 1.0)
    np.testing.assert_allclose(pairwise_distances(triangle), [1.0, 1.0, 1.0])  # equilateral
    assert np.all(triangle[1:, 0] < 0)  # leader at the apex, in front
    with pytest.raises(ValueError):
        formation_offsets(4, 1.0)


def test_actions_are_per_drone_body_frame():
    env = gym.make("DroneFormationWaypoint-v0")
    base = env.unwrapped
    env.reset(seed=0)
    base._set_drone_states([[0.0, 0.0], [0.0, 1.5], [0.0, -1.5]], yaw=[0.0, np.pi / 2, np.pi])
    for _ in range(20):
        env.step([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])  # 1 m/s forward / left
    np.testing.assert_allclose(base.drone_vel[:, :2], [[0.0, 0.0], [0.0, 1.0], [0.0, -1.0]], atol=0.02)
    assert np.all(np.abs(base.drone_pos[:, 2] - base.flight_altitude) < 1e-3)
    env.close()


def test_teammates_are_in_body_frame():
    env = gym.make("DroneFormationTracking-v0", num_drones=2)
    base = env.unwrapped
    env.reset(seed=0)
    base._set_drone_states([[0.0, 0.0], [0.0, 1.0]], yaw=[np.pi / 2, 0.0])
    # Drone 0 faces +y, so drone 1 is straight ahead of it; drone 0 is on drone 1's right.
    np.testing.assert_allclose(base.teammates(), [[1.0, 0.0], [0.0, -1.0]], atol=1e-9)
    env.close()


def test_collision_ends_episode():
    env = gym.make("DroneFormationWaypoint-v0", num_drones=2)
    base = env.unwrapped
    env.reset(seed=0)
    base._set_drone_states([[0.0, 0.0], [1.0, 0.0]], yaw=[0.0, 0.0])
    terminated, steps = False, 0
    while not terminated:
        _, reward, terminated, _, info = env.step([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # ram drone 1
        steps += 1
    assert info["collision"] and steps < 20 and reward < -5
    env.close()


@pytest.mark.parametrize("num_drones", [2, 3])
def test_waypoint_scripted_team_reaches_formations(num_drones):
    env = gym.make("DroneFormationWaypoint-v0", num_drones=num_drones)
    base = env.unwrapped
    for seed in range(5):
        _, info, infos = rollout(env, formation_policy, seed)
        assert info["is_success"], (seed, info)
        assert "collision" not in info
        assert any(i["target_in_view"].all() for i in infos)
        # All drones sit on their waypoints together: the formation shape.
        expected = pairwise_distances(base.formation)
        np.testing.assert_allclose(pairwise_distances(base.drone_pos[:, :2]), expected, atol=2 * base.reach_radius)
    env.close()


@pytest.mark.parametrize("num_drones", [2, 3])
def test_landing_scripted_team_hands_off_in_formation(num_drones):
    env = gym.make("DroneFormationLanding-v0", num_drones=num_drones)
    base = env.unwrapped
    for seed in range(5):
        _, info, infos = rollout(env, formation_landing_policy, seed)
        assert info["is_success"], (seed, info)
        assert infos[-1]["target_in_view"].all()
        # Every drone is handoff_distance from its landing spot, and the spots form the shape,
        # with the leader's on the pad.
        np.testing.assert_allclose(np.linalg.norm(info["handoff_offset"], axis=1), base.handoff_distance, atol=0.15)
        spots = base.landing_spots()
        np.testing.assert_allclose(pairwise_distances(spots), pairwise_distances(base.formation), atol=1e-9)
        np.testing.assert_allclose(spots[0], base._pad_top[:2])
    env.close()


@pytest.mark.parametrize("num_drones", [2, 3])
def test_tracking_scripted_team_holds_formation(num_drones):
    env = gym.make("DroneFormationTracking-v0", num_drones=num_drones)
    for seed in range(3):
        followed, info, infos = rollout(env, formation_policy, seed)
        hovered, _, _ = rollout(env, lambda base: np.zeros((num_drones, 3)), seed)
        assert len(infos) == 500 and "collision" not in info and "out_of_bounds" not in info, (seed, info)
        assert np.max([i["formation_error"] for i in infos[100:]]) < 0.3
        assert np.mean([i["target_in_view"] for i in infos]) > 0.9
        assert followed > 2 * hovered
    env.close()


def test_rgb_render():
    env = gym.make("DroneFormationTracking-v0", render_mode="rgb_array", width=64, height=48)
    env.reset(seed=0)
    assert env.render().shape == (48, 64, 3)
    env.close()


def test_teammate_detection():
    env = gym.make("DroneFormationTracking-v0")
    base = env.unwrapped
    env.reset(seed=0)
    # Drone 0 at the origin facing +x: drone 1 straight ahead, drone 2 behind it.
    base._set_drone_states([[0.0, 0.0], [1.5, 0.0], [-1.5, 0.0]], yaw=[0.0, 0.0, 0.0])
    mujoco.mj_forward(base.model, base.data)
    ahead, behind = base.detect_teammates(0)
    assert ahead[0] == 1.0 and behind[0] == 0.0
    assert abs(ahead[1] - 0.5) < 0.01 and 0.4 < ahead[2] <= 0.5  # centred; the masts on top raise it a little
    assert ahead[3] > ahead[4]  # a drone is wider than it is tall
    # Drone 2 sees both teammates ahead (drone 0 at 1.5 m, drone 1 at 3 m), the nearer one bigger.
    near, far = base.detect_teammates(2)
    assert far[0] == near[0] == 1.0 and near[3] > far[3]
    env.close()


def test_teammate_detection_matches_rendered_pixels():
    env = gym.make("DroneFormationTracking-v0", num_drones=2, obs_mode="pixels", image_size=128)
    base = env.unwrapped
    env.reset(seed=0)
    base.data.mocap_pos[base._target] = [50.0, 0.0, 0.0]  # out of the way
    base._set_drone_states([[0.0, 0.0], [1.2, 0.3]], yaw=[0.0, 0.4])
    mujoco.mj_forward(base.model, base.data)
    empty = base.onboard_image(0).astype(int)
    base._set_drone_states([[0.0, 0.0], [100.0, 0.0]], yaw=[0.0, 0.0])  # teammate far away
    mujoco.mj_forward(base.model, base.data)
    changed = np.abs(empty - base.onboard_image(0)).sum(axis=-1) > 30
    base._set_drone_states([[0.0, 0.0], [1.2, 0.3]], yaw=[0.0, 0.4])
    mujoco.mj_forward(base.model, base.data)
    ys, xs = np.nonzero(changed)
    pixel_box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]) / 128
    _, cx, cy, w, h = base.detect_teammates(0)[0]
    np.testing.assert_allclose([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], pixel_box, atol=3 / 128)
    env.close()
