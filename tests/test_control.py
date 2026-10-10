import gymnasium as gym
import numpy as np
import pytest

import streampilot.env  # noqa: F401  (registers the environments)
from streampilot.control import MPC, PID, TAU, ControllerPolicy

SINGLE = ["DroneWaypoint-v0", "DroneLanding-v0"]
FORMATION = ["DroneFormationWaypoint-v0", "DroneFormationLanding-v0"]


def rollout(env, policy, seed):
    base = env.unwrapped
    env.reset(seed=seed)
    policy.reset(base)
    while True:
        _, _, terminated, truncated, info = env.step(policy(base))
        if terminated or truncated:
            return info


@pytest.mark.parametrize("kind", ["pid", "mpc"])
@pytest.mark.parametrize("env_id", SINGLE)
def test_single_drone_tasks_succeed(env_id, kind):
    env = gym.make(env_id, obs_mode="state")
    policy = ControllerPolicy(kind)
    for seed in range(3):
        assert rollout(env, policy, seed)["is_success"], seed
    env.close()


@pytest.mark.parametrize("kind", ["pid", "mpc"])
@pytest.mark.parametrize("env_id", FORMATION)
def test_formation_tasks_succeed_without_collisions(env_id, kind):
    env = gym.make(env_id, obs_mode="state")
    policy = ControllerPolicy(kind)
    for seed in range(3):
        info = rollout(env, policy, seed)
        assert "collision" not in info and np.all(info["is_success"]), (seed, info)
    env.close()


@pytest.mark.parametrize("env_id", ["DroneTracking-v0", "DroneFormationTracking-v0"])
def test_tracking_stays_in_bounds(env_id):
    env = gym.make(env_id, obs_mode="state")
    policy = ControllerPolicy("mpc")
    info = rollout(env, policy, 0)
    assert "out_of_bounds" not in info and "collision" not in info
    env.close()


def test_mpc_plans_within_speed_limit_and_reaches_goal_from_rest():
    dt = 0.05
    mpc, pid = MPC(dt, max_speed=1.0), PID(max_speed=1.0)
    for tracker in (mpc, pid):
        tracker.reset()
        pos, vel, goal = np.zeros((1, 2)), np.zeros((1, 2)), np.array([[2.0, -1.0]])
        a = np.exp(-dt / TAU)  # the simulated velocity lag
        for _ in range(200):
            u = tracker(pos, vel, goal, np.zeros((1, 2)), dt)
            assert np.linalg.norm(u) <= 1.0 + 1e-9
            pos = pos + dt * vel  # close enough to the plant for a convergence check
            vel = a * vel + (1 - a) * u
        assert np.linalg.norm(pos - goal) < 0.05, type(tracker).__name__
