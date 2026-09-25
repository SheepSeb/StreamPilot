import gymnasium as gym
import numpy as np
import pytest
import torch

import streampilot.env  # noqa: F401  (registers the environments)
from streampilot.baselines.mappo import MAPPO, TeamHistory
from streampilot.policy import TeamPolicy
from streampilot.stream_x.wrappers import ObservationHistory, RunningMeanStd
from streampilot.train_formation import TASKS, Team, evaluate, save_checkpoint
from streampilot.vec_env import FormationVecEnv


@pytest.fixture(scope="module")
def venv():
    env = FormationVecEnv(TASKS["formation-waypoint"], {"num_drones": 2}, num_envs=3, num_workers=2, with_state=True)
    yield env
    env.close()


def test_running_mean_std_batch_update_matches_sequential():
    rng = np.random.default_rng(0)
    sequential, batched = RunningMeanStd((4,)), RunningMeanStd((4,))
    for chunk in (rng.normal(2.0, 3.0, size=(n, 4)) for n in (1, 7, 30)):
        for row in chunk:
            sequential.update(row)
        batched.update_batch(chunk)
    np.testing.assert_allclose(batched.mean, sequential.mean)
    np.testing.assert_allclose(batched.var, sequential.var)
    assert batched.count == sequential.count


def test_team_history_matches_per_drone_history():
    rng = np.random.default_rng(0)
    team = TeamHistory(num_envs=2, num_drones=3, obs_dim=5, action_dim=3, num_frames=4)
    drones = [[ObservationHistory(5, 3, 4) for _ in range(3)] for _ in range(2)]
    obs = rng.normal(size=(2, 3, 5)).astype(np.float32)
    features = team.reset(obs)
    expected = [[h.reset(obs[e, d]) for d, h in enumerate(row)] for e, row in enumerate(drones)]
    np.testing.assert_allclose(features, expected)
    for step in range(10):
        actions = rng.normal(size=(2, 3, 3)).astype(np.float32) * 2  # some outside [-1, 1]
        obs = rng.normal(size=(2, 3, 5)).astype(np.float32)
        features = team.push(actions, obs)
        expected = [[h.push(actions[e, d], obs[e, d]) for d, h in enumerate(row)] for e, row in enumerate(drones)]
        if step == 5:  # environment 1 starts a new episode
            done = np.array([False, True])
            features = team.reset(obs, done)
            expected[1] = [h.reset(obs[1, d]) for d, h in enumerate(drones[1])]
        np.testing.assert_allclose(features, expected, atol=1e-6)


def test_vec_env_matches_sequential_envs():
    """Same seeds and actions: the workers reproduce plain environments, including autoresets
    (a short time limit, so every environment resets)."""
    rng = np.random.default_rng(0)
    envs = [gym.make(TASKS["formation-waypoint"], num_drones=2, max_episode_steps=15) for _ in range(3)]
    short = FormationVecEnv(
        TASKS["formation-waypoint"], {"num_drones": 2, "max_episode_steps": 15}, num_envs=3, num_workers=2, with_state=True
    )
    try:
        obs = short.reset(seeds=[5, 6, 7])
        expected = [env.reset(seed=s)[0] for env, s in zip(envs, [5, 6, 7])]
        np.testing.assert_array_equal(obs, expected)
        ended = 0
        for _ in range(40):
            actions = rng.uniform(-1, 1, size=(3, 2, 3)).astype(np.float32)
            obs, reward, terminated, truncated, episodes = short.step(actions)
            ended += len(episodes)
            for i, env in enumerate(envs):
                o, r, te, tr, _ = env.step(actions[i])
                assert reward[i] == r and terminated[i] == te and truncated[i] == tr
                if te or tr:
                    np.testing.assert_array_equal(short.final_obs[i], o)
                    np.testing.assert_array_equal(short.final_state[i], env.unwrapped.state_obs())
                    o, _ = env.reset()
                np.testing.assert_array_equal(obs[i], o)
                np.testing.assert_array_equal(short.state[i], env.unwrapped.state_obs())
        assert ended >= 3  # every environment finished at least one (15-step) episode
    finally:
        short.close()


def make_mappo(team: Team, num_envs: int, rollout_steps: int = 16) -> MAPPO:
    return MAPPO(
        team.actor_dim,
        team.critic_dim,
        3,
        team.venv.num_drones,
        num_envs,
        total_steps=10 * rollout_steps * num_envs,
        hidden_size=32,
        critic_hidden_size=32,
        rollout_steps=rollout_steps,
        epochs=2,
        num_minibatches=2,
    )


def collect(venv, team, agent, actor_x, critic_x):
    while not agent.rollout_full:
        actions = agent.act(actor_x)
        obs, reward, terminated, truncated, _ = venv.step(actions)
        done = terminated | truncated
        next_actor_x, next_critic_x, final_critic_x = team.step(actions, obs, done)
        agent.store(actor_x, critic_x, actions, reward, terminated, done, final_critic_x)
        actor_x, critic_x = next_actor_x, next_critic_x
    return actor_x, critic_x


def test_mappo_update(venv):
    torch.manual_seed(0)
    team = Team(venv, num_frames=2, critic_state=True)
    assert team.critic_dim == 2 * team.actor_dim + 2 * venv.state_dim
    agent = make_mappo(team, venv.num_envs)
    actor_x, critic_x = team.reset(venv.reset(seeds=[0, 1, 2]))
    before = [p.clone() for p in agent.actor.parameters()]
    for _ in range(2):
        actor_x, critic_x = collect(venv, team, agent, actor_x, critic_x)
        metrics = agent.update(critic_x)
        assert np.isfinite(list(metrics.values())).all()
    assert any(not torch.equal(b, p) for b, p in zip(before, agent.actor.parameters()))
    assert agent.optim.param_groups[0]["lr"] < 3e-4  # linearly annealed


def test_checkpoint_team_policy_matches_training_agent(venv, tmp_path):
    """The deployed per-drone policies (raw rows, frozen stats) reproduce the training agent's
    deterministic actions."""
    torch.manual_seed(0)
    team = Team(venv, num_frames=3, critic_state=False)
    agent = make_mappo(team, venv.num_envs)
    actor_x, critic_x = team.reset(venv.reset(seeds=[0, 1, 2]))
    collect(venv, team, agent, actor_x, critic_x)
    agent.update(critic_x)
    config = {
        "task": "formation-waypoint",
        "algo": "mappo",
        "obs_mode": "detection",
        "env_kwargs": {"num_drones": 2},
        "num_frames": 3,
        "hidden_size": 32,
        "gamma": 0.99,
    }
    save_checkpoint(tmp_path / "ckpt.pt", config, agent, team)
    policy = TeamPolicy.load(tmp_path / "ckpt.pt")

    env = gym.make(TASKS["formation-waypoint"], num_drones=2)
    history = TeamHistory(1, 2, venv.obs_dim, 3, 3)
    obs, _ = env.reset(seed=1)
    policy.reset()
    features = history.reset(obs[None])
    for _ in range(20):
        actor_x = team.actor_norm(features.reshape(-1, team.actor_dim), update=False).reshape(features.shape)
        expected = agent.act_deterministic(actor_x)[0]
        action = policy(obs)
        np.testing.assert_allclose(action, expected, atol=1e-5)
        obs, *_ = env.step(action)
        features = history.push(action[None], obs[None])

    result = evaluate(venv, agent, team, episodes=4, seed=100)
    assert 0 < result["eval/length"] <= 600 and np.isfinite(result["eval/return"])
