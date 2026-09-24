import argparse

import numpy as np
import pytest
import torch

from streampilot.baselines.ppo import PPO
from streampilot.baselines.sac import SAC
from streampilot.policy import Policy
from streampilot.stream_x.wrappers import ObservationHistory
from streampilot.train import ALGOS, make_env, save_checkpoint


def make_config(algo: str) -> dict:
    return {
        "task": "waypoint",
        "algo": algo,
        "obs_mode": "detection",
        "env_kwargs": {},
        "num_frames": 4,
        "hidden_size": 32,
        "gamma": 0.99,
    }


def make_agent(algo: str, env, steps: int):
    parser = argparse.ArgumentParser()
    ALGOS[algo].add_args(parser)
    args = parser.parse_args(["--hidden-size", "32"])
    args.steps, args.seed = steps, 0
    if algo == "ppo":
        args.rollout_steps, args.minibatch_size, args.epochs = 64, 16, 2
    if algo == "sac":
        args.learning_starts, args.batch_size = 32, 16
    return ALGOS[algo].make_agent(args, env.observation_space.shape[0], env.action_space.shape[0])


def run(agent, env, steps: int) -> list[dict]:
    metrics = []
    obs, _ = env.reset(seed=0)
    for _ in range(steps):
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        metrics.append(agent.observe(obs, action, reward, next_obs, terminated, terminated or truncated))
        obs = next_obs if not (terminated or truncated) else env.reset()[0]
    return [m for m in metrics if m]


def test_ppo_updates_once_per_rollout():
    env = make_env(make_config("ppo"))
    agent = make_agent("ppo", env, steps=200)
    assert isinstance(agent, PPO)
    before = [p.clone() for p in agent.actor.parameters()]
    metrics = run(agent, env, 200)
    assert len(metrics) == 200 // 64
    assert all(np.isfinite(list(m.values())).all() for m in metrics)
    assert any(not torch.equal(b, p) for b, p in zip(before, agent.actor.parameters()))
    assert agent.optim.param_groups[0]["lr"] < 3e-4  # linearly annealed


def test_sac_explores_randomly_then_updates_every_step():
    env = make_env(make_config("sac"))
    agent = make_agent("sac", env, steps=100)
    assert isinstance(agent, SAC)
    metrics = run(agent, env, 100)
    assert len(metrics) == 100 - 32 + 1
    assert sum("actor_loss" in m for m in metrics) == len(range(32, 101, 2))  # delayed policy updates
    assert all(np.isfinite(list(m.values())).all() for m in metrics)
    # The targets trail the online Q-networks.
    for target, online in zip(agent.q_targets.parameters(), agent.qs.parameters()):
        assert not torch.equal(target, online)


@pytest.mark.parametrize("algo", ["stream_ac", "ppo", "sac"])
def test_checkpoint_policy_matches_training_agent(tmp_path, algo):
    config = make_config(algo)
    env = make_env(config)
    agent = make_agent(algo, env, steps=200)
    run(agent, env, 200)
    save_checkpoint(tmp_path / "ckpt.pt", config, agent, env)

    # Replay a fresh episode through both: the deployed policy (raw obs, frozen stats) must
    # reproduce the training agent's deterministic actions on the same (normalized) features.
    policy = Policy.load(tmp_path / "ckpt.pt")
    raw_env = env.unwrapped
    if ALGOS[algo].NORMALIZE:
        stats = env.get_wrapper_attr("obs_stats")
        mean, std = stats.mean, np.sqrt(stats.var + 1e-8)
    else:
        mean, std = 0.0, 1.0
    history = ObservationHistory(5, 3, 4)
    raw_obs, _ = raw_env.reset(seed=1)
    policy.reset()
    features = history.reset(raw_obs)
    for _ in range(20):
        normalized = torch.as_tensor(((features - mean) / std).astype(np.float32))
        with torch.no_grad():
            expected = agent.actor.deterministic(normalized).numpy()
        action = policy(raw_obs)
        np.testing.assert_allclose(action, expected, atol=1e-5)
        raw_obs, *_ = raw_env.step(action)
        features = history.push(action, raw_obs)
