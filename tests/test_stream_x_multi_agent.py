import gymnasium as gym
import numpy as np
import torch

import streampilot.env  # noqa: F401  (registers the environments)
from streampilot.policy import TeamPolicy
from streampilot.stream_x.agents import Actor, Critic, StreamAC
from streampilot.stream_x.multi_agent import CentralizedStreamAC, IndependentStreamAC
from streampilot.stream_x.optim import ObGD
from streampilot.stream_x.wrappers import ObservationHistory, RunningMeanStd
from streampilot.train_formation_stream import TASKS, evaluate, save_checkpoint

CONFIG = {
    "task": "formation-waypoint",
    "algo": "istream_ac",
    "obs_mode": "detection",
    "env_kwargs": {"num_drones": 2},
    "num_frames": 3,
    "hidden_size": 32,
    "gamma": 0.99,
}


def make_team(env, num_frames=3):
    return IndependentStreamAC(env.unwrapped.num_drones, env.observation_space.shape[1], 3, num_frames, hidden_size=32)


def run(env, team, steps: int, seed: int = 0) -> None:
    obs, _ = env.reset(seed=seed)
    team.reset(obs)
    for _ in range(steps):
        actions = team.act()
        obs, reward, terminated, truncated, _ = env.step(actions)
        done = terminated or truncated
        assert np.isfinite(team.observe(actions, reward, obs, terminated, done)).all()
        if done:
            obs, _ = env.reset()
            team.reset(obs)


class SingleDrone:
    """The reference: a plain ``StreamAC`` with the normalization of ``train.make_env``."""

    def __init__(self, agent: StreamAC, obs_dim: int, num_frames: int):
        self.agent, self.history = agent, ObservationHistory(obs_dim, 3, num_frames)
        self.obs_stats, self.return_stats, self._return = RunningMeanStd((self.history.dim,)), RunningMeanStd(), 0.0

    def normalize(self, x):
        self.obs_stats.update(x)
        return ((x - self.obs_stats.mean) / np.sqrt(self.obs_stats.var + 1e-8)).astype(np.float32)

    def reset(self, obs):
        self.x = self.normalize(self.history.reset(obs))

    def observe(self, action, reward, next_obs, terminated, done):
        next_x = self.normalize(self.history.push(action, next_obs))
        self._return = self._return * 0.99 * (1.0 - terminated) + reward
        self.return_stats.update(self._return)
        if done:
            self._return = 0.0
        delta = self.agent.update(self.x, action, reward / np.sqrt(self.return_stats.var + 1e-8), next_x, terminated, done)
        self.x = next_x
        return delta


def test_batched_team_matches_separate_stream_ac_learners():
    """Same initial weights and transitions: the batched team does exactly what one StreamAC per
    drone would, including a trace reset at the end of an episode."""
    torch.manual_seed(0)
    env = gym.make(TASKS["formation-waypoint"], num_drones=3, max_episode_steps=40)
    team = make_team(env)
    obs_dim = env.observation_space.shape[1]
    singles = []
    for i in range(3):
        drone = SingleDrone(StreamAC(team.histories[0].dim, 3, hidden_size=32), obs_dim, 3)
        drone.agent.actor.load_state_dict(team.actor.drone_state_dict(i))
        drone.agent.critic.load_state_dict(team.critic.drone_state_dict(i))
        singles.append(drone)

    rng = np.random.default_rng(0)
    obs, _ = env.reset(seed=0)
    team.reset(obs)
    for drone, row in zip(singles, obs):
        drone.reset(row)
    for _ in range(60):
        actions = rng.normal(size=(3, 3)).astype(np.float32)
        obs, reward, terminated, truncated, _ = env.step(actions)
        done = terminated or truncated
        delta = team.observe(actions, reward, obs, terminated, done)
        expected = [d.observe(a, reward, row, terminated, done) for d, a, row in zip(singles, actions, obs)]
        np.testing.assert_allclose(delta, expected, rtol=1e-4, atol=1e-5)
        if done:
            obs, _ = env.reset()
            team.reset(obs)
            for drone, row in zip(singles, obs):
                drone.reset(row)
    for i, drone in enumerate(singles):
        for net, ref in ((team.actor, drone.agent.actor), (team.critic, drone.agent.critic)):
            for k, v in net.drone_state_dict(i).items():
                torch.testing.assert_close(v, ref.state_dict()[k], rtol=1e-4, atol=1e-5)


def test_learners_are_independent():
    torch.manual_seed(0)
    env = gym.make(TASKS["formation-waypoint"], num_drones=2)
    team = make_team(env)
    before = [team.actor.drone_state_dict(i) for i in range(2)]
    run(env, team, 50)
    for i in range(2):
        after = team.actor.drone_state_dict(i)
        assert any(not torch.equal(before[i][k], after[k]) for k in after)
    assert team.obs_stats.count == 51  # the first observation and one per step
    assert not torch.equal(team.actor.heads[0].weight[0], team.actor.heads[0].weight[1])


def test_checkpoint_team_policy_matches_learners(tmp_path):
    """The deployed per-drone policies (raw rows, frozen stats) reproduce the team's
    deterministic actions."""
    torch.manual_seed(0)
    env = gym.make(TASKS["formation-waypoint"], num_drones=2)
    team = make_team(env)
    run(env, team, 30)
    save_checkpoint(tmp_path / "ckpt.pt", CONFIG, team, env)
    policy = TeamPolicy.load(tmp_path / "ckpt.pt")

    obs, _ = env.reset(seed=1)
    policy.reset()
    features = np.stack([h.reset(row) for h, row in zip(team.histories, obs)])
    stats = team.obs_stats
    for _ in range(20):
        team.features = ((features - stats.mean) / np.sqrt(stats.var + 1e-8)).astype(np.float32)
        expected = team.act(deterministic=True).clip(-1.0, 1.0)
        action = policy(obs)
        np.testing.assert_allclose(action, expected, atol=1e-5)
        obs, *_ = env.step(action)
        features = np.stack([h.push(a, row) for h, a, row in zip(team.histories, action, obs)])

    result = evaluate(tmp_path / "ckpt.pt", CONFIG, episodes=2, seed=100)
    assert 0 < result["eval/length"] <= 600 and np.isfinite(result["eval/return"])


def test_centralized_matches_shared_critic_and_separate_actors():
    """Same initial weights and transitions: the CTDE team does exactly what one ``Critic`` of the
    joint features (its own ObGD and trace, the team TD error) and one ``Actor`` per drone (each
    with its own ObGD and trace, the shared TD error) would, including trace resets."""
    torch.manual_seed(0)
    env = gym.make(TASKS["formation-waypoint"], num_drones=3, max_episode_steps=40)
    obs_dim, n = env.observation_space.shape[1], 3
    team = CentralizedStreamAC(n, obs_dim, 3, 3, hidden_size=32)
    critic = Critic(n * team.dim, hidden_size=32)
    critic.load_state_dict(team.critic.state_dict())
    critic_optim = ObGD(critic.parameters(), kappa=2.0)
    actors = [Actor(team.dim, 3, hidden_size=32) for _ in range(n)]
    for i, actor in enumerate(actors):
        actor.load_state_dict(team.actor.drone_state_dict(i))
    actor_optims = [ObGD(a.parameters(), kappa=3.0) for a in actors]
    histories = [ObservationHistory(obs_dim, 3, 3) for _ in range(n)]
    obs_stats, return_stats, ret = RunningMeanStd((n, team.dim)), RunningMeanStd(), 0.0

    def normalize(features):
        obs_stats.update(features)
        return ((features - obs_stats.mean) / np.sqrt(obs_stats.var + 1e-8)).astype(np.float32)

    rng = np.random.default_rng(0)
    obs, _ = env.reset(seed=0)
    team.reset(obs)
    x = normalize(np.stack([h.reset(row) for h, row in zip(histories, obs)]))
    resets = 0
    for _ in range(60):
        actions = rng.normal(size=(n, 3)).astype(np.float32)
        obs, reward, terminated, truncated, _ = env.step(actions)
        done = terminated or truncated
        delta = team.observe(actions, reward, obs, terminated, done)

        next_x = normalize(np.stack([h.push(a, row) for h, a, row in zip(histories, actions, obs)]))
        ret = ret * 0.99 * (1.0 - terminated) + reward
        return_stats.update(ret)
        scaled = reward / np.sqrt(return_stats.var + 1e-8)
        value, next_value = critic(torch.as_tensor(np.stack([x.ravel(), next_x.ravel()])))
        expected = float(scaled + 0.99 * (1.0 - terminated) * next_value.detach() - value.detach())
        critic_optim.zero_grad()
        (-value).backward()
        critic_optim.step(expected, reset=done)
        for actor, optim, row, a in zip(actors, actor_optims, x, actions):
            mu, std = actor(torch.as_tensor(row))
            dist = torch.distributions.Normal(mu, std)
            loss = -dist.log_prob(torch.as_tensor(a)).sum() - 0.01 * np.sign(expected) * dist.entropy().sum()
            optim.zero_grad()
            loss.backward()
            optim.step(expected, reset=done)
        x = next_x
        np.testing.assert_allclose(delta, expected, rtol=1e-4, atol=1e-5)

        if done:
            resets += 1
            ret = 0.0
            obs, _ = env.reset()
            team.reset(obs)
            x = normalize(np.stack([h.reset(row) for h, row in zip(histories, obs)]))
    assert resets >= 1
    for k, v in critic.state_dict().items():
        torch.testing.assert_close(team.critic.state_dict()[k], v, rtol=1e-4, atol=1e-5)
    for i, actor in enumerate(actors):
        for k, v in team.actor.drone_state_dict(i).items():
            torch.testing.assert_close(v, actor.state_dict()[k], rtol=1e-4, atol=1e-5)


def test_centralized_advantages_drive_only_the_actors():
    """Per-drone advantages replace the TD error in the actors' updates; the critic still learns
    from the team TD error, so it ends up the same. A zero advantage leaves that drone's actor
    unchanged."""
    env = gym.make(TASKS["formation-waypoint"], num_drones=2)
    teams = []
    for advantages in (None, np.array([0.0, 0.5])):
        torch.manual_seed(0)
        team = CentralizedStreamAC(2, env.observation_space.shape[1], 3, 3, hidden_size=32)
        before = team.actor.drone_state_dict(0)
        rng = np.random.default_rng(0)
        obs, _ = env.reset(seed=0)
        team.reset(obs)
        for _ in range(20):
            actions = rng.normal(size=(2, 3)).astype(np.float32)
            obs, reward, terminated, truncated, _ = env.step(actions)
            team.observe(actions, reward, obs, terminated, terminated or truncated, advantages=advantages)
        teams.append((team, before))
    (plain, _), (advantaged, before) = teams
    for k, v in plain.critic.state_dict().items():
        torch.testing.assert_close(advantaged.critic.state_dict()[k], v)
    for k, v in advantaged.actor.drone_state_dict(0).items():
        torch.testing.assert_close(v, before[k])
    assert any(
        not torch.equal(v, plain.actor.drone_state_dict(1)[k]) for k, v in advantaged.actor.drone_state_dict(1).items()
    )


def test_centralized_checkpoint_team_policy_matches_learners(tmp_path):
    """Deployment needs only each drone's actor and its own observation row."""
    torch.manual_seed(0)
    env = gym.make(TASKS["formation-waypoint"], num_drones=2)
    team = CentralizedStreamAC(2, env.observation_space.shape[1], 3, 3, hidden_size=32)
    run(env, team, 30)
    save_checkpoint(tmp_path / "ckpt.pt", {**CONFIG, "algo": "cstream_ac"}, team, env)
    checkpoint = torch.load(tmp_path / "ckpt.pt", weights_only=False)
    assert checkpoint["critic"]["hidden.0.weight"].shape[1] == 2 * team.dim  # the joint observation
    policy = TeamPolicy.load(tmp_path / "ckpt.pt")

    obs, _ = env.reset(seed=1)
    policy.reset()
    features = np.stack([h.reset(row) for h, row in zip(team.histories, obs)])
    stats = team.obs_stats
    for _ in range(20):
        team.features = ((features - stats.mean) / np.sqrt(stats.var + 1e-8)).astype(np.float32)
        expected = team.act(deterministic=True).clip(-1.0, 1.0)
        action = policy(obs)
        np.testing.assert_allclose(action, expected, atol=1e-5)
        obs, *_ = env.step(action)
        features = np.stack([h.push(a, row) for h, a, row in zip(team.histories, action, obs)])
