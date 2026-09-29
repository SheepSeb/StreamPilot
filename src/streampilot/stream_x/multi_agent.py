"""Streaming multi-agent learners for the formation tasks, both built from Stream AC(lambda).

``IndependentStreamAC`` is the fully decentralised baseline. Every drone is its own streaming
learner, as in the single-drone tasks:

- its own actor and critic, with no parameter sharing and no centralised critic; the critic sees
  the same features as the actor, so training is decentralised too, not only execution;
- its own features: the last ``num_frames`` rows of its own observation plus its previous action
  (``ObservationHistory``), standardized with its own running statistics;
- its own reward scaling, of the team reward it receives.

Each learner treats its teammates as part of the environment. It learns from every transition
once, as it arrives, so the same loop could run on every drone of a real team, each with only its
own observation row and the team reward.

``CentralizedStreamAC`` keeps the per-drone actors and features but trains them against one shared
critic of the joint observation (centralised training, decentralised execution).

For speed, the drones' networks are stacked into batched weights (``(num_drones, out, in)``) and
updated together: one forward, backward and optimizer step per team step instead of one per
drone. A single sample per drone is all overhead and no arithmetic, so this is ~2x faster with
three drones. The math is that of ``num_drones`` separate ``StreamAC`` learners (``BatchedObGD``
bounds each drone's step with its own trace and TD error), and each drone's slice is saved as a
plain ``stream_ac.Actor``.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from streampilot.stream_x.agents import Critic, sparse_init_
from streampilot.stream_x.optim import BatchedObGD, ObGD
from streampilot.stream_x.wrappers import ObservationHistory, RunningMeanStd

EPSILON = 1e-8


class BatchedLinear(nn.Module):
    """``n`` independent ``nn.Linear`` layers: ``(n, batch, in) -> (n, batch, out)``."""

    def __init__(self, n: int, in_features: int, out_features: int):
        super().__init__()
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(n, out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(n, out_features))
        for w in self.weight:
            sparse_init_(w)

    def forward(self, x):
        return torch.baddbmm(self.bias.unsqueeze(1), x, self.weight.transpose(1, 2))


class BatchedMLP(nn.Module):
    """``n`` copies of ``stream_ac.MLP`` with independent weights, under the same parameter names
    (so ``drone_state_dict(i)`` loads into an ``MLP``)."""

    def __init__(self, n: int, in_dim: int, out_dims: tuple[int, ...], hidden_size: int = 128):
        super().__init__()
        self.hidden = nn.ModuleList([BatchedLinear(n, in_dim, hidden_size), BatchedLinear(n, hidden_size, hidden_size)])
        self.heads = nn.ModuleList([BatchedLinear(n, hidden_size, d) for d in out_dims])

    def forward(self, x):
        for layer in self.hidden:
            x = F.leaky_relu(F.layer_norm(layer(x), (layer.out_features,)))
        return tuple(head(x) for head in self.heads)

    def drone_state_dict(self, i: int) -> dict:
        return {k: v[i].clone() for k, v in self.state_dict().items()}


class _StreamTeam:
    """What both learners share: per drone, an observation history, observation normalization and
    an actor (batched, with ``BatchedObGD``: one trace and step-size bound per drone); and the
    scaling of the team reward. Subclasses add the critic and ``observe``."""

    def __init__(
        self,
        num_drones: int,
        obs_dim: int,
        action_dim: int,
        num_frames: int,
        hidden_size: int,
        lr: float,
        gamma: float,
        lamda: float,
        kappa_policy: float,
        entropy_coeff: float,
        num_critics: int,
    ):
        self.num_drones, self.gamma, self.entropy_coeff = num_drones, gamma, entropy_coeff
        self.histories = [ObservationHistory(obs_dim, action_dim, num_frames) for _ in range(num_drones)]
        self.dim = self.histories[0].dim
        self.obs_stats = RunningMeanStd((num_drones, self.dim))  # elementwise, so one set per drone
        self.return_stats = RunningMeanStd((num_critics,))  # one reward scale per critic
        self._return = np.zeros(num_critics)
        self.actor = BatchedMLP(num_drones, self.dim, (action_dim, action_dim), hidden_size)
        self.actor_optim = BatchedObGD(self.actor.parameters(), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_policy)
        self.features: np.ndarray | None = None  # (num_drones, dim), normalized

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        self.obs_stats.update(features)
        return ((features - self.obs_stats.mean) / np.sqrt(self.obs_stats.var + EPSILON)).astype(np.float32)

    def _policy(self, x: torch.Tensor) -> Normal:
        mu, pre_std = self.actor(x.unsqueeze(1))
        return Normal(mu.squeeze(1), F.softplus(pre_std.squeeze(1)))

    def _push(self, actions, next_obs) -> np.ndarray:
        return self._normalize(np.stack([h.push(a, row) for h, a, row in zip(self.histories, actions, next_obs)]))

    def _scale_reward(self, reward: float, terminated: bool, done: bool) -> torch.Tensor:
        self._return = self._return * self.gamma * (1.0 - terminated) + reward
        self.return_stats.update(self._return)
        if done:
            self._return[:] = 0.0
        return torch.as_tensor(reward / np.sqrt(self.return_stats.var + EPSILON), dtype=torch.float32)

    def _policy_loss(self, x: torch.Tensor, actions, signal: torch.Tensor) -> torch.Tensor:
        """Per drone, as in ``StreamAC.update``, where ``signal`` (``(num_drones,)``) is what ObGD
        will scale that drone's trace by. Summing over drones keeps the gradients apart, since each
        drone's loss depends only on its own slice of the weights."""
        dist = self._policy(x)
        log_prob = dist.log_prob(torch.as_tensor(actions, dtype=torch.float32)).sum(-1)
        # The entropy bonus follows the sign of the signal, since ObGD scales the whole trace by it.
        return (-log_prob - self.entropy_coeff * torch.sign(signal) * dist.entropy().sum(-1)).sum()

    def reset(self, obs) -> None:
        """Start an episode from its first observation, ``(num_drones, obs_dim)``."""
        self.features = self._normalize(np.stack([h.reset(row) for h, row in zip(self.histories, obs)]))

    @torch.no_grad()
    def act(self, deterministic: bool = False) -> np.ndarray:
        dist = self._policy(torch.as_tensor(self.features))
        return (dist.mean if deterministic else dist.sample()).numpy()

    def _drone_state_dict(self, i: int) -> dict:
        s = self.obs_stats
        return {
            "actor": self.actor.drone_state_dict(i),
            "obs_stats": {"mean": s.mean[i], "var": s.var[i], "m2": s._m2[i], "count": s.count},
        }


class IndependentStreamAC(_StreamTeam):
    """One Stream AC(lambda) learner per drone (defaults: the paper's MuJoCo hyperparameters), each
    with its own observation history, observation normalization and reward scaling, as
    ``train.make_env`` gives the single-drone learner."""

    def __init__(
        self,
        num_drones: int,
        obs_dim: int,
        action_dim: int,
        num_frames: int,
        hidden_size: int = 128,
        lr: float = 1.0,
        gamma: float = 0.99,
        lamda: float = 0.8,
        kappa_policy: float = 3.0,
        kappa_value: float = 2.0,
        entropy_coeff: float = 0.01,
    ):
        super().__init__(
            num_drones, obs_dim, action_dim, num_frames, hidden_size, lr, gamma, lamda, kappa_policy, entropy_coeff,
            num_critics=num_drones,
        )  # fmt: skip
        self.critic = BatchedMLP(num_drones, self.dim, (1,), hidden_size)
        self.critic_optim = BatchedObGD(self.critic.parameters(), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_value)

    def observe(self, actions, reward: float, next_obs, terminated: bool, done: bool) -> np.ndarray:
        """Every drone learns from its own row of the transition and the team reward. After
        ``done``, call ``reset`` with the next episode's first observation. Returns each drone's
        TD error."""
        next_features = self._push(actions, next_obs)
        scaled = self._scale_reward(reward, terminated, done)

        x = torch.as_tensor(self.features)
        both = torch.stack([x, torch.as_tensor(next_features)], dim=1)  # (num_drones, 2, dim)
        value, next_value = self.critic(both)[0].squeeze(-1).unbind(1)
        delta = (scaled + self.gamma * (1.0 - terminated) * next_value - value).detach()

        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        (self._policy_loss(x, actions, delta) - value.sum()).backward()
        self.actor_optim.step(delta, reset=done)
        self.critic_optim.step(delta, reset=done)
        self.features = next_features
        return delta.numpy()

    def state_dict(self) -> dict:
        """Per drone, what a single-drone Stream AC checkpoint holds: ``stream_ac.Actor`` and
        ``Critic`` weights and the observation statistics."""
        return {
            "drones": [
                {**self._drone_state_dict(i), "critic": self.critic.drone_state_dict(i)} for i in range(self.num_drones)
            ]
        }


class CentralizedStreamAC(_StreamTeam):
    """Stream AC(lambda) with centralised training and decentralised execution (CTDE):

    - one shared critic ``V(s)`` of the joint observation (every drone's normalized features,
      concatenated), with its own eligibility trace and ObGD step, updated with the team TD error
      ``delta = r + gamma V(s') - V(s)`` of the team reward (scaled by one running return scale);
    - one actor per drone, of that drone's features only, each with its own eligibility trace and
      ObGD step-size bound (``BatchedObGD``), updated with the shared ``delta``, or with a
      per-drone advantage estimate when ``observe`` is given one.

    The critic is only needed for training. Deployed, each drone runs its own actor on its own
    observation row, exactly as with ``IndependentStreamAC``, and checkpoints load the same way.
    """

    def __init__(
        self,
        num_drones: int,
        obs_dim: int,
        action_dim: int,
        num_frames: int,
        hidden_size: int = 128,
        lr: float = 1.0,
        gamma: float = 0.99,
        lamda: float = 0.8,
        kappa_policy: float = 3.0,
        kappa_value: float = 2.0,
        entropy_coeff: float = 0.01,
    ):
        super().__init__(
            num_drones, obs_dim, action_dim, num_frames, hidden_size, lr, gamma, lamda, kappa_policy, entropy_coeff,
            num_critics=1,
        )  # fmt: skip
        self.critic = Critic(num_drones * self.dim, hidden_size)
        self.critic_optim = ObGD(self.critic.parameters(), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_value)

    def observe(self, actions, reward: float, next_obs, terminated: bool, done: bool, advantages=None) -> float:
        """The critic learns from the joint transition, each actor from its own row of it.
        ``advantages`` (``(num_drones,)``, optional) replaces the team TD error as the actors'
        update signal, one value per drone. After ``done``, call ``reset`` with the next episode's
        first observation. Returns the team TD error."""
        next_features = self._push(actions, next_obs)
        scaled = self._scale_reward(reward, terminated, done)[0]

        x = torch.as_tensor(self.features)
        joint = torch.stack([x.flatten(), torch.as_tensor(next_features).flatten()])  # (2, num_drones * dim)
        value, next_value = self.critic(joint)
        delta = (scaled + self.gamma * (1.0 - terminated) * next_value - value).detach()
        if advantages is None:
            signal = delta.expand(self.num_drones)
        else:
            signal = torch.as_tensor(advantages, dtype=torch.float32).reshape(self.num_drones)

        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        (self._policy_loss(x, actions, signal) - value).backward()  # disjoint parameters, as in StreamAC
        self.actor_optim.step(signal, reset=done)
        self.critic_optim.step(float(delta), reset=done)
        self.features = next_features
        return float(delta)

    def state_dict(self) -> dict:
        """Per drone, the actor and observation statistics (what ``TeamPolicy`` runs); the shared
        critic once, as a ``stream_ac.Critic`` of ``num_drones * dim`` inputs."""
        return {
            "drones": [self._drone_state_dict(i) for i in range(self.num_drones)],
            "critic": self.critic.state_dict(),
        }
