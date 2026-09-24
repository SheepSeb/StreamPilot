"""Stream AC(lambda) from Elsayed, Vasan & Mahmood (2024), "Streaming Deep Reinforcement Learning
Finally Works": actor-critic that learns from each transition once, as it arrives, with no replay
buffer and no batches. Stability comes from ObGD, sparse initialization, LayerNorm and online
observation/reward normalization (see ``wrappers``)."""

import argparse
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from streampilot.stream_x.optim import ObGD

# Online observation normalization and reward scaling are part of the algorithm.
NORMALIZE = True


@torch.no_grad()
def sparse_init_(weight: torch.Tensor, sparsity: float = 0.9) -> None:
    """LeCun-uniform init, then zero a fraction ``sparsity`` of each unit's incoming weights."""
    fan_out, fan_in = weight.shape
    weight.uniform_(-math.sqrt(1.0 / fan_in), math.sqrt(1.0 / fan_in))
    num_zeros = min(math.ceil(sparsity * fan_in), fan_in - 1)  # keep at least one input per unit
    for row in weight:
        row[torch.randperm(fan_in)[:num_zeros]] = 0.0


class MLP(nn.Module):
    """Two hidden layers, each Linear -> LayerNorm (no affine parameters) -> LeakyReLU."""

    def __init__(self, in_dim: int, out_dims: tuple[int, ...], hidden_size: int = 128):
        super().__init__()
        self.hidden = nn.ModuleList([nn.Linear(in_dim, hidden_size), nn.Linear(hidden_size, hidden_size)])
        self.heads = nn.ModuleList([nn.Linear(hidden_size, d) for d in out_dims])
        for layer in [*self.hidden, *self.heads]:
            sparse_init_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer in self.hidden:
            x = F.leaky_relu(F.layer_norm(layer(x), (layer.out_features,)))
        return tuple(head(x) for head in self.heads)


class Actor(MLP):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 128):
        super().__init__(obs_dim, (action_dim, action_dim), hidden_size)

    def forward(self, x):
        mu, pre_std = super().forward(x)
        return mu, F.softplus(pre_std)

    def deterministic(self, x):
        return self(x)[0].clamp(-1.0, 1.0)


class Critic(MLP):
    def __init__(self, obs_dim: int, hidden_size: int = 128):
        super().__init__(obs_dim, (1,), hidden_size)

    def forward(self, x):
        return super().forward(x)[0].squeeze(-1)


class StreamAC:
    """Stream AC(lambda) for continuous actions. Defaults are the paper's MuJoCo hyperparameters."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_size: int = 128,
        lr: float = 1.0,
        gamma: float = 0.99,
        lamda: float = 0.8,
        kappa_policy: float = 3.0,
        kappa_value: float = 2.0,
        entropy_coeff: float = 0.01,
    ):
        self.gamma, self.entropy_coeff = gamma, entropy_coeff
        self.actor = Actor(obs_dim, action_dim, hidden_size)
        self.critic = Critic(obs_dim, hidden_size)
        self.actor_optim = ObGD(self.actor.parameters(), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_policy)
        self.critic_optim = ObGD(self.critic.parameters(), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_value)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        mu, std = self.actor(torch.as_tensor(obs))
        return (mu if deterministic else Normal(mu, std).sample()).numpy()

    def update(self, obs, action, reward: float, next_obs, terminated: bool, done: bool) -> float:
        """One TD(lambda) actor-critic update. ``terminated`` cuts the bootstrap; ``done`` (terminated
        or truncated) clears the eligibility traces. Returns the TD error."""
        obs, action = torch.as_tensor(obs), torch.as_tensor(action)
        value, next_value = self.critic(torch.stack([obs, torch.as_tensor(next_obs)]))
        delta = float(reward + self.gamma * (1.0 - terminated) * next_value.detach() - value.detach())

        mu, std = self.actor(obs)
        dist = Normal(mu, std)
        # The entropy bonus follows the sign of delta, since ObGD scales the whole trace by delta.
        policy_loss = -dist.log_prob(action).sum() - self.entropy_coeff * np.sign(delta) * dist.entropy().sum()

        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        (policy_loss - value).backward()  # disjoint parameters: one pass fills both sets of grads
        self.actor_optim.step(delta, reset=done)
        self.critic_optim.step(delta, reset=done)
        return delta

    def observe(self, obs, action, reward: float, next_obs, terminated: bool, done: bool) -> dict:
        return {"abs_td_error": abs(self.update(obs, action, reward, next_obs, terminated, done))}

    def state_dict(self) -> dict:
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lamda", type=float, default=0.8)
    parser.add_argument("--kappa-policy", type=float, default=3.0)
    parser.add_argument("--kappa-value", type=float, default=2.0)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)


def make_agent(args: argparse.Namespace, obs_dim: int, action_dim: int) -> StreamAC:
    return StreamAC(
        obs_dim,
        action_dim,
        hidden_size=args.hidden_size,
        lr=args.lr,
        gamma=args.gamma,
        lamda=args.lamda,
        kappa_policy=args.kappa_policy,
        kappa_value=args.kappa_value,
        entropy_coeff=args.entropy_coeff,
    )
