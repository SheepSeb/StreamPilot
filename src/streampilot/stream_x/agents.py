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


# softplus(0): what the std head gives while its output is zero
DEFAULT_INIT_STD = math.log(2.0)


def softplus_inverse(y: float) -> float:
    return math.log(math.expm1(y))


def std_bias(init_std: float, max_std: float | None) -> float:
    """Bias of the std head that gives ``init_std`` while the head's weights are near zero."""
    if max_std is None:
        return softplus_inverse(init_std)
    assert init_std < max_std, "--init-std must be below --max-std"
    return math.log(init_std / (max_std - init_std))


def to_std(pre_std: torch.Tensor, max_std: float | None) -> torch.Tensor:
    """Standard deviation from the std head's output: softplus, or (``max_std``) a sigmoid that stays below it."""
    return F.softplus(pre_std) if max_std is None else max_std * torch.sigmoid(pre_std)


class Actor(MLP):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_size: int = 128,
        init_std: float = DEFAULT_INIT_STD,
        bound_mean: bool = False,
        max_std: float | None = None,
    ):
        super().__init__(obs_dim, (action_dim, action_dim), hidden_size)
        self.bound_mean, self.max_std = bound_mean, max_std
        nn.init.constant_(self.heads[1].bias, std_bias(init_std, max_std))  # the std head; weights start near zero

    def forward(self, x):
        mu, pre_std = super().forward(x)
        return (torch.tanh(mu) if self.bound_mean else mu), to_std(pre_std, self.max_std)

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
        init_std: float = DEFAULT_INIT_STD,
        bound_mean: bool = False,
        max_std: float | None = None,
    ):
        self.gamma, self.entropy_coeff = gamma, entropy_coeff
        self.actor = Actor(obs_dim, action_dim, hidden_size, init_std, bound_mean, max_std)
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
        delta = self.update(obs, action, reward, next_obs, terminated, done)
        return {
            "abs_td_error": abs(delta),
            "actor_step_scale": self.actor_optim.last_scale,
            "critic_step_scale": self.critic_optim.last_scale,
        }

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
    parser.add_argument("--init-std", type=float, default=DEFAULT_INIT_STD, help="action standard deviation at the start")
    parser.add_argument(
        "--max-std",
        type=float,
        default=None,
        help="cap the action standard deviation (a sigmoid instead of the softplus), so exploration cannot grow without bound",
    )
    parser.add_argument(
        "--bound-mean",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="pass the policy mean through tanh, so it stays inside the action range (the environment clips actions, "
        "so a mean beyond it only makes the policy saturate and unable to recover)",
    )


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
        init_std=args.init_std,
        bound_mean=args.bound_mean,
        max_std=args.max_std,
    )
