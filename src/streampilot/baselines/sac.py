"""SAC (Haarnoja et al. 2018), a replay-buffer off-policy baseline for Stream AC. Follows CleanRL's
``sac_continuous_action``: twin Q-networks with Polyak-averaged targets, a tanh-squashed Gaussian
policy updated every ``policy_frequency`` steps, and automatic entropy tuning."""

import argparse

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# Running observation statistics would drift away from those of the transitions already in the
# replay buffer, so SAC sees the raw features (detections are already in [0, 1]) and raw rewards.
NORMALIZE = False

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


def mlp(in_dim: int, out_dim: int, hidden_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, out_dim),
    )


class Actor(nn.Module):
    """Gaussian over pre-tanh actions; ``forward`` returns its mean and standard deviation."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 256):
        super().__init__()
        self.net = mlp(obs_dim, 2 * action_dim, hidden_size)

    def forward(self, x):
        mu, log_std = self.net(x).chunk(2, dim=-1)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            torch.tanh(log_std) + 1
        )
        return mu, log_std.exp()

    def sample(self, x):
        """Reparameterized squashed action and its log-probability."""
        mu, std = self(x)
        u = mu + std * torch.randn_like(mu)
        action = torch.tanh(u)
        log_prob = torch.distributions.Normal(mu, std).log_prob(u) - torch.log(
            1 - action.pow(2) + 1e-6
        )
        return action, log_prob.sum(-1)

    def deterministic(self, x):
        return torch.tanh(self(x)[0])


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 256):
        super().__init__()
        self.net = mlp(obs_dim + action_dim, 1, hidden_size)

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1)).squeeze(-1)


class ReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, capacity: int):
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.capacity, self.size, self._next = capacity, 0, 0

    def add(self, obs, action, reward, next_obs, terminated) -> None:
        i = self._next
        (
            self.obs[i],
            self.actions[i],
            self.rewards[i],
            self.next_obs[i],
            self.terminated[i],
        ) = (obs, action, reward, next_obs, terminated)
        self._next = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(0, self.size, size=batch_size)
        return tuple(
            torch.as_tensor(x[idx])
            for x in (
                self.obs,
                self.actions,
                self.rewards,
                self.next_obs,
                self.terminated,
            )
        )


class SAC:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_size: int = 256,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        learning_starts: int = 5000,
        policy_lr: float = 3e-4,
        q_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_frequency: int = 2,
        seed: int = 0,
    ):
        self.action_dim = action_dim
        self.batch_size, self.learning_starts = batch_size, learning_starts
        self.gamma, self.tau, self.policy_frequency = gamma, tau, policy_frequency
        self.rng = np.random.default_rng(seed)
        self.actor = Actor(obs_dim, action_dim, hidden_size)
        self.qs = nn.ModuleList(
            [QNetwork(obs_dim, action_dim, hidden_size) for _ in range(2)]
        )
        self.q_targets = nn.ModuleList(
            [QNetwork(obs_dim, action_dim, hidden_size) for _ in range(2)]
        )
        self.q_targets.load_state_dict(self.qs.state_dict())
        self.q_targets.requires_grad_(False)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=policy_lr)
        self.q_optim = torch.optim.Adam(self.qs.parameters(), lr=q_lr)
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=q_lr)
        self.buffer = ReplayBuffer(obs_dim, action_dim, buffer_size)
        self.steps = 0

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        if self.steps < self.learning_starts:
            return self.rng.uniform(-1.0, 1.0, self.action_dim).astype(np.float32)
        return self.actor.sample(torch.as_tensor(obs))[0].numpy()

    def observe(
        self, obs, action, reward: float, next_obs, terminated: bool, done: bool
    ) -> dict:
        # Only termination cuts the bootstrap; truncated episodes keep their final transition as is.
        self.buffer.add(obs, action, reward, next_obs, terminated)
        self.steps += 1
        if self.steps < self.learning_starts:
            return {}
        return self._update()

    def _update(self) -> dict:
        obs, actions, rewards, next_obs, terminated = self.buffer.sample(
            self.batch_size, self.rng
        )
        alpha = self.log_alpha.exp().item()
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_obs)
            next_q = (
                torch.min(*(q(next_obs, next_actions) for q in self.q_targets))
                - alpha * next_log_probs
            )
            target = rewards + self.gamma * (1.0 - terminated) * next_q
        q_loss = sum(F.mse_loss(q(obs, actions), target) for q in self.qs)
        self.q_optim.zero_grad()
        q_loss.backward()
        self.q_optim.step()
        metrics = {"q_loss": q_loss.item()}

        if self.steps % self.policy_frequency == 0:
            # CleanRL's TD3-style delayed update: compensate with policy_frequency actor steps.
            for _ in range(self.policy_frequency):
                new_actions, log_probs = self.actor.sample(obs)
                q = torch.min(*(q(obs, new_actions) for q in self.qs))
                actor_loss = (alpha * log_probs - q).mean()
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                alpha_loss = -(
                    self.log_alpha.exp() * (log_probs.detach() + self.target_entropy)
                ).mean()
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()
                alpha = self.log_alpha.exp().item()
            metrics.update(
                actor_loss=actor_loss.item(),
                alpha=alpha,
                entropy=-log_probs.mean().item(),
            )

        with torch.no_grad():
            target_params, params = (
                list(self.q_targets.parameters()),
                list(self.qs.parameters()),
            )
            torch._foreach_lerp_(target_params, params, self.tau)
        return metrics

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "qs": self.qs.state_dict(),
            "log_alpha": self.log_alpha.detach(),
        }


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-starts", type=int, default=5000)
    parser.add_argument("--policy-lr", type=float, default=3e-4)
    parser.add_argument("--q-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy-frequency", type=int, default=2)


def make_agent(args: argparse.Namespace, obs_dim: int, action_dim: int) -> SAC:
    return SAC(
        obs_dim,
        action_dim,
        hidden_size=args.hidden_size,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        policy_lr=args.policy_lr,
        q_lr=args.q_lr,
        gamma=args.gamma,
        tau=args.tau,
        policy_frequency=args.policy_frequency,
        seed=args.seed,
    )
