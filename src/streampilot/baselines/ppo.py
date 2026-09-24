"""PPO (Schulman et al. 2017), a batch on-policy baseline for Stream AC. Follows CleanRL's
``ppo_continuous_action``: rollouts of ``rollout_steps`` transitions, several epochs of minibatch
updates, GAE, clipped value loss and a linearly annealed learning rate. Unlike CleanRL, time-limit
truncation bootstraps from the value of the final observation instead of cutting the return."""

import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

# Same observation and reward normalization as Stream AC (CleanRL's PPO uses the equivalent wrappers).
NORMALIZE = True


def layer_init(layer: nn.Linear, std: float = np.sqrt(2)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.zeros_(layer.bias)
    return layer


def mlp(in_dim: int, out_dim: int, hidden_size: int, out_std: float) -> nn.Sequential:
    return nn.Sequential(
        layer_init(nn.Linear(in_dim, hidden_size)),
        nn.Tanh(),
        layer_init(nn.Linear(hidden_size, hidden_size)),
        nn.Tanh(),
        layer_init(nn.Linear(hidden_size, out_dim), std=out_std),
    )


class Actor(nn.Module):
    """Gaussian policy with a state-independent log standard deviation."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 64):
        super().__init__()
        self.mean = mlp(obs_dim, action_dim, hidden_size, out_std=0.01)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        mu = self.mean(x)
        return mu, self.log_std.exp().expand_as(mu)

    def deterministic(self, x):
        return self(x)[0].clamp(-1.0, 1.0)


class PPO:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        total_steps: int,
        hidden_size: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        rollout_steps: int = 2048,
        epochs: int = 10,
        minibatch_size: int = 64,
        clip_coef: float = 0.2,
        entropy_coeff: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
    ):
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.rollout_steps, self.epochs, self.minibatch_size = rollout_steps, epochs, minibatch_size
        self.clip_coef, self.entropy_coeff, self.vf_coef, self.max_grad_norm = clip_coef, entropy_coeff, vf_coef, max_grad_norm
        self.actor = Actor(obs_dim, action_dim, hidden_size)
        self.critic = mlp(obs_dim, 1, hidden_size, out_std=1.0)
        self.optim = torch.optim.Adam([*self.actor.parameters(), *self.critic.parameters()], lr=lr, eps=1e-5)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optim, 1.0, 0.0, total_iters=max(total_steps // rollout_steps, 1)
        )
        self._rollout: list[tuple] = []

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        mu, std = self.actor(torch.as_tensor(obs))
        return Normal(mu, std).sample().numpy()

    def observe(self, obs, action, reward: float, next_obs, terminated: bool, done: bool) -> dict:
        self._rollout.append((obs, action, reward, next_obs, terminated, done))
        if len(self._rollout) < self.rollout_steps:
            return {}
        metrics = self._update()
        self._rollout.clear()
        return metrics

    def _value(self, obs):
        return self.critic(obs).squeeze(-1)

    def _update(self) -> dict:
        obs, actions, rewards, next_obs, terminated, done = (
            torch.as_tensor(np.array(x), dtype=torch.float32) for x in zip(*self._rollout)
        )
        # The policy is unchanged since the rollout started, so the behaviour log-probs and values
        # can be computed now, in one batch.
        with torch.no_grad():
            mu, std = self.actor(obs)
            old_log_probs = Normal(mu, std).log_prob(actions).sum(-1)
            values, next_values = self._value(obs), self._value(next_obs)
            deltas = rewards + self.gamma * (1.0 - terminated) * next_values - values
            advantages = torch.zeros_like(rewards)
            gae = 0.0
            for t in reversed(range(len(rewards))):
                gae = deltas[t] + self.gamma * self.gae_lambda * (1.0 - done[t]) * gae
                advantages[t] = gae
            returns = advantages + values

        stats = []
        for _ in range(self.epochs):
            for idx in torch.randperm(len(rewards)).split(self.minibatch_size):
                mu, std = self.actor(obs[idx])
                dist = Normal(mu, std)
                log_ratio = dist.log_prob(actions[idx]).sum(-1) - old_log_probs[idx]
                ratio = log_ratio.exp()
                adv = advantages[idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                policy_loss = torch.max(
                    -adv * ratio, -adv * ratio.clamp(1 - self.clip_coef, 1 + self.clip_coef)
                ).mean()

                value = self._value(obs[idx])
                clipped = values[idx] + (value - values[idx]).clamp(-self.clip_coef, self.clip_coef)
                value_loss = 0.5 * torch.max((value - returns[idx]) ** 2, (clipped - returns[idx]) ** 2).mean()
                entropy = dist.entropy().sum(-1).mean()

                loss = policy_loss - self.entropy_coeff * entropy + self.vf_coef * value_loss
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_([*self.actor.parameters(), *self.critic.parameters()], self.max_grad_norm)
                self.optim.step()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clip_frac = ((ratio - 1).abs() > self.clip_coef).float().mean()
                stats.append([policy_loss.item(), value_loss.item(), entropy.item(), approx_kl.item(), clip_frac.item()])
        self.scheduler.step()
        names = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac")
        return dict(zip(names, np.mean(stats, axis=0)))

    def state_dict(self) -> dict:
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--entropy-coeff", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)


def make_agent(args: argparse.Namespace, obs_dim: int, action_dim: int) -> PPO:
    return PPO(
        obs_dim,
        action_dim,
        total_steps=args.steps,
        hidden_size=args.hidden_size,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        rollout_steps=args.rollout_steps,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        clip_coef=args.clip_coef,
        entropy_coeff=args.entropy_coeff,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
    )
