"""MAPPO (Yu et al. 2022, *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*)
for the formation tasks, trained on many environments at once.

- **Actor:** one Gaussian policy shared by every drone (``ppo.Actor``), run on each drone's own
  features: its last ``num_frames`` observation rows and its previous action, as in the
  single-drone tasks. The row's slot one-hot tells the drones apart. It is decentralised, so
  each drone runs it on its own row at deployment (``streampilot.policy.TeamPolicy``).
- **Critic:** centralised, used only in training. It sees every drone's features and, with
  ``critic_state``, the privileged state of every drone, and predicts one value for the team
  reward.
- **Update:** as ``ppo.PPO`` (CleanRL's ``ppo_continuous_action``): GAE, clipped policy and value
  losses, several epochs of minibatches, a linearly annealed learning rate, and bootstrapping
  through time-limit truncation. The team advantage of a step is shared by its drones, and each
  drone has its own probability ratio.

Rollouts are sampled with a CPU copy of the actor, since a small batch per step is faster there
than a round trip to the GPU. The rollout is copied to ``device`` once per update, where the
values, advantages and gradient steps are computed in large batches.
"""

import argparse
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from streampilot.baselines.ppo import Actor, mlp

__all__ = ["MAPPO", "Actor", "TeamHistory", "add_args", "make_agent"]


class TeamHistory:
    """``ObservationHistory`` for a batch of teams: per environment and drone, the features
    ``[obs_{t-k+1}, ..., obs_t, a_{t-1}]`` of that drone's observation rows and previous action,
    exactly as ``streampilot.policy.Policy`` builds them from one drone's row at deployment."""

    def __init__(self, num_envs: int, num_drones: int, obs_dim: int, action_dim: int, num_frames: int):
        self.num_frames = num_frames
        self._frames = np.zeros((num_envs, num_drones, num_frames, obs_dim), dtype=np.float32)
        self._last_action = np.zeros((num_envs, num_drones, action_dim), dtype=np.float32)

    @property
    def dim(self) -> int:
        return self._frames.shape[-1] * self.num_frames + self._last_action.shape[-1]

    def reset(self, obs, mask=None) -> np.ndarray:
        """Start new episodes (in the environments where ``mask`` is true, or all of them)."""
        mask = slice(None) if mask is None else mask
        self._frames[mask] = np.asarray(obs)[mask][:, :, None]
        self._last_action[mask] = 0.0
        return self.features()

    def push(self, action, obs) -> np.ndarray:
        self._frames[:, :, :-1] = self._frames[:, :, 1:]
        self._frames[:, :, -1] = obs
        self._last_action[:] = np.clip(action, -1.0, 1.0)
        return self.features()

    def features(self) -> np.ndarray:
        e, n = self._last_action.shape[:2]
        return np.concatenate([self._frames.reshape(e, n, -1), self._last_action], axis=-1)


class MAPPO:
    def __init__(
        self,
        actor_dim: int,
        critic_dim: int,
        action_dim: int,
        num_drones: int,
        num_envs: int,
        total_steps: int,
        hidden_size: int = 64,
        critic_hidden_size: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        rollout_steps: int = 128,
        epochs: int = 5,
        num_minibatches: int = 4,
        clip_coef: float = 0.2,
        entropy_coeff: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: float | None = None,
        device: str | torch.device = "cpu",
    ):
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.rollout_steps, self.epochs, self.num_minibatches = rollout_steps, epochs, num_minibatches
        self.clip_coef, self.entropy_coeff, self.vf_coef = clip_coef, entropy_coeff, vf_coef
        self.max_grad_norm, self.target_kl = max_grad_norm, target_kl
        self.device = torch.device(device)

        self.actor = Actor(actor_dim, action_dim, hidden_size).to(self.device)
        self.critic = mlp(critic_dim, 1, critic_hidden_size, out_std=1.0).to(self.device)
        self._params = [*self.actor.parameters(), *self.critic.parameters()]
        # Fused Adam: one kernel for all the parameters instead of several per tensor.
        self.optim = torch.optim.Adam(self._params, lr=lr, eps=1e-5, fused=self.device.type == "cuda")
        updates = max(total_steps // (rollout_steps * num_envs), 1)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optim, 1.0, 0.0, total_iters=updates)
        self.rollout_actor = self.actor if self.device.type == "cpu" else copy.deepcopy(self.actor).cpu()

        # Rollout storage on the host (pinned, so the copy to the GPU is one fast transfer).
        pin = self.device.type == "cuda"
        T, E, n = rollout_steps, num_envs, num_drones

        def buffer(*shape, dtype=torch.float32):
            return torch.zeros(shape, dtype=dtype, pin_memory=pin)

        self._buf = {
            "actor_x": buffer(T, E, n, actor_dim),
            "critic_x": buffer(T, E, critic_dim),
            "actions": buffer(T, E, n, action_dim),
            "rewards": buffer(T, E),
            "terminated": buffer(T, E),
            "done": buffer(T, E),
            # Critic input of the terminal observation, where an episode ended (for bootstrapping).
            "final_critic_x": buffer(T, E, critic_dim),
            "next_critic_x": buffer(E, critic_dim),
        }
        self._np = {k: v.numpy() for k, v in self._buf.items()}
        self._t = 0

    @torch.inference_mode()
    def act(self, actor_x: np.ndarray) -> np.ndarray:
        mu, std = self.rollout_actor(torch.from_numpy(actor_x))
        return torch.normal(mu, std).numpy()

    @torch.inference_mode()
    def act_deterministic(self, actor_x: np.ndarray) -> np.ndarray:
        return self.rollout_actor.deterministic(torch.from_numpy(actor_x)).numpy()

    def store(self, actor_x, critic_x, actions, rewards, terminated, done, final_critic_x=None) -> None:
        """Record one step of every environment. ``final_critic_x``: the critic input of the terminal
        observation of each environment where ``done`` (in environment order)."""
        b, t = self._np, self._t
        b["actor_x"][t], b["critic_x"][t], b["actions"][t] = actor_x, critic_x, actions
        b["rewards"][t], b["terminated"][t], b["done"][t] = rewards, terminated, done
        if final_critic_x is not None:
            b["final_critic_x"][t, done] = final_critic_x
        self._t += 1

    @property
    def rollout_full(self) -> bool:
        return self._t == self.rollout_steps

    def update(self, next_critic_x: np.ndarray) -> dict:
        """One PPO update on the stored rollout; ``next_critic_x`` is the critic input after its last step."""
        assert self.rollout_full
        self._np["next_critic_x"][:] = next_critic_x
        b = {k: v.to(self.device, non_blocking=True) for k, v in self._buf.items()}
        T, E = b["rewards"].shape

        # The policy is unchanged since the rollout started, so the behaviour log-probs and the
        # values can be computed now, in one batch.
        with torch.no_grad():
            mu, std = self.actor(b["actor_x"])
            old_log_probs = Normal(mu, std, validate_args=False).log_prob(b["actions"]).sum(-1)  # (T, E, n)
            values = self.critic(b["critic_x"]).squeeze(-1)  # (T, E)
            next_values = torch.cat([values[1:], self.critic(b["next_critic_x"]).T])
            ended = b["done"].bool()
            next_values[ended] = self.critic(b["final_critic_x"][ended]).squeeze(-1)
            not_terminal = 1.0 - b["terminated"]
            deltas = b["rewards"] + self.gamma * not_terminal * next_values - values
            carry = self.gamma * self.gae_lambda * (1.0 - b["done"])
            advantages = torch.empty_like(deltas)
            gae = torch.zeros(E, device=self.device)
            for t in reversed(range(T)):
                gae = deltas[t] + carry[t] * gae
                advantages[t] = gae
            returns = advantages + values

        actor_x, actions = b["actor_x"].flatten(0, 1), b["actions"].flatten(0, 1)
        critic_x, old_log_probs = b["critic_x"].flatten(0, 1), old_log_probs.flatten(0, 1)
        advantages, returns, values = advantages.flatten(), returns.flatten(), values.flatten()

        stats = []
        for _ in range(self.epochs):
            for idx in torch.randperm(T * E, device=self.device).chunk(self.num_minibatches):
                mu, std = self.actor(actor_x[idx])
                dist = Normal(mu, std, validate_args=False)
                log_ratio = dist.log_prob(actions[idx]).sum(-1) - old_log_probs[idx]  # (B, n)
                ratio = log_ratio.exp()
                adv = advantages[idx]
                adv = ((adv - adv.mean()) / (adv.std() + 1e-8)).unsqueeze(-1)  # shared by the drones
                policy_loss = torch.max(
                    -adv * ratio, -adv * ratio.clamp(1 - self.clip_coef, 1 + self.clip_coef)
                ).mean()

                value = self.critic(critic_x[idx]).squeeze(-1)
                clipped = values[idx] + (value - values[idx]).clamp(-self.clip_coef, self.clip_coef)
                value_loss = 0.5 * torch.max((value - returns[idx]) ** 2, (clipped - returns[idx]) ** 2).mean()
                entropy = dist.entropy().sum(-1).mean()

                loss = policy_loss - self.entropy_coeff * entropy + self.vf_coef * value_loss
                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self._params, self.max_grad_norm)
                self.optim.step()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clip_frac = ((ratio - 1).abs() > self.clip_coef).float().mean()
                # Kept on the device: one synchronisation per update, not per minibatch.
                stats.append(torch.stack([policy_loss, value_loss, entropy, approx_kl, clip_frac]).detach())
            if self.target_kl is not None and approx_kl.item() > self.target_kl:
                break
        self.scheduler.step()
        if self.rollout_actor is not self.actor:
            self.rollout_actor.load_state_dict(self.actor.state_dict())
        self._t = 0

        with torch.no_grad():
            explained_var = 1.0 - (returns - values).var() / (returns.var() + 1e-8)
        stats = torch.stack(stats).mean(0).tolist()
        names = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac")
        return {
            **dict(zip(names, stats)),
            "explained_variance": explained_var.item(),
            "action_std": self.actor.log_std.exp().mean().item(),
            "lr": self.optim.param_groups[0]["lr"],
        }

    def state_dict(self) -> dict:
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hidden-size", type=int, default=64, help="actor")
    parser.add_argument("--critic-hidden-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--rollout-steps", type=int, default=128, help="steps per environment per update")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--entropy-coeff", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None, help="stop the epochs early above this KL")


def make_agent(args: argparse.Namespace, actor_dim: int, critic_dim: int, action_dim: int, num_drones: int, device):
    return MAPPO(
        actor_dim,
        critic_dim,
        action_dim,
        num_drones,
        num_envs=args.num_envs,
        total_steps=args.steps,
        hidden_size=args.hidden_size,
        critic_hidden_size=args.critic_hidden_size,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        rollout_steps=args.rollout_steps,
        epochs=args.epochs,
        num_minibatches=args.num_minibatches,
        clip_coef=args.clip_coef,
        entropy_coeff=args.entropy_coeff,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        device=device,
    )
