"""Run a trained policy (Stream AC, PPO or SAC) from raw environment (or real detector) observations."""

from pathlib import Path

import numpy as np
import torch

from streampilot.baselines import ppo, sac
from streampilot.stream_x import agents as stream_ac
from streampilot.stream_x.wrappers import ObservationHistory

ACTORS = {"stream_ac": stream_ac.Actor, "ppo": ppo.Actor, "sac": sac.Actor}


class Policy:
    """Deterministic policy loaded from a training checkpoint, with the observation history and
    (frozen) normalization it was trained with. Feed it the raw observation every control step::

        policy = Policy.load("runs/waypoint_seed0/final.pt")
        policy.reset()
        action = policy(obs)  # obs = [visible, cx, cy, w, h] from the env or the detector
    """

    def __init__(self, checkpoint: dict):
        self.config = checkpoint["config"]
        obs_dim, action_dim = checkpoint["obs_dim"], checkpoint["action_dim"]
        self.history = ObservationHistory(obs_dim, action_dim, self.config["num_frames"])
        actor_cls = ACTORS[self.config.get("algo", "stream_ac")]  # checkpoints before baselines were Stream AC
        self.actor = actor_cls(self.history.dim, action_dim, self.config["hidden_size"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor.eval()
        stats = checkpoint["obs_stats"]  # None when the algorithm does not normalize observations
        self.obs_mean = stats["mean"] if stats else 0.0
        self.obs_std = np.sqrt(stats["var"] + 1e-8) if stats else 1.0
        self._last_action: np.ndarray | None = None

    @classmethod
    def load(cls, path: str | Path) -> "Policy":
        return cls(torch.load(path, map_location="cpu", weights_only=False))

    def reset(self) -> None:
        self._last_action = None

    @torch.no_grad()
    def __call__(self, obs) -> np.ndarray:
        if self._last_action is None:
            features = self.history.reset(obs)
        else:
            features = self.history.push(self._last_action, obs)
        features = ((features - self.obs_mean) / self.obs_std).astype(np.float32)
        self._last_action = self.actor.deterministic(torch.as_tensor(features)).numpy()
        return self._last_action
