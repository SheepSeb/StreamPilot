"""Observation history and the online normalization streaming RL relies on.

``ObservationHistory`` is used both by the training wrapper and by ``Policy`` at
deployment, so the policy sees the same features in sim and on the real drone.
"""

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


class RunningMeanStd:
    """Welford's running mean and (sample) variance, one sample at a time."""

    def __init__(self, shape=()):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self._m2 = np.zeros(shape)
        self.count = 0

    def update(self, x) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean = self.mean + delta / self.count
        self._m2 = self._m2 + delta * (x - self.mean)
        if self.count > 1:
            self.var = self._m2 / (self.count - 1)

    def update_batch(self, x) -> None:
        """Add every row of ``x`` at once (Chan et al.'s parallel update; equal to ``update`` per row)."""
        x = np.asarray(x, dtype=np.float64).reshape(-1, *np.shape(self.mean))
        n = len(x)
        if n == 0:
            return
        batch_mean = x.mean(axis=0)
        delta = batch_mean - self.mean
        total = self.count + n
        self.mean = self.mean + delta * (n / total)
        self._m2 = self._m2 + ((x - batch_mean) ** 2).sum(axis=0) + delta**2 * (self.count * n / total)
        self.count = total
        if total > 1:
            self.var = self._m2 / (total - 1)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "m2": self._m2, "count": self.count}

    def load_state_dict(self, state: dict) -> None:
        self.mean, self.var, self._m2, self.count = state["mean"], state["var"], state["m2"], state["count"]


class ObservationHistory:
    """Features ``[obs_{t-k+1}, ..., obs_t, a_{t-1}]``: the last ``num_frames`` observations
    (the first one repeated after a reset) and the previous action (zeros after a reset)."""

    def __init__(self, obs_dim: int, action_dim: int, num_frames: int):
        self.obs_dim, self.action_dim, self.num_frames = obs_dim, action_dim, num_frames
        self._frames: deque = deque(maxlen=num_frames)
        self._last_action = np.zeros(action_dim)

    @property
    def dim(self) -> int:
        return self.obs_dim * self.num_frames + self.action_dim

    def reset(self, obs) -> np.ndarray:
        self._frames.extend([np.asarray(obs, dtype=np.float64).ravel()] * self.num_frames)
        self._last_action = np.zeros(self.action_dim)
        return self.features()

    def push(self, action, obs) -> np.ndarray:
        self._frames.append(np.asarray(obs, dtype=np.float64).ravel())
        self._last_action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        return self.features()

    def features(self) -> np.ndarray:
        return np.concatenate([*self._frames, self._last_action]).astype(np.float32)


class HistoryObservation(gym.Wrapper):
    def __init__(self, env: gym.Env, num_frames: int):
        super().__init__(env)
        self.history = ObservationHistory(
            int(np.prod(env.observation_space.shape)), int(np.prod(env.action_space.shape)), num_frames
        )
        self.observation_space = Box(-np.inf, np.inf, shape=(self.history.dim,), dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.history.reset(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.history.push(action, obs), reward, terminated, truncated, info


class NormalizeObservation(gym.ObservationWrapper):
    """Standardize observations with running statistics, updated on every observation."""

    def __init__(self, env: gym.Env, epsilon: float = 1e-8):
        super().__init__(env)
        self.obs_stats = RunningMeanStd(env.observation_space.shape)
        self.epsilon = epsilon

    def observation(self, obs):
        self.obs_stats.update(obs)
        return ((obs - self.obs_stats.mean) / np.sqrt(self.obs_stats.var + self.epsilon)).astype(np.float32)


class ScaleReward(gym.Wrapper):
    """Divide rewards by the running standard deviation of the discounted return."""

    def __init__(self, env: gym.Env, gamma: float = 0.99, epsilon: float = 1e-8):
        super().__init__(env)
        self.return_stats = RunningMeanStd()
        self.gamma, self.epsilon = gamma, epsilon
        self._return = 0.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._return = self._return * self.gamma * (1.0 - terminated) + reward
        self.return_stats.update(self._return)
        if terminated or truncated:
            self._return = 0.0
        return obs, reward / np.sqrt(self.return_stats.var + self.epsilon), terminated, truncated, info
