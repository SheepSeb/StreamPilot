"""The firmware's C learner (compiled for the PC) against the PyTorch reference in streampilot.stream_x.

    uv run pytest pico/tests
"""

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium.spaces import Box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "host"))

from pico_link import EmulatedTransport, PicoAgent, flatten, unflatten  # noqa: E402
from streampilot.stream_x.agents import StreamAC  # noqa: E402
from streampilot.stream_x.wrappers import HistoryObservation, NormalizeObservation, ScaleReward  # noqa: E402

OBS, ACT, FRAMES = 5, 3, 4
IN = OBS * FRAMES + ACT


def make_pair(hidden: int, seed: int = 0) -> tuple[StreamAC, PicoAgent]:
    torch.manual_seed(seed)
    ref = StreamAC(IN, ACT, hidden_size=hidden)
    pico = PicoAgent(EmulatedTransport())
    pico.init(OBS, ACT, num_frames=FRAMES, hidden_size=hidden, seed=seed + 1)
    pico.load_torch(ref.actor, ref.critic)
    return ref, pico


def params_of(ref: StreamAC) -> np.ndarray:
    return np.concatenate([flatten(ref.actor), flatten(ref.critic)])


def test_flatten_roundtrip():
    ref, pico = make_pair(hidden=16)
    state = pico.state_dict()
    for net, key in ((ref.actor, "actor"), (ref.critic, "critic")):
        for name, value in net.state_dict().items():
            torch.testing.assert_close(state[key][name], value)
    assert pico.n_actor + pico.n_critic == params_of(ref).size
    assert unflatten(flatten(ref.critic), IN, 16, (1,)).keys() == ref.critic.state_dict().keys()


def test_device_sparse_init():
    pico = PicoAgent(EmulatedTransport())
    pico.init(OBS, ACT, num_frames=FRAMES, hidden_size=64, seed=3)
    w = pico.state_dict()["actor"]["hidden.1.weight"]
    assert ((w == 0).sum(1) == 58).all()  # ceil(0.9 * 64) zeros per unit, as sparse_init_
    assert w.abs().max() <= (1 / 64) ** 0.5


@pytest.mark.parametrize("hidden", [32, 128])
def test_forward_matches_torch(hidden):
    ref, pico = make_pair(hidden)
    rng = np.random.default_rng(0)
    for _ in range(10):
        x = rng.normal(size=IN).astype(np.float32) * 2
        mu, std, value = pico.forward(x)
        with torch.no_grad():
            ref_mu, ref_std = ref.actor(torch.as_tensor(x))
            ref_value = ref.critic(torch.as_tensor(x))
        np.testing.assert_allclose(mu, ref_mu.numpy(), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(std, ref_std.numpy(), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(value, ref_value.item(), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("hidden", [32, 128])
def test_update_matches_torch(hidden):
    ref, pico = make_pair(hidden)
    rng = np.random.default_rng(1)
    obs = rng.normal(size=IN).astype(np.float32)
    for t in range(300):
        next_obs = rng.normal(size=IN).astype(np.float32)
        action = rng.normal(size=ACT).astype(np.float32)
        reward = float(rng.normal() * 3)
        terminated = bool(rng.random() < 0.03)
        done = terminated or bool(rng.random() < 0.02)
        ref_delta = ref.update(obs, action, reward, next_obs, terminated, done)
        delta = pico.update(obs, action, reward, next_obs, terminated, done)
        assert delta == pytest.approx(ref_delta, rel=1e-3, abs=1e-4), t
        obs = next_obs
    np.testing.assert_allclose(pico.get_params(), params_of(ref), rtol=1e-3, atol=1e-4)


class FakeEnv(gym.Env):
    """Random observations and rewards with detection-like shapes, and random episode ends."""

    observation_space = Box(0.0, 1.0, shape=(OBS,), dtype=np.float32)
    action_space = Box(-1.0, 1.0, shape=(ACT,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        reward = float(self.np_random.normal() * 2 - 0.1 * np.sum(np.clip(action, -1, 1) ** 2))
        return self._obs(), reward, bool(self.np_random.random() < 0.02), self.t >= 60, {}

    def _obs(self):
        return self.np_random.random(OBS).astype(np.float32)


class Recorder(gym.Wrapper):
    """Keeps the raw observation and reward, which is what the PC sends to the Pico."""

    def reset(self, **kwargs):
        self.obs, info = self.env.reset(**kwargs)
        return self.obs, info

    def step(self, action):
        self.obs, self.reward, terminated, truncated, info = self.env.step(action)
        return self.obs, self.reward, terminated, truncated, info


def test_full_pipeline_matches_torch():
    """History, observation normalization, reward scaling and updates on the Pico, against the Python
    wrappers and StreamAC fed the actions the Pico sampled."""
    ref, pico = make_pair(hidden=64)
    raw = Recorder(FakeEnv())
    env = ScaleReward(NormalizeObservation(HistoryObservation(raw, FRAMES)), gamma=0.99)

    obs, _ = env.reset(seed=0)
    action = pico.reset(raw.obs)
    for t in range(400):
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        ref_delta = ref.update(obs, action, reward, next_obs, terminated, done)
        delta, _, next_action = pico.step(raw.obs, raw.reward, terminated, truncated)
        assert delta == pytest.approx(ref_delta, rel=1e-3, abs=1e-4), t
        assert (next_action is None) == done
        if done:
            obs, _ = env.reset()
            action = pico.reset(raw.obs)
        else:
            obs, action = next_obs, next_action

    obs_stats, ret_stats = pico.stats()
    ref_obs = env.get_wrapper_attr("obs_stats")
    ref_ret = env.get_wrapper_attr("return_stats")
    assert obs_stats["count"] == ref_obs.count and ret_stats["count"] == ref_ret.count
    np.testing.assert_allclose(obs_stats["mean"], ref_obs.mean, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(obs_stats["var"], ref_obs.var, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(ret_stats["var"], [ref_ret.var], rtol=1e-5)
    np.testing.assert_allclose(pico.get_params(), params_of(ref), rtol=1e-3, atol=1e-4)


def test_step_before_reset_is_rejected():
    pico = PicoAgent(EmulatedTransport())
    pico.init(OBS, ACT, num_frames=FRAMES, hidden_size=8)
    with pytest.raises(RuntimeError, match="wrong state"):
        pico.step(np.zeros(OBS), 0.0, False, False)
