import numpy as np
import pytest
import torch

from streampilot.stream_x.agents import sparse_init_
from streampilot.stream_x.optim import ObGD
from streampilot.stream_x.wrappers import ObservationHistory, RunningMeanStd


@pytest.mark.parametrize("delta_target", [0.5, 5.0, 500.0])
def test_obgd_does_not_overshoot(delta_target):
    # Linear value v(x) = w.x: plain SGD with lr=1 would overshoot the target by ||x||^2 - 1.
    # The bound replaces ||z||_2^2 with ||z||_1, which is only an upper bound for |z_i| <= 1
    # (what input normalization and LayerNorm provide in practice).
    torch.manual_seed(0)
    w = torch.zeros(64, requires_grad=True)
    x = torch.rand(64) * 2 - 1
    optim = ObGD([w], lr=1.0, lamda=0.0, kappa=2.0)
    delta = delta_target - float(w.detach() @ x)
    (-(w @ x)).backward()
    optim.step(delta)
    new_delta = delta_target - float(w.detach() @ x)
    assert 0.0 <= new_delta / delta < 1.0


def test_sparse_init_keeps_an_input_per_unit():
    small, large = torch.empty(64, 5), torch.empty(64, 128)
    sparse_init_(small)
    sparse_init_(large)
    assert ((small != 0).sum(dim=1) == 1).all()
    assert (large == 0).float().mean().item() == pytest.approx(0.9, abs=0.01)


def test_running_mean_std_matches_numpy():
    xs = np.random.default_rng(0).normal(3.0, 2.0, size=(500, 4))
    stats = RunningMeanStd(4)
    for x in xs:
        stats.update(x)
    np.testing.assert_allclose(stats.mean, xs.mean(axis=0))
    np.testing.assert_allclose(stats.var, xs.var(axis=0, ddof=1))


def test_observation_history():
    history = ObservationHistory(obs_dim=2, action_dim=1, num_frames=3)
    np.testing.assert_array_equal(history.reset([1, 2]), [1, 2, 1, 2, 1, 2, 0])
    np.testing.assert_array_equal(history.push([5.0], [3, 4]), [1, 2, 1, 2, 3, 4, 1])
