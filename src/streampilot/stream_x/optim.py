"""Overshooting-bounded gradient descent (ObGD), from Elsayed, Vasan & Mahmood (2024),
"Streaming Deep Reinforcement Learning Finally Works" (https://arxiv.org/abs/2410.14606)."""

import torch


class ObGD(torch.optim.Optimizer):
    """Semi-gradient TD(lambda) with accumulating eligibility traces and a step size that is
    shrunk whenever a full step would overshoot the TD target.

    ``p.grad`` must hold the gradient of the quantity to *decrease* (e.g. ``-v(s)`` or
    ``-log pi(a|s)``); ``step(delta)`` then moves the parameters by ``alpha * delta * trace``.
    """

    def __init__(self, params, lr: float = 1.0, gamma: float = 0.99, lamda: float = 0.8, kappa: float = 2.0):
        super().__init__(params, dict(lr=lr, gamma=gamma, lamda=lamda, kappa=kappa))

    @torch.no_grad()
    def step(self, delta: float, reset: bool = False) -> None:
        for group in self.param_groups:
            params = group["params"]
            for p in params:
                if not self.state[p]:
                    self.state[p]["trace"] = torch.zeros_like(p)
            traces = [self.state[p]["trace"] for p in params]
            torch._foreach_mul_(traces, group["gamma"] * group["lamda"])
            torch._foreach_add_(traces, [p.grad for p in params])
            z_sum = float(sum(torch._foreach_norm(traces, 1)))

            # Effective step size bound: alpha * kappa * max(|delta|, 1) * ||z||_1 <= 1.
            bound = group["lr"] * group["kappa"] * max(abs(delta), 1.0) * z_sum
            step_size = group["lr"] / bound if bound > 1.0 else group["lr"]
            torch._foreach_add_(params, traces, alpha=-step_size * delta)
            if reset:
                torch._foreach_zero_(traces)


class BatchedObGD(torch.optim.Optimizer):
    """``ObGD`` for ``n`` independent learners whose parameters are stacked along dim 0 (learner
    ``i`` owns ``p[i]`` of every parameter). Each learner has its own trace, TD error and step-size
    bound, so ``step`` does exactly what ``n`` separate ``ObGD`` optimizers would."""

    def __init__(self, params, lr: float = 1.0, gamma: float = 0.99, lamda: float = 0.8, kappa: float = 2.0):
        super().__init__(params, dict(lr=lr, gamma=gamma, lamda=lamda, kappa=kappa))

    @torch.no_grad()
    def step(self, delta: torch.Tensor, reset: bool = False) -> None:
        """``delta``: ``(n,)``, one TD error per learner."""
        for group in self.param_groups:
            params = group["params"]
            for p in params:
                if not self.state[p]:
                    self.state[p]["trace"] = torch.zeros_like(p)
            traces = [self.state[p]["trace"] for p in params]
            torch._foreach_mul_(traces, group["gamma"] * group["lamda"])
            torch._foreach_add_(traces, [p.grad for p in params])
            z_sum = sum(t.abs().flatten(1).sum(1) for t in traces)  # (n,): each learner's ||z||_1

            bound = group["lr"] * group["kappa"] * delta.abs().clamp(min=1.0) * z_sum
            step_size = torch.where(bound > 1.0, group["lr"] / bound, torch.full_like(bound, group["lr"]))
            scale = -step_size * delta
            for p, t in zip(params, traces):
                p.add_(t * scale.view(-1, *[1] * (p.dim() - 1)))
            if reset:
                torch._foreach_zero_(traces)
