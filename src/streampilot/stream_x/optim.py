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
        self.last_scale = 1.0  # step size taken in the last step / lr: below 1, the bound was active

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
            self.last_scale = step_size / group["lr"]
            torch._foreach_add_(params, traces, alpha=-step_size * delta)
            if reset:
                torch._foreach_zero_(traces)


def _batched_obgd_step(params, trace, grads, delta, decay, lr, kappa):
    """Decay and update the traces, bound the step per learner, move the parameters. Returns each
    learner's step size over ``lr``. A free function so that ``torch.compile`` can fuse it."""
    trace.mul_(decay)
    trace.add_(torch.cat([g.flatten(1) for g in grads], dim=1))
    z_sum = trace.abs().sum(1)  # (n,): each learner's ||z||_1
    bound = lr * kappa * delta.abs().clamp(min=1.0) * z_sum
    step_size = torch.where(bound > 1.0, lr / bound, torch.full_like(bound, lr))
    update = trace * (-step_size * delta).unsqueeze(1)
    for p, u in zip(params, update.split([q[0].numel() for q in params], dim=1)):
        p.add_(u.reshape(p.shape))
    return step_size / lr


class BatchedObGD(torch.optim.Optimizer):
    """``ObGD`` for ``n`` independent learners whose parameters are stacked along dim 0 (learner
    ``i`` owns ``p[i]`` of every parameter). Each learner has its own trace, TD error and step-size
    bound, so ``step`` does exactly what ``n`` separate ``ObGD`` optimizers would.

    The traces live in one ``(n, total)`` buffer (each parameter's slice flattened, in order), so a
    step is a handful of tensor ops instead of a few per parameter. ``compile`` fuses them."""

    def __init__(self, params, lr: float = 1.0, gamma: float = 0.99, lamda: float = 0.8, kappa: float = 2.0):
        super().__init__(params, dict(lr=lr, gamma=gamma, lamda=lamda, kappa=kappa))
        self.last_scale = torch.ones(1)  # (n,): each learner's step size / lr in the last step
        self._core = _batched_obgd_step
        for group in self.param_groups:
            ps = group["params"]
            group["_trace"] = torch.zeros(ps[0].shape[0], sum(p[0].numel() for p in ps))

    def compile(self) -> None:
        self._core = torch.compile(_batched_obgd_step, dynamic=False)

    @torch.no_grad()
    def step(self, delta: torch.Tensor, reset: bool = False) -> None:
        """``delta``: ``(n,)``, one TD error per learner."""
        for group in self.param_groups:
            params = group["params"]
            self.last_scale = self._core(
                params, group["_trace"], [p.grad for p in params], delta,
                group["gamma"] * group["lamda"], group["lr"], group["kappa"],
            )  # fmt: skip
            if reset:
                group["_trace"].zero_()
