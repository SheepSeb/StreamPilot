"""Classical (non-learning) controllers for the drone tasks: a PID and a linear MPC.

The drone is a velocity-commanded point mass: ``ctrl`` is a world-frame velocity, tracked with a
first-order lag (time constant ``TAU`` ~ 0.2 s), so per axis ``v' = (u - v) / TAU``, ``p' = v``.
Both controllers track a *reference* (a goal position and a feedforward velocity) with that model:

- ``PID``: ``u = v_ff + Kp e + Ki integral(e) - Kd v``, with the integral clamped (anti-windup).
- ``MPC``: minimises tracking error over a receding horizon with the exact discretised model and a
  speed limit ``|u| <= max_speed``, solved as a condensed QP by projected gradient (numpy only).

The references come from ``guidance``, the same task logic as the scripted controllers in
``streampilot.visualize``. Like those, these baselines read the privileged state
(``env.state_obs()``), not the detection: they show what a good model-based controller reaches
with perfect state estimation, the upper reference for the vision policies.

    controller = ControllerPolicy("mpc")
    controller.reset(env)
    action = controller(env)   # (3,) or, for formation tasks, (num_drones, 3)
"""

import numpy as np

from streampilot.env.base import wrap_angle
from streampilot.env.formation import FormationBaseEnv

TAU = 0.2  # s, velocity lag of the simulated flight controller (see assets/skydio_x2/scene.xml)
YAW_GAIN = 2.0  # 1/s, proportional heading control, shared by both controllers


def body_action(env, state, world_vel, desired_yaw) -> np.ndarray:
    """Normalized ``[vx, vy, yaw_rate]`` body-frame action for a horizontal world-frame velocity."""
    yaw = np.arctan2(state[5], state[4])
    c, s = np.cos(yaw), np.sin(yaw)
    vx, vy = world_vel
    yaw_rate = YAW_GAIN * wrap_angle(desired_yaw - yaw)
    return np.clip(np.array([c * vx + s * vy, -s * vx + c * vy, yaw_rate]) / env.action_scale, -1.0, 1.0)


def limit_norm(v: np.ndarray, limit: float) -> np.ndarray:
    """Scale each row of ``v`` down to at most ``limit`` long."""
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v * np.minimum(1.0, limit / np.maximum(norm, 1e-9))


class PID:
    """Position PID producing world-frame velocity commands for ``n`` drones at once."""

    def __init__(self, kp=2.0, ki=0.2, kd=0.4, max_speed=1.5, integral_limit=0.5):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.max_speed, self.integral_limit = max_speed, integral_limit
        self.integral = 0.0

    def reset(self) -> None:
        self.integral = 0.0

    def __call__(self, pos, vel, goal, ff, dt) -> np.ndarray:
        error = goal - pos
        # Clamped integral: no windup during long approaches.
        self.integral = limit_norm(self.integral + error * dt, self.integral_limit)
        u = ff + self.kp * error + self.ki * self.integral - self.kd * vel
        return limit_norm(u, self.max_speed)


class MPC:
    """Receding-horizon tracking of ``goal + ff * t`` for ``n`` drones (each solved independently).

    Cost per step ``k`` of the ``horizon``: ``q |p_k - ref_k|^2 + qv |v_k - ff|^2 + r |u_k - ff|^2``,
    subject to ``|u_k| <= max_speed``. The model decouples in x and y, so one ``horizon`` x
    ``horizon`` Hessian serves both axes and all drones; the speed limit is a per-step ball, whose
    projection is trivial, hence projected gradient with Nesterov momentum instead of a QP solver.
    """

    def __init__(self, dt, horizon=20, q=10.0, qv=1.0, r=0.5, max_speed=1.5, iterations=60):
        self.horizon, self.max_speed, self.iterations, self.dt = horizon, max_speed, iterations, dt
        a = np.exp(-dt / TAU)
        # Exact zero-order-hold step of v' = (u - v) / TAU, p' = v:  x+ = A x + B u, x = [p, v].
        A = np.array([[1.0, TAU * (1 - a)], [0.0, a]])
        B = np.array([[dt - TAU * (1 - a)], [1 - a]])
        n = horizon
        # Stacked predictions x_1..x_N = free @ x_0 + G @ u_0..u_{N-1}, with p and v rows interleaved.
        free = np.zeros((2 * n, 2))
        G = np.zeros((2 * n, n))
        power = np.eye(2)
        for k in range(n):
            power = A @ power
            free[2 * k : 2 * k + 2] = power
            reach = np.eye(2)
            for j in range(k, -1, -1):
                G[2 * k : 2 * k + 2, j] = (reach @ B)[:, 0]
                reach = A @ reach
        self._free = free
        self._Gp, self._Gv = G[0::2], G[1::2]  # position and velocity rows, (N, N)
        self._Ap, self._Av = free[0::2], free[1::2]  # (N, 2) each
        self.q, self.qv, self.r = q, qv, r
        self._H = q * self._Gp.T @ self._Gp + qv * self._Gv.T @ self._Gv + r * np.eye(n)
        self._step = 1.0 / np.linalg.eigvalsh(self._H).max()
        self._warm: np.ndarray | None = None

    def reset(self) -> None:
        self._warm = None

    def __call__(self, pos, vel, goal, ff, dt=None) -> np.ndarray:
        n, drones = self.horizon, len(pos)
        t = self.dt * np.arange(1, n + 1)
        # Reference and free response, each (N, drones, 2).
        ref = goal[None] + t[:, None, None] * ff[None]
        free_p = self._Ap[:, 0, None, None] * pos[None] + self._Ap[:, 1, None, None] * vel[None]
        free_v = self._Av[:, 0, None, None] * pos[None] + self._Av[:, 1, None, None] * vel[None]
        rp, rv = (ref - free_p).reshape(n, -1), (ff[None] - free_v).reshape(n, -1)
        ff_flat = np.broadcast_to(ff[None], (n, drones, 2)).reshape(n, -1)
        # grad(u) = H u - (q Gp' rp + qv Gv' rv + r ff); columns are (drone, axis) pairs.
        linear = self.q * self._Gp.T @ rp + self.qv * self._Gv.T @ rv + self.r * ff_flat

        u = self._start(drones, ff)
        y, momentum = u.copy(), 1.0
        for _ in range(self.iterations):
            step = y - self._step * (self._H @ y - linear)
            new = limit_norm(step.reshape(n, drones, 2), self.max_speed).reshape(n, -1)
            new_momentum = (1 + np.sqrt(1 + 4 * momentum**2)) / 2
            y = new + (momentum - 1) / new_momentum * (new - u)
            u, momentum = new, new_momentum
        u = u.reshape(n, drones, 2)
        self._warm = np.concatenate([u[1:], u[-1:]])  # shift: the next solve starts from this plan
        return u[0]

    def _start(self, drones, ff) -> np.ndarray:
        if self._warm is not None and self._warm.shape[1] == drones:
            return self._warm.reshape(self.horizon, -1)
        return np.broadcast_to(ff[None], (self.horizon, drones, 2)).reshape(self.horizon, -1).copy()


def guidance(env) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per drone: goal position ``(n, 2)``, feedforward world velocity ``(n, 2)`` and heading to
    hold ``(n,)``, from privileged state. ``n`` is 1 for the single-drone tasks."""
    if isinstance(env, FormationBaseEnv):
        return _formation_guidance(env)
    state = env.state_obs()
    pos, rel = state[0:2], state[7:9]
    name = type(env).__name__
    ff = np.zeros(2)
    if name == "WaypointEnv":
        goal, heading = pos + rel, np.arctan2(rel[1], rel[0])
    elif name == "LandingEnv":
        pad = pos + rel
        # Hand-off point on the arena-centre side of the pad, so it stays in bounds.
        side = -pad / np.linalg.norm(pad) if np.linalg.norm(pad) > 0.3 else -rel / max(np.linalg.norm(rel), 1e-6)
        goal, heading = pad + side * env.handoff_distance, np.arctan2(rel[1], rel[0])
    elif name == "TrackingEnv":
        away = -rel / max(np.linalg.norm(rel), 1e-6)
        limit = env.arena_half_extent - 0.2
        goal = np.clip(pos + rel + away * env.follow_distance, -limit, limit)
        ff, heading = state[9:11], np.arctan2(rel[1], rel[0])
    else:
        raise ValueError(f"no guidance for {name}")
    return goal[None], ff[None], np.array([heading])


def _formation_guidance(env) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    goals, look_at = env.goals()
    goals = goals.copy()
    if hasattr(env, "handoff_distance"):
        # Landing: the leader approaches the pad from the arena-centre side (as in the scripted policy).
        pad = env._pad_top[:2]
        norm = np.linalg.norm(pad)
        side = -pad / norm if norm > 0.3 else (env.drone_pos[0, :2] - pad) / max(np.linalg.norm(env.drone_pos[0, :2] - pad), 1e-6)
        goals[0] = pad + side * env.handoff_distance
    pos, state = env.drone_pos[:, :2], env.state_obs()
    ff = np.tile(getattr(env, "_target_vel", np.zeros(3))[:2], (env.num_drones, 1))

    headings = []
    for i in range(env.num_drones):
        target = goals[i] if look_at is None else look_at
        rel = target - pos[i]
        if look_at is None and np.linalg.norm(rel) < 0.3:
            headings.append(np.arctan2(state[i, 5], state[i, 4]))  # at the waypoint: any heading
        else:
            headings.append(np.arctan2(rel[1], rel[0]))
    return goals, ff, np.array(headings)


def avoid_collisions(env, vel_cmd: np.ndarray, safe_distance: float = 0.95) -> np.ndarray:
    """Formation collision avoidance on the tracker's velocity commands: within ``safe_distance``
    of a teammate a drone increasingly cancels its velocity towards it (fully at ``min_separation``),
    is pushed away and slides sideways, so two drones meeting head-on pass each other."""
    pos = env.drone_pos[:, :2]
    ahead = pos + 0.3 * env.drone_vel[:, :2]  # where the drones will be, given the velocity lag
    out = vel_cmd.copy()
    for i in range(env.num_drones):
        for j in range(env.num_drones):
            dist = np.linalg.norm(ahead[i] - ahead[j])
            if j == i or dist >= safe_distance:
                continue
            away = (ahead[i] - ahead[j]) / max(dist, 1e-6)
            weight = np.clip((safe_distance - dist) / (safe_distance - env.min_separation - 0.1), 0.0, 1.0)
            out[i] += weight * (max(0.0, -out[i] @ away) + 0.5) * away + weight * 0.65 * np.array([-away[1], away[0]])
    return out


class ControllerPolicy:
    """``policy(env) -> action`` for any task, with a ``"pid"`` or ``"mpc"`` velocity tracker."""

    def __init__(self, kind: str, max_speed: float = 1.0, **kwargs):
        assert kind in ("pid", "mpc"), kind
        self.kind, self.max_speed, self.kwargs = kind, max_speed, kwargs
        self.tracker = None

    def reset(self, env) -> None:
        if self.tracker is None:
            self.tracker = (
                PID(max_speed=self.max_speed, **self.kwargs)
                if self.kind == "pid"
                else MPC(env.dt, max_speed=self.max_speed, **self.kwargs)
            )
        self.tracker.reset()

    def __call__(self, env) -> np.ndarray:
        goal, ff, heading = guidance(env)
        state = np.atleast_2d(env.state_obs())
        pos, vel = state[:, 0:2].astype(np.float64), state[:, 2:4].astype(np.float64)
        vel_cmd = self.tracker(pos, vel, goal, ff, env.dt)
        if isinstance(env, FormationBaseEnv):
            vel_cmd = avoid_collisions(env, vel_cmd)
        actions = np.array([body_action(env, state[i], vel_cmd[i], heading[i]) for i in range(len(state))])
        return actions if isinstance(env, FormationBaseEnv) else actions[0]
