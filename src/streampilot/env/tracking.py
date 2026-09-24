"""Target tracking: follow a target that wanders across the floor while keeping it in view."""

import mujoco
import numpy as np

from streampilot.env.base import DroneBaseEnv


class TrackingEnv(DroneBaseEnv):
    """The target (a red, person-sized pillar standing on the floor) is the detector target. It
    moves with an Ornstein-Uhlenbeck velocity and bounces off the walls of a square that leaves
    room for the standoff inside the (larger, by default) arena.

    The drone should hold ``follow_distance`` (horizontal) from the target while facing it; at the
    default altitude the level camera then sees the pillar near the image centre.
    Reward per step, in [0, 1]: ``exp(-standoff_error / reward_scale) * (1 + cos(heading_error)) / 2``.
    There is no success termination. Privileged task obs: horizontal vector to the target and
    target velocity.
    """

    task_obs_dim = 4

    def __init__(
        self,
        follow_distance: float = 1.5,
        target_height: float = 1.6,
        target_max_speed: float = 0.6,
        target_speed_reversion: float = 0.5,
        target_speed_noise: float = 0.4,
        reward_scale: float = 0.3,
        arena_half_extent: float = 4.0,
        **kwargs,
    ):
        self.follow_distance = follow_distance
        self.target_half_height = target_height / 2
        self.target_max_speed = target_max_speed
        self.target_speed_reversion = target_speed_reversion
        self.target_speed_noise = target_speed_noise
        self.reward_scale = reward_scale
        self.target_half_width = 0.15
        self.target_half_extent = arena_half_extent - follow_distance - 0.5
        assert self.target_half_extent > 0, "arena too small for the follow distance"
        super().__init__(arena_half_extent=arena_half_extent, **kwargs)
        self._target = self._mocap_id("target")
        self._target_geom = self.model.geom("target").id
        self._target_vel = np.zeros(3)

    def _build_scene(self, spec: mujoco.MjSpec) -> None:
        target = spec.worldbody.add_body(name="target", mocap=True)
        target.add_geom(
            name="target",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[self.target_half_width, self.target_half_width, self.target_half_height],
            rgba=[0.9, 0.1, 0.1, 1.0],
            contype=0,
            conaffinity=0,
        )

    @property
    def _target_pos(self) -> np.ndarray:
        return self.data.mocap_pos[self._target]

    def _standoff_error(self) -> float:
        return float(abs(np.linalg.norm(self.drone_pos[:2] - self._target_pos[:2]) - self.follow_distance))

    def _task_reset(self, options: dict) -> None:
        target_xy = self.np_random.uniform(-1.0, 1.0, size=2) * self.target_half_extent
        self.data.mocap_pos[self._target] = [*target_xy, self.target_half_height]
        self._target_vel = np.zeros(3)
        # Start near the standoff, roughly facing the target, so it begins in (or near) view.
        bearing = self.np_random.uniform(-np.pi, np.pi)
        distance = self.follow_distance + self.np_random.uniform(-0.5, 0.5)
        start_xy = target_xy + distance * np.array([np.cos(bearing), np.sin(bearing)])
        yaw = bearing + np.pi + self.np_random.uniform(-0.5, 0.5)
        self._set_drone_state(start_xy, yaw)

    def _task_before_step(self) -> None:
        # Ornstein-Uhlenbeck velocity in the floor plane, clipped to the maximum speed.
        noise = self.np_random.normal(size=2) * self.target_speed_noise * np.sqrt(self.dt)
        vel = self._target_vel[:2] * (1.0 - self.target_speed_reversion * self.dt) + noise
        speed = np.linalg.norm(vel)
        if speed > self.target_max_speed:
            vel *= self.target_max_speed / speed

        pos = self._target_pos[:2] + vel * self.dt
        limit = self.target_half_extent
        bounced = np.abs(pos) > limit
        vel[bounced] *= -1.0
        pos = np.clip(pos, -limit, limit)

        self._target_vel[:2] = vel
        self.data.mocap_pos[self._target][:2] = pos

    def _task_target_geom(self) -> int:
        return self._target_geom

    def _task_obs(self) -> np.ndarray:
        return np.concatenate([self._target_pos[:2] - self.drone_pos[:2], self._target_vel[:2]])

    def _task_step(self):
        facing = (1.0 + np.cos(self._heading_error(self._target_pos))) / 2
        reward = float(np.exp(-self._standoff_error() / self.reward_scale) * facing)
        return reward, False, self._task_info()

    def _task_info(self) -> dict:
        return {
            "standoff_error": self._standoff_error(),
            "heading_error": abs(self._heading_error(self._target_pos)),
        }
