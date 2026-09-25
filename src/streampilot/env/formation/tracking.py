"""Formation tracking: follow a wandering target while holding a formation behind the leader."""

import mujoco
import numpy as np

from streampilot.env.formation.base import FormationBaseEnv, rotation, unit
from streampilot.env.tracking import wander


class FormationTrackingEnv(FormationBaseEnv):
    """The team version of ``TrackingEnv``: the same wandering red pillar is every drone's detector
    target. The leader (drone 0) holds ``follow_distance`` (horizontal) from it; the others hold
    their formation slots behind the leader (a column for two drones, a triangle for three) in the
    frame pointing from the leader to the target, and every drone faces the target.

    Reward per step, in [0, 1]: the mean over drones of
    ``exp(-slot_error / reward_scale) * (1 + cos(heading_error)) / 2``, where the leader's slot
    error is its standoff error. There is no success termination. Privileged task obs: horizontal
    vector to the target and target velocity.
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
        arena_half_extent: float = 5.0,
        **kwargs,
    ):
        self.follow_distance = follow_distance
        self.target_half_height = target_height / 2
        self.target_max_speed = target_max_speed
        self.target_speed_reversion = target_speed_reversion
        self.target_speed_noise = target_speed_noise
        self.reward_scale = reward_scale
        self.target_half_width = 0.15
        super().__init__(arena_half_extent=arena_half_extent, **kwargs)
        # Leave room for the whole formation between the target and the arena edge.
        reach = np.max(np.linalg.norm(self.formation - [follow_distance, 0.0], axis=1))
        self.target_half_extent = arena_half_extent - reach - 0.5
        assert self.target_half_extent > 0, "arena too small for the formation"
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

    def _slots(self) -> np.ndarray:
        target = self._target_pos[:2]
        forward = unit(target - self.drone_pos[0, :2])
        return target - self.follow_distance * forward + self.formation @ rotation(forward).T

    def _slot_errors(self) -> np.ndarray:
        return np.linalg.norm(self._slots() - self.drone_pos[:, :2], axis=1)

    def _standoff_error(self) -> float:
        return float(abs(np.linalg.norm(self.drone_pos[0, :2] - self._target_pos[:2]) - self.follow_distance))

    def _task_reset(self, options: dict) -> None:
        target_xy = self.np_random.uniform(-1.0, 1.0, size=2) * self.target_half_extent
        self.data.mocap_pos[self._target] = [*target_xy, self.target_half_height]
        self._target_vel = np.zeros(3)
        # Start near the formation, roughly facing the target, so it begins in (or near) view.
        bearing = self.np_random.uniform(-np.pi, np.pi)
        distance = self.follow_distance + self.np_random.uniform(-0.5, 0.5)
        forward = -np.array([np.cos(bearing), np.sin(bearing)])
        xy = target_xy - distance * forward + self.formation @ rotation(forward).T
        # Jitter of at most 0.1 m per axis keeps the drones clear of min_separation.
        xy[1:] += self.np_random.uniform(-0.1, 0.1, size=(self.num_drones - 1, 2))
        rel = target_xy - xy
        yaw = np.arctan2(rel[:, 1], rel[:, 0]) + self.np_random.uniform(-0.5, 0.5, size=self.num_drones)
        self._set_drone_states(xy, yaw)

    def _task_before_step(self) -> None:
        pos, vel = wander(
            self.np_random,
            self._target_pos[:2],
            self._target_vel[:2],
            self.dt,
            self.target_max_speed,
            self.target_speed_reversion,
            self.target_speed_noise,
            self.target_half_extent,
        )
        self._target_vel[:2] = vel
        self.data.mocap_pos[self._target][:2] = pos

    def _task_target_geom(self, drone: int) -> int:
        return self._target_geom

    def _task_obs(self, drone: int) -> np.ndarray:
        return np.concatenate([self._target_pos[:2] - self.drone_pos[drone, :2], self._target_vel[:2]])

    def _task_step(self):
        facing = (1.0 + np.cos(self._heading_error(self._target_pos))) / 2
        # The leader's slot is on its own bearing, so its slot error is its standoff error.
        reward = float(np.mean(np.exp(-self._slot_errors() / self.reward_scale) * facing))
        return reward, False, self._task_info()

    def _task_info(self) -> dict:
        return {
            "standoff_error": self._standoff_error(),
            "formation_error": self._slot_errors(),
            "heading_error": np.abs(self._heading_error(self._target_pos)),
        }

    def goals(self):
        return self._slots(), self._target_pos[:2].copy()
