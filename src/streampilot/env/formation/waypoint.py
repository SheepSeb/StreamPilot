"""Formation waypoints: the team flies to a sequence of formation-shaped sets of waypoints."""

import mujoco
import numpy as np

from streampilot.env.formation.base import DRONE_COLORS, FormationBaseEnv, rotation


class FormationWaypointEnv(FormationBaseEnv):
    """Each stage shows one waypoint per drone (a ball at flight altitude in the drone's colour, its
    detector target), laid out in the formation shape (triangle for three drones, column for two)
    at a random position and rotation. Drone ``i`` flies to ball ``i``; the stage is complete once
    every drone is within ``reach_radius`` of its ball at the same time, and the next set appears.
    Privileged task obs: horizontal vector from the drone to its waypoint.

    Reward: mean progress towards the waypoints, plus a bonus for each completed stage. The
    episode terminates successfully after the last stage.
    """

    task_obs_dim = 2

    def __init__(
        self,
        num_stages: int = 3,
        reach_radius: float = 0.15,
        progress_weight: float = 10.0,
        stage_bonus: float = 10.0,
        **kwargs,
    ):
        self.num_stages = num_stages
        self.reach_radius = reach_radius
        self.progress_weight = progress_weight
        self.stage_bonus = stage_bonus
        super().__init__(**kwargs)
        self._markers = [self._mocap_id(f"waypoint{i}") for i in range(self.num_drones)]
        self._marker_geoms = [self.model.geom(f"waypoint{i}").id for i in range(self.num_drones)]
        self._stages = np.zeros((num_stages, self.num_drones, 2))
        self._reached = 0
        self._prev_dist = np.zeros(self.num_drones)

    def _build_scene(self, spec: mujoco.MjSpec) -> None:
        for i in range(self.num_drones):
            marker = spec.worldbody.add_body(name=f"waypoint{i}", mocap=True)
            marker.add_geom(
                name=f"waypoint{i}",
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[self.reach_radius, 0, 0],
                rgba=[*DRONE_COLORS[i], 1.0],
                contype=0,
                conaffinity=0,
            )

    @property
    def _targets(self) -> np.ndarray:
        # Stays on the last stage after it is complete, so the final observation is valid.
        return self._stages[min(self._reached, self.num_stages - 1)]

    def _distances(self) -> np.ndarray:
        return np.linalg.norm(self._targets - self.drone_pos[:, :2], axis=1)

    def _show_targets(self) -> None:
        for marker, xy in zip(self._markers, self._targets):
            self.data.mocap_pos[marker] = [*xy, self.flight_altitude]

    def _sample_stage(self) -> np.ndarray:
        shape = self.formation - self.formation.mean(axis=0)
        radius = np.max(np.linalg.norm(shape, axis=1))
        angle = self.np_random.uniform(-np.pi, np.pi)
        return self._sample_xy(margin=0.5 + radius) + shape @ rotation([np.cos(angle), np.sin(angle)]).T

    def _task_reset(self, options: dict) -> None:
        self._set_drone_states(self._sample_separated_xy(), self._sample_yaw())
        self._stages = np.stack([self._sample_stage() for _ in range(self.num_stages)])
        self._reached = 0
        self._show_targets()
        self._prev_dist = self._distances()

    def _task_target_geom(self, drone: int) -> int:
        return self._marker_geoms[drone]

    def _task_obs(self, drone: int) -> np.ndarray:
        return self._targets[drone] - self.drone_pos[drone, :2]

    def _task_step(self):
        dist = self._distances()
        reward = self.progress_weight * float(np.mean(self._prev_dist - dist))
        if np.all(dist < self.reach_radius):
            reward += self.stage_bonus
            self._reached += 1
            self._show_targets()
            dist = self._distances()
        self._prev_dist = dist
        return reward, self._reached == self.num_stages, self._task_info()

    def _task_info(self) -> dict:
        return {
            "stages_reached": self._reached,
            "distance": self._prev_dist.copy(),
            "is_success": self._reached == self.num_stages,
        }

    def goals(self):
        return self._targets.copy(), None
