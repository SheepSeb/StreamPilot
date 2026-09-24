"""Waypoint reaching: visit a random sequence of waypoints in order."""

import mujoco
import numpy as np

from streampilot.env.base import DroneBaseEnv


class WaypointEnv(DroneBaseEnv):
    """Only the current waypoint is shown (an opaque green ball at flight altitude, the detector
    target); the next one appears once it is reached. Privileged task obs: horizontal vector from
    the drone to the waypoint.

    Reward: progress towards the current waypoint, plus a bonus for each waypoint reached.
    The episode terminates successfully once the last waypoint is reached.
    """

    task_obs_dim = 2

    def __init__(
        self,
        num_waypoints: int = 3,
        reach_radius: float = 0.15,
        progress_weight: float = 10.0,
        reach_bonus: float = 10.0,
        **kwargs,
    ):
        self.num_waypoints = num_waypoints
        self.reach_radius = reach_radius
        self.progress_weight = progress_weight
        self.reach_bonus = reach_bonus
        super().__init__(**kwargs)
        self._marker = self._mocap_id("waypoint")
        self._marker_geom = self.model.geom("waypoint").id
        self._waypoints = np.zeros((num_waypoints, 3))
        self._reached = 0
        self._prev_dist = 0.0

    def _build_scene(self, spec: mujoco.MjSpec) -> None:
        marker = spec.worldbody.add_body(name="waypoint", mocap=True)
        marker.add_geom(
            name="waypoint",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[self.reach_radius, 0, 0],
            rgba=[0.1, 0.9, 0.2, 1.0],
            contype=0,
            conaffinity=0,
        )

    @property
    def _target(self) -> np.ndarray:
        # Stays on the last waypoint after it is reached, so the final observation is valid.
        return self._waypoints[min(self._reached, self.num_waypoints - 1)]

    def _distance(self) -> float:
        return float(np.linalg.norm(self._target - self.drone_pos))

    def _task_reset(self, options: dict) -> None:
        self._set_drone_state(self._sample_xy(), self._sample_yaw())
        self._waypoints = np.stack([[*self._sample_xy(), self.flight_altitude] for _ in range(self.num_waypoints)])
        self._reached = 0
        self.data.mocap_pos[self._marker] = self._target
        self._prev_dist = self._distance()

    def _task_target_geom(self) -> int:
        return self._marker_geom

    def _task_obs(self) -> np.ndarray:
        return (self._target - self.drone_pos)[:2]

    def _task_step(self):
        dist = self._distance()
        reward = self.progress_weight * (self._prev_dist - dist)
        if dist < self.reach_radius:
            reward += self.reach_bonus
            self._reached += 1
            self.data.mocap_pos[self._marker] = self._target
            dist = self._distance()
        self._prev_dist = dist
        return reward, self._reached == self.num_waypoints, self._task_info()

    def _task_info(self) -> dict:
        return {
            "waypoints_reached": self._reached,
            "distance": self._prev_dist,
            "is_success": self._reached == self.num_waypoints,
        }
