"""Landing approach: fly to the hand-off point in front of a pad on the floor, then hand over."""

import mujoco
import numpy as np

from streampilot.env.base import DroneBaseEnv


class LandingEnv(DroneBaseEnv):
    """The drone flies at a fixed altitude, so it cannot land itself. Its job is the visual
    approach: find the pad (the detector target), then hover ``handoff_distance`` (horizontal)
    from it, facing it, with the pad centre in the camera view, for ``hold_steps`` steps. At that
    point a real drone hands over to the flight controller's own landing, flying the remaining
    ``info["handoff_offset"]`` (body frame) and descending. The level camera loses the pad centre
    once the drone is closer than ~1.4x its height above the pad, so the hand-off has to happen
    from a distance.

    Reward: progress on ``|distance - handoff_distance| + heading_weight * |heading error|``, plus
    ``view_reward`` per step with the pad in view and ``handoff_bonus`` on success, which ends the
    episode. Privileged task obs: horizontal vector to the pad centre.
    """

    task_obs_dim = 2

    def __init__(
        self,
        flight_altitude: float = 0.6,
        pad_half_size: float = 0.25,
        handoff_distance: float = 1.2,
        distance_tolerance: float = 0.15,
        heading_tolerance_deg: float = 10.0,
        max_handoff_speed: float = 0.15,
        hold_steps: int = 10,
        progress_weight: float = 10.0,
        heading_weight: float = 0.5,
        view_reward: float = 0.05,
        handoff_bonus: float = 20.0,
        **kwargs,
    ):
        self.pad_half_size = pad_half_size
        self.pad_half_height = 0.01
        self.handoff_distance = handoff_distance
        self.distance_tolerance = distance_tolerance
        self.heading_tolerance = np.deg2rad(heading_tolerance_deg)
        self.max_handoff_speed = max_handoff_speed
        self.hold_steps = hold_steps
        self.progress_weight = progress_weight
        self.heading_weight = heading_weight
        self.view_reward = view_reward
        self.handoff_bonus = handoff_bonus
        super().__init__(flight_altitude=flight_altitude, **kwargs)
        self._pad = self._mocap_id("pad")
        self._pad_geom = self.model.geom("pad").id
        self._prev_error = 0.0
        self._held = 0

    def _build_scene(self, spec: mujoco.MjSpec) -> None:
        pad = spec.worldbody.add_body(name="pad", mocap=True)
        pad.add_geom(
            name="pad",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[self.pad_half_size, self.pad_half_size, self.pad_half_height],
            rgba=[0.9, 0.6, 0.1, 1.0],
        )

    @property
    def _pad_top(self) -> np.ndarray:
        return self.data.mocap_pos[self._pad] + [0.0, 0.0, self.pad_half_height]

    def _distance(self) -> float:
        return float(np.linalg.norm(self._pad_top[:2] - self.drone_pos[:2]))

    def _approach_error(self) -> float:
        heading_error = abs(self._heading_error(self._pad_top))
        return abs(self._distance() - self.handoff_distance) + self.heading_weight * heading_error

    def _task_reset(self, options: dict) -> None:
        self._set_drone_state(self._sample_xy(), self._sample_yaw())
        self.data.mocap_pos[self._pad] = [*self._sample_xy(), self.pad_half_height]
        self._prev_error = self._approach_error()
        self._held = 0

    def _task_target_geom(self) -> int:
        return self._pad_geom

    def _task_obs(self) -> np.ndarray:
        return (self._pad_top - self.drone_pos)[:2]

    def _in_handoff_zone(self) -> bool:
        return (
            abs(self._distance() - self.handoff_distance) <= self.distance_tolerance
            and abs(self._heading_error(self._pad_top)) <= self.heading_tolerance
            and np.linalg.norm(self.drone_vel[:2]) <= self.max_handoff_speed
            and self._point_in_view(self._pad_top)
        )

    def _task_step(self):
        error = self._approach_error()
        reward = self.progress_weight * (self._prev_error - error)
        self._prev_error = error
        if self._point_in_view(self._pad_top):
            reward += self.view_reward
        self._held = self._held + 1 if self._in_handoff_zone() else 0
        success = self._held >= self.hold_steps
        if success:
            reward += self.handoff_bonus
        return reward, success, self._task_info()

    def _task_info(self) -> dict:
        rel = self._pad_top[:2] - self.drone_pos[:2]
        c, s = np.cos(self.drone_yaw), np.sin(self.drone_yaw)
        return {
            "distance": self._distance(),
            "heading_error": abs(self._heading_error(self._pad_top)),
            "handoff_offset": np.array([c * rel[0] + s * rel[1], -s * rel[0] + c * rel[1]]),
            "is_success": self._held >= self.hold_steps,
        }
