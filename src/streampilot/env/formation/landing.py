"""Formation landing approach: the team hovers in formation in front of one pad, then hands over."""

import mujoco
import numpy as np

from streampilot.env.formation.base import DRONE_COLORS, FormationBaseEnv, rotation, unit


class FormationLandingEnv(FormationBaseEnv):
    """The team version of ``LandingEnv``: one pad (the detector target of every drone), and the
    drones land in the formation shape, a column for two drones and a triangle for three.

    The formation frame points from the leader (drone 0) towards the pad, so the approach
    direction is free, as in the single-drone task. The leader lands on the pad; drone ``i``
    lands on the spot ``formation[i]`` in that frame, behind it (shown as a faint disc in the
    drone's colour). Each drone hovers ``handoff_distance`` in front of its landing spot (along
    the formation's forward axis), facing the pad with the pad in view. Once all of them have
    held that, slowly, for ``hold_steps`` steps, each hands over to its flight controller's own
    landing with ``info["handoff_offset"][i]`` (body frame, to its landing spot).

    Reward: mean progress on ``slot distance + heading_weight * |heading error|``, plus
    ``view_reward`` times the fraction of drones seeing the pad and ``handoff_bonus`` on success,
    which ends the episode. Privileged task obs: horizontal vector to the pad centre.
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
        self._spot_markers = [self._mocap_id(f"spot{i}") for i in range(1, self.num_drones)]
        self._prev_error = np.zeros(self.num_drones)
        self._held = 0

    def _build_scene(self, spec: mujoco.MjSpec) -> None:
        pad = spec.worldbody.add_body(name="pad", mocap=True)
        pad.add_geom(
            name="pad",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[self.pad_half_size, self.pad_half_size, self.pad_half_height],
            rgba=[0.9, 0.6, 0.1, 1.0],
        )
        for i in range(1, self.num_drones):
            spot = spec.worldbody.add_body(name=f"spot{i}", mocap=True)
            spot.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[0.15, 0.002, 0],
                rgba=[*DRONE_COLORS[i], 0.35],
                contype=0,
                conaffinity=0,
            )

    @property
    def _pad_top(self) -> np.ndarray:
        return self.data.mocap_pos[self._pad] + [0.0, 0.0, self.pad_half_height]

    def _forward(self) -> np.ndarray:
        return unit(self._pad_top[:2] - self.drone_pos[0, :2])

    def landing_spots(self) -> np.ndarray:
        """``(num_drones, 2)``: where each drone lands, the leader on the pad."""
        return self._pad_top[:2] + self.formation @ rotation(self._forward()).T

    def _slots(self, spots=None) -> np.ndarray:
        spots = self.landing_spots() if spots is None else spots
        return spots - self.handoff_distance * self._forward()

    def _slot_distances(self, spots=None) -> np.ndarray:
        return np.linalg.norm(self._slots(spots) - self.drone_pos[:, :2], axis=1)

    def _approach_error(self) -> np.ndarray:
        return self._slot_distances() + self.heading_weight * np.abs(self._heading_error(self._pad_top))

    def _pad_in_view(self) -> np.ndarray:
        return np.array([self._point_in_view(i, self._pad_top) for i in range(self.num_drones)])

    def _task_reset(self, options: dict) -> None:
        self._set_drone_states(self._sample_separated_xy(), self._sample_yaw())
        self.data.mocap_pos[self._pad] = [*self._sample_xy(), self.pad_half_height]
        self._update_spots()
        self._prev_error = self._approach_error()
        self._held = 0

    def _update_spots(self, spots=None) -> None:
        spots = self.landing_spots() if spots is None else spots
        for marker, xy in zip(self._spot_markers, spots[1:]):
            self.data.mocap_pos[marker] = [*xy, 0.002]

    def _task_target_geom(self, drone: int) -> int:
        return self._pad_geom

    def _task_obs(self, drone: int) -> np.ndarray:
        return self._pad_top[:2] - self.drone_pos[drone, :2]

    def _in_handoff_formation(self) -> bool:
        return bool(
            np.all(self._slot_distances() <= self.distance_tolerance)
            and np.all(np.abs(self._heading_error(self._pad_top)) <= self.heading_tolerance)
            and np.all(np.linalg.norm(self.drone_vel[:, :2], axis=1) <= self.max_handoff_speed)
            and np.all(self._pad_in_view())
        )

    def _task_step(self):
        # Everything below is evaluated at the same state: compute each term once.
        spots = self.landing_spots()
        self._update_spots(spots)
        slot_distances = self._slot_distances(spots)
        heading_error = np.abs(self._heading_error(self._pad_top))
        pad_in_view = self._pad_in_view()
        error = slot_distances + self.heading_weight * heading_error
        reward = self.progress_weight * float(np.mean(self._prev_error - error))
        self._prev_error = error
        reward += self.view_reward * float(np.mean(pad_in_view))
        in_formation = bool(
            np.all(slot_distances <= self.distance_tolerance)
            and np.all(heading_error <= self.heading_tolerance)
            and np.all(np.linalg.norm(self.drone_vel[:, :2], axis=1) <= self.max_handoff_speed)
            and np.all(pad_in_view)
        )
        self._held = self._held + 1 if in_formation else 0
        success = self._held >= self.hold_steps
        if success:
            reward += self.handoff_bonus
        return reward, success, self._task_info(spots, slot_distances, heading_error)

    def _task_info(self, spots=None, slot_distances=None, heading_error=None) -> dict:
        spots = self.landing_spots() if spots is None else spots
        return {
            "distance": np.linalg.norm(self._pad_top[:2] - self.drone_pos[:, :2], axis=1),
            "formation_error": self._slot_distances(spots) if slot_distances is None else slot_distances,
            "heading_error": np.abs(self._heading_error(self._pad_top)) if heading_error is None else heading_error,
            "handoff_offset": self._body_frame(spots - self.drone_pos[:, :2]),
            "is_success": self._held >= self.hold_steps,
        }

    def goals(self):
        return self._slots(), self._pad_top[:2].copy()
