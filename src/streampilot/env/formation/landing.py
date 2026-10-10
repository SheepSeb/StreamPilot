"""Formation landing approach: the team hovers in formation in front of one pad, then hands over."""

import mujoco
import numpy as np

from streampilot.env.formation.base import (
    DRONE_COLORS,
    FormationBaseEnv,
    rotation,
    unit,
)


class FormationLandingEnv(FormationBaseEnv):
    """The team version of ``LandingEnv``: one pad (the detector target of every drone), and the
    team hovers in ``formation_shape`` before handing over:

    - ``"arc"`` (default): every drone ``handoff_distance`` from the pad, facing it, on an arc with
      the leader in the middle and the followers ``formation_spacing`` apart along it (left, right,
      then further out). A follower's slot is then what the leader already learns (the pad at the
      hand-off distance, judged from its box) plus the leader at ``formation_spacing``, which its
      teammate observation gives exactly. Followers land on spots ``formation_spacing`` from the pad
      in their direction.
    - ``"abreast"``: side by side, the followers ``formation_spacing`` to the leader's side. Their
      slot depends on the leader-to-pad direction, which they can only judge from a small pad box
      2 m away; they stopped 0.4-0.5 m short of it.
    - ``"triangle"``: the base formation shape, the followers far behind the pad.

    The formation frame points from the leader (drone 0) towards the pad, so the approach
    direction is free, as in the single-drone task. The leader lands on the pad; drone ``i``
    lands on the spot ``formation[i]`` in that frame, behind it (shown as a faint disc in the
    drone's colour). The leader hovers ``handoff_distance`` in front of the pad, and each follower
    at its formation offset from the leader's actual position (so it can hold its slot from what it
    observes, the leader's relative position and the pad's bearing); once the leader is in place,
    that is ``handoff_distance`` in front of the follower's landing spot. All of them face the pad
    with the pad in view. Once all of them have
    held that, slowly, for ``hold_steps`` steps, each hands over to its flight controller's own
    landing with ``info["handoff_offset"][i]`` (body frame, to its landing spot). Since the
    flight controller gets that exact offset, the hand-off only needs the team roughly in place and
    steady, so the tolerances are loose (0.6 m, 25 deg, 0.4 m/s for 3 steps): at 0.25 m, 10 deg,
    0.15 m/s for 10 steps, policies that held all three drones near their slots never succeeded.

    Reward: mean progress on ``slot distance + heading_weight * |heading error|``, plus
    potential-based shaping on ``view_reward`` times the fraction of drones seeing the pad and
    ``slot_reward`` times the fraction meeting their own share of the hand-off condition (in the
    slot, facing the pad, slow, pad in view): gaining either pays once, losing it costs as much,
    and hovering pays nothing (paid every step, it made never handing off pay better than the
    bonus). Since "slow" is part of a drone's share, speeding up in the slot loses it again. Plus
    ``handoff_bonus`` on success, which ends the episode. Privileged task obs: horizontal vector to
    the pad centre.

    Episodes start with every hand-off slot inside the arena (by ``slot_margin``), so the formation
    the leader's approach implies is reachable. As in ``FormationWaypointEnv``, the drones spawn
    further apart, each roughly facing the pad, and pay a wall cost near the arena edge, a small
    proximity cost and the separation shaping (the base env's collision penalty, as in tracking), and the formation is wider
    than the base env's, so drones within the hand-off tolerance of neighbouring slots stay apart.
    """

    task_obs_dim = 2

    def __init__(
        self,
        flight_altitude: float = 0.6,
        pad_half_size: float = 0.25,
        handoff_distance: float = 1.5,
        distance_tolerance: float = 0.6,
        heading_tolerance_deg: float = 25.0,
        max_handoff_speed: float = 0.4,
        hold_steps: int = 3,
        progress_weight: float = 10.0,
        heading_weight: float = 0.5,
        view_reward: float = 1.0,
        handoff_bonus: float = 20.0,
        formation_shape: str = "arc",
        slot_reward: float = 3.0,
        slot_margin: float = 0.3,
        spawn_separation: float = 1.5,
        formation_spacing: float = 1.5,
        spawn_heading_noise_deg: float | None = 30.0,
        wall_weight: float = 2.0,
        geofence: float | None = 0.3,
        separation_weight: float = 5.0,
        proximity_weight: float = 0.25,
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
        self.slot_reward = slot_reward
        self.slot_margin = slot_margin
        super().__init__(
            flight_altitude=flight_altitude,
            spawn_separation=spawn_separation,
            formation_spacing=formation_spacing,
            spawn_heading_noise_deg=spawn_heading_noise_deg,
            wall_weight=wall_weight,
            geofence=geofence,
            proximity_weight=proximity_weight,
            separation_weight=separation_weight,
            **kwargs,
        )
        assert formation_shape in ("arc", "abreast", "triangle")
        self.formation_shape = formation_shape
        side = np.array([0, 1, -1, 2, -2])[: self.num_drones]  # leader, left, right, further out
        if formation_shape == "abreast":  # x forward (towards the pad), y to the left
            self.formation = np.column_stack([np.zeros(self.num_drones), formation_spacing * side])
        # Arc: angle along the circle around the pad between neighbours formation_spacing apart.
        self._arc_angles = side * 2 * np.arcsin(min(formation_spacing / (2 * handoff_distance), 1.0))
        self._pad = self._mocap_id("pad")
        self._pad_geom = self.model.geom("pad").id
        self._spot_markers = [self._mocap_id(f"spot{i}") for i in range(1, self.num_drones)]
        self._prev_error = np.zeros(self.num_drones)
        self._held = 0
        self._bonus_potential = 0.0

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

    def _arc_directions(self) -> np.ndarray:
        """Unit vectors from the pad to each drone's place on the arc, the leader's towards itself."""
        angle = np.arctan2(*(-self._forward())[::-1]) + self._arc_angles
        return np.column_stack([np.cos(angle), np.sin(angle)])

    def landing_spots(self) -> np.ndarray:
        """``(num_drones, 2)``: where each drone lands, the leader on the pad."""
        if self.formation_shape == "arc":
            return self._pad_top[:2] + self.formation_spacing * self._arc_directions() * (self._arc_angles != 0)[:, None]
        return self._pad_top[:2] + self.formation @ rotation(self._forward()).T

    def _slots(self, spots=None) -> np.ndarray:
        """Where each drone hovers: the leader ``handoff_distance`` in front of the pad, each follower
        at its formation offset from the leader's actual position (in the frame facing the pad)."""
        if self.formation_shape == "arc":
            return self._pad_top[:2] + self.handoff_distance * self._arc_directions()
        spots = self.landing_spots() if spots is None else spots
        forward = self._forward()
        slots = spots - self.handoff_distance * forward
        slots[1:] = self.drone_pos[0, :2] + self.formation[1:] @ rotation(forward).T
        return slots

    def _slot_distances(self, spots=None) -> np.ndarray:
        return np.linalg.norm(self._slots(spots) - self.drone_pos[:, :2], axis=1)

    def _approach_error(self) -> np.ndarray:
        return self._slot_distances() + self.heading_weight * np.abs(self._heading_error(self._pad_top))

    def _pad_in_view(self) -> np.ndarray:
        return np.array([self._point_in_view(i, self._pad_top) for i in range(self.num_drones)])

    def _task_reset(self, options: dict) -> None:
        limit = self.arena_half_extent - self.slot_margin
        for _ in range(1000):  # a wide team may not always fit: then take the last draw
            self._set_drone_states(self._sample_separated_xy(), 0.0)
            self.data.mocap_pos[self._pad] = [*self._sample_xy(), self.pad_half_height]
            if np.max(np.abs(self._slots())) <= limit:
                break
        pad = np.tile(self._pad_top[:2], (self.num_drones, 1))
        self._set_drone_states(self.drone_pos[:, :2].copy(), self._sample_yaw(look_at=pad))
        self._update_spots()
        self._prev_error = self._approach_error()
        self._held = 0
        self._bonus_potential = self.view_reward * float(np.mean(self._pad_in_view()))

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
        speed = np.linalg.norm(self.drone_vel[:, :2], axis=1)
        # Each drone's share of the hand-off condition: in its slot, facing the pad, slow, pad in view.
        in_slot = (
            (slot_distances <= self.distance_tolerance)
            & (heading_error <= self.heading_tolerance)
            & (speed <= self.max_handoff_speed)
            & pad_in_view
        )
        in_formation = bool(np.all(in_slot))
        self._held = self._held + 1 if in_formation else 0
        success = self._held >= self.hold_steps
        if success:
            reward += self.handoff_bonus
        # Potential-based (gamma * phi(s') - phi(s)) on the pad in view and each drone's share of the
        # hand-off: gaining them pays once, losing them costs, and hovering pays nothing, so the team
        # cannot farm them by never handing off. The hand-off (success) has potential 0.
        potential = 0.0 if success else self.view_reward * float(np.mean(pad_in_view)) + self.slot_reward * float(np.mean(in_slot))
        reward += self.shaping_gamma * potential - self._bonus_potential
        self._bonus_potential = potential
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
