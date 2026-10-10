"""The single-drone tasks set in the warehouse of the formation tasks. Everything added is visual only: task,
reward, detection and observations are those of the plain task, so policies trained there fly these as they are.
The default render camera is the ``hall`` overview."""

import mujoco

from streampilot.env.formation.hall import build_hall
from streampilot.env.formation.warehouse import (
    drone_dock,
    inventory_robot,
    scan_point_post,
)
from streampilot.env.landing import LandingEnv
from streampilot.env.morphing import MorphingTrackingEnv
from streampilot.env.tracking import TrackingEnv
from streampilot.env.waypoint import WaypointEnv


class WarehouseWaypointEnv(WaypointEnv):
    """``WaypointEnv`` in the warehouse: the waypoint's ball (the detector target) stands on a scan-point post."""

    def __init__(self, **kwargs):
        kwargs.setdefault("camera", "hall")
        super().__init__(**kwargs)

    def _build_scene(self, spec: mujoco.MjSpec) -> None:
        build_hall(spec, self.arena_half_extent, side_racks=True)
        super()._build_scene(spec)
        scan_point_post(spec.body("waypoint"), -self.flight_altitude, -self.reach_radius)  # the ball is at flight altitude


class WarehouseLandingEnv(LandingEnv):
    """``LandingEnv`` in the warehouse: the pad (the detector target) is a drone dock."""

    def __init__(self, **kwargs):
        kwargs.setdefault("camera", "hall")
        super().__init__(**kwargs)

    def _build_scene(self, spec: mujoco.MjSpec) -> None:
        build_hall(spec, self.arena_half_extent, side_racks=True)
        super()._build_scene(spec)
        drone_dock(spec, "pad", self.pad_half_size, self.pad_half_height)


class WarehouseTrackingEnv(TrackingEnv):
    """``TrackingEnv`` in the warehouse: the pillar, now invisible and still the detector target, is filled by a
    mobile inventory robot."""

    def __init__(self, **kwargs):
        kwargs.setdefault("camera", "hall")
        super().__init__(**kwargs)

    def _build_scene(self, spec: mujoco.MjSpec) -> None:
        build_hall(spec, self.arena_half_extent, side_racks=True)
        super()._build_scene(spec)
        inventory_robot(spec, "target", self.target_half_width, self.target_half_height)


class WarehouseMorphingTrackingEnv(MorphingTrackingEnv):
    """``MorphingTrackingEnv`` in the warehouse. The target keeps its red shapes, since changing them is the task."""

    def __init__(self, **kwargs):
        kwargs.setdefault("camera", "hall")
        super().__init__(**kwargs)

    def _build_scene(self, spec: mujoco.MjSpec) -> None:
        build_hall(spec, self.arena_half_extent, side_racks=True)
        super()._build_scene(spec)
