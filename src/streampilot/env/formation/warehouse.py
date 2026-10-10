"""The waypoint, landing and tracking formation tasks set in a warehouse."""

import mujoco
import numpy as np

from streampilot.env.formation.hall import (
    DARK,
    ROBOT,
    SAFETY_YELLOW,
    STEEL,
    STEEL_LIGHT,
    WHITE,
    build_hall,
    prop,
)
from streampilot.env.formation.landing import FormationLandingEnv
from streampilot.env.formation.tracking import FormationTrackingEnv
from streampilot.env.formation.waypoint import FormationWaypointEnv

AGV_ORANGE = [0.95, 0.55, 0.1, 1.0]
STATUS_GREEN = [0.1, 0.85, 0.3, 1.0]


class FormationWarehouseWaypointEnv(FormationWaypointEnv):
    """``FormationWaypointEnv`` in a warehouse lined with stocked pallet racks: each waypoint is an
    inventory scan point, its ball (still the drone's detector target, unchanged) on a marker post
    standing on the floor. Task, reward and observations are unchanged; the default render camera is
    the ``hall`` overview."""

    def __init__(self, **kwargs):
        kwargs.setdefault("camera", "hall")
        super().__init__(**kwargs)

    def _build_scene(self, spec: mujoco.MjSpec) -> None:
        build_hall(spec, self.arena_half_extent, side_racks=True)
        super()._build_scene(spec)
        spec.geom("target").rgba = [0.0, 0.0, 0.0, 0.0]
        robot, w, h = spec.body("target"), self.target_half_width, self.target_half_height
        cylinder = mujoco.mjtGeom.mjGEOM_CYLINDER
        prop(robot, [w, w, 0.11], [0.0, 0.0, -h + 0.13], AGV_ORANGE)  # base
        prop(robot, [w, w, 0.025], [0.0, 0.0, -h + 0.025], DARK)  # bumper
        for x in (-1, 1):
            for y in (-1, 1):
                prop(robot, [0.045, 0.02], [x * (w - 0.05), y * (w - 0.02), -h + 0.045], DARK, cylinder,
                     quat=[0.7071, 0.7071, 0.0, 0.0])
        mast_bottom, head = -h + 0.24, h - 0.2
        prop(robot, [0.05, 0.05, (head - mast_bottom) / 2], [0.0, 0.0, (head + mast_bottom) / 2], STEEL)
        for z in np.linspace(mast_bottom + 0.2, head - 0.15, 4):  # scanner cameras, both sides
            for x in (-1, 1):
                prop(robot, [0.02, 0.07, 0.03], [x * 0.06, 0.0, z], DARK)
        prop(robot, [w, w, 0.2], [0.0, 0.0, h - 0.2], ROBOT)  # head
        prop(robot, [w + 0.002, w + 0.002, 0.02], [0.0, 0.0, h - 0.33], STATUS_GREEN)
        for i in range(self.num_drones):
            post = spec.body(f"waypoint{i}")  # at flight altitude
            floor = -self.flight_altitude
            prop(post, [0.015], None, STEEL_LIGHT, mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, 0, floor, 0, 0, -self.ball_radius])
            prop(post, [0.12, 0.01], [0.0, 0.0, floor + 0.01], SAFETY_YELLOW, mujoco.mjtGeom.mjGEOM_CYLINDER)


class FormationWarehouseLandingEnv(FormationLandingEnv):
    """``FormationLandingEnv`` in a warehouse lined with stocked pallet racks: the pad is a drone
    dock (dark, with a hazard border and an H). The pad's box, the detector target, is unchanged,
    as are the task, reward and observations; the default render camera is the ``hall`` overview."""

    def __init__(self, **kwargs):
        kwargs.setdefault("camera", "hall")
        super().__init__(**kwargs)

    def _build_scene(self, spec: mujoco.MjSpec) -> None:
        build_hall(spec, self.arena_half_extent, side_racks=True)
        super()._build_scene(spec)
        spec.geom("target").rgba = [0.0, 0.0, 0.0, 0.0]
        robot, w, h = spec.body("target"), self.target_half_width, self.target_half_height
        cylinder = mujoco.mjtGeom.mjGEOM_CYLINDER
        prop(robot, [w, w, 0.11], [0.0, 0.0, -h + 0.13], AGV_ORANGE)  # base
        prop(robot, [w, w, 0.025], [0.0, 0.0, -h + 0.025], DARK)  # bumper
        for x in (-1, 1):
            for y in (-1, 1):
                prop(robot, [0.045, 0.02], [x * (w - 0.05), y * (w - 0.02), -h + 0.045], DARK, cylinder,
                     quat=[0.7071, 0.7071, 0.0, 0.0])
        mast_bottom, head = -h + 0.24, h - 0.2
        prop(robot, [0.05, 0.05, (head - mast_bottom) / 2], [0.0, 0.0, (head + mast_bottom) / 2], STEEL)
        for z in np.linspace(mast_bottom + 0.2, head - 0.15, 4):  # scanner cameras, both sides
            for x in (-1, 1):
                prop(robot, [0.02, 0.07, 0.03], [x * 0.06, 0.0, z], DARK)
        prop(robot, [w, w, 0.2], [0.0, 0.0, h - 0.2], ROBOT)  # head
        prop(robot, [w + 0.002, w + 0.002, 0.02], [0.0, 0.0, h - 0.33], STATUS_GREEN)
        spec.geom("pad").rgba = [0.22, 0.23, 0.25, 1.0]
        pad, s = spec.body("pad"), self.pad_half_size
        top = self.pad_half_height + 0.001
        for sign in (-1, 1):  # hazard border
            prop(pad, [s + 0.04, 0.03, 0.002], [0.0, sign * (s + 0.01), top], SAFETY_YELLOW)
            prop(pad, [0.03, s + 0.04, 0.002], [sign * (s + 0.01), 0.0, top], SAFETY_YELLOW)
        for sign in (-1, 1):  # H
            prop(pad, [0.025, 0.4 * s, 0.002], [sign * 0.3 * s, 0.0, top], WHITE)
        prop(pad, [0.3 * s, 0.025, 0.002], [0.0, 0.0, top], WHITE)
        prop(pad, [s + 0.12, s + 0.12, 0.004], [0.0, 0.0, -self.pad_half_height + 0.004], DARK)  # dock plinth


class FormationWarehouseTrackingEnv(FormationTrackingEnv):
    """``FormationTrackingEnv`` in a warehouse lined with stocked pallet racks, the target dressed as
    a mobile inventory robot (wheeled base, mast with scanners, head) that fills the red pillar's
    box. The pillar, now invisible, is still the detector target, and the hall and robot are visual
    only, so task, reward and observations are unchanged and policies trained on
    ``FormationTrackingEnv`` transfer as they are; the default render camera is the ``hall`` overview."""

    def __init__(self, **kwargs):
        kwargs.setdefault("camera", "hall")
        super().__init__(**kwargs)

    def _build_scene(self, spec: mujoco.MjSpec) -> None:
        build_hall(spec, self.arena_half_extent, side_racks=True)
        super()._build_scene(spec)
        spec.geom("target").rgba = [0.0, 0.0, 0.0, 0.0]
        robot, w, h = spec.body("target"), self.target_half_width, self.target_half_height
        cylinder = mujoco.mjtGeom.mjGEOM_CYLINDER
        prop(robot, [w, w, 0.11], [0.0, 0.0, -h + 0.13], AGV_ORANGE)  # base
        prop(robot, [w, w, 0.025], [0.0, 0.0, -h + 0.025], DARK)  # bumper
        for x in (-1, 1):
            for y in (-1, 1):
                prop(robot, [0.045, 0.02], [x * (w - 0.05), y * (w - 0.02), -h + 0.045], DARK, cylinder,
                     quat=[0.7071, 0.7071, 0.0, 0.0])
        mast_bottom, head = -h + 0.24, h - 0.2
        prop(robot, [0.05, 0.05, (head - mast_bottom) / 2], [0.0, 0.0, (head + mast_bottom) / 2], STEEL)
        for z in np.linspace(mast_bottom + 0.2, head - 0.15, 4):  # scanner cameras, both sides
            for x in (-1, 1):
                prop(robot, [0.02, 0.07, 0.03], [x * 0.06, 0.0, z], DARK)
        prop(robot, [w, w, 0.2], [0.0, 0.0, h - 0.2], ROBOT)  # head
        prop(robot, [w + 0.002, w + 0.002, 0.02], [0.0, 0.0, h - 0.33], STATUS_GREEN)
