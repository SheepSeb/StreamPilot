"""Factory escort: a formation follows and inspects an AGV carrying end-of-life products through a
remanufacturing hall."""

from collections.abc import Sequence

import mujoco
import numpy as np

from streampilot.env.formation.hall import (
    DARK,
    SAFETY_YELLOW,
    build_hall,
    prop,
)
from streampilot.env.formation.tracking import FormationTrackingEnv

# End-of-life products the AGV carries: half-size (x along the AGV, y, z) in metres and colour.
PRODUCTS = {
    "ev_battery": ((0.30, 0.20, 0.10), (0.15, 0.25, 0.55)),
    "weee_crate": ((0.22, 0.22, 0.18), (0.20, 0.60, 0.25)),
    "e_motor": ((0.14, 0.14, 0.14), (0.55, 0.55, 0.60)),
    "solar_panel": ((0.40, 0.26, 0.03), (0.10, 0.12, 0.20)),
    "gearbox": ((0.20, 0.15, 0.15), (0.60, 0.35, 0.15)),
}
AGV_HALF_SIZE = (0.35, 0.25, 0.12)
NUM_STATIONS = 4  # disassembly, inspection, sorting, storage: the corners of the AGV's loop
LANE_HALF_WIDTH = 0.04
PRODUCT_BASE = 2 * AGV_HALF_SIZE[2] + 0.055  # top of the AGV's load deck
AGV_ORANGE = [0.95, 0.55, 0.1, 1.0]


class FormationFactoryEnv(FormationTrackingEnv):
    """``FormationTrackingEnv`` set in a remanufacturing hall. The team escorts an AGV (an orange
    mobile robot) that carries one end-of-life product per episode (EV battery module, WEEE crate,
    electric motor, solar panel or gearbox, with its colour worn by ``product_color_noise``) around a
    rectangular intralogistics loop marked on the floor. At each corner (a station) the AGV stops
    for a random ``station_dwell`` (seconds), then turns and drives on, accelerating and braking at ``agv_accel`` up to a cruise speed drawn per episode from
    ``agv_speed_range`` (fractions of ``target_max_speed``), clockwise or not. The loop's size and
    position are drawn per episode, and the AGV starts at a random point on it. Robot workcells
    stand outside the arena's corners, where the drones cannot fly.

    The detector target is the product, so its apparent size changes with the product: with
    ``products`` (a subset of ``PRODUCTS``) the catalogue can drift between training phases, for
    continual learning under product variability. ``reset(options={"product": name})`` picks one.

    Slots, reward, termination and observations are those of ``FormationTrackingEnv`` (the leader
    holds ``follow_distance`` from the AGV, the others their formation slots, every drone faces it).
    ``info["inspection_coverage"]`` is the fraction of ``inspection_sectors`` equal sectors around
    the product (in its frame) from which some drone has seen it so far: a multi-view inspection
    measure, reported but not rewarded. ``info["product"]`` is the product's index in ``PRODUCTS``.
    """

    def __init__(
        self,
        products: Sequence[str] | None = None,
        product_color_noise: float = 0.1,
        agv_speed_range: tuple[float, float] = (0.5, 1.0),
        agv_accel: float = 0.5,
        station_dwell: tuple[float, float] = (0.0, 3.0),
        inspection_sectors: int = 8,
        arena_half_extent: float = 6.0,
        **kwargs,
    ):
        self.products = list(PRODUCTS) if products is None else list(products)
        assert self.products and all(p in PRODUCTS for p in self.products), f"products must be in {list(PRODUCTS)}"
        self.product_color_noise = product_color_noise
        self.agv_speed_range = agv_speed_range
        self.agv_accel = agv_accel
        self.station_dwell = station_dwell
        self.inspection_sectors = inspection_sectors
        kwargs.setdefault("camera", "team")
        super().__init__(arena_half_extent=arena_half_extent, **kwargs)
        self._product_geom = self._target_geom
        self._lanes = [self.model.geom(f"lane{k}").id for k in range(NUM_STATIONS)]
        self._lane_bodies = [self._mocap_id(f"lane{k}") for k in range(NUM_STATIONS)]
        self._corners = np.zeros((NUM_STATIONS, 2))
        self._covered = np.zeros(inspection_sectors, dtype=bool)

    def _build_scene(self, spec: mujoco.MjSpec) -> None:
        build_hall(spec, self.arena_half_extent, corner_cells=True)
        # Default render camera: moves with the leader (without turning), behind and above the team.
        spec.body("d0_x2").add_camera(
            name="team",
            pos=[0.0, -4.5, 3.5],
            xyaxes=[1, 0, 0, 0, 0.6, 0.8],
            mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM,
            fovy=55,
        )
        agv = spec.worldbody.add_body(name="target", mocap=True)
        prop(agv, AGV_HALF_SIZE, [0.0, 0.0, AGV_HALF_SIZE[2] + 0.04], AGV_ORANGE, name="agv")
        for x in (-0.25, 0.25):  # wheels
            for y in (-0.26, 0.26):
                prop(agv, [0.06, 0.03], [x, y, 0.06], DARK, mujoco.mjtGeom.mjGEOM_CYLINDER, quat=[1, 1, 0, 0])
        prop(agv, [0.36, 0.26, 0.015], [0.0, 0.0, 2 * AGV_HALF_SIZE[2] + 0.04], DARK)  # load deck
        prop(agv, [0.05, 0.05, 0.03], [0.3, 0.0, 2 * AGV_HALF_SIZE[2] + 0.07], [0.1, 0.8, 0.3, 1.0])  # status light
        size, color = PRODUCTS[self.products[0]]
        prop(agv, size, [0.0, 0.0, PRODUCT_BASE + size[2]], [*color, 1.0], name="target")
        for k in range(NUM_STATIONS):
            # Lanes are resized to the episode's loop in ``_task_reset``.
            lane = spec.worldbody.add_body(name=f"lane{k}", mocap=True)
            prop(lane, [1.0, LANE_HALF_WIDTH, 0.002], [0.0, 0.0, 0.0], SAFETY_YELLOW, name=f"lane{k}")

    def _set_box_size(self, geom: int, size) -> None:
        self.model.geom_size[geom] = size
        self.model.geom_aabb[geom, 3:] = size  # used by the detector's projection
        self.model.geom_rbound[geom] = np.linalg.norm(size)

    def _task_reset(self, options: dict) -> None:
        rng = self.np_random
        # The product: its shape, worn colour and height on the AGV.
        name = options.get("product", self.products[rng.integers(len(self.products))])
        self._product = list(PRODUCTS).index(name)
        size, color = PRODUCTS[name]
        self._set_box_size(self._product_geom, size)
        self.model.geom_pos[self._product_geom, 2] = PRODUCT_BASE + size[2]
        worn = np.clip(np.array(color) + rng.normal(scale=self.product_color_noise, size=3), 0.0, 1.0)
        self.model.geom_rgba[self._product_geom, :3] = worn

        # The loop: a rectangle inside the target's square, its corners the stations.
        ext = self.target_half_extent
        half = rng.uniform(0.5, 1.0, size=2) * ext
        center = rng.uniform(-1.0, 1.0, size=2) * (ext - half)
        signs = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
        corners = center + signs * half
        self._corners = corners if rng.random() < 0.5 else corners[::-1].copy()
        for k in range(NUM_STATIONS):
            a, b = corners[k], corners[(k + 1) % NUM_STATIONS]
            edge = np.abs(b - a) / 2 + LANE_HALF_WIDTH
            self._set_box_size(self._lanes[k], [*edge, 0.002])
            self.data.mocap_pos[self._lane_bodies[k]] = [*(a + b) / 2, 0.002]

        # The AGV: somewhere on the loop, heading for the next station, at rest.
        self._leg = int(rng.integers(NUM_STATIONS))
        start = self._corners[self._leg - 1] + rng.uniform() * (self._corners[self._leg] - self._corners[self._leg - 1])
        self._cruise = rng.uniform(*self.agv_speed_range) * self.target_max_speed
        self._speed, self._dwell = 0.0, 0.0
        self._target_vel = np.zeros(3)
        self.data.mocap_pos[self._target] = [*start, 0.0]
        self._face(self._corners[self._leg] - start)
        self._covered[:] = False
        self._spawn_team(start)

    def _face(self, direction) -> None:
        yaw = np.arctan2(direction[1], direction[0])
        self._agv_yaw = yaw
        self.data.mocap_quat[self._target] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]

    def _task_before_step(self) -> None:
        self._target_vel[:2] = 0.0
        if self._dwell > 0.0:
            self._dwell -= self.dt
            return
        pos = self._target_pos[:2]
        goal = self._corners[self._leg]
        remaining = np.linalg.norm(goal - pos)
        # Accelerate to cruise, braking to stop at the station.
        self._speed = min(self._cruise, self._speed + self.agv_accel * self.dt, np.sqrt(2 * self.agv_accel * remaining))
        step = self._speed * self.dt
        if step >= remaining:
            self.data.mocap_pos[self._target][:2] = goal
            self._speed = 0.0
            self._dwell = self.np_random.uniform(*self.station_dwell)
            self._leg = (self._leg + 1) % NUM_STATIONS
            self._face(self._corners[self._leg] - goal)
            return
        direction = (goal - pos) / remaining
        self.data.mocap_pos[self._target][:2] = pos + step * direction
        self._target_vel[:2] = self._speed * direction

    def _get_info(self, info, dets):
        info = super()._get_info(info, dets)
        # Mark the sector (in the product's frame) of every drone that sees the product.
        rel = self.drone_pos[:, :2] - self._target_pos[:2]
        bearing = (np.arctan2(rel[:, 1], rel[:, 0]) - self._agv_yaw) % (2 * np.pi)
        sectors = (bearing / (2 * np.pi) * self.inspection_sectors).astype(int) % self.inspection_sectors
        self._covered[sectors[dets[:, 0] > 0]] = True
        info["inspection_coverage"] = float(self._covered.mean())
        info["product"] = self._product
        return info
