"""A purely visual industrial hall around the arena, shared by the factory and warehouse tasks."""

import mujoco
import numpy as np

SAFETY_YELLOW = [0.95, 0.8, 0.1, 1.0]
DARK = [0.12, 0.12, 0.13, 1.0]
STEEL = [0.3, 0.33, 0.37, 1.0]
STEEL_LIGHT = [0.62, 0.64, 0.67, 1.0]
WALL = [0.82, 0.82, 0.8, 1.0]
WALL_BAND = [0.35, 0.45, 0.55, 1.0]
LAMP = [1.0, 1.0, 0.92, 1.0]
RACK_BLUE = [0.15, 0.3, 0.6, 1.0]
RACK_ORANGE = [0.9, 0.45, 0.1, 1.0]
CARTON = [0.65, 0.5, 0.32, 1.0]
WOOD = [0.6, 0.45, 0.28, 1.0]
ROBOT = [0.92, 0.92, 0.9, 1.0]
WHITE = [0.95, 0.95, 0.95, 1.0]

WALL_HEIGHT = 2.5  # half height
RACK_DEPTH = 0.4  # half depth
RACK_BAY = 1.4


def prop(body, size, pos, rgba, geom_type=mujoco.mjtGeom.mjGEOM_BOX, **kwargs):
    """Add a purely visual (non-colliding) geom to ``body``."""
    if pos is not None:
        kwargs["pos"] = pos
    return body.add_geom(type=geom_type, size=size, rgba=rgba, contype=0, conaffinity=0, **kwargs)


def _rack_row(world, rng, along_x: bool, offset: float, extent: float) -> None:
    """A row of stocked pallet racks centred ``offset`` off the hall axis, its bays covering
    ``[-extent, extent]`` along x (``along_x``) or y."""

    def place(half, pos):  # (along, across, z) to world axes
        if along_x:
            return [half[0], half[1], half[2]], [pos[0], pos[1], pos[2]]
        return [half[1], half[0], half[2]], [pos[1], pos[0], pos[2]]

    for a in np.arange(-extent + RACK_BAY / 2, extent - RACK_BAY / 2 + 1e-6, RACK_BAY):
        for da in (-RACK_BAY / 2, RACK_BAY / 2):
            prop(world, *place([0.04, RACK_DEPTH, 1.4], [a + da, offset, 1.4]), RACK_BLUE)
        for z in (0.05, 0.95, 1.85):
            prop(world, *place([RACK_BAY / 2, RACK_DEPTH, 0.03], [a, offset, z]), RACK_ORANGE)
            for ca in (a - RACK_BAY / 4, a + RACK_BAY / 4):
                if rng.random() < 0.85:
                    half = rng.uniform([0.18, 0.18, 0.12], [0.3, 0.32, 0.32])
                    color = CARTON if rng.random() < 0.7 else [*rng.uniform(0.2, 0.7, 3), 1.0]
                    prop(world, *place(half, [ca, offset, z + 0.03 + half[2]]), color)


def _workcell(world, pos, yaw: float) -> None:
    """A workbench with a robot arm reaching over it, at ``pos`` on the floor, its front (with the
    arm's reach) towards ``yaw``. Footprint: 0.4 m in front of ``pos`` and 0.5 m to either side."""
    cell = world.add_body(pos=[*pos, 0.0], quat=[np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    prop(cell, [0.4, 0.5, 0.03], [0.0, 0.0, 0.82], STEEL_LIGHT)
    for x in (-0.35, 0.35):
        for y in (-0.45, 0.45):
            prop(cell, [0.03, 0.03, 0.4], [x, y, 0.4], DARK)
    prop(cell, [0.12, 0.08], [-0.15, 0.0, 0.93], DARK, mujoco.mjtGeom.mjGEOM_CYLINDER)
    arm = mujoco.mjtGeom.mjGEOM_CAPSULE
    shoulder, elbow, wrist = [-0.15, 0.0, 1.0], [-0.1, 0.0, 1.45], [0.3, 0.0, 1.25]
    prop(cell, [0.07], None, ROBOT, arm, fromto=[*shoulder, *elbow])
    prop(cell, [0.055], None, ROBOT, arm, fromto=[*elbow, *wrist])
    prop(cell, [0.04, 0.05], [0.3, 0.0, 1.15], DARK, mujoco.mjtGeom.mjGEOM_CYLINDER)
    prop(cell, [0.15, 0.12, 0.08], [0.1, 0.1, 0.93], CARTON)  # a part being taken apart


def build_hall(spec: mujoco.MjSpec, half_extent: float, side_racks: bool = False, corner_cells: bool = False) -> None:
    """Concrete floor, a safety line at the arena edge (``half_extent``), back and side walls (the
    front is open for the cameras), steel columns, roof beams and lamps, pallet racks along the
    back wall and either the side walls (``side_racks``) or pallets of goods by them. Adds the ``hall``
    overview camera. Every geom is visual only, so the task's physics and detection are unchanged.

    ``corner_cells`` adds a robot workcell (bench and arm, up to 1.5 m tall) diagonally outside each
    corner of the arena, facing its centre, and widens the hall to make room. Its nearest point is
    0.7 m outside the arena, where a drone cannot fly: leaving the arena ends the episode.
    """
    floor = spec.texture("groundplane")
    floor.rgb1, floor.rgb2, floor.markrgb = [0.47, 0.47, 0.45], [0.44, 0.44, 0.42], [0.38, 0.38, 0.36]
    for texture in spec.textures:
        if texture.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            texture.rgb1, texture.rgb2 = [0.75, 0.77, 0.8], [0.45, 0.47, 0.5]
    world, h = spec.worldbody, half_extent
    wall = h + (2.0 if corner_cells else 1.2)
    # Overview from the open front, under the roof beams.
    world.add_camera(name="hall", pos=[0.0, -wall - 0.3, 4.3], xyaxes=[1, 0, 0, 0, 0.6, 0.8], fovy=75)
    for x in (-h / 2, h / 2):
        world.add_light(pos=[x, 0.0, 4.5], dir=[0, 0, -1], diffuse=[0.35, 0.35, 0.33], castshadow=False)
    for sign in (-1, 1):  # safety line at the arena edge
        prop(world, [h, 0.05, 0.001], [0.0, sign * h, 0.001], SAFETY_YELLOW)
        prop(world, [0.05, h, 0.001], [sign * h, 0.0, 0.001], SAFETY_YELLOW)
    prop(world, [wall, 0.1, WALL_HEIGHT], [0.0, wall, WALL_HEIGHT], WALL)
    prop(world, [wall, 0.11, 0.6], [0.0, wall, 0.6], WALL_BAND)
    for sign in (-1, 1):
        prop(world, [0.1, wall, WALL_HEIGHT], [sign * wall, 0.0, WALL_HEIGHT], WALL)
        prop(world, [0.11, wall, 0.6], [sign * wall, 0.0, 0.6], WALL_BAND)
    for y in np.linspace(-wall + 0.3, wall - 0.3, 5):
        for sign in (-1, 1):
            prop(world, [0.12, 0.12, WALL_HEIGHT], [sign * (wall - 0.25), y, WALL_HEIGHT], STEEL)
            prop(world, [0.16, 0.16, 0.3], [sign * (wall - 0.25), y, 0.3], SAFETY_YELLOW)
        prop(world, [wall, 0.07, 0.15], [0.0, y, 2 * WALL_HEIGHT - 0.15], STEEL)  # roof beam
        prop(world, [0.6, 0.15, 0.03], [0.0, y, 2 * WALL_HEIGHT - 0.35], LAMP)
    rng = np.random.default_rng(0)  # the same stock in every episode
    _rack_row(world, rng, True, wall - 0.55, h - 0.6 if corner_cells else wall - 1.0)
    if corner_cells:
        for sx in (-1, 1):
            for sy in (-1, 1):
                _workcell(world, [sx * (h + 0.8), sy * (h + 0.8)], np.arctan2(-sy, -sx))
    if side_racks:
        for sign in (-1, 1):
            _rack_row(world, rng, False, sign * (wall - 0.75), wall - 1.6)
        return
    for x, y in [(sign * (wall - 0.65), y) for sign in (-1, 1) for y in (-h + 1.0, -h + 2.6)]:  # pallets of goods
        prop(world, [0.4, 0.6, 0.07], [x, y, 0.07], WOOD)
        half = rng.uniform([0.25, 0.25, 0.2], [0.35, 0.35, 0.45])
        prop(world, half, [x, y, 0.14 + half[2]], CARTON)
