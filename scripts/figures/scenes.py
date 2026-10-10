"""Figures rendered with MuJoCo (EGL).

rl_schematic   factory escort scene, task geometry to scale, and the streaming learning loop
setup          warehouse scene, team close-up, and every onboard view with clean / noisy / dropped detections
"""

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
from matplotlib.patches import (
    Arc,
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
)

from streampilot.env.formation import FormationFactoryEnv
from streampilot.env.formation.base import DRONE_COLORS, formation_offsets, rotation
from streampilot.env.formation.hall import DARK, ROBOT, STEEL, prop
from streampilot.env.formation.warehouse import AGV_ORANGE, STATUS_GREEN
from streampilot.policy import TeamPolicy
from streampilot.train_formation_stream import make_env

from .common import save
from .diagrams import (
    ACTOR_EDGE,
    ACTOR_FACE,
    CRITIC_EDGE,
    CRITIC_FACE,
    ENV_EDGE,
    ENV_FACE,
    INK,
    UPDATE_EDGE,
    UPDATE_FACE,
    arrow,
    box,
)

OUTPUTS = ["rl_schematic", "setup"]
FOLLOW_DISTANCE, SPACING, FOV_DEG = 1.5, 1.0, 70.0  # FormationTrackingEnv / scene.xml defaults
AGV, LANE, HEAD = "#f28c1a", "#e6c700", "#ebebe6"
CLEAN, NOISY = "#1baf7a", "#eb6834"  # detection boxes
RED_NOTE = {"facecolor": "#c0392b", "alpha": 0.85, "edgecolor": "none", "boxstyle": "round,pad=0.3"}


def add_args(parser) -> None:
    parser.add_argument("--drones", type=int, default=3, choices=[2, 3, 4, 5], help="rl_schematic: team size")
    parser.add_argument("--schematic-seed", type=int, default=3, help="rl_schematic: episode seed")
    parser.add_argument("--schematic-steps", type=int, default=160, help="rl_schematic: steps before the frame")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/istream_ac_formation-tracking_3d_seed1/final.pt"),
                        help="setup: the team that flies")
    parser.add_argument("--task", default="formation-warehouse-tracking", help="setup: the scene (a tracking task)")
    parser.add_argument("--camera", default="hall", help="setup: overview camera (hall: warehouse; overview: arena)")
    parser.add_argument("--episode-seed", type=int, default=10_000, help="setup: episode seed")
    parser.add_argument("--steps", type=int, default=150, help="setup: steps flown before the frame is taken")
    parser.add_argument("--noise", type=float, default=0.3, help="setup: detection noise of the noisy panel")
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--crop", nargs=4, type=float, default=[0.12, 0.88, 0.05, 0.75], metavar=("X0", "X1", "Y0", "Y1"),
                        help="setup: the part of the overview shown, as fractions of its width and height")
    parser.add_argument("--margin", type=float, default=0.25, help="setup close-up: padding around drones and target")
    parser.add_argument("--width", type=float, default=7.0, help="setup: figure width in inches")


def run(args, outputs: list[str]) -> None:
    if "rl_schematic" in outputs:
        name = "rl_schematic.pdf" if args.drones == 3 else f"rl_schematic_{args.drones}d.pdf"
        plot_schematic(args.drones, args.schematic_seed, args.schematic_steps, args.out_dir / name)
    if "setup" in outputs:
        plot_setup(args, args.out_dir / "setup_combined.pdf")


def box_patch(box, size, color, **kwargs) -> Rectangle:
    """``[visible, cx, cy, w, h]`` (normalised, y down) as a rectangle in pixel coordinates."""
    _, cx, cy, w, h = box
    return Rectangle(((cx - w / 2) * size, (cy - h / 2) * size), w * size, h * size, fill=False, color=color, **kwargs)


# --- (a) rendered scene ------------------------------------------------------------------------------------------------

class RobotFactoryEnv(FormationFactoryEnv):
    """The factory escort with the setup figures' target: the AGV and its product are swapped for the warehouse tracking
    robot (``FormationWarehouseTrackingEnv``), driven around the factory loop. As there, the detector target is the
    invisible red pillar the robot fills, so detections are those the tracking policies were trained on."""

    def _build_scene(self, spec) -> None:
        super()._build_scene(spec)
        robot, w, h = spec.body("target"), self.target_half_width, self.target_half_height
        for geom in list(robot.geoms):
            spec.delete(geom)
        # The body sits on the floor here (the pillar's centre in the tracking task), so every part is raised by h.
        prop(robot, [w, w, h], [0.0, 0.0, h], [0.0, 0.0, 0.0, 0.0], name="target")
        cylinder = mujoco.mjtGeom.mjGEOM_CYLINDER
        prop(robot, [w, w, 0.11], [0.0, 0.0, 0.13], AGV_ORANGE)  # base
        prop(robot, [w, w, 0.025], [0.0, 0.0, 0.025], DARK)  # bumper
        for x in (-1, 1):
            for y in (-1, 1):
                prop(robot, [0.045, 0.02], [x * (w - 0.05), y * (w - 0.02), 0.045], DARK, cylinder,
                     quat=[0.7071, 0.7071, 0.0, 0.0])
        mast_bottom, head = 0.24, 2 * h - 0.2
        prop(robot, [0.05, 0.05, (head - mast_bottom) / 2], [0.0, 0.0, (head + mast_bottom) / 2], STEEL)
        for z in np.linspace(mast_bottom + 0.2, head - 0.15, 4):  # scanner cameras, both sides
            for x in (-1, 1):
                prop(robot, [0.02, 0.07, 0.03], [x * 0.06, 0.0, z], DARK)
        prop(robot, [w, w, 0.2], [0.0, 0.0, 2 * h - 0.2], ROBOT)  # head
        prop(robot, [w + 0.002, w + 0.002, 0.02], [0.0, 0.0, 2 * h - 0.33], STATUS_GREEN)

    def _task_reset(self, options: dict) -> None:
        super()._task_reset(options)
        # Undo the product's size, height and colour: the detector target stays the invisible pillar.
        w, h = self.target_half_width, self.target_half_height
        self._set_box_size(self._product_geom, [w, w, h])
        self.model.geom_pos[self._product_geom] = [0.0, 0.0, h]
        self.model.geom_rgba[self._product_geom] = 0.0


def fly_scripted(env: FormationFactoryEnv, steps: int) -> None:
    """Fly every drone to its slot (P control plus the robot's velocity) while turning it to face the robot."""
    for _ in range(steps):
        slots, look = env.goals()
        velocity = 1.5 * (slots - env.drone_pos[:, :2]) + env._target_vel[:2]
        yaw_rate = 2.0 * env._heading_error(look)
        action = np.column_stack([env._body_frame(velocity) / env.action_scale[0], yaw_rate / env.action_scale[2]])
        env.step(np.clip(action, -1.0, 1.0))


def scene_panel(ax, num_drones: int, seed: int, steps: int) -> None:
    env = RobotFactoryEnv(num_drones=num_drones)
    env.reset(seed=seed)
    fly_scripted(env, steps)
    scene = env.camera_image("team", 960, 640)
    size = 320
    onboard = env.onboard_image(0, size, size)
    detection = env.target_detections()[0]
    env.close()

    ax.imshow(scene)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"(a) Factory escort, N = {num_drones}: drones follow a mobile robot", fontsize=9)
    inset = ax.inset_axes([0.665, 0.015, 0.32, 0.48])
    inset.imshow(onboard)
    if detection[0]:
        inset.add_patch(box_patch(detection, size, CLEAN, linewidth=1.8))
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color("white")
        spine.set_linewidth(1.5)
    inset.text(0.04, 0.96, "leader's onboard view\n+ detection box", transform=inset.transAxes, fontsize=6.5,
               color="white", ha="left", va="top", bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none",
                                                             "boxstyle": "round,pad=0.25"})


# --- (b) geometry ------------------------------------------------------------------------------------------------------

def draw_drone(ax, xy, yaw, color, size=0.15) -> None:
    """Top view of a quadrotor: four rotors on an X, a body and a nose tick, pointing along ``yaw``."""
    rot = rotation([np.cos(yaw), np.sin(yaw)])
    for angle in np.deg2rad([45, 135, 225, 315]):
        arm = rot @ (size * np.array([np.cos(angle), np.sin(angle)]))
        ax.plot(*np.c_[xy, xy + arm], color=INK, lw=1.0, zorder=5)
        ax.add_patch(Circle(xy + arm, size * 0.42, facecolor="white", edgecolor=INK, lw=0.7, zorder=5))
    ax.add_patch(Circle(xy, size * 0.42, facecolor=color, edgecolor=INK, lw=0.8, zorder=6))
    ax.plot(*np.c_[xy, xy + rot @ [size * 1.2, 0]], color=INK, lw=1.5, zorder=6)


def draw_robot(ax, xy, half_width=0.15) -> None:
    """Top view of the mobile robot: its head (light) over the orange base, with the green status band."""
    w = half_width
    ax.add_patch(Rectangle(xy - (w + 0.03), 2 * (w + 0.03), 2 * (w + 0.03), facecolor=AGV, edgecolor=INK, lw=0.8,
                           zorder=6))
    ax.add_patch(Rectangle(xy - w, 2 * w, 2 * w, facecolor=HEAD, edgecolor="#1bbf55", lw=1.6, zorder=7))


def geometry_panel(ax, num_drones: int) -> None:
    # The robot drives +x along the top lane of its loop, towards a station; the team trails it from the upper left.
    target = np.array([0.0, 0.0])
    lane_y, station_x = target[1], 1.9
    ax.plot([-4.0, station_x], [lane_y, lane_y], color=LANE, lw=4, solid_capstyle="butt", zorder=0)
    ax.plot([station_x, station_x], [lane_y, lane_y - 2.0], color=LANE, lw=4, solid_capstyle="butt", zorder=0)
    ax.add_patch(Rectangle((station_x - 0.25, lane_y - 0.25), 0.5, 0.5, facecolor="none", edgecolor="#8a7700", lw=1.0,
                           ls="--", zorder=1))
    ax.text(station_x, lane_y + 0.33, "station\n(stop 0–3 s)", fontsize=6.5, ha="center", va="bottom", color="#6b5c00")

    heading = np.deg2rad(-28)  # direction from the leader to the robot
    forward = np.array([np.cos(heading), np.sin(heading)])
    slots = target - FOLLOW_DISTANCE * forward + formation_offsets(num_drones, SPACING) @ rotation(forward).T
    # Drones near (not on) their slots, facing the robot up to a small heading error, as mid-episode.
    rng = np.random.default_rng(3)
    drones = slots + rng.uniform(-0.15, 0.15, size=slots.shape)
    drones[0] = slots[0] + np.array([-0.1, 0.07])
    to_target = target - drones
    yaws = np.arctan2(to_target[:, 1], to_target[:, 0]) + np.deg2rad([4, -16, 9, -6, 10][:num_drones])

    # Cameras: a 70 deg wedge along each drone's nose.
    half = np.deg2rad(FOV_DEG / 2)
    for i, (xy, yaw) in enumerate(zip(drones, yaws)):
        tips = [xy + 3.4 * np.array([np.cos(yaw + a), np.sin(yaw + a)]) for a in (-half, half)]
        ax.add_patch(Polygon([xy, *tips], facecolor=DRONE_COLORS[i], alpha=0.09, edgecolor="none", zorder=1))

    # Formation frame: the leader-to-robot axis.
    ax.plot(*np.c_[slots[0] - 1.0 * forward, target + 0.6 * forward], color="#999999", lw=0.8, ls=(0, (6, 3)), zorder=2)

    # Slots and slot errors.
    for i, (slot, xy) in enumerate(zip(slots, drones)):
        ax.add_patch(Circle(slot, 0.07, facecolor="none", edgecolor=DRONE_COLORS[i], lw=1.3, ls="--", zorder=3))
        ax.plot(*np.c_[slot, xy], color=DRONE_COLORS[i], lw=1.0, ls=":", zorder=3)

    # The robot and its velocity.
    draw_robot(ax, target)
    ax.add_patch(FancyArrowPatch(target + [0.35, 0.0], target + [1.3, 0.0], arrowstyle="-|>", mutation_scale=10,
                                 color=AGV, lw=1.6, zorder=6))
    ax.text(0.85, -0.3, "robot $\\leq$0.6 m/s", color="#b35f00", fontsize=6.5, ha="center", va="top")

    # Standoff: leader's slot to the robot, drawn beside the formation axis.
    normal = np.array([-forward[1], forward[0]])
    a, b = slots[0] + 0.45 * normal, target + 0.45 * normal
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="<|-|>", mutation_scale=7, color=INK, lw=0.8, shrinkA=0, shrinkB=0,
                                 zorder=4))
    ax.text(*((a + b) / 2 + 0.1 * normal), f"$d^* = {FOLLOW_DISTANCE:g}$ m", fontsize=7.5, ha="center", va="bottom",
            rotation=np.rad2deg(heading), rotation_mode="anchor")

    # Heading error of the first follower.
    if num_drones > 1:
        xy, yaw = drones[1], yaws[1]
        bearing = np.arctan2(*(target - xy)[::-1])
        ax.plot(*np.c_[xy, xy + 0.9 * np.array([np.cos(yaw), np.sin(yaw)])], color=INK, lw=0.8, zorder=4)
        ax.plot(*np.c_[xy, xy + 0.9 * np.array([np.cos(bearing), np.sin(bearing)])], color=INK, lw=0.6, ls="--", zorder=4)
        lo, hi = sorted(np.rad2deg([yaw, bearing]))
        ax.add_patch(Arc(xy, 1.4, 1.4, theta1=lo, theta2=hi, color=INK, lw=0.9, zorder=4))
        mid = np.deg2rad((lo + hi) / 2)
        ax.text(*(xy + 0.8 * np.array([np.cos(mid), np.sin(mid)])), "$\\psi_1$", fontsize=8, ha="left", va="center")

    # Teammate vectors seen by the last drone.
    last = num_drones - 1
    for j in range(last):
        ax.add_patch(FancyArrowPatch(drones[last], drones[j], arrowstyle="-|>", mutation_scale=7, color="#7f7f7f", lw=0.8,
                                     ls=(0, (2, 2)), shrinkA=5, shrinkB=7, zorder=3))
    if last:
        ax.text(*(drones[last] + [0.0, -0.3]), "teammate positions\n(body frame)", fontsize=6.5, color="#555555",
                ha="center", va="top")

    for i, (xy, yaw) in enumerate(zip(drones, yaws)):
        draw_drone(ax, xy, yaw, DRONE_COLORS[i])
        if i == 0:
            ax.text(*(xy + [0.05, -0.27]), "leader 0", fontsize=7, ha="center", va="top", color=INK, zorder=7)
        else:
            ax.text(*(xy + [-0.27, 0.0]), str(i), fontsize=7, ha="right", va="center", color=INK, zorder=7)

    ax.plot([], [], marker="o", ms=5, mfc="none", mec="#777777", lw=0, label="formation slot")
    ax.plot([], [], color="#777777", ls=":", label="slot error $e_i$")
    ax.add_patch(Polygon([[0, 0]] * 3, facecolor="#777777", alpha=0.2, label=f"camera FOV {FOV_DEG:g}°"))
    ax.legend(loc="upper right", fontsize=6.5, framealpha=0.85, edgecolor="none", handlelength=1.6, borderaxespad=0.3)
    ax.text(0.015, 0.03,
            "$r_i = \\exp(-e_i/0.3)\\cdot\\frac{1+\\cos\\psi_i}{2}$\n"
            "$R = \\mathrm{mean}_i\\, r_i - \\mathrm{mean}_i\\, \\mathrm{cost}_i$\n"
            "$-10$ and end: collision ($<$0.6 m) or exit",
            transform=ax.transAxes, fontsize=7, ha="left", va="bottom", linespacing=1.5,
            bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "boxstyle": "round,pad=0.35"})

    ax.set_xlim(-3.6, 2.6)
    ax.set_ylim(-1.45, 2.35)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#bbbbbb")
    ax.set_title("(b) Task geometry (top view, to scale)", fontsize=9)


# --- (c) learning loop -------------------------------------------------------------------------------------------------

def note(ax, xy, text, ha="left") -> None:
    ax.text(*xy, text, fontsize=6.5, ha=ha, va="center", color="#444444", zorder=5, linespacing=1.3)


def learning_panel(ax, num_drones: int) -> None:
    n = num_drones
    obs_dim = 5 + n + 2 * (n - 1)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    box(ax, (5.25, 9.3), 9.3, 1.1,
        "Environment: remanufacturing hall (MuJoCo, 20 Hz, 30 s episodes)\n"
        f"{n} Skydio X2 (velocity control, altitude hold) $\\cdot$ mobile robot on a loop",
        ENV_FACE, ENV_EDGE)

    # One learner per drone: a stack of cards.
    top, bottom = 7.75, 0.2
    for k in (2, 1, 0):
        d = 0.09 * k
        ax.add_patch(FancyBboxPatch((1.0 + d, bottom - d), 8.85, top - bottom,
                                    boxstyle="round,pad=0.02,rounding_size=0.2", facecolor="white", edgecolor="#9a9a9a",
                                    lw=0.9, ls="-" if k == 0 else ":", zorder=1 + (2 - k) * 0.1))
    ax.text(5.42, 0.45, f"learner of drone $i$  ($\\times${n}: own weights, no replay buffer)", fontsize=7,
            ha="center", va="center", color="#555555", style="italic", zorder=4)

    left, right, w_left, w_right, h = 3.3, 7.6, 4.0, 3.9, 1.05
    rows = [6.3, 4.65, 3.0]
    box(ax, (left, rows[0]), w_left, h, f"observation $o_i$ ({obs_dim})\ndet. box 5 | slot id {n} | team {2 * (n - 1)}",
        "white", "#888888")
    box(ax, (left, rows[1]), w_left, h, "features $x_i = [o_{t-3:t},\\, a_{t-1}]$\nrunning normalisation", "white",
        "#888888")
    box(ax, (left, rows[2]), w_left, h, "actor $\\pi_i(a \\mid x_i)$\nGaussian, MLP 2$\\times$128", ACTOR_FACE, ACTOR_EDGE)
    box(ax, (right, rows[0]), w_right, h, "team reward $R_t$\nsame for every drone", "white", "#888888")
    box(ax, (right, rows[1]), w_right, h, "critic $V_i(x_i)$\nMLP 2$\\times$128", CRITIC_FACE, CRITIC_EDGE)
    box(ax, (right, rows[2]), w_right, h, "TD error\n$\\delta_i = R_t/\\sigma_i + \\gamma V_i(x_i') - V_i(x_i)$",
        CRITIC_FACE, CRITIC_EDGE)
    box(ax, (5.42, 1.35), 8.3, 1.05, "ObGD step on actor and critic, every step\neligibility traces $\\lambda$ = 0.8, "
        "$\\gamma$ = 0.99, step size bounded per update", UPDATE_FACE, UPDATE_EDGE, fontsize=7)

    env_bottom, edge = 8.75, h / 2
    arrow(ax, (left, env_bottom), (left, rows[0] + edge))
    note(ax, (left + 0.12, 8.25), "own camera detection,\nteammate positions")
    arrow(ax, (right, env_bottom), (right, rows[0] + edge))
    note(ax, (right + 0.12, 8.25), "mean $r_i$ $-$ costs\n$-$ penalties")
    for x in (left, right):
        arrow(ax, (x, rows[0] - edge), (x, rows[1] + edge))
        arrow(ax, (x, rows[1] - edge), (x, rows[2] + edge))
    note(ax, (right + 0.12, (rows[0] + rows[1]) / 2), "$\\div\\,\\sigma_i$")
    arrow(ax, (left + w_left / 2, rows[1]), (right - w_right / 2, rows[1]))
    arrow(ax, (right, rows[2] - edge), (right, 1.35 + 0.525))
    arrow(ax, (left, 1.35 + 0.525), (left, rows[2] - edge), dashed=True)
    note(ax, (left + 0.12, (rows[2] - edge + 1.875) / 2), "update")

    # The action, around the left side, back to the environment.
    ax.plot([left - w_left / 2, 0.45], [rows[2], rows[2]], color=INK, lw=1.0, zorder=5)
    arrow(ax, (0.45, rows[2]), (0.45, env_bottom))
    ax.text(0.22, (rows[2] + env_bottom) / 2, "action $a_i = [v_x, v_y, \\dot\\psi] \\in [-1, 1]^3$ (body frame)",
            fontsize=7, rotation=90, ha="center", va="center", color=INK)
    ax.set_title("(c) Streaming learning loop (independent Stream AC($\\lambda$))", fontsize=9)


def plot_schematic(num_drones: int, seed: int, steps: int, out: Path) -> None:
    fig = plt.figure(figsize=(11.0, 7.4))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.95], height_ratios=[1.0, 0.95], wspace=0.06, hspace=0.12)
    scene_panel(fig.add_subplot(grid[0, 0]), num_drones, seed, steps)
    geometry_panel(fig.add_subplot(grid[1, 0]), num_drones)
    learning_panel(fig.add_subplot(grid[:, 1]), num_drones)
    save(fig, out)


# --- setup ---------------------------------------------------------------------------------------------------------

def pixels(base, camera: str, points: np.ndarray, width: int, height: int) -> np.ndarray:
    """World points (k, 3) to (u, v) pixels of ``camera`` (MuJoCo cameras look along -z, x right, y up)."""
    cam = base.model.camera(camera).id
    pos, rot = base.data.cam_xpos[cam], base.data.cam_xmat[cam].reshape(3, 3)
    local = (points - pos) @ rot
    f = (height / 2) / np.tan(np.deg2rad(base.model.cam_fovy[cam]) / 2)
    return np.column_stack([width / 2 + f * local[:, 0] / -local[:, 2], height / 2 - f * local[:, 1] / -local[:, 2]])


def auto_crop(base, camera: str, width: int, height: int, margin: float, aspect: float = 4 / 3) -> list[int]:
    """[x0, x1, y0, y1] around the drones and the target (its foot and top), padded by ``margin`` of the
    larger side and widened to ``aspect``, kept inside the image."""
    target = base._target_pos
    points = np.vstack([base.drone_pos, target - [0, 0, base.target_half_height], target + [0, 0, base.target_half_height]])
    uv = pixels(base, camera, points, width, height)
    (u0, v0), (u1, v1) = uv.min(0), uv.max(0)
    pad = margin * max(u1 - u0, v1 - v0)
    u0, u1, v0, v1 = u0 - pad, u1 + pad, v0 - pad, v1 + pad
    if (u1 - u0) < aspect * (v1 - v0):  # widen
        grow = (aspect * (v1 - v0) - (u1 - u0)) / 2
        u0, u1 = u0 - grow, u1 + grow
    else:  # heighten
        grow = ((u1 - u0) / aspect - (v1 - v0)) / 2
        v0, v1 = v0 - grow, v1 + grow
    return [int(max(u0, 0)), int(min(u1, width)), int(max(v0, 0)), int(min(v1, height))]


def plot_setup(args, out: Path) -> None:
    config = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["config"]
    config["task"] = args.task
    config["env_kwargs"] = {**config["env_kwargs"], "detection_dropout": 0.0, "detection_noise": 0.0}
    env = make_env(config)
    policy = TeamPolicy.load(args.checkpoint)
    obs, _ = env.reset(seed=args.episode_seed)
    policy.reset()
    for _ in range(args.steps):
        obs, *_ = env.step(policy(obs))
    base = env.unwrapped

    W, H = 2560, 1920  # high-res: the close-up is a small crop of this
    base.model.vis.global_.offwidth, base.model.vis.global_.offheight = W, H  # the scene XML caps it at 1024
    full = base.camera_image(args.camera, width=W, height=H)
    fx0, fx1, fy0, fy1 = args.crop
    ox0, ox1, oy0, oy1 = int(fx0 * W), int(fx1 * W), int(fy0 * H), int(fy1 * H)
    overview = full[oy0:oy1, ox0:ox1]
    cx0, cx1, cy0, cy1 = auto_crop(base, args.camera, W, H, args.margin)
    closeup = full[cy0:cy1, cx0:cx1]

    size = 400
    n = base.drone_pos.shape[0]
    views = [base.onboard_image(i, size, size) for i in range(n)]
    exact = base.target_detections()
    if not exact[0, 0]:
        raise SystemExit(f"the target is not in the leader's view at step {args.steps}; try other --steps")
    base.detection_noise = args.noise
    base.np_random = np.random.default_rng(args.noise_seed)
    noisy = base.detection_obs(exact)[0]
    env.close()

    # Layout in inches: bottom row of n + 2 squares, top row of the two scene images at a common height,
    # both rows spanning the full width.
    gap, title, pad = 0.06, 0.24, 0.02
    k = n + 2
    s = (args.width - (k - 1) * gap) / k
    aspects = [overview.shape[1] / overview.shape[0], closeup.shape[1] / closeup.shape[0]]
    h = (args.width - gap) / sum(aspects)
    height = pad + s + title + 0.04 + h + title
    fig = plt.figure(figsize=(args.width, height))

    def axes(x, y, w, hh):
        ax = fig.add_axes([x / args.width, y / height, w / args.width, hh / height])
        ax.set_xticks([])
        ax.set_yticks([])
        return ax

    fs = 8
    top_y = pad + s + title + 0.04
    scene = "Warehouse tracking" if "warehouse" in args.task else "Formation tracking"
    ax_a = axes(0, top_y, aspects[0] * h, h)
    ax_a.imshow(overview)
    ax_a.add_patch(Rectangle((cx0 - ox0, cy0 - oy0), cx1 - cx0, cy1 - cy0, fill=False, color="w", linewidth=1.2,
                             linestyle="--"))
    ax_a.set_title(f"(a) {scene}, N = {n}", fontsize=fs + 1)
    ax_b = axes(aspects[0] * h + gap, top_y, aspects[1] * h, h)
    ax_b.imshow(closeup)
    ax_b.set_title("(b) Close-up of the team", fontsize=fs + 1)

    panels = [(f"Drone {i} ({'leader' if i == 0 else f'follower {i}'})", views[i], [(exact[i], CLEAN, "-")], None)
              for i in range(n)]
    panels += [
        (f"Leader, noise σ = {args.noise:g}", views[0], [(exact[0], CLEAN, ":"), (noisy, NOISY, "-")], None),
        ("Leader, dropout", views[0], [], "no detection\n(observation = 0)"),
    ]
    for j, (name, view, boxes, note) in enumerate(panels):
        ax = axes(j * (s + gap), pad, s, s)
        ax.imshow(view)
        for det, color, style in boxes:
            if det[0]:
                ax.add_patch(box_patch(det, size, color, linewidth=1.6, linestyle=style))
        if j < n and not boxes[0][0][0]:
            note = "target not in view"
        if len(boxes) > 1:
            ax.text(size * 0.04, size * 0.05, "dotted: true box", fontsize=fs - 1, va="top",
                    bbox={"facecolor": "w", "alpha": 0.8, "edgecolor": "none", "boxstyle": "round,pad=0.2"})
        if note:
            ax.text(size / 2, size * 0.92, note, color="w", ha="center", va="bottom", fontsize=fs - 1, bbox=RED_NOTE)
        ax.set_title(f"({chr(ord('c') + j)}) {name}", fontsize=fs)

    save(fig, out, dpi=300, pad_inches=0.02)
    print("in view:", [bool(d[0]) for d in exact], f" clean {np.round(exact[0], 3)}  noisy {np.round(noisy, 3)}")

