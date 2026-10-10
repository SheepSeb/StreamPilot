"""Base environment for a team of Skydio X2 drones flying in formation."""

from typing import Any

import mujoco
import numpy as np
from gymnasium.spaces import Box

from streampilot.env.base import (
    ALTITUDE_GAIN,
    ONBOARD_CAMERA,
    SCENE_XML,
    SceneEnv,
    point_in_view,
    project_box,
    project_points,
    wrap_angle,
)

# One colour per drone, used for its waypoint and its landing spot.
# Per-step ``info`` entries reported as episode means (``mean_<key>``); ``target_in_view`` gives the
# fraction of steps the target is visible, ``on_target`` the tracking success rate.
EPISODE_MEANS = (
    "target_in_view",
    "on_target",
    "formation_error",
    "heading_error",
    "standoff_error",
    "distance",
    "separation",
    "inter_drone_distance",
    "proximity_cost",
)

DRONE_COLORS = np.array([[0.1, 0.9, 0.2], [0.1, 0.6, 0.95], [0.95, 0.85, 0.1], [0.9, 0.2, 0.7], [0.95, 0.5, 0.1]])


def formation_offsets(num_drones: int, spacing: float) -> np.ndarray:
    """Slot of each drone in the formation frame (x = forward, towards the target), in metres.

    Drone 0 leads at the origin. Two drones fly in a column (one in front of the other); three
    in an equilateral triangle of side ``spacing`` with the leader at the apex; four in a diamond
    (the triangle plus a tail drone), five in a wedge (two rows of two behind the leader). Every
    pair of neighbouring slots is ``spacing`` apart.
    """
    back = -spacing * np.sqrt(3) / 2
    shapes = {
        1: [[0.0, 0.0]],
        2: [[0.0, 0.0], [-spacing, 0.0]],
        3: [[0.0, 0.0], [back, spacing / 2], [back, -spacing / 2]],
        4: [[0.0, 0.0], [back, spacing / 2], [back, -spacing / 2], [2 * back, 0.0]],
        5: [[0.0, 0.0], [back, spacing / 2], [back, -spacing / 2], [2 * back, spacing], [2 * back, -spacing]],
    }
    if num_drones not in shapes:
        raise ValueError(f"num_drones must be between 1 and 5, got {num_drones}")
    return np.array(shapes[num_drones])


def rotation(forward) -> np.ndarray:
    """2D rotation taking the formation frame's x axis to the unit vector ``forward``."""
    c, s = forward
    return np.array([[c, -s], [s, c]])


def unit(v, fallback=(1.0, 0.0)) -> np.ndarray:
    norm = np.linalg.norm(v)
    return np.asarray(v) / norm if norm > 1e-6 else np.array(fallback)


class FormationBaseEnv(SceneEnv):
    """``num_drones`` drones, each the same velocity-commanded, altitude-holding body with a level
    onboard camera as in ``DroneBaseEnv``, sharing one arena. Drone ``i`` is named ``d{i}_x2``
    and its cameras ``d{i}_onboard`` and ``d{i}_chase``.

    Every array below has one row per drone, in drone order. A decentralised policy runs on one
    row; a centralised one on all of them.

    Action: ``(num_drones, 3)``, each row ``[vx, vy, yaw_rate]`` in ``[-1, 1]`` in that drone's
    body frame, as in the single-drone tasks.

    Observation, by ``obs_mode``:

    - ``"detection"`` (default): ``(num_drones, 5 + num_drones + 2 * (num_drones - 1))``. Each row
      is the drone's own detection of its task target ``[visible, cx, cy, w, h]`` (as in the
      single-drone tasks), a one-hot of the drone's index (its slot in the formation), and the
      horizontal positions of its teammates relative to it, in its body frame and drone order.
      On a real team the teammate positions come from shared position estimates (GPS, VIO, UWB).
    - ``"pixels"``: ``uint8 (num_drones, image_size, image_size, 3)``, each drone's onboard image.
    - ``"state"``: privileged state, one row per drone:
      ``[x, y, vx, vy, cos(yaw), sin(yaw), yaw_rate, task_obs, one-hot, teammates]``.

    Reward is the team's: the task's per-drone terms averaged over the drones, minus the average
    control cost and the average proximity cost. The episode ends, with a penalty, if any drone
    leaves the arena or two drones come closer than ``min_separation`` (horizontally; ~0.62 m is
    rotor-to-rotor contact). The proximity cost warns before that: each drone pays
    ``proximity_weight`` times, summed over its teammates, how far into the band between
    ``proximity_distance`` and ``min_separation`` the teammate is (0 at the outer edge, 1 at contact).

    ``separation_weight`` adds a longer-range warning that does not penalise holding a formation: potential-based
    shaping (Ng et al. 1999, so it leaves the optimal behaviour unchanged) of the same band measure, but out to
    ``separation_distance``. Each drone is rewarded ``separation_weight`` times the decrease, summed over its
    teammates, of how far into that band they are (``gamma * phi(s') - phi(s)``, ``gamma = shaping_gamma``).
    Closing in costs, moving apart pays, and standing still costs nothing at any distance.

    ``wall_weight`` warns before leaving the arena in the same way: each drone pays it times how far
    into the band ``wall_distance`` inside the arena edge it is (0 at the inner edge, 1 at the wall).

    ``geofence`` (metres, ``None``: off) works like a flight controller's: the velocity command
    away from the arena centre is cut as a drone nears ``geofence`` inside the edge, so it stays
    in the arena instead of ending the episode, and can still recover.

    Subclasses define a task through ``_build_scene`` and the ``_task_*`` hooks, as in
    ``DroneBaseEnv``, with ``_task_target_geom(i)`` and ``_task_obs(i)`` per drone.
    """

    task_obs_dim = 0

    def __init__(
        self,
        num_drones: int = 3,
        formation_spacing: float = 1.0,
        min_separation: float = 0.6,
        spawn_separation: float | None = None,
        spawn_heading_noise_deg: float | None = None,
        collision_penalty: float = 10.0,
        proximity_distance: float = 0.9,
        proximity_weight: float = 0.0,
        separation_distance: float = 1.8,
        separation_weight: float = 0.0,
        shaping_gamma: float = 0.99,
        wall_distance: float = 0.5,
        wall_weight: float = 0.0,
        geofence: float | None = None,
        obs_mode: str = "detection",
        image_size: int = 84,
        detection_noise: float = 0.0,
        detection_dropout: float = 0.0,
        min_detection_size: float = 0.02,
        frame_skip: int = 5,
        flight_altitude: float = 1.0,
        max_speed: float = 2.0,
        max_yaw_rate: float = np.pi / 2,
        arena_half_extent: float = 3.0,
        ctrl_cost_weight: float = 0.01,
        out_of_bounds_penalty: float = 10.0,
        render_mode: str | None = None,
        width: int = 480,
        height: int = 480,
        camera: str | None = "overview",
    ):
        assert obs_mode in ("detection", "pixels", "state")
        assert formation_spacing > min_separation, "formation slots would count as collisions"
        assert proximity_distance > min_separation
        self.num_drones = num_drones
        self.formation = formation_offsets(num_drones, formation_spacing)
        self.formation_spacing = formation_spacing
        self.min_separation = min_separation
        self.spawn_separation = formation_spacing if spawn_separation is None else spawn_separation
        self.spawn_heading_noise = None if spawn_heading_noise_deg is None else np.deg2rad(spawn_heading_noise_deg)
        self.collision_penalty = collision_penalty
        self.proximity_distance = proximity_distance
        self.proximity_weight = proximity_weight
        assert separation_distance > min_separation
        self.separation_distance = separation_distance
        self.separation_weight = separation_weight
        self.shaping_gamma = shaping_gamma
        self._separation_potential = np.zeros(num_drones)
        self.wall_distance = wall_distance
        self.wall_weight = wall_weight
        self.geofence = geofence
        self.obs_mode = obs_mode
        self.detection_noise = detection_noise
        self.detection_dropout = detection_dropout
        self.min_detection_size = min_detection_size
        self.flight_altitude = flight_altitude
        self.action_scale = np.array([max_speed, max_speed, max_yaw_rate])
        self.arena_half_extent = arena_half_extent
        self.ctrl_cost_weight = ctrl_cost_weight
        self.out_of_bounds_penalty = out_of_bounds_penalty

        spec = mujoco.MjSpec.from_file(str(SCENE_XML))
        spec.delete(spec.body("x2"))
        for actuator in list(spec.actuators):
            spec.delete(actuator)
        for i in range(num_drones):
            drone = mujoco.MjSpec.from_file(str(SCENE_XML)).body("x2")
            spec.worldbody.add_frame().attach_body(drone, f"d{i}_", "")
        self._build_scene(spec)
        self._init_scene(spec.compile(), frame_skip, image_size, render_mode, width, height, camera)

        slides = [self.model.joint(f"d{i}_slide_x") for i in range(num_drones)]
        yaws = [self.model.joint(f"d{i}_yaw") for i in range(num_drones)]
        self._pos_adr = np.array([j.qposadr[0] + np.arange(3) for j in slides])
        self._vel_adr = np.array([j.dofadr[0] + np.arange(3) for j in slides])
        self._yaw_qpos_adr = np.array([j.qposadr[0] for j in yaws])
        self._yaw_qvel_adr = np.array([j.dofadr[0] for j in yaws])
        actuators = ("vx", "vy", "vz", "yaw_rate")
        self._ctrl_adr = np.array([[self.model.actuator(f"d{i}_{a}").id for a in actuators] for i in range(num_drones)])
        self._onboard_cam_ids = [self.model.camera(f"d{i}_{ONBOARD_CAMERA}").id for i in range(num_drones)]
        # Teammates of each drone in drone order, (num_drones, num_drones - 1), and every pair.
        self._teammate_idx = np.array([[j for j in range(num_drones) if j != i] for i in range(num_drones)], dtype=int)
        self._pairs = np.triu_indices(num_drones, k=1)
        body_ids = [self.model.body(f"d{i}_x2").id for i in range(num_drones)]
        # Each drone's visual mesh (vertices in its geom frame), for teammate detection.
        meshes = self.model.geom_type == mujoco.mjtGeom.mjGEOM_MESH
        self._drone_meshes = []
        for b in body_ids:
            geom = int(np.flatnonzero((self.model.geom_bodyid == b) & meshes)[0])
            mesh = self.model.geom_dataid[geom]
            adr, num = self.model.mesh_vertadr[mesh], self.model.mesh_vertnum[mesh]
            self._drone_meshes.append((geom, self.model.mesh_vert[adr : adr + num].copy()))

        self.action_space = Box(-1.0, 1.0, shape=(num_drones, 3), dtype=np.float32)
        shared_dim = num_drones + 2 * (num_drones - 1)  # one-hot and teammates
        if obs_mode == "detection":
            low = np.r_[np.zeros(5 + num_drones), np.full(2 * (num_drones - 1), -np.inf)].astype(np.float32)
            high = np.r_[np.ones(5 + num_drones), np.full(2 * (num_drones - 1), np.inf)].astype(np.float32)
            self.observation_space = Box(np.tile(low, (num_drones, 1)), np.tile(high, (num_drones, 1)))
        elif obs_mode == "pixels":
            self.observation_space = Box(0, 255, shape=(num_drones, image_size, image_size, 3), dtype=np.uint8)
        else:
            shape = (num_drones, 7 + self.task_obs_dim + shared_dim)
            self.observation_space = Box(-np.inf, np.inf, shape=shape, dtype=np.float32)
        self.step_count = 0

    @property
    def drone_pos(self) -> np.ndarray:
        return self.data.qpos[self._pos_adr]

    @property
    def drone_vel(self) -> np.ndarray:
        return self.data.qvel[self._vel_adr]

    @property
    def drone_yaw(self) -> np.ndarray:
        return wrap_angle(self.data.qpos[self._yaw_qpos_adr])

    @property
    def drone_yaw_rate(self) -> np.ndarray:
        return self.data.qvel[self._yaw_qvel_adr]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.step_count = 0
        self._episode_sums = {}
        self._drone_return = np.zeros(self.num_drones)
        mujoco.mj_resetData(self.model, self.data)
        self._task_reset(options or {})
        mujoco.mj_forward(self.model, self.data)
        self._separation_potential = self.separation_potential()
        if self.render_mode == "human":
            self.render()
        dets = self.target_detections()
        return self._get_obs(dets), self._get_info(self._task_info(), dets)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(self.num_drones, 3), -1.0, 1.0)
        self._task_before_step()
        vx, vy, yaw_rate = (action * self.action_scale).T
        yaw = self.drone_yaw
        c, s = np.cos(yaw), np.sin(yaw)
        vz = ALTITUDE_GAIN * (self.flight_altitude - self.drone_pos[:, 2])
        world = np.stack([c * vx - s * vy, s * vx + c * vy], axis=1)
        if self.geofence is not None:
            # Outward speed allowed: shrinks to 0 at ``geofence`` inside the arena edge (2/s gain).
            room = (self.arena_half_extent - self.geofence) - np.abs(self.drone_pos[:, :2])
            outward = np.sign(world) == np.sign(self.drone_pos[:, :2])
            world = np.where(outward, np.sign(world) * np.minimum(np.abs(world), np.maximum(2.0 * room, 0.0)), world)
        self.data.ctrl[self._ctrl_adr] = np.column_stack([world, vz, yaw_rate])
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self.step_count += 1

        reward, terminated, info = self._task_step()
        proximity_cost = self.proximity_cost() + self.wall_cost()
        info["proximity_cost"] = proximity_cost
        potential = self.separation_potential()
        shaping = self.shaping_gamma * potential - self._separation_potential
        self._separation_potential = potential
        cost = self.ctrl_cost_weight * np.sum(action**2, axis=1) + proximity_cost - shaping
        reward -= float(np.mean(cost))
        drone_reward = info.pop("drone_reward", None)
        separation = self.min_pairwise_distance()
        info["separation"] = separation
        info["inter_drone_distance"] = self.mean_pairwise_distance()
        if not terminated and separation < self.min_separation:
            reward -= self.collision_penalty
            terminated = True
            info["collision"] = True
        if not terminated and self._out_of_bounds():
            reward -= self.out_of_bounds_penalty
            terminated = True
            info["out_of_bounds"] = True

        if self.render_mode == "human":
            self.render()
        if drone_reward is None:
            drone_reward = np.full(self.num_drones, reward)
        else:  # each drone's own task reward, control and proximity costs; the terminal penalties are shared
            drone_reward = drone_reward - cost
            drone_reward += reward - float(np.mean(drone_reward))
        info["drone_reward"] = drone_reward
        dets = self.target_detections()
        info = self._get_info(info, dets)
        self._accumulate(info)
        return self._get_obs(dets), reward, terminated, False, info

    def detect(self, drone: int, geom_id: int) -> np.ndarray:
        """Exact ``[visible, cx, cy, w, h]`` box of ``geom_id`` in ``drone``'s onboard camera."""
        return project_box(self.model, self.data, geom_id, self._onboard_cam_ids[drone], self.min_detection_size)

    def detect_teammates(self, drone: int) -> np.ndarray:
        """``(num_drones - 1, 5)``: exact ``[visible, cx, cy, w, h]`` box of each teammate (in drone
        order, without ``drone`` itself) in ``drone``'s onboard camera, as a detector would see
        it with no occlusion. The box bounds the teammate's rendered mesh."""
        boxes = []
        for j in range(self.num_drones):
            if j != drone:
                geom, verts = self._drone_meshes[j]
                points = self.data.geom_xpos[geom] + verts @ self.data.geom_xmat[geom].reshape(3, 3).T
                cam = self._onboard_cam_ids[drone]
                boxes.append(project_points(self.model, self.data, points, cam, self.min_detection_size))
        return np.array(boxes).reshape(self.num_drones - 1, 5)

    def target_detections(self) -> np.ndarray:
        """``(num_drones, 5)``: each drone's exact detection of its task target."""
        return np.array([self.detect(i, self._task_target_geom(i)) for i in range(self.num_drones)])

    def detection_obs(self, exact: np.ndarray | None = None) -> np.ndarray:
        """Each drone's detection of its task target, as a noisy detector would report it.
        ``exact``: the output of ``target_detections``, if already computed this step."""
        exact = self.target_detections() if exact is None else exact
        dets = np.zeros((self.num_drones, 5))
        for i, det in enumerate(exact):
            if det[0] == 0.0 or self.np_random.random() < self.detection_dropout:
                continue
            det = det.copy()
            det[1:] += self.np_random.normal(scale=self.detection_noise, size=4)
            dets[i] = np.clip(det, 0.0, 1.0)
        return dets

    def teammates(self) -> np.ndarray:
        """``(num_drones, 2 * (num_drones - 1))``: teammate positions relative to each drone, in
        its body frame."""
        xy, yaw = self.drone_pos[:, :2], self.drone_yaw
        rel = xy[self._teammate_idx] - xy[:, None]  # (num_drones, num_drones - 1, 2), world frame
        c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
        x, y = rel[..., 0], rel[..., 1]
        return np.stack([x * c + y * s, -x * s + y * c], axis=-1).reshape(self.num_drones, -1)

    def state_obs(self) -> np.ndarray:
        yaw = self.drone_yaw
        own = np.column_stack(
            [self.drone_pos[:, :2], self.drone_vel[:, :2], np.cos(yaw), np.sin(yaw), self.drone_yaw_rate]
        )
        task = np.array([self._task_obs(i) for i in range(self.num_drones)]).reshape(self.num_drones, -1)
        return np.hstack([own, task, np.eye(self.num_drones), self.teammates()]).astype(np.float32)

    def camera_image(self, camera: str | int = f"d0_{ONBOARD_CAMERA}", width=None, height=None):
        """Render ``camera`` (default: the leader's onboard camera) to an RGB array."""
        return super().camera_image(camera, width, height)

    def onboard_image(self, drone: int, width: int | None = None, height: int | None = None) -> np.ndarray:
        return self.camera_image(f"d{drone}_{ONBOARD_CAMERA}", width, height)

    def min_pairwise_distance(self) -> float:
        xy = self.drone_pos[:, :2]
        a, b = self._pairs
        return float(np.min(np.linalg.norm(xy[a] - xy[b], axis=-1), initial=np.inf))

    def proximity_cost(self) -> np.ndarray:
        """``(num_drones,)``: each drone's proximity cost this step (see the class docstring)."""
        xy = self.drone_pos[:, :2]
        dist = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
        np.fill_diagonal(dist, np.inf)
        band = self.proximity_distance - self.min_separation
        return self.proximity_weight * np.clip((self.proximity_distance - dist) / band, 0.0, 1.0).sum(axis=1)

    def wall_cost(self) -> np.ndarray:
        """``(num_drones,)``: each drone's cost for flying near the arena edge (see the class docstring)."""
        if not self.wall_weight:
            return np.zeros(self.num_drones)
        depth = np.max(np.abs(self.drone_pos[:, :2]), axis=1) - (self.arena_half_extent - self.wall_distance)
        return self.wall_weight * np.clip(depth / self.wall_distance, 0.0, 1.0)

    def separation_potential(self) -> np.ndarray:
        """``(num_drones,)``: each drone's potential for the separation shaping, minus its band measure."""
        if not self.separation_weight:
            return np.zeros(self.num_drones)
        xy = self.drone_pos[:, :2]
        dist = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
        np.fill_diagonal(dist, np.inf)
        band = self.separation_distance - self.min_separation
        return -self.separation_weight * np.clip((self.separation_distance - dist) / band, 0.0, 1.0).sum(axis=1)

    def mean_pairwise_distance(self) -> float:
        xy = self.drone_pos[:, :2]
        a, b = self._pairs
        return float(np.mean(np.linalg.norm(xy[a] - xy[b], axis=-1))) if len(a) else 0.0

    # --- helpers for subclasses -------------------------------------------------------

    def _set_drone_states(self, xy, yaw) -> None:
        self.data.qpos[self._pos_adr] = np.column_stack([xy, np.full(self.num_drones, self.flight_altitude)])
        self.data.qpos[self._yaw_qpos_adr] = yaw
        self.data.qvel[:] = 0.0

    def _sample_xy(self, margin: float = 0.5) -> np.ndarray:
        return self.np_random.uniform(-1.0, 1.0, size=2) * (self.arena_half_extent - margin)

    def _sample_separated_xy(self, margin: float = 0.5) -> np.ndarray:
        """Random start positions, at least ``spawn_separation`` (default ``formation_spacing``) apart."""
        while True:
            xy = np.array([self._sample_xy(margin) for _ in range(self.num_drones)])
            dists = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
            if np.all(dists[np.triu_indices(self.num_drones, k=1)] >= self.spawn_separation):
                return xy

    def _sample_yaw(self, look_at=None) -> np.ndarray:
        """Start headings: uniform, or, with ``spawn_heading_noise_deg`` and ``look_at`` (one point per
        drone, ``(num_drones, 2)``, given its start ``xy`` already set), each drone facing its point
        up to that much noise, so its target starts in view."""
        if look_at is None or self.spawn_heading_noise is None:
            return self.np_random.uniform(-np.pi, np.pi, size=self.num_drones)
        rel = np.asarray(look_at) - self.drone_pos[:, :2]
        noise = self.np_random.uniform(-self.spawn_heading_noise, self.spawn_heading_noise, size=self.num_drones)
        return np.arctan2(rel[:, 1], rel[:, 0]) + noise

    def _point_in_view(self, drone: int, point) -> bool:
        return point_in_view(self.model, self.data, self._onboard_cam_ids[drone], point)

    def _heading_error(self, point) -> np.ndarray:
        """Angle from each drone's nose to ``point`` in the horizontal plane."""
        rel = np.asarray(point)[:2] - self.drone_pos[:, :2]
        return wrap_angle(np.arctan2(rel[:, 1], rel[:, 0]) - self.drone_yaw)

    def _body_frame(self, world_xy) -> np.ndarray:
        """Rotate one horizontal world-frame vector per drone into that drone's body frame."""
        c, s = np.cos(self.drone_yaw), np.sin(self.drone_yaw)
        x, y = np.asarray(world_xy).T
        return np.column_stack([c * x + s * y, -s * x + c * y])

    def _out_of_bounds(self) -> bool:
        return bool(np.max(np.abs(self.drone_pos[:, :2])) > self.arena_half_extent)

    def _get_obs(self, dets: np.ndarray) -> np.ndarray:
        if self.obs_mode == "pixels":
            return np.stack([self.onboard_image(i) for i in range(self.num_drones)])
        if self.obs_mode == "state":
            return self.state_obs()
        return np.hstack([self.detection_obs(dets), np.eye(self.num_drones), self.teammates()]).astype(np.float32)

    def _get_info(self, info: dict[str, Any], dets: np.ndarray) -> dict[str, Any]:
        info["target_in_view"] = dets[:, 0].astype(bool)
        return info

    def _accumulate(self, info: dict[str, Any]) -> None:
        """Add this step's metrics to the running sums and report the episode means so far
        (``mean_*``; over the whole episode once it ends), averaged over the drones. Per-drone
        metrics also get ``mean_*_per_drone`` (one entry per drone), and ``drone_return`` is each
        drone's return so far."""
        self._drone_return += info["drone_reward"]
        info["drone_return"] = self._drone_return.copy()
        for key in EPISODE_MEANS:
            if key in info:
                value = np.asarray(info[key], dtype=float)
                self._episode_sums[key] = self._episode_sums.get(key, 0.0) + value
                mean = self._episode_sums[key] / self.step_count
                info[f"mean_{key}"] = float(np.mean(mean))
                if value.ndim > 0:
                    info[f"mean_{key}_per_drone"] = mean

    # --- task hooks -------------------------------------------------------------------

    def _build_scene(self, spec: mujoco.MjSpec) -> None:
        """Add task-specific bodies to the scene before it is compiled."""

    def _task_reset(self, options: dict) -> None:
        """Place the drones and the task props. Runs after ``mj_resetData``."""
        self._set_drone_states(self._sample_separated_xy(), self._sample_yaw())

    def _task_before_step(self) -> None:
        """Advance task props (e.g. moving targets) once per control step."""

    def _task_obs(self, drone: int) -> np.ndarray:
        """Privileged task state of one drone, used by ``obs_mode="state"``."""
        return np.zeros(0)

    def _task_target_geom(self, drone: int) -> int:
        """Id of the geom ``drone``'s detector looks for."""
        raise NotImplementedError

    def _task_step(self) -> tuple[float, bool, dict[str, Any]]:
        """Return ``(team reward, terminated, info)`` after the physics step."""
        return 0.0, False, self._task_info()

    def _task_info(self) -> dict[str, Any]:
        return {}

    def goals(self) -> tuple[np.ndarray, np.ndarray | None]:
        """Where each drone should be now (``(num_drones, 2)``) and the point it should face
        (``None``: any heading). For scripted controllers and diagnostics."""
        raise NotImplementedError
