"""Base environment for the Skydio X2 drone with an onboard camera."""

import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium.spaces import Box

SCENE_XML = Path(__file__).resolve().parent.parent / "assets" / "skydio_x2" / "scene.xml"
ONBOARD_CAMERA = "onboard"
ALTITUDE_GAIN = 2.0  # 1/s, altitude-hold correction


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class DroneBaseEnv(gym.Env):
    """Drone flying at a fixed altitude in a bounded arena, commanded like a real drone's offboard
    velocity API with altitude hold.

    Action: ``[vx, vy, yaw_rate]`` in ``[-1, 1]``, body frame (x = nose), scaled by ``max_speed``
    and ``max_yaw_rate``. The drone holds ``flight_altitude``; it cannot climb or descend.

    Camera: one level onboard camera looking straight ahead from the nose.

    Observation, by ``obs_mode``:

    - ``"detection"`` (default): what an object detector (e.g. YOLO) run on the onboard camera
      would report for the task target: ``[visible, cx, cy, w, h]`` with the box in YOLO's
      normalized ``xywhn`` format (origin top-left, y down) and zeros when not detected. In sim
      the box comes from projecting the target geom into the camera, with optional noise
      (``detection_noise``, std as a fraction of the image) and missed detections
      (``detection_dropout``). On the real drone, feed the detector's output instead.
    - ``"pixels"``: the onboard camera image, ``uint8 (image_size, image_size, 3)``.
    - ``"state"``: privileged state ``[x, y, vx, vy, cos(yaw), sin(yaw), yaw_rate, task_obs]``,
      for debugging, scripted controllers and asymmetric critics (also via ``state_obs()``).

    One observation carries no motion information; stack frames with a wrapper for that.

    Subclasses define a task by overriding ``_build_scene`` and the ``_task_*`` hooks.
    Episode length is limited by the ``TimeLimit`` wrapper added at registration.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    task_obs_dim = 0

    def __init__(
        self,
        obs_mode: str = "detection",
        image_size: int = 84,
        detection_noise: float = 0.0,
        detection_dropout: float = 0.0,
        min_detection_size: float = 0.02,
        frame_skip: int = 5,
        flight_altitude: float = 1.0,
        max_speed: float = 2.0,
        max_yaw_rate: float = np.pi / 2,
        arena_half_extent: float = 2.0,
        ctrl_cost_weight: float = 0.01,
        out_of_bounds_penalty: float = 10.0,
        render_mode: str | None = None,
        width: int = 480,
        height: int = 480,
        camera: str | None = "chase",
    ):
        assert obs_mode in ("detection", "pixels", "state")
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.obs_mode = obs_mode
        self.image_size = image_size
        self.detection_noise = detection_noise
        self.detection_dropout = detection_dropout
        self.min_detection_size = min_detection_size
        self.frame_skip = frame_skip
        self.flight_altitude = flight_altitude
        self.action_scale = np.array([max_speed, max_speed, max_yaw_rate])
        self.arena_half_extent = arena_half_extent
        self.ctrl_cost_weight = ctrl_cost_weight
        self.out_of_bounds_penalty = out_of_bounds_penalty
        self.render_mode = render_mode
        self.width, self.height, self.camera = width, height, camera

        spec = mujoco.MjSpec.from_file(str(SCENE_XML))
        self._build_scene(spec)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)

        slide_x, yaw = self.model.joint("slide_x"), self.model.joint("yaw")
        self._qpos_adr, self._qvel_adr = int(slide_x.qposadr[0]), int(slide_x.dofadr[0])
        self._yaw_qpos_adr, self._yaw_qvel_adr = int(yaw.qposadr[0]), int(yaw.dofadr[0])
        self._drone_body_id = self.model.body("x2").id
        self._onboard_cam_id = self.model.camera(ONBOARD_CAMERA).id
        self._half_fov = np.deg2rad(self.model.cam_fovy[self._onboard_cam_id]) / 2

        self.metadata = {**self.metadata, "render_fps": int(np.round(1.0 / self.dt))}
        self.action_space = Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        if obs_mode == "detection":
            self.observation_space = Box(0.0, 1.0, shape=(5,), dtype=np.float32)
        elif obs_mode == "pixels":
            self.observation_space = Box(0, 255, shape=(image_size, image_size, 3), dtype=np.uint8)
        else:
            self.observation_space = Box(-np.inf, np.inf, shape=(7 + self.task_obs_dim,), dtype=np.float32)

        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self._viewer = None
        self._last_render_time: float | None = None
        self.step_count = 0

    @property
    def dt(self) -> float:
        return self.model.opt.timestep * self.frame_skip

    @property
    def drone_pos(self) -> np.ndarray:
        return self.data.qpos[self._qpos_adr : self._qpos_adr + 3]

    @property
    def drone_vel(self) -> np.ndarray:
        return self.data.qvel[self._qvel_adr : self._qvel_adr + 3]

    @property
    def drone_yaw(self) -> float:
        return float(wrap_angle(self.data.qpos[self._yaw_qpos_adr]))

    @property
    def drone_yaw_rate(self) -> float:
        return float(self.data.qvel[self._yaw_qvel_adr])

    @property
    def viewer_running(self) -> bool:
        """False once the human-mode viewer window has been closed."""
        return self._viewer is None or self._viewer.is_running()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.step_count = 0
        mujoco.mj_resetData(self.model, self.data)
        self._task_reset(options or {})
        mujoco.mj_forward(self.model, self.data)
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), self._get_info(self._task_info())

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self._task_before_step()
        vx, vy, yaw_rate = action * self.action_scale
        c, s = np.cos(self.drone_yaw), np.sin(self.drone_yaw)
        # Altitude hold: the body is gravity compensated, so this only corrects numerical drift.
        vz = ALTITUDE_GAIN * (self.flight_altitude - self.drone_pos[2])
        self.data.ctrl[:] = [c * vx - s * vy, s * vx + c * vy, vz, yaw_rate]
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self.step_count += 1

        reward, terminated, info = self._task_step()
        reward -= self.ctrl_cost_weight * float(np.sum(action**2))
        if not terminated and self._out_of_bounds():
            reward -= self.out_of_bounds_penalty
            terminated = True
            info["out_of_bounds"] = True

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, False, self._get_info(info)

    def camera_image(self, camera: str | int = ONBOARD_CAMERA, width: int | None = None, height: int | None = None):
        """Render ``camera`` (name, id, or -1 for the free camera) to an RGB array."""
        size = (height or self.image_size, width or self.image_size)
        if size not in self._renderers:
            self._renderers[size] = mujoco.Renderer(self.model, *size)
            # The first frame of a new GL context shades the floor texture slightly differently
            # (a few intensity levels); discard it so identical states give identical images.
            self._renderers[size].update_scene(self.data, camera=camera)
            self._renderers[size].render()
        renderer = self._renderers[size]
        renderer.update_scene(self.data, camera=camera)
        return renderer.render()

    def detect(self, geom_id: int) -> np.ndarray:
        """Exact ``[visible, cx, cy, w, h]`` box of ``geom_id`` in the onboard camera (no occlusion)."""
        center, half = self.model.geom_aabb[geom_id, :3], self.model.geom_aabb[geom_id, 3:]
        signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).reshape(3, -1).T
        geom_mat = self.data.geom_xmat[geom_id].reshape(3, 3)
        corners = self.data.geom_xpos[geom_id] + (center + signs * half) @ geom_mat.T

        cam_mat = self.data.cam_xmat[self._onboard_cam_id].reshape(3, 3)
        local = (corners - self.data.cam_xpos[self._onboard_cam_id]) @ cam_mat
        local = local[local[:, 2] < -1e-3]  # corners in front of the camera
        if len(local) == 0:
            return np.zeros(5)
        # Normalized image coordinates in [0, 1], origin top-left (camera x right, y up).
        scale = 0.5 / np.tan(self._half_fov)
        u = 0.5 + scale * local[:, 0] / -local[:, 2]
        v = 0.5 - scale * local[:, 1] / -local[:, 2]
        u0, u1 = np.clip([u.min(), u.max()], 0.0, 1.0)
        v0, v1 = np.clip([v.min(), v.max()], 0.0, 1.0)
        w, h = u1 - u0, v1 - v0
        if min(w, h) < self.min_detection_size:
            return np.zeros(5)
        return np.array([1.0, (u0 + u1) / 2, (v0 + v1) / 2, w, h])

    def detection_obs(self) -> np.ndarray:
        """Detection of the task target as a noisy detector would report it."""
        det = self.detect(self._task_target_geom())
        if det[0] == 0.0 or self.np_random.random() < self.detection_dropout:
            return np.zeros(5, dtype=np.float32)
        det[1:] += self.np_random.normal(scale=self.detection_noise, size=4)
        det[1:] = np.clip(det[1:], 0.0, 1.0)
        return det.astype(np.float32)

    def state_obs(self) -> np.ndarray:
        yaw = self.drone_yaw
        return np.concatenate(
            [self.drone_pos[:2], self.drone_vel[:2], [np.cos(yaw), np.sin(yaw), self.drone_yaw_rate], self._task_obs()]
        ).astype(np.float32)

    def render(self):
        if self.render_mode == "rgb_array":
            return self.camera_image(-1 if self.camera is None else self.camera, self.width, self.height)
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
                if self.camera is not None:
                    self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    self._viewer.cam.fixedcamid = self.model.camera(self.camera).id
            # Pace the passive viewer to real time.
            if self._last_render_time is not None:
                time.sleep(max(0.0, self.dt - (time.perf_counter() - self._last_render_time)))
            self._last_render_time = time.perf_counter()
            self._viewer.sync()
        return None

    def close(self):
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # --- helpers for subclasses -------------------------------------------------------

    def _set_drone_state(self, xy, yaw: float = 0.0) -> None:
        self.data.qpos[self._qpos_adr : self._qpos_adr + 3] = [*xy, self.flight_altitude]
        self.data.qpos[self._yaw_qpos_adr] = yaw
        self.data.qvel[:] = 0.0

    def _sample_xy(self, margin: float = 0.5) -> np.ndarray:
        return self.np_random.uniform(-1.0, 1.0, size=2) * (self.arena_half_extent - margin)

    def _sample_yaw(self) -> float:
        return float(self.np_random.uniform(-np.pi, np.pi))

    def _mocap_id(self, body_name: str) -> int:
        return int(self.model.body_mocapid[self.model.body(body_name).id])

    def _drone_contacts(self, geom_id: int) -> bool:
        """Whether any drone geom is currently touching ``geom_id``."""
        g1, g2 = self.data.contact.geom1, self.data.contact.geom2  # length ncon
        other = np.concatenate([g2[g1 == geom_id], g1[g2 == geom_id]])
        return bool(np.any(self.model.geom_bodyid[other] == self._drone_body_id))

    def _point_in_view(self, point) -> bool:
        """Whether ``point`` (world frame) projects inside the onboard camera image."""
        cam_mat = self.data.cam_xmat[self._onboard_cam_id].reshape(3, 3)
        x, y, z = (np.asarray(point) - self.data.cam_xpos[self._onboard_cam_id]) @ cam_mat
        limit = np.tan(self._half_fov) * -z  # square image: same FOV both ways
        return bool(z < 0 and abs(x) <= limit and abs(y) <= limit)

    def _heading_error(self, point) -> float:
        """Angle from the drone's nose to ``point`` in the horizontal plane."""
        dx, dy = np.asarray(point)[:2] - self.drone_pos[:2]
        return float(wrap_angle(np.arctan2(dy, dx) - self.drone_yaw))

    def _out_of_bounds(self) -> bool:
        return bool(np.max(np.abs(self.drone_pos[:2])) > self.arena_half_extent)

    def _get_obs(self) -> np.ndarray:
        if self.obs_mode == "detection":
            return self.detection_obs()
        return self.camera_image() if self.obs_mode == "pixels" else self.state_obs()

    def _get_info(self, info: dict[str, Any]) -> dict[str, Any]:
        info["target_in_view"] = bool(self.detect(self._task_target_geom())[0])
        return info

    # --- task hooks -------------------------------------------------------------------

    def _build_scene(self, spec: mujoco.MjSpec) -> None:
        """Add task-specific bodies to the scene before it is compiled."""

    def _task_reset(self, options: dict) -> None:
        """Place the drone and the task props. Runs after ``mj_resetData``."""
        self._set_drone_state(self._sample_xy(), self._sample_yaw())

    def _task_before_step(self) -> None:
        """Advance task props (e.g. moving targets) once per control step."""

    def _task_obs(self) -> np.ndarray:
        """Privileged task state, used by ``obs_mode="state"``."""
        return np.zeros(0)

    def _task_target_geom(self) -> int:
        """Id of the geom the detector looks for (the waypoint, pad or tracked target)."""
        raise NotImplementedError

    def _task_step(self) -> tuple[float, bool, dict[str, Any]]:
        """Return ``(reward, terminated, info)`` after the physics step."""
        return 0.0, False, self._task_info()

    def _task_info(self) -> dict[str, Any]:
        return {}
