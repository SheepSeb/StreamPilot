"""Morphing target tracking: the tracked target changes its shape and its motion mid-episode."""

import mujoco
import numpy as np

from streampilot.env.tracking import TrackingEnv, wander

RED = [0.9, 0.1, 0.1, 1.0]
# name: (geom type, size, height of the geom centre above the floor). All stand on the floor
# except the ball, which floats at about camera height like another drone would.
SHAPES = {
    "pillar": (mujoco.mjtGeom.mjGEOM_BOX, [0.15, 0.15, 0.8], 0.8),
    "cylinder": (mujoco.mjtGeom.mjGEOM_CYLINDER, [0.3, 0.6, 0.0], 0.6),
    "crate": (mujoco.mjtGeom.mjGEOM_BOX, [0.45, 0.25, 0.4], 0.4),
    "ball": (mujoco.mjtGeom.mjGEOM_SPHERE, [0.3, 0.0, 0.0], 1.0),
}
MOTIONS = ("wander", "circle", "line", "stop")


class MorphingTrackingEnv(TrackingEnv):
    """``TrackingEnv`` with a target that changes mid-run. Every ``switch_steps`` (a range, sampled
    anew after each switch) the target takes a new shape from ``shapes`` and a new motion from
    ``motions``, both different from the current ones:

    - shapes: ``pillar`` (the tracking pillar), ``cylinder`` (wide and short), ``crate`` (a low,
      wide box) and ``ball`` (floating at camera height);
    - motions: ``wander`` (the tracking random walk), ``circle`` (orbits a point, radius 0.5 to
      1.5 m), ``line`` (constant velocity, bouncing off the walls) and ``stop`` (stands still).

    So the detector box changes size and aspect ratio abruptly, and the target's velocity changes
    character, which a policy has to adapt to online. Task, reward and privileged task obs are
    those of ``TrackingEnv``; ``info`` also has the current ``shape`` and ``motion`` and
    ``switched`` (whether they changed before this step).
    """

    def __init__(
        self,
        shapes: tuple[str, ...] = tuple(SHAPES),
        motions: tuple[str, ...] = MOTIONS,
        switch_steps: tuple[int, int] = (80, 160),
        target_min_speed: float = 0.3,
        **kwargs,
    ):
        assert set(shapes) <= set(SHAPES) and set(motions) <= set(MOTIONS)
        self.shapes, self.motions = tuple(shapes), tuple(motions)
        self.switch_steps = switch_steps
        self.target_min_speed = target_min_speed
        super().__init__(**kwargs)
        self._shape_geoms = {name: self.model.geom(f"target_{name}").id for name in self.shapes}
        self._shape = self.shapes[0]
        self._motion = self.motions[0]
        self._next_switch = 0
        self._switched = False
        self._circle_center = np.zeros(2)
        self._circle_rate = 0.0

    def _build_scene(self, spec: mujoco.MjSpec) -> None:
        # One mocap body on the floor carrying every shape; only the active one is visible.
        target = spec.worldbody.add_body(name="target", mocap=True)
        for name in self.shapes:
            geom_type, size, height = SHAPES[name]
            target.add_geom(
                name=f"target_{name}",
                type=geom_type,
                size=size,
                pos=[0.0, 0.0, height],
                rgba=RED,
                contype=0,
                conaffinity=0,
            )
        # TrackingEnv looks up a geom named "target"; point it at a zero-size marker.
        target.add_geom(
            name="target",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[1e-3, 0.0, 0.0],
            rgba=[0.0, 0.0, 0.0, 0.0],
            contype=0,
            conaffinity=0,
        )

    def _set_shape(self, name: str) -> None:
        self._shape = name
        for shape, geom in self._shape_geoms.items():
            self.model.geom_rgba[geom, 3] = 1.0 if shape == name else 0.0

    def _set_motion(self, name: str) -> None:
        self._motion = name
        pos = self._target_pos[:2]
        speed = self.np_random.uniform(self.target_min_speed, self.target_max_speed)
        if name == "stop":
            self._target_vel[:2] = 0.0
        elif name == "line":
            angle = self.np_random.uniform(-np.pi, np.pi)
            self._target_vel[:2] = speed * np.array([np.cos(angle), np.sin(angle)])
        elif name == "circle":
            # Orbit a centre towards the middle of the arena, so the circle mostly stays inside.
            radius = self.np_random.uniform(0.5, 1.5)
            inward = -pos / max(np.linalg.norm(pos), 1e-6) if np.linalg.norm(pos) > 1e-6 else np.array([1.0, 0.0])
            self._circle_center = pos + radius * inward
            self._circle_rate = self.np_random.choice([-1.0, 1.0]) * speed / radius
        # "wander" keeps the current velocity as the initial state of its random walk.

    def _switch(self, first: bool = False) -> None:
        shapes = [s for s in self.shapes if first or s != self._shape] or [self._shape]
        motions = [m for m in self.motions if first or m != self._motion] or [self._motion]
        self._set_shape(str(self.np_random.choice(shapes)))
        self._set_motion(str(self.np_random.choice(motions)))
        self._next_switch = self.step_count + int(self.np_random.integers(*self.switch_steps, endpoint=True))

    def _task_reset(self, options: dict) -> None:
        super()._task_reset(options)
        self.data.mocap_pos[self._target][2] = 0.0  # the shapes carry their own height
        self._switch(first=True)
        self._switched = False

    def _task_before_step(self) -> None:
        self._switched = self.step_count >= self._next_switch
        if self._switched:
            self._switch()
        pos, vel = self._target_pos[:2].copy(), self._target_vel[:2].copy()
        limit = self.target_half_extent
        if self._motion == "wander":
            pos, vel = wander(
                self.np_random,
                pos,
                vel,
                self.dt,
                self.target_max_speed,
                self.target_speed_reversion,
                self.target_speed_noise,
                limit,
            )
        elif self._motion == "line":
            pos = pos + vel * self.dt
            bounced = np.abs(pos) > limit
            vel[bounced] *= -1.0
        elif self._motion == "circle":
            rel = pos - self._circle_center
            angle = self._circle_rate * self.dt
            c, s = np.cos(angle), np.sin(angle)
            new = self._circle_center + np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]])
            new = np.clip(new, -limit, limit)
            vel, pos = (new - pos) / self.dt, new
        self._target_vel[:2] = vel
        self.data.mocap_pos[self._target][:2] = np.clip(pos, -limit, limit)

    def _task_target_geom(self) -> int:
        return self._shape_geoms[self._shape]

    def _task_info(self) -> dict:
        return {**super()._task_info(), "shape": self._shape, "motion": self._motion, "switched": self._switched}
