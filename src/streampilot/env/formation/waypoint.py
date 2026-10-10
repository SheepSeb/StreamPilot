"""Formation waypoints: the team flies to a sequence of formation-shaped sets of waypoints."""

from itertools import permutations

import mujoco
import numpy as np

from streampilot.env.formation.base import DRONE_COLORS, FormationBaseEnv, rotation


def _segment_distance(a0, a1, b0, b1) -> float:
    """Shortest distance between the 2D segments ``a0 a1`` and ``b0 b1``."""
    d1, d2, r = a1 - a0, b1 - b0, a0 - b0
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) > 1e-9:  # they intersect if both crossing parameters fall inside the segments
        t = (d2[0] * r[1] - d2[1] * r[0]) / cross
        u = (d1[0] * r[1] - d1[1] * r[0]) / cross
        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
            return 0.0

    def to_segment(p, q0, q1):
        d = q1 - q0
        t = np.clip((p - q0) @ d / max(d @ d, 1e-12), 0.0, 1.0)
        return np.linalg.norm(p - (q0 + t * d))

    return min(to_segment(a0, b0, b1), to_segment(a1, b0, b1), to_segment(b0, a0, a1), to_segment(b1, a0, a1))


class FormationWaypointEnv(FormationBaseEnv):
    """Each stage shows one waypoint per drone (a ball at flight altitude in the drone's colour, its
    detector target), laid out in the formation shape (triangle for three drones, column for two)
    at a random position and rotation. Drone ``i`` flies to ball ``i``; the stage is complete once
    every drone is within ``reach_radius`` of its ball at the same time, and the next set appears.
    The ball is drawn smaller (``ball_radius``) than ``reach_radius``: up close, a ball fills the
    camera and its box stops telling the drone how far it is, so a smaller ball keeps the box
    informative for longer while arrival still counts at the larger radius.
    The balls of each stage are handed out to minimise the team's total travel (from the start
    positions for the first stage, from where the drones are after that), so the straight paths
    rarely cross: with a fixed assignment, 3 drones' paths come within contact distance on
    most stages. Each stage after the first appears once the previous one is complete, with every
    drone's ball in its camera view as far as possible (``next_stage_in_view``), as the drones start facing their first
    balls: with only a detector, a ball behind a drone leaves it nothing to fly by. Privileged task
    obs: horizontal vector from the drone to its waypoint. Every stage is also placed so that the
    drones' straight paths to their balls stay ``path_clearance`` apart: most collisions were a
    drone flying straight at its ball into a teammate waiting at (or starting from) its own.

    Reward: mean progress towards the waypoints, plus a bonus for each completed stage, plus
    potential-based shaping on ``hold_reward`` times the fraction of drones inside their ball and
    ``view_reward`` times the fraction seeing it: entering a ball or getting it in view pays once,
    leaving or losing it costs as much, and staying pays nothing. (Paid every step instead, the
    team learned to park two drones in their balls and never complete the stage.) The episode terminates successfully after the last stage. The drones
    spawn further apart than in the base env, each roughly facing its first ball, and pay a wall
    cost near the arena edge and a proximity cost (``proximity_weight``), which matters most when a
    stage changes and the drones cross to the next set of balls, plus the base env's separation
    shaping (``separation_weight``): drones close in on each other at over 1 m/s, so the proximity
    band alone is crossed in a few steps, while the shaping charges for closing in from 1.8 m. The collision penalty is the base
    env's, as in tracking. The formation is wider than the base env's (``formation_spacing``): at
    1.0 m, two drones each inside their ``reach_radius`` but off towards each other already touch.
    """

    task_obs_dim = 2

    def __init__(
        self,
        num_stages: int = 3,
        reach_radius: float = 0.3,
        ball_radius: float = 0.15,
        progress_weight: float = 10.0,
        stage_bonus: float = 10.0,
        hold_reward: float = 2.0,
        view_reward: float = 1.0,
        assign_nearest: bool = True,
        next_stage_in_view: bool = True,
        min_stage_distance: float = 0.8,
        path_clearance: float = 0.8,
        spawn_separation: float = 1.5,
        formation_spacing: float = 1.5,
        spawn_heading_noise_deg: float | None = 30.0,
        wall_weight: float = 2.0,
        geofence: float | None = 0.3,
        separation_weight: float = 5.0,
        proximity_weight: float = 1.0,
        **kwargs,
    ):
        self.num_stages = num_stages
        self.reach_radius = reach_radius
        self.ball_radius = ball_radius
        self.progress_weight = progress_weight
        self.stage_bonus = stage_bonus
        self.hold_reward = hold_reward
        self.view_reward = view_reward
        self.assign_nearest = assign_nearest
        self.next_stage_in_view = next_stage_in_view
        self.min_stage_distance = min_stage_distance
        self.path_clearance = path_clearance
        super().__init__(
            spawn_separation=spawn_separation,
            formation_spacing=formation_spacing,
            spawn_heading_noise_deg=spawn_heading_noise_deg,
            wall_weight=wall_weight,
            geofence=geofence,
            proximity_weight=proximity_weight,
            separation_weight=separation_weight,
            **kwargs,
        )
        self._markers = [self._mocap_id(f"waypoint{i}") for i in range(self.num_drones)]
        self._marker_geoms = [self.model.geom(f"waypoint{i}").id for i in range(self.num_drones)]
        self._stages = np.zeros((num_stages, self.num_drones, 2))
        self._reached = 0
        self._prev_dist = np.zeros(self.num_drones)
        self._bonus_potential = 0.0

    def _build_scene(self, spec: mujoco.MjSpec) -> None:
        for i in range(self.num_drones):
            marker = spec.worldbody.add_body(name=f"waypoint{i}", mocap=True)
            marker.add_geom(
                name=f"waypoint{i}",
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[self.ball_radius, 0, 0],
                rgba=[*DRONE_COLORS[i], 1.0],
                contype=0,
                conaffinity=0,
            )

    @property
    def _targets(self) -> np.ndarray:
        # Stays on the last stage after it is complete, so the final observation is valid.
        return self._stages[min(self._reached, self.num_stages - 1)]

    def _distances(self) -> np.ndarray:
        return np.linalg.norm(self._targets - self.drone_pos[:, :2], axis=1)

    def _show_targets(self) -> None:
        for marker, xy in zip(self._markers, self._targets):
            self.data.mocap_pos[marker] = [*xy, self.flight_altitude]

    def _sample_stage(self) -> np.ndarray:
        shape = self.formation - self.formation.mean(axis=0)
        radius = np.max(np.linalg.norm(shape, axis=1))
        angle = self.np_random.uniform(-np.pi, np.pi)
        return self._sample_xy(margin=0.5 + radius) + shape @ rotation([np.cos(angle), np.sin(angle)]).T

    def _assign(self, stage: np.ndarray, start: np.ndarray) -> np.ndarray:
        """``stage`` reordered so that drone ``i``, starting at ``start[i]``, gets the ball that
        minimises the team's total travel."""
        if not self.assign_nearest:
            return stage
        order = min(permutations(range(self.num_drones)), key=lambda p: np.linalg.norm(stage[list(p)] - start, axis=1).sum())
        return stage[list(order)]

    def _clearance(self, start: np.ndarray, stage: np.ndarray) -> float:
        """Closest any two drones' straight paths (``start[i]`` to ``stage[i]``) come: a drone that
        flies straight at its ball must not pass where a teammate starts, flies or waits."""
        a, b = self._pairs
        return min((_segment_distance(start[i], stage[i], start[j], stage[j]) for i, j in zip(a, b)), default=np.inf)

    def _clear_stage(self, start: np.ndarray, tries: int = 200) -> np.ndarray:
        """A stage anywhere in the arena whose paths from ``start`` keep ``path_clearance`` (the
        clearest candidate, if none of ``tries`` does)."""
        best, best_clear = None, -1.0
        for _ in range(tries):
            stage = self._assign(self._sample_stage(), start)
            clear = self._clearance(start, stage)
            if clear > best_clear:
                best, best_clear = stage, clear
                if clear >= self.path_clearance:
                    break
        return best

    def _visible_stage(self, tries: int = 200) -> np.ndarray:
        """A new stage ahead of the team, assigned from the drones' current positions, with every
        drone's ball in its camera view and at least ``min_stage_distance`` away (the candidate
        meeting that for the most drones, if none of ``tries`` does for all: at a stage change the
        drones face different ways, often too far apart for one stage to be in every view)."""
        pos = self.drone_pos[:, :2]
        forward = np.array([np.cos(self.drone_yaw).mean(), np.sin(self.drone_yaw).mean()])
        forward /= max(np.linalg.norm(forward), 1e-6)  # the team's mean heading
        shape = self.formation - self.formation.mean(axis=0)
        limit = self.arena_half_extent - 0.5 - np.max(np.linalg.norm(shape, axis=1))
        best, best_score = None, (False, -1)
        for _ in range(tries):
            # Centred ahead of the team, where most of its cameras look, at a random rotation.
            centre = np.clip(pos.mean(axis=0) + forward * self.np_random.uniform(1.5, 3.0), -limit, limit)
            angle = self.np_random.uniform(-np.pi, np.pi)
            stage = self._assign(centre + shape @ rotation([np.cos(angle), np.sin(angle)]).T, pos)
            far = np.linalg.norm(stage - pos, axis=1) >= self.min_stage_distance
            seen = [self._point_in_view(i, [*xy, self.flight_altitude]) for i, xy in enumerate(stage)]
            # Clear paths first (collisions end the episode), then as many balls in view as possible.
            score = (self._clearance(pos, stage) >= self.path_clearance, int(np.sum(far & np.array(seen))))
            if score > best_score:
                best, best_score = stage, score
                if score == (True, self.num_drones):
                    break
        return best

    def _task_reset(self, options: dict) -> None:
        xy = self._sample_separated_xy()
        start, stages = xy, []
        for _ in range(self.num_stages):
            stages.append(self._clear_stage(start))
            start = stages[-1]
        self._stages = np.stack(stages)
        self._set_drone_states(xy, 0.0)  # positions first: the headings are relative to them
        self._set_drone_states(xy, self._sample_yaw(look_at=self._stages[0]))
        self._reached = 0
        self._show_targets()
        self._prev_dist = self._distances()
        self._bonus_potential = self._potential(self._prev_dist)

    def _task_target_geom(self, drone: int) -> int:
        return self._marker_geoms[drone]

    def _task_obs(self, drone: int) -> np.ndarray:
        return self._targets[drone] - self.drone_pos[drone, :2]

    def _potential(self, dist: np.ndarray) -> float:
        """Shaping potential: ``hold_reward`` times the fraction of drones inside their ball plus
        ``view_reward`` times the fraction seeing it (a drone at its ball counts as seeing it: up
        close the ball leaves the camera's view, and losing the view there would make arriving
        cost)."""
        inside = dist < self.reach_radius
        in_view = [self._point_in_view(i, [*xy, self.flight_altitude]) for i, xy in enumerate(self._targets)]
        return self.hold_reward * float(np.mean(inside)) + self.view_reward * float(np.mean(np.array(in_view) | inside))

    def _task_step(self):
        dist = self._distances()
        reward = self.progress_weight * float(np.mean(self._prev_dist - dist))
        if np.all(dist < self.reach_radius):
            reward += self.stage_bonus
            self._reached += 1
            if self.next_stage_in_view and self._reached < self.num_stages:
                self._stages[self._reached] = self._visible_stage()
            self._show_targets()
            dist = self._distances()
        # Potential-based (gamma * phi(s') - phi(s)): seeing and entering a ball pay once, losing
        # them costs, and staying pays nothing, so the team cannot farm them by never finishing.
        # The episode's end (the last stage) has potential 0.
        potential = 0.0 if self._reached == self.num_stages else self._potential(dist)
        reward += self.shaping_gamma * potential - self._bonus_potential
        self._bonus_potential = potential
        self._prev_dist = dist
        return reward, self._reached == self.num_stages, self._task_info()

    def _task_info(self) -> dict:
        return {
            "stages_reached": self._reached,
            "distance": self._prev_dist.copy(),
            "is_success": self._reached == self.num_stages,
        }

    def goals(self):
        return self._targets.copy(), None
