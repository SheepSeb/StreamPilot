from gymnasium.envs.registration import register

from streampilot.env.base import DroneBaseEnv
from streampilot.env.landing import LandingEnv
from streampilot.env.tracking import TrackingEnv
from streampilot.env.waypoint import WaypointEnv

__all__ = ["DroneBaseEnv", "LandingEnv", "TrackingEnv", "WaypointEnv"]

# 20 Hz control (timestep 0.01 s, frame_skip 5).
register(id="DroneWaypoint-v0", entry_point="streampilot.env:WaypointEnv", max_episode_steps=400)
register(id="DroneLanding-v0", entry_point="streampilot.env:LandingEnv", max_episode_steps=400)
register(id="DroneTracking-v0", entry_point="streampilot.env:TrackingEnv", max_episode_steps=500)
