from gymnasium.envs.registration import register

from streampilot.env.base import DroneBaseEnv
from streampilot.env.formation import FormationLandingEnv, FormationTrackingEnv, FormationWaypointEnv
from streampilot.env.landing import LandingEnv
from streampilot.env.tracking import TrackingEnv
from streampilot.env.waypoint import WaypointEnv

__all__ = [
    "DroneBaseEnv",
    "FormationLandingEnv",
    "FormationTrackingEnv",
    "FormationWaypointEnv",
    "LandingEnv",
    "TrackingEnv",
    "WaypointEnv",
]

# 20 Hz control (timestep 0.01 s, frame_skip 5).
register(id="DroneWaypoint-v0", entry_point="streampilot.env:WaypointEnv", max_episode_steps=400)
register(id="DroneLanding-v0", entry_point="streampilot.env:LandingEnv", max_episode_steps=400)
register(id="DroneTracking-v0", entry_point="streampilot.env:TrackingEnv", max_episode_steps=500)

# Multi-drone formation tasks (num_drones=3 by default, or 2).
register(id="DroneFormationWaypoint-v0", entry_point="streampilot.env:FormationWaypointEnv", max_episode_steps=600)
register(id="DroneFormationLanding-v0", entry_point="streampilot.env:FormationLandingEnv", max_episode_steps=500)
register(id="DroneFormationTracking-v0", entry_point="streampilot.env:FormationTrackingEnv", max_episode_steps=500)
