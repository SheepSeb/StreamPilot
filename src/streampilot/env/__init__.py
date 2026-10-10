from gymnasium.envs.registration import register

from streampilot.env.base import DroneBaseEnv
from streampilot.env.formation import (
    FormationFactoryEnv,
    FormationLandingEnv,
    FormationTrackingEnv,
    FormationWarehouseLandingEnv,
    FormationWarehouseTrackingEnv,
    FormationWarehouseWaypointEnv,
    FormationWaypointEnv,
)
from streampilot.env.landing import LandingEnv
from streampilot.env.morphing import MorphingTrackingEnv
from streampilot.env.tracking import TrackingEnv
from streampilot.env.warehouse import (
    WarehouseLandingEnv,
    WarehouseMorphingTrackingEnv,
    WarehouseTrackingEnv,
    WarehouseWaypointEnv,
)
from streampilot.env.waypoint import WaypointEnv

__all__ = [
    "DroneBaseEnv",
    "FormationFactoryEnv",
    "FormationLandingEnv",
    "FormationTrackingEnv",
    "FormationWarehouseLandingEnv",
    "FormationWarehouseTrackingEnv",
    "FormationWarehouseWaypointEnv",
    "FormationWaypointEnv",
    "LandingEnv",
    "MorphingTrackingEnv",
    "TrackingEnv",
    "WarehouseLandingEnv",
    "WarehouseMorphingTrackingEnv",
    "WarehouseTrackingEnv",
    "WarehouseWaypointEnv",
    "WaypointEnv",
]

# 20 Hz control (timestep 0.01 s, frame_skip 5).
register(id="DroneWaypoint-v0", entry_point="streampilot.env:WaypointEnv", max_episode_steps=400)
register(id="DroneLanding-v0", entry_point="streampilot.env:LandingEnv", max_episode_steps=400)
register(id="DroneTracking-v0", entry_point="streampilot.env:TrackingEnv", max_episode_steps=500)
register(id="DroneMorphingTracking-v0", entry_point="streampilot.env:MorphingTrackingEnv", max_episode_steps=600)
# The same, in the warehouse (visual only).
register(id="DroneWarehouseWaypoint-v0", entry_point="streampilot.env:WarehouseWaypointEnv", max_episode_steps=400)
register(id="DroneWarehouseLanding-v0", entry_point="streampilot.env:WarehouseLandingEnv", max_episode_steps=400)
register(id="DroneWarehouseTracking-v0", entry_point="streampilot.env:WarehouseTrackingEnv", max_episode_steps=500)
register(
    id="DroneWarehouseMorphingTracking-v0", entry_point="streampilot.env:WarehouseMorphingTrackingEnv", max_episode_steps=600
)

# Multi-drone formation tasks (num_drones=3 by default, or 2).
register(id="DroneFormationWaypoint-v0", entry_point="streampilot.env:FormationWaypointEnv", max_episode_steps=600)
register(id="DroneFormationLanding-v0", entry_point="streampilot.env:FormationLandingEnv", max_episode_steps=500)
register(id="DroneFormationTracking-v0", entry_point="streampilot.env:FormationTrackingEnv", max_episode_steps=500)
register(id="DroneFormationFactory-v0", entry_point="streampilot.env:FormationFactoryEnv", max_episode_steps=600)
register(id="DroneFormationWarehouseWaypoint-v0", entry_point="streampilot.env:FormationWarehouseWaypointEnv", max_episode_steps=600)
register(id="DroneFormationWarehouseLanding-v0", entry_point="streampilot.env:FormationWarehouseLandingEnv", max_episode_steps=500)
register(id="DroneFormationWarehouseTracking-v0", entry_point="streampilot.env:FormationWarehouseTrackingEnv", max_episode_steps=500)
