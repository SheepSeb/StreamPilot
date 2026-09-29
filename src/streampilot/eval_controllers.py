"""Evaluate the classical controllers (PID, MPC) and the scripted policy on the drone tasks.

    uv run streampilot-eval-controllers                       # every task, every controller
    uv run streampilot-eval-controllers waypoint formation-landing --controllers mpc --episodes 50

Episodes use seeds ``0, 1, ...`` (as ``streampilot`` does), so results are comparable across
controllers. Reports the mean return, success rate and, for formations, the collision rate.
"""

import argparse
import time

import gymnasium as gym
import numpy as np

import streampilot.env  # noqa: F401  (registers the environments)
from streampilot.control import ControllerPolicy
from streampilot.visualize import FORMATION_POLICIES, FORMATION_TASKS, SCRIPTED_POLICIES, TASKS


def scripted(task):
    if task in FORMATION_TASKS:
        return lambda env: FORMATION_POLICIES[task](env)
    return lambda env: SCRIPTED_POLICIES[task](env, env.state_obs())


def run(task: str, controller: str, episodes: int, drones: int, seed: int) -> dict:
    formation = task in FORMATION_TASKS
    kwargs = {"num_drones": drones} if formation else {}
    env = gym.make((FORMATION_TASKS if formation else TASKS)[task], obs_mode="state", **kwargs)
    base = env.unwrapped
    policy = scripted(task) if controller == "scripted" else ControllerPolicy(controller)
    results = []
    for episode in range(episodes):
        env.reset(seed=seed + episode)
        if controller != "scripted":
            policy.reset(base)
        total, done = 0.0, False
        while not done:
            _, reward, terminated, truncated, info = env.step(policy(base))
            total += reward
            done = terminated or truncated
        results.append(
            (total, base.step_count, float(np.all(info.get("is_success", False))), float(np.any(info.get("collision", False))))
        )
    env.close()
    ret, length, success, collision = np.mean(results, axis=0)
    return {"return": ret, "length": length, "success": success, "collision": collision}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tasks", nargs="*", default=[*TASKS, *FORMATION_TASKS])
    parser.add_argument("--controllers", nargs="+", default=["scripted", "pid", "mpc"], choices=["scripted", "pid", "mpc"])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--drones", type=int, default=3, choices=[2, 3])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"{'task':<20}{'controller':<11}{'return':>9}{'length':>8}{'success':>9}{'collision':>11}{'ms/step':>9}")
    for task in args.tasks:
        for controller in args.controllers:
            start = time.perf_counter()
            r = run(task, controller, args.episodes, args.drones, args.seed)
            ms = 1000 * (time.perf_counter() - start) / (r["length"] * args.episodes)
            print(
                f"{task:<20}{controller:<11}{r['return']:9.2f}{r['length']:8.1f}{r['success']:9.2f}{r['collision']:11.2f}{ms:9.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
