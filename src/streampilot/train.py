"""Train a drone task with Stream AC(lambda) or one of the non-streaming baselines (PPO, SAC).

    uv run streampilot-train waypoint --steps 2000000
    uv run streampilot-train landing --detection-noise 0.01 --detection-dropout 0.05
    uv run streampilot-train tracking --obs-mode state   # privileged state, for debugging
    uv run streampilot-train waypoint --algo ppo         # batch baselines: ppo, sac
    uv run streampilot-train waypoint --algo sac --help  # lists the algorithm's hyperparameters

All algorithms see the same features (observation history plus previous action), run the same
loop (one environment, one transition at a time) and are evaluated the same way, so runs compare
directly. Logs episodes and periodic deterministic evaluations to Trackio (``uv run trackio show``
opens the dashboard) and writes ``latest.pt`` (at every evaluation) and ``final.pt`` to ``--out``.
Watch a checkpoint with ``uv run streampilot TASK --policy RUN_DIR/final.pt``.
"""

import argparse
import os
import time
from collections import defaultdict, deque
from pathlib import Path

# Trackio logs locally; online, importing it blocks on Hugging Face Hub network calls.
# Set HF_HUB_OFFLINE=0 to sync to a Hub Space instead.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import trackio  # noqa: E402

import streampilot.env  # noqa: E402, F401  (registers the environments)
from streampilot.baselines import ppo, sac  # noqa: E402
from streampilot.policy import Policy  # noqa: E402
from streampilot.stream_x import agents as stream_ac  # noqa: E402
from streampilot.stream_x.wrappers import HistoryObservation, NormalizeObservation, ScaleReward  # noqa: E402

TASKS = {"waypoint": "DroneWaypoint-v0", "landing": "DroneLanding-v0", "tracking": "DroneTracking-v0"}
# Each module provides NORMALIZE, add_args(parser), make_agent(args, obs_dim, action_dim) and Actor.
# Agents provide act(obs), observe(obs, action, reward, next_obs, terminated, done) -> metrics and
# state_dict() (with the actor's weights under "actor").
ALGOS = {"stream_ac": stream_ac, "ppo": ppo, "sac": sac}


def make_env(config: dict) -> gym.Env:
    env = gym.make(TASKS[config["task"]], obs_mode=config["obs_mode"], **config["env_kwargs"])
    env = gym.wrappers.RecordEpisodeStatistics(env)  # raw (unscaled) returns in info["episode"]
    env = HistoryObservation(env, config["num_frames"])
    if not ALGOS[config.get("algo", "stream_ac")].NORMALIZE:
        return env
    env = NormalizeObservation(env)
    return ScaleReward(env, gamma=config["gamma"])


def save_checkpoint(path: Path, config: dict, agent, env: gym.Env) -> None:
    base = env.unwrapped
    normalized = ALGOS[config.get("algo", "stream_ac")].NORMALIZE
    torch.save(
        {
            "config": config,
            "obs_dim": int(np.prod(base.observation_space.shape)),
            "action_dim": int(np.prod(base.action_space.shape)),
            **agent.state_dict(),
            "obs_stats": env.get_wrapper_attr("obs_stats").state_dict() if normalized else None,
        },
        path,
    )


def evaluate(checkpoint: Path, config: dict, episodes: int, seed: int) -> dict:
    """Deterministic rollouts of a saved policy on fresh seeds."""
    env = gym.make(TASKS[config["task"]], obs_mode=config["obs_mode"], **config["env_kwargs"])
    policy = Policy.load(checkpoint)
    returns, successes = [], []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        policy.reset()
        total, done = 0.0, False
        while not done:
            obs, reward, terminated, truncated, info = env.step(policy(obs))
            total += reward
            done = terminated or truncated
        returns.append(total)
        successes.append(info.get("is_success", np.nan))
    env.close()
    return {"return": float(np.mean(returns)), "success": float(np.mean(successes))}


def log_eval(checkpoint: Path, config: dict, episodes: int, step: int) -> dict:
    # Same seeds every time, so evaluations are comparable across checkpoints.
    result = evaluate(checkpoint, config, episodes, seed=10_000)
    trackio.log({f"eval/{k}": v for k, v in result.items() if not np.isnan(v)}, step=step)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a drone task with Stream AC(lambda) or a non-streaming baseline (PPO, SAC)."
    )
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("--algo", choices=ALGOS, default="stream_ac")
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--obs-mode", choices=["detection", "state"], default="detection")
    parser.add_argument("--frames", type=int, default=4, help="observations stacked into the input")
    parser.add_argument("--detection-noise", type=float, default=0.0)
    parser.add_argument("--detection-dropout", type=float, default=0.0)
    parser.add_argument("--out", type=Path, default=None, help="default: runs/[ALGO_]TASK_seedSEED")
    parser.add_argument("--project", default="streampilot", help="Trackio project")
    parser.add_argument("--log-every", type=int, default=50, help="episodes between progress lines")
    parser.add_argument("--eval-every", type=int, default=100_000, help="steps between evaluations and checkpoints")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--final-eval-episodes", type=int, default=50)
    # The hyperparameters depend on the algorithm, so add them once --algo is known.
    known, _ = parser.parse_known_args()
    ALGOS[known.algo].add_args(parser.add_argument_group(f"{known.algo} hyperparameters"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)  # small networks and batches: threads only add overhead
    torch.manual_seed(args.seed)
    prefix = "" if args.algo == "stream_ac" else f"{args.algo}_"
    out = args.out or Path("runs") / f"{prefix}{args.task}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    config = {
        "task": args.task,
        "algo": args.algo,
        "obs_mode": args.obs_mode,
        "env_kwargs": {"detection_noise": args.detection_noise, "detection_dropout": args.detection_dropout}
        if args.obs_mode == "detection"
        else {},
        "num_frames": args.frames,
        "hidden_size": args.hidden_size,
        "gamma": args.gamma,
    }
    env = make_env(config)
    env.action_space.seed(args.seed)
    agent = ALGOS[args.algo].make_agent(args, env.observation_space.shape[0], env.action_space.shape[0])

    trackio.init(project=args.project, name=out.name, group=args.task, config={**config, **vars(args), "out": str(out)})
    recent = deque(maxlen=args.log_every)
    train_metrics = defaultdict(list)
    start, episode = time.perf_counter(), 0

    obs, _ = env.reset(seed=args.seed)
    for step in range(1, args.steps + 1):
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # Truncation (time limit) is not a real end, so it still bootstraps from next_obs.
        for key, value in agent.observe(obs, action, reward, next_obs, terminated, done).items():
            train_metrics[key].append(value)
        obs = next_obs

        if done:
            ret, length = float(info["episode"]["r"]), int(info["episode"]["l"])
            success = float(info.get("is_success", np.nan))
            metrics = {"episode/return": ret, "episode/length": length}
            metrics.update({f"train/{k}": np.mean(v) for k, v in train_metrics.items()})
            if not np.isnan(success):
                metrics["episode/success"] = success
            metrics["episode/out_of_bounds"] = float(info.get("out_of_bounds", False))
            for key in ("standoff_error", "distance", "heading_error"):
                if key in info:
                    metrics[f"episode/final_{key}"] = info[key]
            trackio.log(metrics, step=step)
            recent.append((ret, length, success))
            train_metrics.clear()
            episode += 1
            if episode % args.log_every == 0:
                r, l, s = np.mean(recent, axis=0)  # success is NaN for tracking (no success criterion)
                print(
                    f"step {step:>9}  episode {episode:>6}  return {r:8.2f}  length {l:6.1f}  "
                    f"success {s:5.2f}  {step / (time.perf_counter() - start):6.0f} steps/s",
                    flush=True,
                )
                trackio.log({"train/steps_per_sec": step / (time.perf_counter() - start)}, step=step)
            obs, _ = env.reset()

        if step % args.eval_every == 0:
            save_checkpoint(out / "latest.pt", config, agent, env)
            log_eval(out / "latest.pt", config, args.eval_episodes, step)

    save_checkpoint(out / "final.pt", config, agent, env)
    env.close()
    result = log_eval(out / "final.pt", config, args.final_eval_episodes, args.steps)
    print(f"eval ({args.final_eval_episodes} episodes, deterministic): {result}", flush=True)
    trackio.finish()


if __name__ == "__main__":
    main()
