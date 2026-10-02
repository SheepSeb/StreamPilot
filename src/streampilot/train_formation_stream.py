"""Train a formation task (multiple drones) with streaming Stream AC(lambda) learners. By default
(``--critic independent``) one learner per drone, fully decentralised (no shared parameters, no
centralised critic). With ``--critic centralized``, per-drone actors of local observations and one
shared critic of the joint observation, trained on the team TD error (CTDE).

    uv run streampilot-train-formation-stream formation-waypoint
    uv run streampilot-train-formation-stream formation-landing --drones 2
    uv run streampilot-train-formation-stream formation-waypoint --critic centralized
    uv run streampilot-train-formation-stream formation-tracking --help   # all hyperparameters

One environment, one transition at a time, as in ``streampilot-train``: every drone's learner
updates on every team step, from its own observation row and the team reward (the centralised
critic, from the joint observation). ``--steps`` counts team steps.

Episodes (team return, success, collisions, out-of-bounds, formation error, each drone's or the
team's TD error) and periodic deterministic evaluations are logged to Trackio (project
``streampilot``, group = task). ``latest.pt`` is written at every evaluation and ``final.pt`` at
the end. Watch a checkpoint with ``uv run streampilot TASK --policy RUN_DIR/final.pt``.
"""

import argparse
import os
import time
from collections import defaultdict, deque
from pathlib import Path

# Trackio logs locally; online, importing it blocks on Hugging Face Hub network calls.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import trackio  # noqa: E402

import streampilot.env  # noqa: E402, F401  (registers the environments)
from streampilot.policy import TeamPolicy  # noqa: E402
from streampilot.stream_x import agents as stream_ac  # noqa: E402
from streampilot.stream_x.multi_agent import CentralizedStreamAC, IndependentStreamAC  # noqa: E402
from streampilot.train_formation import TASKS, RunLog, summarize  # noqa: E402
from streampilot.vec_env import episode_summary  # noqa: E402


def make_env(config: dict) -> gym.Env:
    return gym.make(TASKS[config["task"]], obs_mode=config["obs_mode"], **config["env_kwargs"])


def save_checkpoint(path: Path, config: dict, team: IndependentStreamAC | CentralizedStreamAC, env: gym.Env) -> None:
    base = env.unwrapped
    torch.save(
        {
            "config": config,
            "obs_dim": int(np.prod(base.observation_space.shape[1:])),  # one drone's observation row
            "action_dim": int(base.action_space.shape[1]),
            **team.state_dict(),
        },
        path,
    )


def evaluate(checkpoint: Path, config: dict, episodes: int, seed: int) -> dict:
    """Deterministic episodes of a saved team on seeds ``seed, seed + 1, ...``."""
    env = make_env(config)
    policy = TeamPolicy.load(checkpoint)
    results = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        policy.reset()
        total, length, done = 0.0, 0, False
        while not done:
            obs, reward, terminated, truncated, info = env.step(policy(obs))
            total += reward
            length += 1
            done = terminated or truncated
        results.append(episode_summary(info, total, length))
    env.close()
    return summarize(results, "eval")


def log_eval(logger: RunLog, checkpoint: Path, config: dict, episodes: int, step: int, compute: dict) -> dict:
    # Same seeds every time, so evaluations are comparable across checkpoints.
    result = evaluate(checkpoint, config, episodes, seed=10_000)
    logger.log(result | compute, step)
    logger.flush()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a formation task with Stream AC(lambda): per-drone actors, independent or centralised critic."
    )
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("--drones", type=int, default=3, choices=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--critic",
        choices=["independent", "centralized"],
        default="independent",
        help="independent: one critic per drone, of its own features; centralized: one shared critic of the joint observation",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="fuse the update with torch.compile: same math, ~25%% faster, ~1 min compile at the start",
    )
    parser.add_argument("--steps", type=int, default=2_000_000, help="team steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--obs-mode", choices=["detection", "state"], default="detection")
    parser.add_argument("--frames", type=int, default=4, help="observations stacked into each drone's input")
    parser.add_argument("--detection-noise", type=float, default=0.0)
    parser.add_argument("--detection-dropout", type=float, default=0.0)
    parser.add_argument(
        "--curriculum-fraction",
        type=float,
        default=0.0,
        help="ramp the detection dropout and noise linearly from 0 to --detection-dropout/--detection-noise "
        "over this fraction of the steps, then hold them (0: no curriculum, the full levels from the start). "
        "Evaluations always use the full levels",
    )
    parser.add_argument("--out", type=Path, default=None, help="default: runs/ALGO_TASK[_Nd]_seedSEED (ALGO: istream_ac or cstream_ac)")
    parser.add_argument("--project", default="streampilot", help="Trackio project")
    parser.add_argument("--log-every", type=int, default=50, help="episodes between progress lines")
    parser.add_argument("--eval-every", type=int, default=100_000, help="steps between evaluations and checkpoints")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--final-eval-episodes", type=int, default=50)
    stream_ac.add_args(parser.add_argument_group("Stream AC hyperparameters (every learner)"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)  # small networks, one sample per update: threads only add overhead
    torch.manual_seed(args.seed)
    drones = "" if args.drones == 3 else f"_{args.drones}d"
    algo = "cstream_ac" if args.critic == "centralized" else "istream_ac"
    out = args.out or Path("runs") / f"{algo}_{args.task}{drones}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    logger = RunLog(out)

    env_kwargs = {"num_drones": args.drones}
    if args.obs_mode == "detection":
        env_kwargs |= {"detection_noise": args.detection_noise, "detection_dropout": args.detection_dropout}
    config = {
        "task": args.task,
        "algo": algo,
        "obs_mode": args.obs_mode,
        "env_kwargs": env_kwargs,
        "num_frames": args.frames,
        "hidden_size": args.hidden_size,
        "gamma": args.gamma,
        "curriculum_fraction": args.curriculum_fraction,
    }
    env = make_env(config)
    obs_dim, action_dim = int(np.prod(env.observation_space.shape[1:])), env.action_space.shape[1]
    team_cls = CentralizedStreamAC if args.critic == "centralized" else IndependentStreamAC
    team = team_cls(
        args.drones,
        obs_dim,
        action_dim,
        args.frames,
        gamma=args.gamma,
        hidden_size=args.hidden_size,
        lr=args.lr,
        lamda=args.lamda,
        kappa_policy=args.kappa_policy,
        kappa_value=args.kappa_value,
        entropy_coeff=args.entropy_coeff,
    )

    if args.compile:
        team.compile()

    trackio.init(project=args.project, name=out.name, group=args.task, config={**config, **vars(args), "out": str(out)})
    recent = deque(maxlen=args.log_every)
    abs_td, scales = [], []
    start, episode, eval_seconds = time.perf_counter(), 0, 0.0
    total, length = 0.0, 0

    def compute(now: float) -> dict:
        # Cumulative at the current step, for a compute axis next to the team steps; evaluation time excluded.
        return {"train/grad_updates": step * team.updates_per_step, "train/wall_seconds": now - start - eval_seconds}

    ramp = args.curriculum_fraction * args.steps
    obs, _ = env.reset(seed=args.seed)
    team.reset(obs)
    for step in range(1, args.steps + 1):
        if ramp > 0:  # the curriculum: the detection degrades from clean to the full levels
            scale = min(step / ramp, 1.0)
            env.unwrapped.detection_dropout = scale * args.detection_dropout
            env.unwrapped.detection_noise = scale * args.detection_noise
        actions = team.act()
        obs, reward, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        # Truncation (time limit) is not a real end, so each learner still bootstraps from obs.
        abs_td.append(np.abs(team.observe(actions, reward, obs, terminated, done)))
        scales.append(team.step_scales())
        total += reward
        length += 1

        if done:
            summary = episode_summary(info, total, length)
            metrics = {f"episode/{k}": v for k, v in summary.items()}
            mean_td = np.mean(abs_td, axis=0)
            if mean_td.ndim == 0:  # one centralised critic, one team TD error
                metrics["train/abs_td_error"] = float(mean_td)
            else:
                for i, td in enumerate(mean_td):
                    metrics[f"train/abs_td_error_d{i}"] = float(td)
            # ObGD's bound: the mean step size relative to lr, and how often it shrank the step.
            for name in ("actor", "critic"):
                per_step = np.stack([s[name] for s in scales])  # (steps, learners)
                for i, (mean, active) in enumerate(zip(per_step.mean(0), (per_step < 1.0).mean(0))):
                    suffix = "" if per_step.shape[1] == 1 else f"_d{i}"
                    metrics[f"train/{name}_step_scale{suffix}"] = float(mean)
                    metrics[f"train/{name}_bound_active{suffix}"] = float(active)
            if ramp > 0:
                metrics["train/curriculum_scale"] = scale
            logger.log(metrics, step)
            recent.append(summary)
            abs_td.clear()
            scales.clear()
            total, length = 0.0, 0
            episode += 1
            if episode % args.log_every == 0:
                log = summarize(list(recent), "recent")
                steps_per_sec = step / (time.perf_counter() - start)
                print(
                    f"step {step:>9}  episode {episode:>6}  return {log['recent/return']:8.2f}  "
                    f"length {log['recent/length']:6.1f}  success {log.get('recent/is_success', np.nan):5.2f}  "
                    f"collision {log['recent/collision']:4.2f}  {steps_per_sec:6.0f} steps/s",
                    flush=True,
                )
                logger.log({"train/steps_per_sec": steps_per_sec}, step)
            obs, _ = env.reset()
            team.reset(obs)

        if step % args.eval_every == 0:
            t0 = time.perf_counter()
            save_checkpoint(out / "latest.pt", config, team, env)
            log_eval(logger, out / "latest.pt", config, args.eval_episodes, step, compute(t0))
            eval_seconds += time.perf_counter() - t0

    save_checkpoint(out / "final.pt", config, team, env)
    env.close()
    result = log_eval(logger, out / "final.pt", config, args.final_eval_episodes, args.steps, compute(time.perf_counter()))
    print(f"eval ({args.final_eval_episodes} episodes, deterministic): {result}", flush=True)
    trackio.finish()


if __name__ == "__main__":
    main()
