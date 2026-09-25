"""Train a formation task (multiple drones) with MAPPO on many environments in parallel.

    uv run streampilot-train-formation formation-waypoint
    uv run streampilot-train-formation formation-landing --drones 2 --num-envs 128
    uv run streampilot-train-formation formation-tracking --device cpu   # no GPU
    uv run streampilot-train-formation formation-waypoint --help          # all hyperparameters

``--num-envs`` environments run in ``--num-workers`` processes (default: one per CPU thread).
Rollouts are sampled on the CPU; the updates run on ``--device`` (default: the fastest GPU if
PyTorch was installed with CUDA support, see the README, else the CPU). ``--steps`` counts
team steps, summed over the environments.

Every update logs the episodes that ended during its rollout, the losses and the throughput to
Trackio (project ``streampilot``, group = task). Every ``--eval-every`` steps the deterministic
policy is evaluated on the same fixed seeds and ``latest.pt`` is written; ``final.pt`` at the
end. Watch a checkpoint with ``uv run streampilot TASK --policy RUN_DIR/final.pt``.
"""

import argparse
import os
import time
from collections import defaultdict
from pathlib import Path

# Trackio logs locally; online, importing it blocks on Hugging Face Hub network calls.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import trackio  # noqa: E402

from streampilot.baselines import mappo  # noqa: E402
from streampilot.baselines.mappo import TeamHistory  # noqa: E402
from streampilot.stream_x.wrappers import RunningMeanStd  # noqa: E402
from streampilot.vec_env import FormationVecEnv  # noqa: E402

TASKS = {
    "formation-waypoint": "DroneFormationWaypoint-v0",
    "formation-landing": "DroneFormationLanding-v0",
    "formation-tracking": "DroneFormationTracking-v0",
}
EPSILON = 1e-8


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    # The GPU with the most multiprocessors (on a mixed machine, the fastest one).
    best = max(range(torch.cuda.device_count()), key=lambda i: torch.cuda.get_device_properties(i).multi_processor_count)
    return torch.device(f"cuda:{best}")


class Normalizer:
    """Standardizes rows with running statistics (updated on every row it sees in training)."""

    def __init__(self, dim: int):
        self.stats = RunningMeanStd((dim,))

    def __call__(self, x: np.ndarray, update: bool = True) -> np.ndarray:
        if update:
            self.stats.update_batch(x)
        return ((x - self.stats.mean) / np.sqrt(self.stats.var + EPSILON)).astype(np.float32)


class Team:
    """The features of every environment's team: each drone's actor input (history of its
    observation rows and its previous action) and the critic input (all of the drones' actor
    inputs and, with privileged state, their states), both normalized."""

    def __init__(self, venv: FormationVecEnv, num_frames: int, critic_state: bool):
        self.venv, self.critic_state = venv, critic_state
        self.history = TeamHistory(venv.num_envs, venv.num_drones, venv.obs_dim, venv.action_dim, num_frames)
        n = venv.num_drones
        self.actor_dim = self.history.dim
        self.critic_dim = n * self.actor_dim + (n * venv.state_dim if critic_state else 0)
        self.actor_norm = Normalizer(self.actor_dim)
        self.critic_norm = Normalizer(self.critic_dim)

    def _critic_x(self, features, state, update: bool) -> np.ndarray:
        parts = [features.reshape(len(features), -1)]
        if self.critic_state:
            parts.append(state.reshape(len(state), -1))
        return self.critic_norm(np.concatenate(parts, axis=1), update)

    def reset(self, obs, update: bool = True):
        features = self.history.reset(obs)
        actor_x = self.actor_norm(features.reshape(-1, self.actor_dim), update).reshape(features.shape)
        return actor_x, self._critic_x(features, self.venv.state, update)

    def step(self, actions, obs, done, update: bool = True):
        """Push a step; returns ``(actor_x, critic_x, final_critic_x)``, the last for the
        terminal observations of the environments where ``done``."""
        venv = self.venv
        final_critic_x = None
        if done.any():
            # The history's view of the terminal observation, then a fresh history for the new episode.
            obs = obs.copy()
            obs[done] = venv.final_obs[done]
            final = self.history.push(actions, obs)[done]
            if update:
                self.actor_norm.stats.update_batch(final.reshape(-1, self.actor_dim))
            final_critic_x = self._critic_x(final, venv.final_state[done] if self.critic_state else None, update)
            features = self.history.reset(venv.obs, done)
        else:
            features = self.history.push(actions, obs)
        actor_x = self.actor_norm(features.reshape(-1, self.actor_dim), update).reshape(features.shape)
        return actor_x, self._critic_x(features, venv.state, update), final_critic_x


def save_checkpoint(path: Path, config: dict, agent: mappo.MAPPO, team: Team) -> None:
    torch.save(
        {
            "config": config,
            "obs_dim": team.venv.obs_dim,  # one drone's observation row
            "action_dim": team.venv.action_dim,
            **{k: {n: t.cpu() for n, t in v.items()} for k, v in agent.state_dict().items()},
            "obs_stats": team.actor_norm.stats.state_dict(),
            "critic_obs_stats": team.critic_norm.stats.state_dict(),
        },
        path,
    )


def summarize(episodes: list[dict], prefix: str) -> dict:
    by_key = defaultdict(list)
    for summary in episodes:
        for key, value in summary.items():
            by_key[key].append(value)
    return {f"{prefix}/{key}": float(np.mean(values)) for key, values in by_key.items()}


def evaluate(venv: FormationVecEnv, agent: mappo.MAPPO, team: Team, episodes: int, seed: int) -> dict:
    """Deterministic episodes on seeds ``seed, seed + 1, ...``, with the normalization frozen.
    Uses (and resets) the training environments."""
    results = []
    history = TeamHistory(venv.num_envs, venv.num_drones, venv.obs_dim, venv.action_dim, team.history.num_frames)

    def actor_x(features):
        return team.actor_norm(features.reshape(-1, team.actor_dim), update=False).reshape(features.shape)

    for start in range(0, episodes, venv.num_envs):
        pending = np.arange(venv.num_envs) < episodes - start
        features = history.reset(venv.reset(seeds=[seed + start + i for i in range(venv.num_envs)]))
        while pending.any():
            actions = agent.act_deterministic(actor_x(features))
            obs, _, terminated, truncated, ended = venv.step(actions)
            for i, summary in ended:
                if pending[i]:
                    results.append(summary)
                    pending[i] = False
            features = history.push(actions, obs)
            done = terminated | truncated
            if done.any():
                features = history.reset(obs, done)
    return summarize(results, "eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a formation task with MAPPO on parallel environments.")
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("--drones", type=int, default=3, choices=[2, 3])
    parser.add_argument("--steps", type=int, default=20_000_000, help="team steps, over all environments")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--obs-mode", choices=["detection", "state"], default="detection")
    parser.add_argument("--frames", type=int, default=4, help="observations stacked into the actor input")
    parser.add_argument("--detection-noise", type=float, default=0.0)
    parser.add_argument("--detection-dropout", type=float, default=0.0)
    parser.add_argument(
        "--critic-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="give the centralised critic the privileged state of every drone",
    )
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=None, help="default: one per CPU thread")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:1, ... (for the updates)")
    parser.add_argument("--out", type=Path, default=None, help="default: runs/mappo_TASK[_Nd]_seedSEED")
    parser.add_argument("--project", default="streampilot", help="Trackio project")
    parser.add_argument("--log-every", type=int, default=10, help="updates between progress lines")
    parser.add_argument("--eval-every", type=int, default=1_000_000, help="steps between evaluations and checkpoints")
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--final-eval-episodes", type=int, default=128)
    mappo.add_args(parser.add_argument_group("MAPPO hyperparameters"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    # Sampling runs one small batch per step: extra threads only add overhead there. The
    # updates on the CPU get the threads back, since the workers are idle meanwhile.
    update_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    drones = "" if args.drones == 3 else f"_{args.drones}d"
    out = args.out or Path("runs") / f"mappo_{args.task}{drones}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    env_kwargs = {"num_drones": args.drones, "obs_mode": args.obs_mode}
    if args.obs_mode == "detection":
        env_kwargs |= {"detection_noise": args.detection_noise, "detection_dropout": args.detection_dropout}
    config = {
        "task": args.task,
        "algo": "mappo",
        "obs_mode": args.obs_mode,
        "env_kwargs": {k: v for k, v in env_kwargs.items() if k != "obs_mode"},
        "num_frames": args.frames,
        "hidden_size": args.hidden_size,
        "gamma": args.gamma,
        "critic_state": args.critic_state,
    }
    venv = FormationVecEnv(TASKS[args.task], env_kwargs, args.num_envs, args.num_workers, args.critic_state)
    team = Team(venv, args.frames, args.critic_state)
    agent = mappo.make_agent(args, team.actor_dim, team.critic_dim, venv.action_dim, venv.num_drones, device)
    batch = args.rollout_steps * args.num_envs
    num_updates = max(args.steps // batch, 1)
    workers = len(venv._slices)
    print(
        f"{args.task}: {args.num_envs} envs in {workers} workers, {batch} team steps per update, "
        f"{num_updates} updates, device {device}",
        flush=True,
    )

    trackio.init(
        project=args.project, name=out.name, group=args.task, config={**config, **vars(args), "out": str(out)}
    )
    seeds = np.random.SeedSequence(args.seed)

    def fresh_start():
        # Independent seeds for every environment, new after every evaluation (which reseeds them).
        obs = venv.reset(seeds=seeds.spawn(1)[0].generate_state(args.num_envs).tolist())
        return team.reset(obs)

    return_stats = RunningMeanStd()  # of the discounted return, for reward scaling (as ScaleReward)
    running_return = np.zeros(args.num_envs)
    actor_x, critic_x = fresh_start()
    episodes: list[dict] = []
    step, start = 0, time.perf_counter()
    sample_time = update_time = 0.0
    next_eval = args.eval_every

    for update in range(1, num_updates + 1):
        t0 = time.perf_counter()
        for _ in range(args.rollout_steps):
            actions = agent.act(actor_x)
            obs, reward, terminated, truncated, ended = venv.step(actions)
            done = terminated | truncated
            running_return = running_return * args.gamma * (1.0 - terminated) + reward
            return_stats.update_batch(running_return)
            running_return[done] = 0.0
            scaled = reward / np.sqrt(return_stats.var + EPSILON)
            next_actor_x, next_critic_x, final_critic_x = team.step(actions, obs, done)
            agent.store(actor_x, critic_x, actions, scaled, terminated, done, final_critic_x)
            actor_x, critic_x = next_actor_x, next_critic_x
            episodes.extend(summary for _, summary in ended)
        step += batch
        t1 = time.perf_counter()
        if device.type == "cpu":
            torch.set_num_threads(update_threads)
        metrics = agent.update(critic_x)
        torch.set_num_threads(1)
        t2 = time.perf_counter()
        sample_time += t1 - t0
        update_time += t2 - t1

        log = {f"train/{k}": v for k, v in metrics.items()}
        log |= summarize(episodes, "episode")
        log["episode/count"] = len(episodes)
        log["train/steps_per_sec"] = step / (t2 - start)
        log["train/sample_steps_per_sec"] = batch / (t1 - t0)
        log["train/update_seconds"] = t2 - t1
        trackio.log(log, step=step)
        if update % args.log_every == 0 or update == num_updates:
            ret = log.get("episode/return", float("nan"))
            success = log.get("episode/is_success", float("nan"))
            print(
                f"step {step:>10}  update {update:>5}  episodes {len(episodes):>4}  return {ret:8.2f}  "
                f"success {success:5.2f}  {log['train/steps_per_sec']:6.0f} steps/s "
                f"(sampling {sample_time / (sample_time + update_time):.0%})",
                flush=True,
            )
        episodes.clear()

        if step >= next_eval and update < num_updates:
            next_eval += args.eval_every
            save_checkpoint(out / "latest.pt", config, agent, team)
            result = evaluate(venv, agent, team, args.eval_episodes, seed=10_000)
            trackio.log(result, step=step)
            actor_x, critic_x = fresh_start()
            running_return[:] = 0.0

    save_checkpoint(out / "final.pt", config, agent, team)
    result = evaluate(venv, agent, team, args.final_eval_episodes, seed=10_000)
    trackio.log(result, step=step)
    venv.close()
    print(f"eval ({args.final_eval_episodes} episodes, deterministic): {result}", flush=True)
    trackio.finish()


if __name__ == "__main__":
    main()
