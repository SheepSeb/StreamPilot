"""Task sequence: train one team on tracking, then waypoint, then landing, then tracking again,
``--steps-per-task`` team steps each, and track how well it adapts.

    uv run streampilot-continual --method stream --mode continue     # StreamX keeps learning
    uv run streampilot-continual --method stream --mode scratch      # StreamX, a fresh learner on each task
    uv run streampilot-continual --method cstream --mode continue    # StreamX with a centralised critic
    uv run streampilot-continual --method mappo  --mode continue     # MAPPO fine-tuned (or --method ippo)
    uv run streampilot-continual --method stream --swap-phase 1      # also swap a drone mid-phase
    uv run streampilot-continual --method stream --compile --step-scale-end 0.1 --restore-best

Every phase keeps its best evaluation as ``best_phaseN.pt``. ``--restore-best`` rolls the learner back
to it at the end of the phase (before the probes and the next task), so a policy that degrades late in
a phase is not carried forward. ``--step-scale-end`` (StreamX) anneals the step size within each phase.

``--mode continue`` carries the networks, the observation statistics and (MAPPO) nothing else from
task to task; the reward scale and the eligibility traces start over with each task. ``--mode scratch``
builds a new learner for every task. The same run of ``continue`` also scores the *frozen* team: the
policy as it was at the end of the first phase, never trained again, on every later task (``frozen/``
keys). Every task needs the same observation and action sizes, so the critic gets no privileged state
(``--critic-state`` is off): its size would differ per task.

Logged to ``RUN/metrics.parquet`` (and Trackio), x axis = team steps over the whole sequence:

- ``eval/*``: deterministic evaluation of the current policy on the current task, at the start of each
  phase and every ``--eval-every`` steps, on fixed seeds. ``eval/score`` is the task's success (waypoint
  and landing: ``is_success``; tracking, which never ends in success: the fraction of steps on target),
  ``eval/success`` the success rate (tracking: survival), ``eval/time_to_complete`` the mean length of
  the successful episodes, ``eval/accuracy`` the task's final error (waypoint: distance to the ball,
  landing: slot error, tracking: standoff error), all averaged over the drones.
- ``probe/*`` at the end of every phase: the same on all three tasks (``probe_task`` = the task id), for
  forgetting and transfer.
- ``frozen/*``: ``eval/*`` of the frozen policy, on every evaluation row.
- ``phase``, ``phase_step``, ``task_id``; ``event_swap`` (0: just before the swap, 1: just after).
"""

import argparse
import copy
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import trackio  # noqa: E402

from streampilot.baselines import mappo  # noqa: E402
from streampilot.policy import TeamPolicy  # noqa: E402
from streampilot.stream_x import agents as stream_ac  # noqa: E402
from streampilot.stream_x.multi_agent import CentralizedStreamAC, IndependentStreamAC  # noqa: E402
from streampilot.stream_x.wrappers import RunningMeanStd  # noqa: E402
from streampilot.train_formation import EPSILON, TASKS, RunLog, Team, pick_device, summarize  # noqa: E402
from streampilot.train_formation import save_checkpoint as save_mappo  # noqa: E402
from streampilot.train_formation_stream import make_env  # noqa: E402
from streampilot.vec_env import FormationVecEnv, episode_summary  # noqa: E402

STREAM_METHODS = ("stream", "cstream")  # StreamX with independent critics, or one centralised critic
SEQUENCE = ["formation-tracking", "formation-waypoint", "formation-landing", "formation-tracking"]
TASK_IDS = {"formation-tracking": 0, "formation-waypoint": 1, "formation-landing": 2}  # as in the report script
EVAL_SEED = 10_000
# Per task: the episode summary key behind the score, and behind the accuracy (an error, lower is better).
SCORE = {"formation-tracking": "mean_on_target", "formation-waypoint": "is_success", "formation-landing": "is_success"}
ACCURACY = {"formation-tracking": "mean_standoff_error", "formation-waypoint": "distance", "formation-landing": "formation_error"}


def evaluate(checkpoint: Path, config: dict, task: str, episodes: int, seed: int = EVAL_SEED) -> dict:
    """Deterministic episodes of a saved team on ``task``, seeds ``seed, seed + 1, ...``."""
    env = make_env({**config, "task": task})
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
    out = summarize(results, "eval")
    out["eval/score"] = float(np.mean([r[SCORE[task]] for r in results]))
    # Tracking never succeeds: survival is its success.
    success = [r.get("is_success", r["survived"]) for r in results]
    out["eval/success"] = float(np.mean(success))
    won = [r["length"] for r, s in zip(results, success) if s]
    out["eval/time_to_complete"] = float(np.mean(won)) if won else float("nan")
    out["eval/accuracy"] = float(np.mean([r[ACCURACY[task]] for r in results]))
    return out


class StreamRunner:
    """Independent Stream AC(lambda) learners, one transition at a time."""

    unit = 1  # steps per indivisible chunk of training

    def __init__(self, args: argparse.Namespace, config: dict):
        self.args, self.config, self.team, self.env = args, config, None, None
        self.phase_steps = args.steps_per_task

    def begin(self, task: str, fresh: bool, seed: int) -> None:
        a = self.args
        if self.env is not None:
            self.env.close()
        self.env = make_env({**self.config, "task": task})
        obs_dim, action_dim = int(np.prod(self.env.observation_space.shape[1:])), self.env.action_space.shape[1]
        if fresh or self.team is None:
            learner = CentralizedStreamAC if a.method == "cstream" else IndependentStreamAC
            self.team = learner(
                a.drones, obs_dim, action_dim, a.frames, gamma=a.gamma, hidden_size=a.hidden_size, lr=a.lr,
                lamda=a.lamda, kappa_policy=a.kappa_policy, kappa_value=a.kappa_value, entropy_coeff=a.entropy_coeff, init_std=a.init_std, bound_mean=a.bound_mean, max_std=a.max_std,
            )  # fmt: skip
            if a.compile:
                self.team.compile()
        else:
            self.team.new_task()
        obs, _ = self.env.reset(seed=seed)
        self.team.reset(obs)
        self.total, self.length = 0.0, 0
        self.local = 0  # team steps into this phase, for the step-size schedule

    def run(self, steps: int, log) -> int:
        team, env, end = self.team, self.env, self.args.step_scale_end
        for _ in range(steps):
            if end is not None:  # decays within each phase, back to full size when a new task starts
                team.set_step_scale(1.0 + (end - 1.0) * self.local / self.phase_steps)
            self.local += 1
            actions = team.act()
            obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            team.observe(actions, reward, obs, terminated, done)
            self.total += reward
            self.length += 1
            if done:
                log({f"episode/{k}": v for k, v in episode_summary(info, self.total, self.length).items()})
                self.total, self.length = 0.0, 0
                obs, _ = env.reset()
                team.reset(obs)
        return steps

    def swap(self, drone: int) -> None:
        self.team.reset_drone(drone)

    def snapshot(self) -> dict:
        """The team's weights and observation statistics, to ``restore`` later."""
        params = [*self.team.actor.parameters(), *self.team.critic.parameters()]
        return {"params": [p.detach().clone() for p in params], "obs_stats": copy.deepcopy(self.team.obs_stats)}

    def restore(self, snapshot: dict) -> None:
        """Back to a ``snapshot``, in place (the optimizers and compiled updates keep their tensors)."""
        with torch.no_grad():
            for p, saved in zip([*self.team.actor.parameters(), *self.team.critic.parameters()], snapshot["params"]):
                p.copy_(saved)
        self.team.obs_stats = copy.deepcopy(snapshot["obs_stats"])

    def save(self, path: Path) -> None:
        base = self.env.unwrapped
        torch.save(
            {
                "config": self.config,
                "obs_dim": int(np.prod(base.observation_space.shape[1:])),
                "action_dim": int(base.action_space.shape[1]),
                **self.team.state_dict(),
            },
            path,
        )

    def close(self) -> None:
        if self.env is not None:
            self.env.close()


class MappoRunner:
    """MAPPO or IPPO on parallel environments. A task change builds new environments and a new agent
    (and optimizer, and learning-rate schedule) that starts from the previous agent's networks and
    observation normalization, unless ``fresh``."""

    def __init__(self, args: argparse.Namespace, config: dict):
        self.args, self.config, self.venv, self.team, self.agent = args, config, None, None, None
        self.device = pick_device(args.device)
        self.unit = args.rollout_steps * args.num_envs
        self.phase_steps = max(args.steps_per_task // self.unit, 1) * self.unit
        args.steps = self.phase_steps  # the learning-rate schedule's length (mappo.make_agent)
        self.seeds = np.random.SeedSequence(args.seed)

    def begin(self, task: str, fresh: bool, seed: int) -> None:
        a = self.args
        old_team, old_agent = self.team, self.agent
        if self.venv is not None:
            self.venv.close()
        env_kwargs = {"num_drones": a.drones, "obs_mode": a.obs_mode}
        if a.obs_mode == "detection":
            env_kwargs |= {"detection_noise": a.detection_noise, "detection_dropout": a.detection_dropout}
        self.venv = FormationVecEnv(TASKS[task], env_kwargs, a.num_envs, a.num_workers, False)
        self.team = Team(self.venv, a.frames, False, independent=a.method == "ippo")
        self.agent = mappo.make_agent(
            a, self.team.actor_dim, self.team.critic_dim, self.venv.action_dim, self.venv.num_drones, self.device,
            self.team.independent,
        )  # fmt: skip
        if old_agent is not None and not fresh:
            self.team.actor_norm, self.team.critic_norm = old_team.actor_norm, old_team.critic_norm
            self.agent.actor.load_state_dict(old_agent.actor.state_dict())
            self.agent.critic.load_state_dict(old_agent.critic.state_dict())
        self.return_stats, self.running_return = RunningMeanStd(), np.zeros(a.num_envs)
        obs = self.venv.reset(seeds=self.seeds.spawn(1)[0].generate_state(a.num_envs).tolist())
        self.actor_x, self.critic_x = self.team.reset(obs)

    def run(self, steps: int, log) -> int:
        a, venv, team, agent = self.args, self.venv, self.team, self.agent
        updates = max(steps // self.unit, 1)
        for _ in range(updates):
            episodes = []
            for _ in range(a.rollout_steps):
                actions = agent.act(self.actor_x)
                obs, reward, terminated, truncated, ended = venv.step(actions)
                done = terminated | truncated
                self.running_return = self.running_return * a.gamma * (1.0 - terminated) + reward
                self.return_stats.update_batch(self.running_return)
                self.running_return[done] = 0.0
                scaled = reward / np.sqrt(self.return_stats.var + EPSILON)
                next_actor_x, next_critic_x, final_critic_x = team.step(actions, obs, done)
                agent.store(self.actor_x, self.critic_x, actions, scaled, terminated, done, final_critic_x)
                self.actor_x, self.critic_x = next_actor_x, next_critic_x
                episodes.extend(summary for _, summary in ended)
            metrics = {f"train/{k}": v for k, v in agent.update(self.critic_x).items()}
            log(metrics | summarize(episodes, "episode") | {"episode/count": len(episodes)})
        return updates * self.unit

    def swap(self, drone: int) -> None:
        raise NotImplementedError("MAPPO's drones share one actor")

    def snapshot(self) -> dict:
        """The agent's networks and the team's observation normalization, to ``restore`` later."""
        return copy.deepcopy(
            {
                "actor": self.agent.actor.state_dict(),
                "critic": self.agent.critic.state_dict(),
                "norm": (self.team.actor_norm, self.team.critic_norm),
            }
        )

    def restore(self, snapshot: dict) -> None:
        self.agent.actor.load_state_dict(snapshot["actor"])
        self.agent.critic.load_state_dict(snapshot["critic"])
        self.team.actor_norm, self.team.critic_norm = copy.deepcopy(snapshot["norm"])

    def save(self, path: Path) -> None:
        save_mappo(path, self.config, self.agent, self.team)

    def close(self) -> None:
        if self.venv is not None:
            self.venv.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task sequence (tracking, waypoint, landing, tracking).")
    parser.add_argument(
        "--method",
        choices=["stream", "cstream", "mappo", "ippo"],
        default="stream",
        help="stream: StreamX with a critic per drone; cstream: StreamX with one centralised critic (CTDE)",
    )
    parser.add_argument(
        "--mode", choices=["continue", "scratch"], default="continue", help="carry the learner over, or start anew per task"
    )
    parser.add_argument("--sequence", nargs="+", choices=TASKS, default=SEQUENCE)
    parser.add_argument("--steps-per-task", type=int, default=1_000_000, help="K, team steps per task")
    parser.add_argument("--drones", type=int, default=3, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--obs-mode", choices=["detection", "state"], default="detection")
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument(
        "--step-scale-end",
        type=float,
        default=None,
        help="StreamX: within each phase, anneal the effective step size linearly from 1x to this factor, back to "
        "1x at the next task (default: constant). Keeps a policy that works from being unlearned late in a phase",
    )
    parser.add_argument(
        "--restore-best",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="at the end of each phase, roll the learner back to that phase's best evaluation (best_phaseN.pt, "
        "always saved) before the probes and the next task",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="StreamX: fuse each learner's update with torch.compile (same math, faster; ~1 min to compile per new learner)",
    )
    parser.add_argument("--detection-noise", type=float, default=0.0)
    parser.add_argument("--detection-dropout", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=50_000, help="team steps between evaluations")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--probe-episodes", type=int, default=50, help="per task, at the end of each phase")
    parser.add_argument("--swap-phase", type=int, default=None, help="swap a drone for a fresh learner in this phase (0-based)")
    parser.add_argument("--swap-at", type=float, default=0.5, help="fraction of the phase at which to swap")
    parser.add_argument("--swap-drone", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=64, help="MAPPO/IPPO")
    parser.add_argument("--num-workers", type=int, default=None, help="MAPPO/IPPO; default: one per CPU thread")
    parser.add_argument("--device", default="auto", help="MAPPO/IPPO updates")
    parser.add_argument("--out", type=Path, default=None, help="default: runs/continual_METHOD-MODE[_Nd]_seedSEED")
    parser.add_argument("--project", default="streampilot")
    # The two hyperparameter sets share option names (--lr, --gamma, ...) with different defaults: add the chosen one.
    method = parser.parse_known_args()[0].method
    if method in STREAM_METHODS:
        stream_ac.add_args(parser.add_argument_group("Stream AC hyperparameters"))
    else:
        mappo.add_args(parser.add_argument_group("MAPPO / IPPO hyperparameters"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.swap_phase is not None and args.method not in STREAM_METHODS:
        raise SystemExit("--swap-phase needs a StreamX method: MAPPO's drones share one actor, there is no drone to swap")
    torch.set_num_threads(1 if args.method in STREAM_METHODS else torch.get_num_threads())
    torch.manual_seed(args.seed)
    drones = "" if args.drones == 3 else f"_{args.drones}d"
    out = args.out or Path("runs") / f"continual_{args.method}-{args.mode}{drones}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    logger = RunLog(out)

    env_kwargs = {"num_drones": args.drones}
    if args.obs_mode == "detection":
        env_kwargs |= {"detection_noise": args.detection_noise, "detection_dropout": args.detection_dropout}
    config = {
        "task": args.sequence[0],
        "algo": {"stream": "istream_ac", "cstream": "cstream_ac"}.get(args.method, args.method),
        "obs_mode": args.obs_mode,
        "env_kwargs": env_kwargs,
        "num_frames": args.frames,
        "hidden_size": args.hidden_size,
        "gamma": args.gamma,
        "bound_mean": getattr(args, "bound_mean", False),  # StreamX only
        "critic_state": False,
        "curriculum_fraction": 0.0,
    }
    runner = (StreamRunner if args.method in STREAM_METHODS else MappoRunner)(args, config)
    k = runner.phase_steps
    if args.swap_phase is not None and not 0 <= args.swap_phase < len(args.sequence):
        raise SystemExit("--swap-phase is out of the sequence")
    trackio.init(project=args.project, name=out.name, group="continual", config={**config, **vars(args), "out": str(out)})
    print(f"{args.method}-{args.mode}: {' -> '.join(args.sequence)}, {k} team steps each", flush=True)

    start, eval_seconds = time.perf_counter(), 0.0
    snapshot: Path | None = None  # the frozen policy: the end of the first phase
    frozen: dict = {}
    ckpt = out / "latest.pt"

    for phase, task in enumerate(args.sequence):
        base, done_steps = phase * k, 0
        # Task rows carry these, so the report can slice by phase and task.
        tags = {"phase": phase, "task_id": TASK_IDS[task]}
        runner.begin(task, fresh=args.mode == "scratch" or phase == 0, seed=args.seed + phase)
        # This phase's best evaluation: by eval/score, then eval/return (the score is often 0 early on, and a
        # tie must not pick the untrained start of the phase), and the later one on a full tie.
        best_score, best_snapshot = (-np.inf, -np.inf), None

        def log_row(extra: dict, local: int) -> None:
            logger.log(tags | {"phase_step": local} | extra, base + local)

        def evaluate_now(local: int, episodes: int, **extra) -> None:
            nonlocal eval_seconds, best_score, best_snapshot
            t0 = time.perf_counter()
            runner.save(ckpt)
            row = evaluate(ckpt, config, task, episodes) | extra
            score = (row["eval/score"], row["eval/return"])
            if score >= best_score:
                best_score = score
                shutil.copy(ckpt, out / f"best_phase{phase}.pt")
                if args.restore_best:
                    best_snapshot = runner.snapshot()
            if snapshot is not None:
                row |= {k_.replace("eval/", "frozen/", 1): v for k_, v in frozen.items()}
            log_row(row, local)
            logger.flush()
            eval_seconds += time.perf_counter() - t0
            print(
                f"phase {phase} {task:<18} step {base + local:>9}  score {row['eval/score']:5.2f}  "
                f"success {row['eval/success']:5.2f}  accuracy {row['eval/accuracy']:6.3f}"
                + (f"  frozen {row['frozen/score']:5.2f}" if snapshot is not None else ""),
                flush=True,
            )

        if snapshot is not None:  # the frozen policy on this task: it never changes, so once per phase
            frozen = evaluate(snapshot, config, task, args.eval_episodes)
        evaluate_now(0, args.eval_episodes)
        swap_at = int(args.swap_at * k) if args.swap_phase == phase else None
        next_eval = args.eval_every  # PPO advances in whole updates, so done_steps rarely lands on a multiple
        while done_steps < k:
            chunk = min(args.eval_every, k - done_steps)
            if swap_at is not None and done_steps < swap_at:
                chunk = min(chunk, swap_at - done_steps)
            done_steps += runner.run(chunk, lambda m: log_row(m, done_steps))
            if swap_at is not None and done_steps >= swap_at:
                evaluate_now(done_steps, args.eval_episodes, event_swap=0)
                runner.swap(args.swap_drone)
                evaluate_now(done_steps, args.eval_episodes, event_swap=1)
                swap_at = None
            elif done_steps >= next_eval or done_steps >= k:
                evaluate_now(done_steps, args.eval_episodes)
                next_eval = (done_steps // args.eval_every + 1) * args.eval_every

        # End of the phase: the checkpoint, and the policy on every task.
        if args.restore_best and best_snapshot is not None:
            runner.restore(best_snapshot)
            print(f"phase {phase} {task:<18} restored the phase's best evaluation (score {best_score[0]:.2f}, return {best_score[1]:.1f})", flush=True)
        runner.save(out / f"after_phase{phase}.pt")
        for probe_task in TASK_IDS:
            probe = evaluate(out / f"after_phase{phase}.pt", config, probe_task, args.probe_episodes)
            log_row({k_.replace("eval/", "probe/", 1): v for k_, v in probe.items()} | {"probe_task": TASK_IDS[probe_task]}, k)
        logger.flush()
        if phase == 0 and args.mode == "continue":
            snapshot = out / "after_phase0.pt"
    runner.close()
    trackio.finish()


if __name__ == "__main__":
    main()
