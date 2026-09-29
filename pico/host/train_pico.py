"""Train a drone task with Stream AC(lambda) running on a Raspberry Pi Pico 2 W over a serial line.

The PC only steps the MuJoCo environment; the Pico keeps the observation history, the normalization
statistics, both networks and their eligibility traces, and does every update.

    uv run python pico/host/train_pico.py waypoint --port /dev/ttyUSB0
    uv run python pico/host/train_pico.py waypoint --emulate          # the firmware C code on the PC

Checkpoints (``latest.pt``, ``final.pt``) have the same format as ``streampilot-train``, so
``uv run streampilot TASK --policy RUN_DIR/final.pt`` and ``streampilot.policy.Policy`` work unchanged.
"""

import argparse
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import trackio

from pico_link import EmulatedTransport, PicoAgent, SerialTransport
from streampilot.stream_x.agents import StreamAC
from streampilot.train import TASKS, log_eval


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("task", choices=TASKS)
    link = p.add_mutually_exclusive_group(required=True)
    link.add_argument("--port", help="serial device of the Pico's UART, e.g. /dev/ttyUSB0")
    link.add_argument("--emulate", action="store_true", help="run the firmware's C code on the PC instead")
    p.add_argument("--baud", type=int, default=921600)
    p.add_argument("--steps", type=int, default=2_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--obs-mode", choices=["detection", "state"], default="detection")
    p.add_argument("--frames", type=int, default=4)
    p.add_argument("--detection-noise", type=float, default=0.0)
    p.add_argument("--detection-dropout", type=float, default=0.0)
    p.add_argument("--device-init", action="store_true", help="use the Pico's own sparse init instead of PyTorch's")
    p.add_argument("--out", type=Path, default=None, help="default: runs/pico_TASK_seedSEED")
    p.add_argument("--project", default="streampilot")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=100_000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--final-eval-episodes", type=int, default=50)
    h = p.add_argument_group("stream_ac hyperparameters")
    h.add_argument("--hidden-size", type=int, default=128)
    h.add_argument("--lr", type=float, default=1.0)
    h.add_argument("--gamma", type=float, default=0.99)
    h.add_argument("--lamda", type=float, default=0.8)
    h.add_argument("--kappa-policy", type=float, default=3.0)
    h.add_argument("--kappa-value", type=float, default=2.0)
    h.add_argument("--entropy-coeff", type=float, default=0.01)
    return p.parse_args()


def save_checkpoint(path: Path, config: dict, agent: PicoAgent) -> None:
    obs_stats, _ = agent.stats()
    torch.save({"config": config, "obs_dim": agent.obs_dim, "action_dim": agent.action_dim,
                **agent.state_dict(), "obs_stats": obs_stats}, path)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    out = args.out or Path("runs") / f"pico_{args.task}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "task": args.task,
        "algo": "stream_ac",  # same networks, so Policy.load picks stream_x's Actor
        "device": "pico",
        "obs_mode": args.obs_mode,
        "env_kwargs": {"detection_noise": args.detection_noise, "detection_dropout": args.detection_dropout}
        if args.obs_mode == "detection"
        else {},
        "num_frames": args.frames,
        "hidden_size": args.hidden_size,
        "gamma": args.gamma,
    }

    env = gym.make(TASKS[args.task], obs_mode=args.obs_mode, **config["env_kwargs"])
    env = gym.wrappers.RecordEpisodeStatistics(env)  # raw returns; everything else happens on the Pico
    obs_dim, action_dim = int(np.prod(env.observation_space.shape)), int(np.prod(env.action_space.shape))

    agent = PicoAgent(EmulatedTransport() if args.emulate else SerialTransport(args.port, args.baud))
    version, arena = agent.ping()
    agent.init(obs_dim, action_dim, num_frames=args.frames, hidden_size=args.hidden_size, lr=args.lr,
               gamma=args.gamma, lamda=args.lamda, kappa_policy=args.kappa_policy, kappa_value=args.kappa_value,
               entropy_coeff=args.entropy_coeff, seed=args.seed)
    n_params = agent.n_actor + agent.n_critic
    print(f"pico protocol v{version}: {n_params} parameters, {8 * n_params / 1024:.0f} of {arena / 1024:.0f} KB "
          f"arena with traces", flush=True)
    if not args.device_init:
        ref = StreamAC(agent.in_dim, action_dim, hidden_size=args.hidden_size)
        agent.load_torch(ref.actor, ref.critic)

    trackio.init(project=args.project, name=out.name, group=args.task, config={**config, **vars(args), "out": str(out)})
    recent, update_us, deltas = deque(maxlen=args.log_every), [], []
    start, episode = time.perf_counter(), 0

    obs, _ = env.reset(seed=args.seed)
    action = agent.reset(obs)
    for step in range(1, args.steps + 1):
        obs, reward, terminated, truncated, info = env.step(action)
        delta, us, action = agent.step(obs, reward, terminated, truncated)
        update_us.append(us)
        deltas.append(abs(delta))

        if action is None:  # episode over
            ret, length = float(info["episode"]["r"]), int(info["episode"]["l"])
            success = float(info.get("is_success", np.nan))
            metrics = {"episode/return": ret, "episode/length": length, "train/abs_td_error": np.mean(deltas),
                       "train/pico_update_us": np.mean(update_us)}
            if not np.isnan(success):
                metrics["episode/success"] = success
            trackio.log(metrics, step=step)
            recent.append((ret, length, success, np.mean(update_us)))
            update_us.clear(), deltas.clear()
            episode += 1
            if episode % args.log_every == 0:
                r, l, s, us_mean = np.mean(recent, axis=0)
                print(f"step {step:>9}  episode {episode:>6}  return {r:8.2f}  length {l:6.1f}  success {s:5.2f}  "
                      f"pico update {us_mean / 1000:5.2f} ms  {step / (time.perf_counter() - start):6.0f} steps/s",
                      flush=True)
            obs, _ = env.reset()
            action = agent.reset(obs)

        if step % args.eval_every == 0:
            save_checkpoint(out / "latest.pt", config, agent)
            log_eval(out / "latest.pt", config, args.eval_episodes, step)

    save_checkpoint(out / "final.pt", config, agent)
    env.close()
    result = log_eval(out / "final.pt", config, args.final_eval_episodes, args.steps)
    print(f"eval ({args.final_eval_episodes} episodes, deterministic): {result}", flush=True)
    trackio.finish()


if __name__ == "__main__":
    main()
