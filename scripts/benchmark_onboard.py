"""On-board cost of each formation learner (one CPU thread, one team): state memory and step time.

PPO runs its whole update inside one step every ``rollout_steps``, which the max step time shows.

    uv run python scripts/benchmark_onboard.py --steps 5000 --out onboard_benchmark.md
"""

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch

from streampilot.baselines.mappo import MAPPO
from streampilot.stream_x.multi_agent import CentralizedStreamAC, IndependentStreamAC

CONTROL_DT = 0.05  # s, the formation tasks' control period (20 Hz)


def nbytes(*objs) -> int:
    """Bytes of every tensor / array in ``objs`` (modules, optimizers, dicts, lists, tensors)."""
    total = 0
    for o in objs:
        if isinstance(o, torch.nn.Module):
            total += sum(t.numel() * t.element_size() for t in [*o.parameters(), *o.buffers()])
        elif isinstance(o, torch.optim.Optimizer):
            total += nbytes(*[v for s in o.state.values() for v in s.values()])
            total += nbytes(*[g[k] for g in o.param_groups for k in g if k.startswith("_trace")])
        elif isinstance(o, dict):
            total += nbytes(*o.values())
        elif isinstance(o, (list, tuple)):
            total += nbytes(*o)
        elif isinstance(o, torch.Tensor):
            total += o.numel() * o.element_size()
        elif isinstance(o, np.ndarray):
            total += o.nbytes
    return total


def n_params(*modules) -> int:
    return sum(p.numel() for m in modules for p in m.parameters())


def bench_stream(cls, a, steps, rng):
    drones = a.drones
    team = cls(drones, a.obs_dim, a.action_dim, a.frames, hidden_size=a.hidden_size)
    team.reset(rng.standard_normal((drones, a.obs_dim)))
    times = []
    for t in range(steps):
        t0 = time.perf_counter()
        actions = team.act()
        done = (t + 1) % a.episode_steps == 0
        team.observe(actions, float(rng.standard_normal()), rng.standard_normal((drones, a.obs_dim)), False, done)
        if done:
            team.reset(rng.standard_normal((drones, a.obs_dim)))
        times.append(time.perf_counter() - t0)
    state = nbytes(team.actor, team.critic, team.actor_optim, team.critic_optim)
    return times, state, n_params(team.actor, team.critic), 0


def bench_mappo(a, steps, rng, independent: bool):
    num_envs = a.num_envs
    actor_dim = a.frames * a.obs_dim + a.action_dim
    critic_dim = actor_dim * (1 if independent else a.drones)
    m = a.drones if independent else 1
    agent = MAPPO(
        actor_dim, critic_dim, a.action_dim, a.drones, num_envs, total_steps=10**9, independent=independent,
        hidden_size=a.ppo_hidden_size,
    )  # fmt: skip
    E, n = num_envs, a.drones
    times = []
    for t in range(steps):
        t0 = time.perf_counter()
        actor_x = rng.standard_normal((E, n, actor_dim)).astype(np.float32)
        actions = agent.act(actor_x)
        done = np.full(E, (t + 1) % a.episode_steps == 0)
        critic_x = rng.standard_normal((E, m, critic_dim)).astype(np.float32)
        agent.store(actor_x, critic_x, actions, rng.standard_normal(E), np.zeros(E), done, critic_x[done] if done.any() else None)
        if agent.rollout_full:
            agent.update(critic_x)
        times.append(time.perf_counter() - t0)
    state = nbytes(agent.actor, agent.critic, agent.optim, agent._buf)
    return times, state, n_params(agent.actor, agent.critic), nbytes(agent._buf)


ALGOS = {
    # name: (table label, benchmark)
    "istream_ac": ("IStream AC", lambda a, s, r: bench_stream(IndependentStreamAC, a, s, r)),
    "cstream_ac": ("CStream AC", lambda a, s, r: bench_stream(CentralizedStreamAC, a, s, r)),
    "mappo": ("MAPPO", lambda a, s, r: bench_mappo(a, s, r, False)),
    "ippo": ("IPPO", lambda a, s, r: bench_mappo(a, s, r, True)),
}


def run_one(name: str, a: argparse.Namespace, queue) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    times, state, params, buffer = ALGOS[name][1](a, a.steps, rng)
    times = np.asarray(times[a.warmup :] if len(times) > 2 * a.warmup else times) * 1e3
    queue.put(
        {
            "name": name,
            "params": params,
            "state_kb": state / 1024,
            "buffer_kb": buffer / 1024,
            "median_ms": float(np.median(times)),
            "p99_ms": float(np.percentile(times, 99)),
            "max_ms": float(times.max()),
            "mean_ms": float(times.mean()),
        }
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algos", nargs="+", default=list(ALGOS), choices=list(ALGOS))
    p.add_argument("--steps", type=int, default=3000, help="control steps timed per algorithm")
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--drones", type=int, default=3)
    p.add_argument("--obs-dim", type=int, default=12, help="one drone's observation row (detection mode)")
    p.add_argument("--action-dim", type=int, default=3)
    p.add_argument("--frames", type=int, default=4)
    p.add_argument("--hidden-size", type=int, default=128, help="Stream AC")
    p.add_argument("--ppo-hidden-size", type=int, default=64, help="MAPPO / IPPO actor (critic: 128)")
    p.add_argument("--num-envs", type=int, default=1, help="MAPPO / IPPO teams per step (training used 64)")
    p.add_argument("--episode-steps", type=int, default=500)
    p.add_argument("--out", type=Path, default=None, help="also write the table as Markdown")
    a = p.parse_args()

    ctx = mp.get_context("spawn")  # a fresh process per algorithm, so none times another's warm caches
    rows = []
    for name in a.algos:
        q = ctx.Queue()
        proc = ctx.Process(target=run_one, args=(name, a, q))
        proc.start()
        rows.append(q.get())
        proc.join()
        print(f"done: {name}", flush=True)

    budget = CONTROL_DT * 1e3
    header = "| Learner | Params | State (KiB) | of which buffer | Step mean (ms) | median (ms) | p99 (ms) | max (ms) | max / 50 ms |"
    lines = [header, "|" + "---|" * 9]
    for r in rows:
        lines.append(
            f"| {ALGOS[r['name']][0]} | {r['params']:,} | {r['state_kb']:,.0f} | {r['buffer_kb']:,.0f} | {r['mean_ms']:.3f} "
            f"| {r['median_ms']:.3f} | {r['p99_ms']:.3f} | {r['max_ms']:.1f} | {r['max_ms'] / budget:.2f} |"
        )
    table = "\n".join(lines)
    print()
    print(table)
    print(f"\n{a.drones} drones, {a.num_envs} team(s) for MAPPO / IPPO, one CPU thread (torch {torch.__version__}); control period {budget:.0f} ms.")
    if a.out:
        a.out.write_text(table + "\n")


if __name__ == "__main__":
    main()
