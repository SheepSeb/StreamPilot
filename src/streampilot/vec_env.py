"""Vectorized formation environments: ``num_envs`` copies stepped in parallel by ``num_workers``
subprocesses, several environments per process.

Observations, privileged states, rewards and done flags go through shared memory, so a step
costs one small pipe message per worker, not a pickled copy of every observation and info dict.
Only finished episodes (a handful of floats each) travel back through the pipes. Environments
reset themselves when an episode ends, and the terminal observation is kept in ``final_obs``
(and ``final_state``) so the learner can bootstrap through time-limit truncation.
"""

import multiprocessing as mp
import os
from contextlib import contextmanager

import numpy as np

# Episode-end ``info`` entries reported per finished episode (averaged over the drones when
# per-drone); ``collision`` and ``out_of_bounds`` are only in ``info`` when they happen.
EPISODE_INFO = (
    "is_success",
    "collision",
    "out_of_bounds",
    "stages_reached",
    "formation_error",
    "standoff_error",
    "distance",
    "heading_error",
)


def _shared(shape, dtype) -> tuple[mp.Array, tuple, np.dtype]:
    dtype = np.dtype(dtype)
    return mp.get_context("spawn").RawArray("b", max(int(np.prod(shape)) * dtype.itemsize, 1)), shape, dtype


def _view(buffer) -> np.ndarray:
    raw, shape, dtype = buffer
    return np.frombuffer(raw, dtype=dtype, count=int(np.prod(shape))).reshape(shape)


@contextmanager
def _single_threaded_children():
    """Spawned workers inherit the environment: keep their BLAS/OpenMP pools at one thread, since
    every core already runs a worker."""
    keys = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    saved = {k: os.environ.get(k) for k in keys}
    os.environ.update(dict.fromkeys(keys, "1"))
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k)
            else:
                os.environ[k] = v


def _episode_summary(info: dict, ret: float, length: int) -> dict:
    summary = {"return": ret, "length": length}
    for key in EPISODE_INFO:
        if key in info:
            summary[key] = float(np.mean(info[key]))
    summary.setdefault("collision", 0.0)
    summary.setdefault("out_of_bounds", 0.0)
    return summary


def _worker(pipe, env_id: str, env_kwargs: dict, indices: range, buffers: dict, with_state: bool) -> None:
    import gymnasium as gym

    import streampilot.env  # noqa: F401  (registers the environments)

    arrays = {k: _view(v) for k, v in buffers.items()}
    action, obs, final_obs = arrays["action"], arrays["obs"], arrays["final_obs"]
    reward, terminated, truncated = arrays["reward"], arrays["terminated"], arrays["truncated"]
    envs = [gym.make(env_id, disable_env_checker=True, **env_kwargs) for _ in indices]
    returns, lengths = np.zeros(len(envs)), np.zeros(len(envs), dtype=int)

    def write_state(i, env, key="state"):
        if with_state:
            arrays[key][i] = env.unwrapped.state_obs()

    try:
        while True:
            cmd, arg = pipe.recv()
            if cmd == "step":
                episodes = []
                for k, (i, env) in enumerate(zip(indices, envs)):
                    o, r, te, tr, info = env.step(action[i])
                    returns[k] += r
                    lengths[k] += 1
                    reward[i], terminated[i], truncated[i] = r, te, tr
                    if te or tr:
                        final_obs[i] = o
                        write_state(i, env, "final_state")
                        episodes.append((i, _episode_summary(info, float(returns[k]), int(lengths[k]))))
                        returns[k], lengths[k] = 0.0, 0
                        o, _ = env.reset()
                    obs[i] = o
                    write_state(i, env)
                pipe.send(episodes)
            elif cmd == "reset":  # arg: one seed per environment of this worker, or None
                for k, (i, env) in enumerate(zip(indices, envs)):
                    o, _ = env.reset(seed=None if arg is None else int(arg[k]))
                    obs[i] = o
                    write_state(i, env)
                returns[:], lengths[:] = 0.0, 0
                pipe.send(None)
            elif cmd == "close":
                break
    except KeyboardInterrupt:
        pass
    finally:
        for env in envs:
            env.close()
        pipe.close()


class FormationVecEnv:
    """``num_envs`` formation environments in ``num_workers`` processes.

    ``step(actions)`` takes ``(num_envs, num_drones, 3)`` and returns ``(obs, reward, terminated,
    truncated, episodes)``. The arrays are views of shared memory, overwritten by the next call:
    copy what you keep. Where an episode ended, ``obs`` is already the next episode's first
    observation and ``final_obs`` holds the terminal one. ``episodes`` lists ``(env_index,
    summary)`` for every episode that ended in this step. With ``with_state``, ``state`` and
    ``final_state`` hold each environment's privileged ``state_obs()`` (for a centralised critic).
    """

    def __init__(
        self,
        env_id: str,
        env_kwargs: dict,
        num_envs: int,
        num_workers: int | None = None,
        with_state: bool = False,
    ):
        import gymnasium as gym

        import streampilot.env  # noqa: F401

        probe = gym.make(env_id, disable_env_checker=True, **env_kwargs)
        self.num_envs = num_envs
        self.num_drones = probe.unwrapped.num_drones
        self.obs_dim = probe.observation_space.shape[-1]
        self.state_dim = probe.unwrapped.state_obs().shape[-1]
        self.action_dim = probe.action_space.shape[-1]
        probe.close()

        n, d = num_envs, self.num_drones
        specs = {
            "action": ((n, d, self.action_dim), np.float32),
            "obs": ((n, d, self.obs_dim), np.float32),
            "final_obs": ((n, d, self.obs_dim), np.float32),
            "reward": ((n,), np.float64),
            "terminated": ((n,), np.bool_),
            "truncated": ((n,), np.bool_),
        }
        if with_state:
            specs["state"] = ((n, d, self.state_dim), np.float32)
            specs["final_state"] = ((n, d, self.state_dim), np.float32)
        buffers = {k: _shared(*v) for k, v in specs.items()}
        arrays = {k: _view(v) for k, v in buffers.items()}
        self._action = arrays["action"]
        self.obs, self.final_obs = arrays["obs"], arrays["final_obs"]
        self.reward, self.terminated, self.truncated = arrays["reward"], arrays["terminated"], arrays["truncated"]
        self.state, self.final_state = arrays.get("state"), arrays.get("final_state")

        num_workers = min(num_workers or os.cpu_count() or 1, num_envs)
        bounds = np.linspace(0, num_envs, num_workers + 1).round().astype(int)
        self._slices = [range(a, b) for a, b in zip(bounds[:-1], bounds[1:])]
        ctx = mp.get_context("spawn")
        self._pipes, self._procs = [], []
        with _single_threaded_children():
            for indices in self._slices:
                parent, child = ctx.Pipe()
                proc = ctx.Process(
                    target=_worker, args=(child, env_id, env_kwargs, indices, buffers, with_state), daemon=True
                )
                proc.start()
                child.close()
                self._pipes.append(parent)
                self._procs.append(proc)
        self._closed = False

    def reset(self, seeds=None) -> np.ndarray:
        """Reset every environment, with ``seeds[i]`` for environment ``i`` (or unseeded)."""
        for pipe, indices in zip(self._pipes, self._slices):
            pipe.send(("reset", None if seeds is None else [seeds[i] for i in indices]))
        for pipe in self._pipes:
            pipe.recv()
        return self.obs

    def step(self, actions):
        self._action[:] = actions
        for pipe in self._pipes:
            pipe.send(("step", None))
        episodes = []
        for pipe in self._pipes:
            episodes.extend(pipe.recv())
        return self.obs, self.reward, self.terminated, self.truncated, episodes

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for pipe in self._pipes:
            try:
                pipe.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self._procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()

    def __del__(self):
        self.close()
