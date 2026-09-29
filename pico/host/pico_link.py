"""PC side of the Pico Stream AC agent: the serial protocol (see firmware/src/protocol.h) and an agent
that runs the whole learner (history, normalization, networks, ObGD) on the microcontroller.

Two transports speak the same protocol:
  SerialTransport("/dev/ttyUSB0")  the Pico over a UART (stdlib termios, Linux/macOS)
  EmulatedTransport()              the same firmware C sources compiled for the PC, for tests
"""

import ctypes
import os
import struct
import subprocess
import termios
from pathlib import Path

import numpy as np
import torch

FIRMWARE_SRC = Path(__file__).resolve().parents[1] / "firmware" / "src"
REQ_MAGIC, RESP_MAGIC, MAX_PAYLOAD = 0xA5, 0x5A, 4096
ERRORS = {1: "unknown command", 2: "bad length", 3: "out of memory", 4: "wrong state (init or reset first)"}

BAUDS = {b: getattr(termios, f"B{b}") for b in (115200, 230400, 460800, 921600, 1000000, 1500000, 2000000, 3000000)
         if hasattr(termios, f"B{b}")}


class PicoError(RuntimeError):
    pass


class SerialTransport:
    def __init__(self, port: str, baud: int = 921600, timeout: float = 10.0):
        self.fd = os.open(port, os.O_RDWR | os.O_NOCTTY)
        attrs = termios.tcgetattr(self.fd)
        iflag, oflag, cflag, lflag, _, _, cc = attrs
        # Raw 8N1, no flow control or line processing.
        iflag &= ~(termios.IGNBRK | termios.BRKINT | termios.PARMRK | termios.ISTRIP | termios.INLCR
                   | termios.IGNCR | termios.ICRNL | termios.IXON | termios.IXOFF | termios.IXANY)
        oflag &= ~termios.OPOST
        lflag &= ~(termios.ECHO | termios.ECHONL | termios.ICANON | termios.ISIG | termios.IEXTEN)
        cflag &= ~(termios.CSIZE | termios.PARENB | termios.CSTOPB | getattr(termios, "CRTSCTS", 0))
        cflag |= termios.CS8 | termios.CREAD | termios.CLOCAL
        cc[termios.VMIN], cc[termios.VTIME] = 0, min(255, int(timeout * 10))
        termios.tcsetattr(self.fd, termios.TCSANOW, [iflag, oflag, cflag, lflag, BAUDS[baud], BAUDS[baud], cc])
        termios.tcflush(self.fd, termios.TCIOFLUSH)

    def _read(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = os.read(self.fd, n - len(buf))
            if not chunk:
                raise TimeoutError("no answer from the Pico (check wiring, baud rate and that it is flashed)")
            buf += chunk
        return buf

    def transact(self, cmd: str, payload: bytes = b"") -> tuple[int, bytes]:
        os.write(self.fd, struct.pack("<BBH", REQ_MAGIC, ord(cmd), len(payload)) + payload)
        while self._read(1)[0] != RESP_MAGIC:
            pass
        status, length = struct.unpack("<BH", self._read(3))
        return status, self._read(length)

    def close(self) -> None:
        os.close(self.fd)


class EmulatedTransport:
    """The firmware's protocol.c and stream_ac.c compiled into a shared library and called directly."""

    def __init__(self, build_dir: Path | None = None):
        build_dir = build_dir or FIRMWARE_SRC.parents[1] / "build-host"
        build_dir.mkdir(exist_ok=True)
        lib = build_dir / "libstreampilot_pico.so"
        sources = [FIRMWARE_SRC / "protocol.c", FIRMWARE_SRC / "stream_ac.c"]
        headers = list(FIRMWARE_SRC.glob("*.h"))
        if not lib.exists() or lib.stat().st_mtime < max(p.stat().st_mtime for p in sources + headers):
            cmd = ["cc", "-O2", "-std=c11", "-Wall", "-Wextra", "-shared", "-fPIC", "-DSAC_ARENA_BYTES=(8u*1024u*1024u)",
                   *map(str, sources), "-o", str(lib), "-lm"]
            subprocess.run(cmd, check=True)
        self.lib = ctypes.CDLL(str(lib))
        self.lib.proto_handle.restype = ctypes.c_uint8
        self._out = ctypes.create_string_buffer(MAX_PAYLOAD)
        self._out_len = ctypes.c_uint32()

    def transact(self, cmd: str, payload: bytes = b"") -> tuple[int, bytes]:
        status = self.lib.proto_handle(ctypes.c_uint8(ord(cmd)), payload, ctypes.c_uint32(len(payload)),
                                       self._out, ctypes.byref(self._out_len))
        return status, self._out.raw[: self._out_len.value]

    def close(self) -> None:
        pass


def flatten(net: torch.nn.Module) -> np.ndarray:
    """A stream_x ``MLP`` in the firmware's layout: W1, b1, W2, b2, stacked head weights, stacked head biases."""
    heads = net.heads
    parts = [net.hidden[0].weight, net.hidden[0].bias, net.hidden[1].weight, net.hidden[1].bias,
             torch.cat([h.weight for h in heads]), torch.cat([h.bias for h in heads])]
    return torch.cat([p.detach().flatten() for p in parts]).numpy().astype(np.float32)


def unflatten(flat: np.ndarray, in_dim: int, hidden: int, head_dims: tuple[int, ...]) -> dict:
    """Inverse of ``flatten``, as a state dict for ``stream_x.agents.MLP``."""
    out, state, i = sum(head_dims), {}, 0

    def take(*shape):
        nonlocal i
        n = int(np.prod(shape))
        t = torch.from_numpy(flat[i : i + n].copy()).reshape(shape)
        i += n
        return t

    state["hidden.0.weight"], state["hidden.0.bias"] = take(hidden, in_dim), take(hidden)
    state["hidden.1.weight"], state["hidden.1.bias"] = take(hidden, hidden), take(hidden)
    w, b = take(out, hidden), take(out)
    for k, (start, d) in enumerate(zip(np.cumsum((0, *head_dims[:-1])), head_dims)):
        state[f"heads.{k}.weight"], state[f"heads.{k}.bias"] = w[start : start + d], b[start : start + d]
    return state


class PicoAgent:
    """Stream AC(lambda) running on the Pico. The PC sends raw observations and rewards and gets actions back."""

    def __init__(self, transport):
        self.t = transport

    def _call(self, cmd: str, payload: bytes = b"") -> bytes:
        status, data = self.t.transact(cmd, payload)
        if status:
            raise PicoError(f"command {cmd!r}: {ERRORS.get(status, status)}")
        return data

    def ping(self) -> tuple[int, int]:
        return struct.unpack("<II", self._call("P"))

    def init(self, obs_dim: int, action_dim: int, num_frames: int = 4, hidden_size: int = 128, normalize: bool = True,
             lr: float = 1.0, gamma: float = 0.99, lamda: float = 0.8, kappa_policy: float = 3.0,
             kappa_value: float = 2.0, entropy_coeff: float = 0.01, seed: int = 0) -> None:
        payload = struct.pack("<5i6fQ", obs_dim, action_dim, num_frames, hidden_size, int(normalize),
                              lr, gamma, lamda, kappa_policy, kappa_value, entropy_coeff, seed & (2**64 - 1))
        self.n_actor, self.n_critic, self.in_dim = struct.unpack("<3I", self._call("I", payload))
        self.obs_dim, self.action_dim, self.hidden_size = obs_dim, action_dim, hidden_size

    def set_params(self, flat: np.ndarray) -> None:
        flat = np.ascontiguousarray(flat, dtype="<f4")
        assert flat.size == self.n_actor + self.n_critic
        step = (MAX_PAYLOAD - 4) // 4
        for offset in range(0, flat.size, step):
            self._call("W", struct.pack("<I", offset) + flat[offset : offset + step].tobytes())

    def get_params(self) -> np.ndarray:
        total, step = self.n_actor + self.n_critic, MAX_PAYLOAD // 4
        chunks = [self._call("G", struct.pack("<II", o, min(step, total - o))) for o in range(0, total, step)]
        return np.frombuffer(b"".join(chunks), dtype="<f4").copy()

    def load_torch(self, actor: torch.nn.Module, critic: torch.nn.Module) -> None:
        self.set_params(np.concatenate([flatten(actor), flatten(critic)]))

    def state_dict(self) -> dict:
        flat = self.get_params()
        return {
            "actor": unflatten(flat[: self.n_actor], self.in_dim, self.hidden_size, (self.action_dim, self.action_dim)),
            "critic": unflatten(flat[self.n_actor :], self.in_dim, self.hidden_size, (1,)),
        }

    def reset(self, obs) -> np.ndarray:
        return np.frombuffer(self._call("R", np.asarray(obs, dtype="<f4").tobytes()), dtype="<f4").copy()

    def step(self, obs, reward: float, terminated: bool, truncated: bool) -> tuple[float, int, np.ndarray | None]:
        """Returns (TD error, update time on the Pico in us, next action or None when the episode ended)."""
        payload = np.asarray(obs, dtype="<f4").tobytes() + struct.pack("<dBB", reward, terminated, truncated)
        data = self._call("S", payload)
        delta, us, has_action = struct.unpack("<fIB", data[:9])
        return delta, us, np.frombuffer(data[9:], dtype="<f4").copy() if has_action else None

    def stats(self) -> tuple[dict, dict]:
        """(observation stats, return stats) in ``RunningMeanStd.state_dict`` form."""
        data, out = self._call("N"), []
        for n in (self.in_dim, 1):
            (count,), data = struct.unpack("<Q", data[:8]), data[8:]
            mean, var, m2 = np.frombuffer(data[: 24 * n], dtype="<f8").reshape(3, n).copy()
            data = data[24 * n :]
            out.append({"mean": mean, "var": var, "m2": m2, "count": count})
        return out[0], out[1]

    def update(self, obs, action, reward: float, next_obs, terminated: bool, done: bool) -> float:
        """Raw TD(lambda) update on already-normalized features (testing)."""
        f = lambda x: np.asarray(x, dtype="<f4").tobytes()  # noqa: E731
        payload = f(obs) + f(action) + struct.pack("<f", reward) + f(next_obs) + struct.pack("<BB", terminated, done)
        return struct.unpack("<f", self._call("U", payload))[0]

    def forward(self, obs) -> tuple[np.ndarray, np.ndarray, float]:
        """(mu, std, value) on already-normalized features (testing)."""
        v = np.frombuffer(self._call("F", np.asarray(obs, dtype="<f4").tobytes()), dtype="<f4")
        a = self.action_dim
        return v[:a].copy(), v[a : 2 * a].copy(), float(v[2 * a])


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Check a flashed Pico: ping, init, and time training steps.")
    parser.add_argument("port", help="e.g. /dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    agent = PicoAgent(SerialTransport(args.port, args.baud))
    version, arena = agent.ping()
    print(f"ping ok: protocol v{version}, arena {arena // 1024} KB")
    agent.init(obs_dim=5, action_dim=3, hidden_size=args.hidden_size)
    print(f"init ok: {agent.n_actor + agent.n_critic} parameters, {agent.in_dim} input features")
    rng = np.random.default_rng(0)
    agent.reset(rng.random(5))
    update_us, start = [], time.perf_counter()
    for t in range(args.steps):
        _, us, action = agent.step(rng.random(5), float(rng.normal()), False, t % 100 == 99)
        update_us.append(us)
        if action is None:
            agent.reset(rng.random(5))
    wall = (time.perf_counter() - start) / args.steps * 1000
    print(f"{args.steps} steps: update on the Pico {np.mean(update_us) / 1000:.2f} ms, "
          f"round trip {wall:.2f} ms/step ({1000 / wall:.0f} steps/s)")
