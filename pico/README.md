# Stream AC on a Raspberry Pi Pico 2 W

Runs the whole Stream AC(λ) learner on the microcontroller, with the MuJoCo environment staying on the PC.
The Pico holds the observation history, the observation and reward normalization, the actor and critic,
their eligibility traces and the ObGD updates. The PC sends raw observations and rewards over a serial line
and gets actions back.

- `firmware/src/stream_ac.c`: the algorithm in plain C (only `<math.h>`, no heap, one static 384 KB arena).
  It is a port of `streampilot/stream_x/agents.py` and `wrappers.py`, with a hand-written backward pass.
- `firmware/src/protocol.c`: the binary request/response protocol (spec in `protocol.h`), hardware independent.
- `firmware/src/main.c`: the only Pico-specific file. It sets up UART0 and serves the protocol.
  The Pico SDK is used only for boot and UART.
- `host/pico_link.py`: the PC side. Opens the serial port with stdlib `termios` (no pyserial) and provides
  `EmulatedTransport`, which compiles the same C sources into a shared library for tests.
- `host/train_pico.py`: the training loop. Its checkpoints load with `streampilot.policy.Policy`.

## Wiring

Use a 3.3 V USB-to-UART adapter (CP2102, FT232, CH340...):

| Adapter | Pico 2 W        |
|---------|-----------------|
| RX      | GP0 (pin 1, TX) |
| TX      | GP1 (pin 2, RX) |
| GND     | GND (pin 3)     |

Power the Pico from its own USB port or VSYS. The default baud rate is 921600 8N1 with no flow control.

## Build and flash

```sh
sudo apt install cmake gcc-arm-none-eabi libnewlib-arm-none-eabi
git clone --depth 1 https://github.com/raspberrypi/pico-sdk ~/pico-sdk && git -C ~/pico-sdk submodule update --init
export PICO_SDK_PATH=~/pico-sdk
cmake -S pico/firmware -B pico/firmware/build        # -DSAC_BAUD=2000000 if your adapter can
cmake --build pico/firmware/build -j
```

Hold BOOTSEL while plugging the Pico in, then copy `pico/firmware/build/streampilot_pico.uf2` to the drive that appears.
Check it with `uv run python pico/host/pico_link.py /dev/ttyUSB0`, which pings the Pico, initializes a network and times 500 training steps.

## Train

```sh
uv run python pico/host/train_pico.py waypoint --port /dev/ttyUSB0
uv run python pico/host/train_pico.py waypoint --emulate            # no hardware: the C code on the PC
uv run streampilot waypoint --policy runs/pico_waypoint_seed0/final.pt
```

By default the PC builds the initial weights with PyTorch's `StreamAC` and uploads them. `--device-init` uses the
Pico's own sparse init instead. Trackio logs `train/pico_update_us`, the time the Pico spends per update.

## Test

```sh
uv run pytest pico/tests
```

The tests compile the firmware C code for the PC and check it against the PyTorch reference:
the forward pass, 300 TD(λ) updates, and the full pipeline (history, normalization, reward scaling) on a random
environment. They agree to about 1e-7 after 2000 updates.

## Memory

The network weights and eligibility traces are both float32. Input sizes include 4 stacked frames plus the previous action:

| Observation (input) | Hidden | Parameters | Weights + traces |
|---------------------|--------|------------|------------------|
| detection (23)      | 128    | 40,071     | 313 KB           |
| state (39)          | 128    | 44,167     | 345 KB           |
| detection (23)      | 64     | 11,847     | 93 KB            |

The arena is 384 KB of the RP2350's 520 KB of SRAM. Use `--hidden-size 64` to leave room for, for example, the Wi-Fi stack.
