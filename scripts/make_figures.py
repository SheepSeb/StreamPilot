"""Every figure and table of the paper (code in scripts/figures/), into --out-dir (default runs_plot_fig/).

    uv run python scripts/make_figures.py all [--skip scenes]
    uv run python scripts/make_figures.py tracking|robustness|continual|diagrams|scenes [--only NAME ...]
    uv run python scripts/make_figures.py tracking -h    # the group's outputs and options
"""

import argparse
import importlib
from pathlib import Path

GROUPS = ["tracking", "robustness", "continual", "diagrams", "scenes"]


def group_parser(name: str):
    """The module of a group (imported only when asked for: scenes needs MuJoCo and torch) and its options."""
    module = importlib.import_module(f"figures.{name}")
    p = argparse.ArgumentParser(prog=f"make_figures.py {name}", description=module.__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", choices=module.OUTPUTS, help="these outputs only (default: all)")
    p.add_argument("--out-dir", type=Path, default=Path("runs_plot_fig"))
    module.add_args(p)
    return module, p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("group", choices=[*GROUPS, "all"])
    parser.add_argument("options", nargs=argparse.REMAINDER, help="the group's options (see GROUP -h)")
    top = parser.parse_args()
    if top.group != "all":
        module, p = group_parser(top.group)
        args = p.parse_args(top.options)
        module.run(args, args.only or module.OUTPUTS)
        return
    p = argparse.ArgumentParser(prog="make_figures.py all", description="every group with its defaults")
    p.add_argument("--out-dir", type=Path, default=Path("runs_plot_fig"))
    p.add_argument("--skip", nargs="+", choices=GROUPS, default=[], help="e.g. scenes, without a GPU for EGL")
    args = p.parse_args(top.options)
    for name in GROUPS:
        if name not in args.skip:
            print(f"=== {name}")
            module, gp = group_parser(name)
            module.run(gp.parse_args(["--out-dir", str(args.out_dir)]), module.OUTPUTS)


if __name__ == "__main__":
    main()
