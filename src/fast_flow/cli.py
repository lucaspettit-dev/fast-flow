"""Command-line interface for fast-flow."""

from __future__ import annotations

import argparse

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fast-flow",
        description="Fast 2D incompressible flow solver for rapid design iteration.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--demo", action="store_true",
                   help="run the built-in demo (not implemented yet)")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.demo:
        raise NotImplementedError("demo coming soon")
    build_parser().print_help()
