"""Command-line entry point for solving a local PNG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import solve_image


def _threshold_argument(value: str) -> int:
    try:
        threshold = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("threshold must be an integer in [0, 255]") from exc
    if not 0 <= threshold <= 255:
        raise argparse.ArgumentTypeError("threshold must be an integer in [0, 255]")
    return threshold


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fourier-line-art",
        description="Extract a line-art route, solve it, and save Fourier data.",
    )
    parser.add_argument("image", type=Path, help="input PNG or other Pillow-readable image")
    parser.add_argument("--output", type=Path, default=Path("solution.npz"))
    parser.add_argument("--threshold", type=_threshold_argument, default=128)
    parser.add_argument("--invert", action="store_true", help="treat light pixels as foreground")
    parser.add_argument("--connectivity", type=int, choices=(4, 8), default=8)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--lift-height", type=float, default=1.0)
    parser.add_argument("--max-rpp-required-edges", type=int, default=15)
    parser.add_argument("--max-rpp-states", type=int, default=2_000_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = solve_image(
        args.image,
        threshold=args.threshold,
        invert=args.invert,
        connectivity=args.connectivity,
        sample_count=args.samples,
        lift_height=args.lift_height,
        max_required_edges=args.max_rpp_required_edges,
        max_rpp_states=args.max_rpp_states,
    )
    output = result.save_npz(args.output)
    print(
        json.dumps(
            {
                "output": str(output),
                "solver": result.solver_name,
                "components": result.analysis.component_count,
                "route_cost": result.route.cost,
                "samples": len(result.samples.points),
                "fourier_terms": result.term_count,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

