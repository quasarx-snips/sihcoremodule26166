"""LunaX command-line workspace for image-to-image lunar registration.

Examples
--------
Run registration and print the judge-facing report::

    python main.py source.png reference.png

Save all presentation artifacts into a chosen directory::

    python main.py source.png reference.png --save-outputs --output-dir results/demo_01
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path
from typing import Sequence

from module_11 import PipelineConfig, print_lunax_pipeline_report, run_lunax_registration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lunax",
        description="LunaX: verified feature-based lunar image registration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("source", type=Path, help="Source image to warp into the reference coordinate frame.")
    parser.add_argument("reference", type=Path, help="Reference image that defines the output frame.")
    parser.add_argument("--save-outputs", action="store_true", help="Write registration artifacts for presentation.")
    parser.add_argument("--output-dir", type=Path, help="Directory for saved artifacts; required with --save-outputs.")
    parser.add_argument("--sift-features", type=int, default=3000, help="Maximum SIFT features extracted per image.")
    parser.add_argument("--ratio", type=float, default=0.75, help="Descriptor ratio-test threshold in (0, 1].")
    parser.add_argument("--reprojection-threshold", type=float, default=3.0, help="RANSAC inlier reprojection threshold in pixels.")
    parser.add_argument("--max-per-cell", type=int, default=10, help="Maximum spatially selected matches per grid cell.")
    parser.add_argument("--no-refinement", action="store_true", help="Disable local sub-pixel correspondence refinement.")
    parser.add_argument("--quiet", action="store_true", help="Suppress intermediate module output; still print final report.")
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    for label, path in (("source", args.source), ("reference", args.reference)):
        if not path.is_file():
            parser.error(f"{label} image does not exist or is not a file: {path}")
    if args.save_outputs and args.output_dir is None:
        parser.error("--output-dir is required when --save-outputs is used")
    if not 0.0 < args.ratio <= 1.0:
        parser.error("--ratio must be in (0, 1]")
    if args.sift_features < 1 or args.max_per_cell < 1 or args.reprojection_threshold <= 0:
        parser.error("--sift-features, --max-per-cell, and --reprojection-threshold must be positive")


def run_cli(argv: Sequence[str] | None = None) -> int:
    """Run the full pipeline from command-line arguments and return a shell exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)

    config = PipelineConfig(
        verbose=not args.quiet,
        save_outputs=args.save_outputs,
        output_dir=str(args.output_dir) if args.output_dir else None,
        sift_features=args.sift_features,
        matching={"ratio": args.ratio, "mutual_consistency": True},
        verification={"reprojection_threshold": args.reprojection_threshold, "min_inliers": 4, "random_seed": 42},
        spatial={"rows": 4, "cols": 4, "max_matches_per_cell": args.max_per_cell},
        refinement={"enabled": not args.no_refinement, "max_refinement_count": 500},
    )

    print("\n" + "=" * 72)
    print("LUNAX | LUNAR IMAGE REGISTRATION WORKSPACE")
    print("=" * 72)
    print(f"Source    : {args.source}")
    print(f"Reference : {args.reference}")
    print("Workflow  : preprocess -> features -> matches -> geometry -> warp -> metrics")
    print("=" * 72)

    # Some legacy modules print their own debug tables.  Keep them available in
    # normal mode, while quiet mode produces a clean presentation transcript.
    if args.quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            result = run_lunax_registration(args.source, args.reference, config)
    else:
        result = run_lunax_registration(args.source, args.reference, config)
    # `verbose=False` hides pipeline chatter, not the useful presentation report.
    if args.quiet or not result.success:
        print_lunax_pipeline_report(result)
    if result.success and args.save_outputs:
        print(f"\nPresentation artifacts saved to: {result.diagnostics['output_dir']}")
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(run_cli())
