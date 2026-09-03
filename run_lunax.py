"""Single command-line entry point for LunaX registration."""
from __future__ import annotations

import argparse
from pathlib import Path

from lunax import PipelineConfig, print_lunax_pipeline_report, run_lunax_registration


def main() -> int:
    parser = argparse.ArgumentParser(description="LunaX lunar image correspondence and registration")
    parser.add_argument("--image-a", required=True, type=Path, help="Source lunar image")
    parser.add_argument("--image-b", required=True, type=Path, help="Reference lunar image")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/latest_run"), help="One directory for all generated artifacts")
    parser.add_argument("--no-save", action="store_true", help="Do not write artifacts")
    parser.add_argument("--sift-features", type=int, default=3000)
    parser.add_argument("--ratio", type=float, default=.75)
    parser.add_argument("--reprojection-threshold", type=float, default=3.)
    args = parser.parse_args()
    if not args.image_a.is_file() or not args.image_b.is_file(): parser.error("--image-a and --image-b must be existing files")
    if not 0 < args.ratio <= 1 or args.sift_features < 1 or args.reprojection_threshold <= 0: parser.error("invalid matching or RANSAC parameter")
    result = run_lunax_registration(args.image_a, args.image_b, PipelineConfig(
        verbose=False, save_outputs=not args.no_save, output_dir=str(args.output_dir), sift_features=args.sift_features,
        matching={"ratio": args.ratio, "mutual_consistency": True},
        verification={"model":"auto", "reprojection_threshold":args.reprojection_threshold, "confidence":.999, "max_iterations":2000, "min_inliers":4, "random_seed":42},
    ))
    print_lunax_pipeline_report(result)
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
