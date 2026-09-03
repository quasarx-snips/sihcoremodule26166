"""Generate four harsh robustness images for each LunaX sample."""
from __future__ import annotations

import argparse
from pathlib import Path

from lunax.augmentation import generate_harsh_sample_set


def main() -> int:
    parser = argparse.ArgumentParser(description="Create lighting, crop, rotation, and combined harsh lunar samples")
    parser.add_argument("--source-dir", type=Path, default=Path("lunar_samples"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/harsh_test_images"))
    args = parser.parse_args()
    count = generate_harsh_sample_set(args.source_dir, args.output_dir)
    print(f"Generated {count} images in: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
