"""Deterministic harsh-image variants for LunaX robustness testing."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class HarshVariant:
    name: str
    angle_degrees: float
    crop_fraction: float
    gamma: float
    contrast: float


def _lighting(image: np.ndarray, gamma: float, contrast: float) -> np.ndarray:
    values = image.astype(np.float32) / 255.0
    values = np.clip((values ** gamma - 0.5) * contrast + 0.5, 0.0, 1.0)
    return np.rint(values * 255.0).astype(np.uint8)


def _crop_and_resize(image: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    height, width = image.shape[:2]
    crop_h, crop_w = max(2, round(height * fraction)), max(2, round(width * fraction))
    top = int(rng.integers(0, height - crop_h + 1))
    left = int(rng.integers(0, width - crop_w + 1))
    crop = image[top:top + crop_h, left:left + crop_w]
    return cv2.resize(crop, (width, height), interpolation=cv2.INTER_CUBIC)


def _rotate(image: np.ndarray, angle: float) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)


def create_harsh_variants(image: np.ndarray, seed: int) -> Dict[str, np.ndarray]:
    """Create severe yet reproducible lighting, crop, rotation, and combined variants."""
    if image is None or image.size == 0:
        raise ValueError("image must be non-empty")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    rng = np.random.default_rng(seed)
    angle = float(rng.uniform(40.0, 60.0) * rng.choice((-1.0, 1.0)))
    fraction = float(rng.uniform(0.58, 0.72))
    gamma = float(rng.choice((0.32, 2.6)))
    contrast = float(rng.uniform(1.5, 2.2))
    lighting = _lighting(gray, gamma, contrast)
    cropped = _crop_and_resize(gray, fraction, rng)
    rotated = _rotate(gray, angle)
    combined = _lighting(_rotate(_crop_and_resize(gray, fraction, rng), angle), gamma, contrast)
    return {"lighting": lighting, "crop": cropped, "rotation": rotated, "combined": combined}


def generate_harsh_sample_set(source_dir: Path, output_dir: Path, start: int = 2, end: int = 50) -> int:
    """Write four flat, named variants for each ``sample_N.png`` in range."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for index in range(start, end + 1):
        source = source_dir / f"sample_{index}.png"
        image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Could not load required sample image: {source}")
        for name, variant in create_harsh_variants(image, seed=10_000 + index).items():
            target = output_dir / f"sample_{index}_{name}.png"
            if not cv2.imwrite(str(target), variant):
                raise IOError(f"Could not write: {target}")
            written += 1
    return written
