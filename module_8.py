"""LunaX Module 08: bounded local sub-pixel correspondence refinement.

Inputs must already be geometrically verified correspondences.  This module never
performs global feature search: each reference point is searched only inside a
small, configurable window around its supplied location.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass
class RefinementConfig:
    enabled: bool = True
    patch_radius: int = 7
    search_radius: int = 3
    minimum_patch_std: float = 3.0
    minimum_correlation: float = 0.60
    minimum_peak_margin: float = 0.02
    max_refinement_count: Optional[int] = 1000
    transformation: Optional[np.ndarray] = None  # source -> reference, optional residual report


@dataclass
class CorrespondenceDiagnostics:
    valid: bool
    reason: str
    correlation: float = float("nan")
    peak_margin: float = float("nan")
    local_shift: Tuple[float, float] = (0.0, 0.0)
    before_residual: Optional[float] = None
    after_residual: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"valid": self.valid, "reason": self.reason, "correlation": self.correlation,
                "peak_margin": self.peak_margin, "local_shift": self.local_shift,
                "before_residual": self.before_residual, "after_residual": self.after_residual}


def _config(config: Optional[Any]) -> RefinementConfig:
    if config is None:
        return RefinementConfig()
    if isinstance(config, RefinementConfig):
        return config
    if isinstance(config, dict):
        return RefinementConfig(**config)
    raise TypeError("config must be None, a RefinementConfig, or a dictionary")


def _gray(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3:
        if array.shape[2] == 4:
            array = cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)
        elif array.shape[2] == 3:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("multi-channel images must have 3 or 4 channels")
    if array.ndim != 2 or array.size == 0:
        raise ValueError("images must be non-empty grayscale or BGR/BGRA arrays")
    return array.astype(np.float32)


def _inside(point: np.ndarray, radius: int, image: np.ndarray) -> bool:
    x, y = point
    return radius <= x < image.shape[1] - radius and radius <= y < image.shape[0] - radius


def _subpixel_peak(scores: np.ndarray, x: int, y: int) -> Tuple[float, float]:
    """Quadratic interpolation of a correlation peak; bounded for stability."""
    if x == 0 or y == 0 or x == scores.shape[1] - 1 or y == scores.shape[0] - 1:
        return 0.0, 0.0
    center = float(scores[y, x])
    def offset(left: float, right: float) -> float:
        denominator = left - 2.0 * center + right
        if abs(denominator) < 1e-12:
            return 0.0
        return float(np.clip(0.5 * (left - right) / denominator, -0.75, 0.75))
    return offset(float(scores[y, x - 1]), float(scores[y, x + 1])), \
        offset(float(scores[y - 1, x]), float(scores[y + 1, x]))


def _residual(point_a: np.ndarray, point_b: np.ndarray, transformation: Optional[np.ndarray]) -> Optional[float]:
    if transformation is None:
        return None
    matrix = np.asarray(transformation, dtype=np.float64)
    if matrix.shape == (2, 3):
        matrix = np.vstack((matrix, (0.0, 0.0, 1.0)))
    if matrix.shape != (3, 3):
        raise ValueError("transformation must have shape (2, 3) or (3, 3)")
    projected = matrix @ np.array([point_a[0], point_a[1], 1.0])
    if abs(projected[2]) < 1e-12:
        return float("inf")
    return float(np.linalg.norm(projected[:2] / projected[2] - point_b))


def refine_correspondence(source_image, reference_image, source_point, reference_point, config=None):
    """Refine one verified pair by local normalized correlation and sub-pixel peak fitting."""
    cfg = _config(config)
    source, reference = _gray(source_image), _gray(reference_image)
    src = np.asarray(source_point, dtype=np.float64).reshape(-1)
    ref = np.asarray(reference_point, dtype=np.float64).reshape(-1)
    if src.shape != (2,) or ref.shape != (2,) or not np.all(np.isfinite(np.r_[src, ref])):
        raise ValueError("source_point and reference_point must each be finite (x, y) coordinates")
    before = _residual(src, ref, cfg.transformation)
    if not cfg.enabled:
        return src.copy(), ref.copy(), CorrespondenceDiagnostics(True, "disabled", before_residual=before, after_residual=before)
    if cfg.patch_radius < 2 or cfg.search_radius < 0:
        raise ValueError("patch_radius must be at least 2 and search_radius must be non-negative")
    patch_size = 2 * cfg.patch_radius + 1
    required_reference_radius = cfg.patch_radius + cfg.search_radius
    if not _inside(src, cfg.patch_radius, source) or not _inside(ref, required_reference_radius, reference):
        return src.copy(), ref.copy(), CorrespondenceDiagnostics(False, "patch/search window crosses image boundary", before_residual=before)

    template = cv2.getRectSubPix(source, (patch_size, patch_size), tuple(src))
    if float(np.std(template)) < cfg.minimum_patch_std:
        return src.copy(), ref.copy(), CorrespondenceDiagnostics(False, "low-texture source patch", before_residual=before)
    search_size = patch_size + 2 * cfg.search_radius
    search = cv2.getRectSubPix(reference, (search_size, search_size), tuple(ref))
    scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, peak, _, peak_location = cv2.minMaxLoc(scores)
    px, py = peak_location
    suppressed = scores.copy()
    suppressed[max(0, py - 1):py + 2, max(0, px - 1):px + 2] = -np.inf
    second_peak = float(np.max(suppressed)) if np.any(np.isfinite(suppressed)) else -1.0
    margin = float(peak - second_peak)
    if peak < cfg.minimum_correlation:
        return src.copy(), ref.copy(), CorrespondenceDiagnostics(False, "weak local correlation", float(peak), margin, before_residual=before)
    if margin < cfg.minimum_peak_margin:
        return src.copy(), ref.copy(), CorrespondenceDiagnostics(False, "ambiguous local correlation peak", float(peak), margin, before_residual=before)

    dx, dy = _subpixel_peak(scores, px, py)
    refined_ref = ref + np.array([px - cfg.search_radius + dx, py - cfg.search_radius + dy], dtype=np.float64)
    after = _residual(src, refined_ref, cfg.transformation)
    diagnostic = CorrespondenceDiagnostics(True, "ok", float(peak), margin,
                                            (float(refined_ref[0] - ref[0]), float(refined_ref[1] - ref[1])),
                                            before, after)
    return src.copy(), refined_ref, diagnostic


def refine_correspondences(source_image, reference_image, source_points, reference_points, config=None):
    """Refine a bounded number of verified pairs and return fractional coordinates plus validity mask."""
    cfg = _config(config)
    src = np.asarray(source_points, dtype=np.float64)
    ref = np.asarray(reference_points, dtype=np.float64)
    if src.ndim != 2 or src.shape[1] != 2 or ref.shape != src.shape:
        raise ValueError("source_points and reference_points must have matching shape (N, 2)")
    limit = len(src) if cfg.max_refinement_count is None else min(len(src), max(0, cfg.max_refinement_count))
    refined_src, refined_ref = src.copy(), ref.copy()
    valid = np.zeros(len(src), dtype=bool)
    details = []
    for index in range(limit):
        out_src, out_ref, detail = refine_correspondence(source_image, reference_image, src[index], ref[index], cfg)
        refined_src[index], refined_ref[index] = out_src, out_ref
        valid[index] = detail.valid
        details.append(detail.to_dict())
    for index in range(limit, len(src)):
        details.append(CorrespondenceDiagnostics(False, "maximum refinement count reached").to_dict())
    diagnostics: Dict[str, Any] = {"valid_mask": valid, "refined_count": int(limit), "details": details}
    if cfg.transformation is not None:
        before, before_stats = calculate_local_registration_error(src, ref, cfg.transformation)
        after, after_stats = calculate_local_registration_error(refined_src[valid], refined_ref[valid], cfg.transformation)
        diagnostics.update({"before_errors": before, "after_errors": after,
                            "before_statistics": before_stats, "after_statistics": after_stats})
    return refined_src, refined_ref, diagnostics


def calculate_local_registration_error(source_points, reference_points, transformation):
    """Return per-pair source-to-reference residuals and local summary statistics."""
    src, ref = np.asarray(source_points, dtype=np.float64), np.asarray(reference_points, dtype=np.float64)
    if src.ndim != 2 or src.shape[1] != 2 or ref.shape != src.shape:
        raise ValueError("source_points and reference_points must have matching shape (N, 2)")
    if len(src) == 0:
        return np.empty(0), {"count": 0, "rmse": float("nan"), "median": float("nan"), "mean": float("nan"), "p95": float("nan")}
    errors = np.array([_residual(a, b, transformation) for a, b in zip(src, ref)], dtype=np.float64)
    return errors, {"count": int(len(errors)), "rmse": float(np.sqrt(np.mean(errors ** 2))),
                    "median": float(np.median(errors)), "mean": float(np.mean(errors)),
                    "p95": float(np.percentile(errors, 95))}


def test_synthetic_fractional_translation() -> Dict[str, Any]:
    """Verify local refinement reduces coarse correspondence error for a known fractional shift."""
    rng = np.random.default_rng(7)
    source = cv2.GaussianBlur(rng.integers(0, 256, (180, 220), dtype=np.uint8), (0, 0), 1.2)
    shift = np.array([1.35, -0.70])
    transform = np.array([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]], [0.0, 0.0, 1.0]])
    reference = cv2.warpAffine(source, transform[:2], (source.shape[1], source.shape[0]), flags=cv2.INTER_LINEAR)
    source_points = np.array([[40., 40.], [80., 65.], [150., 100.], [180., 130.]])
    true_reference = source_points + shift
    coarse_reference = true_reference + np.array([[0.8, -0.6], [-0.7, 0.5], [0.6, 0.7], [-0.5, -0.8]])
    cfg = RefinementConfig(search_radius=3, transformation=transform)
    _, refined_reference, diagnostics = refine_correspondences(source, reference, source_points, coarse_reference, cfg)
    coarse_error = float(np.median(np.linalg.norm(coarse_reference - true_reference, axis=1)))
    refined_error = float(np.median(np.linalg.norm(refined_reference - true_reference, axis=1)))
    assert np.any(diagnostics["valid_mask"]), "No synthetic local patches were accepted"
    assert refined_error < coarse_error, f"Refinement did not improve: {refined_error} >= {coarse_error}"
    return {"coarse_error": coarse_error, "refined_error": refined_error, "diagnostics": diagnostics}


if __name__ == "__main__":
    result = test_synthetic_fractional_translation()
    print(f"Synthetic refinement passed: coarse={result['coarse_error']:.3f}px, refined={result['refined_error']:.3f}px")
