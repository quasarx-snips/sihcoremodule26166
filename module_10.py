"""LunaX Module 10: quantitative image-based registration evaluation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import cv2
import numpy as np


@dataclass
class EvaluationThresholds:
    max_rmse_px: float = 3.0
    min_inlier_ratio: float = 0.5


def calculate_inlier_ratio(inlier_mask):
    mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)
    return float(np.mean(mask)) if len(mask) else 0.0


def _matrix(transformation):
    matrix = np.asarray(transformation, dtype=np.float64)
    if matrix.shape == (2, 3): matrix = np.vstack((matrix, (0., 0., 1.)))
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)): raise ValueError("transformation must be a finite 2x3 or 3x3 matrix")
    return matrix


def calculate_reprojection_errors(source_points, reference_points, transformation):
    source, reference = np.asarray(source_points, dtype=np.float64), np.asarray(reference_points, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 2 or reference.shape != source.shape: raise ValueError("point arrays must have matching shape (N,2)")
    matrix = _matrix(transformation)
    projected = (matrix @ np.c_[source, np.ones(len(source))].T).T
    if np.any(np.abs(projected[:, 2]) < 1e-12): raise ValueError("transformation projects a point to infinity")
    return np.linalg.norm(projected[:, :2] / projected[:, 2:3] - reference, axis=1)


def calculate_rmse(errors):
    values = np.asarray(errors, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean(values ** 2))) if len(values) else float("nan")


def calculate_error_statistics(errors):
    values = np.asarray(errors, dtype=np.float64).reshape(-1)
    if not len(values): return {"count": 0, "mean": float("nan"), "median": float("nan"), "rmse": float("nan"), "max": float("nan"), "p95": float("nan")}
    return {"count": int(len(values)), "mean": float(np.mean(values)), "median": float(np.median(values)), "rmse": calculate_rmse(values), "max": float(np.max(values)), "p95": float(np.percentile(values, 95))}


def calculate_spatial_coverage(points, image_shape):
    """Independent 4x4 occupancy coverage for evaluation reports."""
    points = np.asarray(points, dtype=np.float64); h, w = int(image_shape[0]), int(image_shape[1]); counts = np.zeros((4, 4), dtype=int)
    valid = np.isfinite(points).all(axis=1) & (points[:, 0] >= 0) & (points[:, 0] < w) & (points[:, 1] >= 0) & (points[:, 1] < h)
    for x, y in points[valid]: counts[min(int(y * 4 / h), 3), min(int(x * 4 / w), 3)] += 1
    return {"occupied_cells": int(np.count_nonzero(counts)), "total_cells": 16, "coverage_percentage": 100.0 * np.count_nonzero(counts) / 16, "points_per_cell": counts}


def _histogram(errors, width=480, height=180):
    canvas = np.full((height, width, 3), 255, np.uint8)
    if len(errors):
        counts, _ = np.histogram(errors, bins=20); scale = (height - 20) / max(int(counts.max()), 1)
        for i, count in enumerate(counts):
            x0, x1 = i * width // 20, (i + 1) * width // 20 - 1
            cv2.rectangle(canvas, (x0, height - 1), (x1, height - round(count * scale)), (200, 80, 20), -1)
    return canvas


def create_residual_vector_visualization(source_points, reference_points, transformation, image_shape, inlier_mask=None):
    """Render predicted-to-observed residual vectors in the reference frame."""
    source, reference = np.asarray(source_points, dtype=np.float64), np.asarray(reference_points, dtype=np.float64)
    matrix = _matrix(transformation); h, w = int(image_shape[0]), int(image_shape[1])
    projected = (matrix @ np.c_[source, np.ones(len(source))].T).T
    predicted = projected[:, :2] / projected[:, 2:3]
    mask = np.ones(len(source), dtype=bool) if inlier_mask is None else np.asarray(inlier_mask, dtype=bool).reshape(-1)
    if len(mask) != len(source): raise ValueError("inlier_mask length must equal point count")
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for start, end in zip(predicted[mask], reference[mask]):
        if np.isfinite(np.r_[start, end]).all():
            cv2.arrowedLine(canvas, tuple(np.rint(start).astype(int)), tuple(np.rint(end).astype(int)), (0, 200, 255), 1, tipLength=0.25)
    return canvas


def evaluate_registration(source_points, reference_points, transformation, inlier_mask, image_shape=None, thresholds=None):
    """Return image-based metrics; these are explicitly not geographic ground truth accuracy."""
    source, reference = np.asarray(source_points, dtype=np.float64), np.asarray(reference_points, dtype=np.float64)
    mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)
    if len(mask) != len(source): raise ValueError("inlier_mask length must equal candidate match count")
    errors = calculate_reprojection_errors(source, reference, transformation)
    inlier_errors = errors[mask]
    stats = calculate_error_statistics(inlier_errors)
    limits = EvaluationThresholds(**thresholds) if isinstance(thresholds, dict) else (thresholds or EvaluationThresholds())
    report: Dict[str, Any] = {"metric_scope": "image-based registration metrics; not absolute geographic accuracy", "candidate_matches": int(len(source)),
        "verified_inliers": int(mask.sum()), "outliers": int((~mask).sum()), "inlier_ratio": calculate_inlier_ratio(mask),
        "evaluated_correspondences": "provided arrays (these may be locally refined; no separate refinement mask was supplied)",
        "candidate_error_statistics": calculate_error_statistics(errors), "inlier_error_statistics": stats,
        "errors": errors, "inlier_errors": inlier_errors, "error_histogram": _histogram(inlier_errors),
        "thresholds": asdict(limits), "passed": bool(stats["rmse"] <= limits.max_rmse_px and calculate_inlier_ratio(mask) >= limits.min_inlier_ratio)}
    if image_shape is not None:
        report["spatial_coverage"] = calculate_spatial_coverage(source[mask], image_shape)
        report["residual_vector_visualization"] = create_residual_vector_visualization(source, reference, transformation, image_shape, mask)
    return report


def print_registration_report(report):
    stats = report["inlier_error_statistics"]
    print("## Registration Report\n")
    print(f"Candidate Matches : {report['candidate_matches']}")
    print(f"Verified Inliers  : {report['verified_inliers']}")
    print(f"Outlier Matches   : {report['outliers']}")
    print(f"Inlier Ratio      : {report['inlier_ratio'] * 100:.2f} %")
    print(f"RMSE              : {stats['rmse']:.4f} px")
    print(f"Median Error      : {stats['median']:.4f} px")
    print(f"Maximum Error     : {stats['max']:.4f} px")
    if "spatial_coverage" in report: print(f"Spatial Coverage  : {report['spatial_coverage']['coverage_percentage']:.2f} %")
    print(f"Status            : {'PASS' if report['passed'] else 'FAIL'}")
    print(f"Note              : {report['metric_scope']}")


def test_synthetic_evaluation():
    rng = np.random.default_rng(3); source = rng.uniform(0, 100, (30, 2)); transform = np.array([[1., 0., 2.], [0., 1., -1.], [0., 0., 1.]])
    reference = source + [2., -1.] + rng.normal(0, 0.2, source.shape); mask = np.ones(len(source), bool); errors = calculate_reprojection_errors(source, reference, transform)
    report = evaluate_registration(source, reference, transform, mask, (100, 100)); independent = float(np.sqrt(np.mean(errors ** 2)))
    assert np.isclose(report["inlier_error_statistics"]["rmse"], independent)
    return report


if __name__ == "__main__": print_registration_report(test_synthetic_evaluation())
