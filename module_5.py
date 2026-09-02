"""
LunaX - Lunar Surface Exploration & Autonav Pipeline
Module 05: Robust Geometric Verification & Registration (Full Production Version)

Optimized for high-throughput execution on real lunar orbital/rover imagery 
via vectorized matrix estimation, zero-allocation inner-loop reprojection 
calculations, fast vector cross-product collinearity checks, and seamless 
integration with Module 4's MatchResult.
"""

from __future__ import annotations

import math
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

# Import Module 4's MatchResult for seamless pipeline integration
try:
    from module_4 import MatchResult
    MATCH_RESULT_AVAILABLE = True
except ImportError:
    MATCH_RESULT_AVAILABLE = False


# =====================================================================
# 1. Configuration & Data Structures
# =====================================================================

class TransformModel(str, Enum):
    """Supported geometric transformation models."""
    AUTO = "auto"              # Automatically select based on point count
    TRANSLATION = "translation"  # 2 DoF: (tx, ty)
    SIMILARITY = "similarity"    # 4 DoF: scale, rotation, translation
    AFFINE = "affine"           # 6 DoF: affine transform
    HOMOGRAPHY = "homography"   # 8 DoF: projective transform
    PROJECTIVE = "projective"   # Alias for homography


@dataclass
class GeometricVerificationConfig:
    """Configuration parameters for geometric verification."""
    model: TransformModel = TransformModel.AUTO
    reprojection_threshold: float = 3.0       # Max pixel distance for inliers
    confidence: float = 0.99                  # RANSAC desired confidence level
    max_iterations: int = 2000                # Maximum RANSAC iterations
    min_inliers: int = 4                      # Minimum required inliers for success
    refit_inliers: bool = True                # Refit model on all inliers via LSQ
    random_seed: Optional[int] = 42           # RNG seed for deterministic execution


@dataclass
class ErrorStatistics:
    """Reprojection error statistics for inlier matches."""
    mean: float = 0.0
    median: float = 0.0
    std: float = 0.0
    rmse: float = 0.0
    min_error: float = 0.0
    max_error: float = 0.0
    percentile_95: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class VerificationDiagnostics:
    """Detailed diagnostics resulting from geometric verification."""
    model_name: str
    total_matches: int
    inlier_count: int
    outlier_count: int
    inlier_ratio: float
    iterations_run: int
    converged: bool
    is_valid: bool
    error_stats: ErrorStatistics
    additional_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["error_stats"] = self.error_stats.to_dict()
        return d

    def summary(self) -> str:
        """Human-readable summary."""
        status = "VALID" if self.is_valid else "INVALID"
        return (f"[{status}] Model: {self.model_name} | "
                f"Inliers: {self.inlier_count}/{self.total_matches} "
                f"({self.inlier_ratio*100:.1f}%) | "
                f"RMSE: {self.error_stats.rmse:.3f}px | "
                f"Iterations: {self.iterations_run}")


@dataclass
class RegistrationResult:
    """
    Complete output from geometric verification and registration.
    Extends the original VerificationResult to include downstream pipeline needs.
    """
    is_valid: bool
    transformation: Optional[np.ndarray]      # 3x3 Transformation matrix
    inlier_mask: np.ndarray                   # 1D boolean array (True = Inlier)
    diagnostics: VerificationDiagnostics
    source_inliers: np.ndarray                # (K, 2) Inlier points in source
    reference_inliers: np.ndarray             # (K, 2) Inlier points in reference
    
    # Optional downstream outputs
    inlier_indices: Optional[np.ndarray] = None  
    registered_image: Optional[np.ndarray] = None
    source_points_warped: Optional[np.ndarray] = None
    
    def get_transformation_2x3(self) -> Optional[np.ndarray]:
        """Return transformation as 2x3 matrix for OpenCV warp functions."""
        if self.transformation is None:
            return None
        if self.transformation.shape == (2, 3):
            return self.transformation.copy()
        if self.transformation.shape == (3, 3):
            return self.transformation[:2, :].copy()
        raise ValueError(f"Unexpected transformation shape: {self.transformation.shape}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (for JSON export)."""
        return {
            "is_valid": self.is_valid,
            "transformation": self.transformation.tolist() if self.transformation is not None else None,
            "inlier_count": int(self.diagnostics.inlier_count),
            "inlier_ratio": float(self.diagnostics.inlier_ratio),
            "rmse": float(self.diagnostics.error_stats.rmse),
            "model": self.diagnostics.model_name,
        }


# =====================================================================
# 2. Vectorized Geometric Solvers (Direct Linear Estimation)
# =====================================================================

MODEL_MIN_SAMPLES = {
    TransformModel.TRANSLATION: 1,
    TransformModel.SIMILARITY: 2,
    TransformModel.AFFINE: 3,
    TransformModel.HOMOGRAPHY: 4,
    TransformModel.PROJECTIVE: 4
}


def _normalize_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Hartley isotropic normalization for numerical stability."""
    centroid = np.mean(points, axis=0)
    shifted = points - centroid
    mean_dist = np.mean(np.sqrt(np.sum(shifted ** 2, axis=1)))
    scale = np.sqrt(2.0) / (mean_dist + 1e-12)

    T = np.array([
        [scale, 0.0, -scale * centroid[0]],
        [0.0, scale, -scale * centroid[1]],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    pts_h = np.hstack([points, np.ones((len(points), 1), dtype=np.float64)])
    normalized_pts = (T @ pts_h.T).T[:, :2]
    return normalized_pts, T


def _fit_translation(src: np.ndarray, dst: np.ndarray) -> Optional[np.ndarray]:
    """Fits 2D Translation: dst = src + t."""
    if len(src) < 1:
        return None
    t = np.mean(dst - src, axis=0)
    T = np.eye(3, dtype=np.float64)
    T[0, 2] = t[0]
    T[1, 2] = t[1]
    return T


def _fit_similarity(src: np.ndarray, dst: np.ndarray) -> Optional[np.ndarray]:
    """
    Vectorized 2D Similarity Solver (Scale + Rotation + Translation).
    dst = s * R * src + t (4 DoF).
    """
    num_pts = len(src)
    if num_pts < 2:
        return None

    x, y = src[:, 0], src[:, 1]
    u, v = dst[:, 0], dst[:, 1]

    A = np.zeros((2 * num_pts, 4), dtype=np.float64)
    A[0::2, 0] = x
    A[0::2, 1] = -y
    A[0::2, 2] = 1.0

    A[1::2, 0] = y
    A[1::2, 1] = x
    A[1::2, 3] = 1.0

    b = np.empty(2 * num_pts, dtype=np.float64)
    b[0::2] = u
    b[1::2] = v

    try:
        sol, residuals, rank, _ = np.linalg.lstsq(A, b, rcond=None)
        if rank < 4:
            return None
        a_val, b_val, tx, ty = sol
        return np.array([
            [a_val, -b_val, tx],
            [b_val,  a_val, ty],
            [0.0,    0.0,   1.0]
        ], dtype=np.float64)
    except (np.linalg.LinAlgError, ValueError):
        return None


def _fit_affine(src: np.ndarray, dst: np.ndarray) -> Optional[np.ndarray]:
    """Vectorized 2D Affine Transformation Solver (6 DoF)."""
    num_pts = len(src)
    if num_pts < 3:
        return None

    src_norm, T_src = _normalize_points(src)
    dst_norm, T_dst = _normalize_points(dst)

    x, y = src_norm[:, 0], src_norm[:, 1]
    u, v = dst_norm[:, 0], dst_norm[:, 1]

    A = np.zeros((2 * num_pts, 6), dtype=np.float64)
    A[0::2, 0] = x
    A[0::2, 1] = y
    A[0::2, 2] = 1.0

    A[1::2, 3] = x
    A[1::2, 4] = y
    A[1::2, 5] = 1.0

    b = np.empty(2 * num_pts, dtype=np.float64)
    b[0::2] = u
    b[1::2] = v

    try:
        sol, residuals, rank, _ = np.linalg.lstsq(A, b, rcond=None)
        if rank < 6:
            return None
        M_norm = np.array([
            [sol[0], sol[1], sol[2]],
            [sol[3], sol[4], sol[5]],
            [0.0,    0.0,    1.0]
        ], dtype=np.float64)
        T = np.linalg.inv(T_dst) @ M_norm @ T_src
        return T / (T[2, 2] + 1e-12)
    except (np.linalg.LinAlgError, ValueError):
        return None


def _fit_homography(src: np.ndarray, dst: np.ndarray) -> Optional[np.ndarray]:
    """Vectorized Homography / Projective Transformation Solver (8 DoF) via normalized DLT."""
    num_pts = len(src)
    if num_pts < 4:
        return None

    src_norm, T_src = _normalize_points(src)
    dst_norm, T_dst = _normalize_points(dst)

    x, y = src_norm[:, 0], src_norm[:, 1]
    u, v = dst_norm[:, 0], dst_norm[:, 1]

    A = np.zeros((2 * num_pts, 9), dtype=np.float64)
    A[0::2, 0] = -x
    A[0::2, 1] = -y
    A[0::2, 2] = -1.0
    A[0::2, 6] = u * x
    A[0::2, 7] = u * y
    A[0::2, 8] = u

    A[1::2, 3] = -x
    A[1::2, 4] = -y
    A[1::2, 5] = -1.0
    A[1::2, 6] = v * x
    A[1::2, 7] = v * y
    A[1::2, 8] = v

    try:
        _, _, Vh = np.linalg.svd(A)
        H_norm = Vh[-1].reshape(3, 3)

        if np.abs(np.linalg.det(H_norm)) < 1e-12:
            return None

        H = np.linalg.inv(T_dst) @ H_norm @ T_src
        if np.abs(H[2, 2]) > 1e-12:
            H = H / H[2, 2]
        return H
    except (np.linalg.LinAlgError, ValueError):
        return None


def _are_points_collinear(points: np.ndarray, eps: float = 1e-6) -> bool:
    """Constant-time vector cross-product collinearity check for minimal sample sets."""
    n = len(points)
    if n < 3:
        return False
    
    if n == 3:
        p0, p1, p2 = points[0], points[1], points[2]
        area = abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]))
        return area < eps
    elif n == 4:
        v01 = points[1] - points[0]
        v02 = points[2] - points[0]
        v03 = points[3] - points[0]
        if abs(v01[0] * v02[1] - v01[1] * v02[0]) < eps: return True
        if abs(v01[0] * v03[1] - v01[1] * v03[0]) < eps: return True
        if abs(v02[0] * v03[1] - v02[1] * v03[0]) < eps: return True
        v12 = points[2] - points[1]
        v13 = points[3] - points[1]
        if abs(v12[0] * v13[1] - v12[1] * v13[0]) < eps: return True
        return False
    
    # Fallback for arbitrary N
    for i in range(n - 2):
        p1, p2, p3 = points[i], points[i + 1], points[i + 2]
        area = abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
        if area < eps:
            return True
    return False


# =====================================================================
# 3. Optimized Projection and Error Computation
# =====================================================================

def apply_transformation(points: np.ndarray, transformation: np.ndarray) -> np.ndarray:
    """
    Fast, zero-allocation projective transformation for (N, 2) points.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected points shape (N, 2), got {points.shape}")

    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float64)

    # Ensure transformation is 3x3
    if transformation.shape == (2, 3):
        transformation = np.vstack([transformation, [0.0, 0.0, 1.0]])

    x, y = points[:, 0], points[:, 1]
    u = transformation[0, 0] * x + transformation[0, 1] * y + transformation[0, 2]
    v = transformation[1, 0] * x + transformation[1, 1] * y + transformation[1, 2]
    w = transformation[2, 0] * x + transformation[2, 1] * y + transformation[2, 2]

    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    
    projected = np.empty_like(points)
    projected[:, 0] = u / w
    projected[:, 1] = v / w
    return projected


def compute_reprojection_errors(
    source_points: np.ndarray,
    reference_points: np.ndarray,
    transformation: np.ndarray
) -> np.ndarray:
    """Computes Euclidean reprojection distance in single-pass NumPy calls."""
    projected = apply_transformation(source_points, transformation)
    return np.hypot(reference_points[:, 0] - projected[:, 0], reference_points[:, 1] - projected[:, 1])


def compute_error_statistics(errors: np.ndarray) -> ErrorStatistics:
    """Calculates summary statistics on an array of errors."""
    if len(errors) == 0:
        return ErrorStatistics()

    return ErrorStatistics(
        mean=float(np.mean(errors)),
        median=float(np.median(errors)),
        std=float(np.std(errors)),
        rmse=float(np.sqrt(np.mean(errors ** 2))),
        min_error=float(np.min(errors)),
        max_error=float(np.max(errors)),
        percentile_95=float(np.percentile(errors, 95)) if len(errors) > 0 else 0.0
    )


# =====================================================================
# 4. Core Verification and Model Estimation API
# =====================================================================

def _fit_model_by_type(
    src: np.ndarray,
    dst: np.ndarray,
    model_type: TransformModel
) -> Optional[np.ndarray]:
    """Dispatches point sets to the appropriate transformation estimator."""
    if model_type == TransformModel.TRANSLATION:
        return _fit_translation(src, dst)
    elif model_type == TransformModel.SIMILARITY:
        return _fit_similarity(src, dst)
    elif model_type == TransformModel.AFFINE:
        return _fit_affine(src, dst)
    elif model_type in (TransformModel.HOMOGRAPHY, TransformModel.PROJECTIVE):
        return _fit_homography(src, dst)
    return None


def classify_matches(
    source_points: np.ndarray,
    reference_points: np.ndarray,
    transformation: Optional[np.ndarray],
    threshold: float
) -> np.ndarray:
    """Classifies correspondences using fast distance thresholding."""
    src = np.asarray(source_points, dtype=np.float64)
    dst = np.asarray(reference_points, dtype=np.float64)

    if len(src) != len(dst):
        raise ValueError(f"Point counts mismatch: {len(src)} vs {len(dst)}")

    if len(src) == 0 or transformation is None:
        return np.zeros(len(src), dtype=bool)

    errors = compute_reprojection_errors(src, dst, transformation)
    return (errors <= threshold) & np.isfinite(errors)


def _ransac_estimator(
    source_points: np.ndarray,
    reference_points: np.ndarray,
    model_type: TransformModel,
    config: GeometricVerificationConfig
) -> Tuple[Optional[np.ndarray], np.ndarray, int, bool]:
    """Internal deterministic RANSAC engine."""
    num_pts = len(source_points)
    sample_size = MODEL_MIN_SAMPLES[model_type]

    if num_pts < sample_size:
        return None, np.zeros(num_pts, dtype=bool), 0, False

    rng = np.random.RandomState(config.random_seed)

    best_inlier_mask = np.zeros(num_pts, dtype=bool)
    best_inlier_count = 0
    best_model = None
    dynamic_max_iters = config.max_iterations
    iterations_run = 0

    for it in range(config.max_iterations):
        iterations_run += 1
        if iterations_run > dynamic_max_iters:
            break

        sample_indices = rng.choice(num_pts, size=sample_size, replace=False)
        src_sample = source_points[sample_indices]
        dst_sample = reference_points[sample_indices]

        if sample_size >= 3 and _are_points_collinear(src_sample):
            continue

        model_candidate = _fit_model_by_type(src_sample, dst_sample, model_type)
        if model_candidate is None or not np.all(np.isfinite(model_candidate)):
            continue

        inlier_mask = classify_matches(
            source_points, reference_points, model_candidate, config.reprojection_threshold
        )
        inlier_count = int(np.sum(inlier_mask))

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inlier_mask = inlier_mask
            best_model = model_candidate

            w = inlier_count / float(num_pts)
            p = config.confidence
            w_sample = max(w ** sample_size, 1e-12)
            if 1.0 - w_sample > 0.0:
                calc_iters = math.log(1.0 - p) / math.log(1.0 - w_sample)
                dynamic_max_iters = min(config.max_iterations, int(math.ceil(calc_iters)))

    if config.refit_inliers and best_inlier_count >= sample_size:
        inlier_src = source_points[best_inlier_mask]
        inlier_dst = reference_points[best_inlier_mask]
        refit_model = _fit_model_by_type(inlier_src, inlier_dst, model_type)
        if refit_model is not None and np.all(np.isfinite(refit_model)):
            best_model = refit_model
            best_inlier_mask = classify_matches(
                source_points, reference_points, best_model, config.reprojection_threshold
            )
            best_inlier_count = int(np.sum(best_inlier_mask))

    is_converged = best_inlier_count >= config.min_inliers
    return best_model, best_inlier_mask, iterations_run, is_converged


def estimate_geometry(
    source_points: np.ndarray,
    reference_points: np.ndarray,
    model: Union[str, TransformModel] = "auto",
    config: Optional[GeometricVerificationConfig] = None
) -> Tuple[Optional[np.ndarray], np.ndarray, VerificationDiagnostics]:
    """Estimates transformation and correspondences between point sets."""
    cfg = config or GeometricVerificationConfig()
    model_str = model if isinstance(model, str) else model.value
    target_model = TransformModel(model_str.lower())

    src = np.asarray(source_points, dtype=np.float64)
    dst = np.asarray(reference_points, dtype=np.float64)
    total_pts = len(src)

    if total_pts != len(dst):
        raise ValueError(f"Mismatch in point sizes: {len(src)} vs {len(dst)}")

    if total_pts < 1:
        diag = VerificationDiagnostics(
            model_name=target_model.value,
            total_matches=0,
            inlier_count=0,
            outlier_count=0,
            inlier_ratio=0.0,
            iterations_run=0,
            converged=False,
            is_valid=False,
            error_stats=ErrorStatistics(),
            additional_info={"reason": "Empty input points"}
        )
        return None, np.zeros(0, dtype=bool), diag

    models_to_evaluate: List[TransformModel] = []
    if target_model == TransformModel.AUTO:
        if total_pts >= 4:
            models_to_evaluate = [TransformModel.AFFINE, TransformModel.HOMOGRAPHY]
        elif total_pts >= 3:
            models_to_evaluate = [TransformModel.AFFINE, TransformModel.SIMILARITY]
        elif total_pts >= 2:
            models_to_evaluate = [TransformModel.SIMILARITY]
        else:
            models_to_evaluate = [TransformModel.TRANSLATION]
    else:
        models_to_evaluate = [target_model]

    best_res = None
    best_inlier_cnt = -1

    for candidate_model in models_to_evaluate:
        T, mask, iters, converged = _ransac_estimator(src, dst, candidate_model, cfg)
        inlier_cnt = int(np.sum(mask))

        if inlier_cnt > best_inlier_cnt:
            best_inlier_cnt = inlier_cnt
            best_res = (T, mask, iters, converged, candidate_model)

    T, inlier_mask, iters, converged, chosen_model = best_res

    if T is not None and best_inlier_cnt > 0:
        inlier_errors = compute_reprojection_errors(src[inlier_mask], dst[inlier_mask], T)
        stats = compute_error_statistics(inlier_errors)
    else:
        stats = ErrorStatistics()

    is_valid = bool(converged and best_inlier_cnt >= cfg.min_inliers)
    inlier_count = int(best_inlier_cnt)
    outlier_count = total_pts - inlier_count
    ratio = float(inlier_count / total_pts) if total_pts > 0 else 0.0

    diagnostics = VerificationDiagnostics(
        model_name=chosen_model.value,
        total_matches=total_pts,
        inlier_count=inlier_count,
        outlier_count=outlier_count,
        inlier_ratio=ratio,
        iterations_run=iters,
        converged=converged,
        is_valid=is_valid,
        error_stats=stats,
        additional_info={"threshold": cfg.reprojection_threshold}
    )

    return (T if is_valid else None), inlier_mask, diagnostics


def verify_matches(
    source_points: np.ndarray,
    reference_points: np.ndarray,
    config: Optional[GeometricVerificationConfig] = None
) -> RegistrationResult:
    """High-level entry point for geometric verification."""
    cfg = config or GeometricVerificationConfig()
    T, mask, diagnostics = estimate_geometry(
        source_points, reference_points, model=cfg.model, config=cfg
    )

    src = np.asarray(source_points, dtype=np.float64)
    dst = np.asarray(reference_points, dtype=np.float64)

    return RegistrationResult(
        is_valid=diagnostics.is_valid,
        transformation=T,
        inlier_mask=mask,
        diagnostics=diagnostics,
        source_inliers=src[mask] if diagnostics.is_valid else np.empty((0, 2), dtype=np.float64),
        reference_inliers=dst[mask] if diagnostics.is_valid else np.empty((0, 2), dtype=np.float64),
        inlier_indices=np.where(mask)[0] if diagnostics.is_valid else None,
    )


def verify_from_match_result(
    match_result: MatchResult,
    config: Optional[GeometricVerificationConfig] = None
) -> RegistrationResult:
    """
    Convenience wrapper that accepts Module 4's MatchResult directly.
    """
    if not hasattr(match_result, 'source_points') or not hasattr(match_result, 'reference_points'):
        raise ValueError("match_result must have source_points and reference_points attributes")
    
    return verify_matches(
        match_result.source_points,
        match_result.reference_points,
        config=config
    )


# =====================================================================
# 5. Image Registration & Warping Functions
# =====================================================================

def warp_image(
    source_image: np.ndarray,
    transformation: np.ndarray,
    reference_shape: Optional[Tuple[int, int]] = None,
    interpolation: str = "linear",
    border_mode: str = "reflect"
) -> np.ndarray:
    """Warp source image using the estimated transformation."""
    if not HAS_OPENCV:
        raise ImportError("OpenCV is required for image warping")

    if transformation.shape == (3, 3):
        transform_2x3 = transformation[:2, :].astype(np.float64)
    elif transformation.shape == (2, 3):
        transform_2x3 = transformation.astype(np.float64)
    else:
        raise ValueError(f"Expected 2x3 or 3x3 transformation, got {transformation.shape}")

    interp_map = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC}
    interp_flag = interp_map.get(interpolation.lower(), cv2.INTER_LINEAR)

    border_map = {"constant": cv2.BORDER_CONSTANT, "reflect": cv2.BORDER_REFLECT101, "replicate": cv2.BORDER_REPLICATE}
    border_flag = border_map.get(border_mode.lower(), cv2.BORDER_REFLECT101)

    if reference_shape is None:
        height, width = source_image.shape[:2]
    else:
        height, width = reference_shape[:2]

    return cv2.warpAffine(source_image, transform_2x3, (width, height), 
                          flags=interp_flag, borderMode=border_flag, borderValue=0)


def register_images(
    source_image: np.ndarray,
    reference_image: np.ndarray,
    transformation: np.ndarray,
    blend: bool = False,
    alpha: float = 0.5
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Register source image to reference image coordinate system."""
    ref_shape = reference_image.shape[:2]
    registered = warp_image(source_image, transformation, reference_shape=ref_shape)
    
    if blend:
        if registered.ndim != reference_image.ndim:
            if registered.ndim == 2: registered = cv2.cvtColor(registered, cv2.COLOR_GRAY2BGR)
            if reference_image.ndim == 2: reference_image = cv2.cvtColor(reference_image, cv2.COLOR_GRAY2BGR)
        blended = cv2.addWeighted(registered, alpha, reference_image, 1.0 - alpha, 0)
        return registered, blended
    
    return registered, None


# =====================================================================
# 6. Diagnostic Visualization Engine (OpenCV Accelerated)
# =====================================================================

def visualize_inliers_outliers(
    source: np.ndarray,
    reference: np.ndarray,
    source_points: np.ndarray,
    reference_points: np.ndarray,
    inlier_mask: np.ndarray,
    max_display_matches: int = 250,
    line_thickness: int = 1,
    circle_radius: int = 3
) -> np.ndarray:
    """
    Visualizes feature matches side-by-side with color-coded inliers and outliers.
    """
    src_img = _to_rgb_uint8(source)
    dst_img = _to_rgb_uint8(reference)

    h1, w1 = src_img.shape[:2]
    h2, w2 = dst_img.shape[:2]

    canvas_h = max(h1, h2)
    canvas_w = w1 + w2
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    canvas[:h1, :w1] = src_img
    canvas[:h2, w1:canvas_w] = dst_img

    src_pts = np.asarray(source_points, dtype=np.float64)
    dst_pts = np.asarray(reference_points, dtype=np.float64)
    mask = np.asarray(inlier_mask, dtype=bool)

    total_pts = len(src_pts)
    if total_pts == 0:
        return canvas

    indices = np.arange(total_pts)
    outlier_indices = indices[~mask]
    inlier_indices = indices[mask]

    if len(outlier_indices) + len(inlier_indices) > max_display_matches:
        np.random.seed(42)
        if len(outlier_indices) > max_display_matches // 2:
            outlier_indices = np.random.choice(outlier_indices, max_display_matches // 2, replace=False)
        if len(inlier_indices) > max_display_matches // 2:
            inlier_indices = np.random.choice(inlier_indices, max_display_matches // 2, replace=False)

    outlier_color = (230, 45, 45)  # Red
    inlier_color = (45, 230, 85)   # Green

    if HAS_OPENCV:
        for idx in outlier_indices:
            p_src = (int(round(src_pts[idx, 0])), int(round(src_pts[idx, 1])))
            p_dst = (int(round(dst_pts[idx, 0] + w1)), int(round(dst_pts[idx, 1])))
            cv2.line(canvas, p_src, p_dst, outlier_color, line_thickness, cv2.LINE_AA)
            cv2.circle(canvas, p_src, circle_radius, outlier_color, -1)
            cv2.circle(canvas, p_dst, circle_radius, outlier_color, -1)

        for idx in inlier_indices:
            p_src = (int(round(src_pts[idx, 0])), int(round(src_pts[idx, 1])))
            p_dst = (int(round(dst_pts[idx, 0] + w1)), int(round(dst_pts[idx, 1])))
            cv2.line(canvas, p_src, p_dst, inlier_color, line_thickness, cv2.LINE_AA)
            cv2.circle(canvas, p_src, circle_radius, inlier_color, -1)
            cv2.circle(canvas, p_dst, circle_radius, inlier_color, -1)
    else:
        for idx in outlier_indices:
            p_src = (int(round(src_pts[idx, 0])), int(round(src_pts[idx, 1])))
            p_dst = (int(round(dst_pts[idx, 0] + w1)), int(round(dst_pts[idx, 1])))
            _draw_line_fallback(canvas, p_src, p_dst, outlier_color, line_thickness)
        for idx in inlier_indices:
            p_src = (int(round(src_pts[idx, 0])), int(round(src_pts[idx, 1])))
            p_dst = (int(round(dst_pts[idx, 0] + w1)), int(round(dst_pts[idx, 1])))
            _draw_line_fallback(canvas, p_src, p_dst, inlier_color, line_thickness)

    return canvas


def _draw_line_fallback(canvas: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int], color: Tuple[int, int, int], thickness: int):
    """Bresenham line fallback for pure NumPy mode."""
    x0, y0 = p1
    x1, y1 = p2
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        if 0 <= y0 < canvas.shape[0] and 0 <= x0 < canvas.shape[1]:
            canvas[max(0, y0-thickness):min(canvas.shape[0], y0+thickness+1),
                   max(0, x0-thickness):min(canvas.shape[1], x0+thickness+1)] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def _to_rgb_uint8(img: np.ndarray) -> np.ndarray:
    """Safely normalizes input 2D or 3D images to standard uint8 RGB."""
    arr = np.asarray(img)
    if arr.ndim == 2:
        if arr.dtype != np.uint8:
            norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-12) * 255.0
            arr = norm.astype(np.uint8)
        return np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3:
        if arr.shape[2] == 1:
            return _to_rgb_uint8(arr[:, :, 0])
        elif arr.shape[2] == 3:
            if arr.dtype != np.uint8:
                norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-12) * 255.0
                return norm.astype(np.uint8)
            return arr.copy()
    raise ValueError(f"Unsupported image shape for visualization: {arr.shape}")


# =====================================================================
# 7. Lunar Image Processing Pipeline & Synthetic Test Suite
# =====================================================================

def process_lunar_image(
    image_path: str,
    output_vis_path: str,
    synthetic_transform_for_eval: bool = True
) -> RegistrationResult:
    """
    Pipeline entry point to execute geometric verification directly on lunar imagery.
    """
    if not HAS_OPENCV:
        raise ImportError("OpenCV (cv2) is required to load and process lunar images directly.")

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load lunar image from path: {image_path}")

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img)

    if synthetic_transform_for_eval:
        h, w = img.shape
        angle = np.radians(4.5)
        scale = 1.02
        tx, ty = 25.0, -18.0
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        T_sim = np.array([
            [scale * cos_a, -scale * sin_a, tx],
            [scale * sin_a,  scale * cos_a, ty],
            [0.0,            0.0,           1.0]
        ], dtype=np.float64)

        ref_img = cv2.warpAffine(img_enhanced, T_sim[:2], (w, h))
    else:
        ref_img = img_enhanced.copy()

    sift = cv2.SIFT_create(nfeatures=2000)
    kp1, des1 = sift.detectAndCompute(img_enhanced, None)
    kp2, des2 = sift.detectAndCompute(ref_img, None)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    ref_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    config = GeometricVerificationConfig(
        model=TransformModel.AFFINE,
        reprojection_threshold=2.5,
        confidence=0.999,
        max_iterations=2000,
        random_seed=42
    )

    result = verify_matches(src_pts, ref_pts, config=config)

    vis_canvas = visualize_inliers_outliers(
        img_enhanced, ref_img, src_pts, ref_pts, result.inlier_mask
    )
    cv2.imwrite(output_vis_path, cv2.cvtColor(vis_canvas, cv2.COLOR_RGB2BGR))

    print("\n" + "=" * 65)
    print(f"LUNAR IMAGE PROCESSING COMPLETE: {image_path}")
    print("=" * 65)
    print(f" -> Keypoints Extracted (Source/Ref) : {len(kp1)} / {len(kp2)}")
    print(f" -> Initial Matches (Ratio Test)   : {len(good_matches)}")
    print(f" -> Robust Verified Inliers        : {result.diagnostics.inlier_count}")
    print(f" -> Inlier Ratio                   : {result.diagnostics.inlier_ratio * 100:.2f}%")
    print(f" -> Reprojection RMSE              : {result.diagnostics.error_stats.rmse:.4f} px")
    print(f" -> Output Saved To                : {output_vis_path}")
    print("=" * 65)

    return result


def generate_synthetic_correspondences(
    num_inliers: int = 100,
    num_outliers: int = 50,
    noise_sigma: float = 0.5,
    transform_type: TransformModel = TransformModel.AFFINE,
    image_shape: Tuple[int, int] = (1024, 1024),
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates synthetic 2D correspondences with ground truth transformation."""
    rng = np.random.RandomState(seed)
    h, w = image_shape

    if transform_type == TransformModel.TRANSLATION:
        tx, ty = rng.uniform(-50, 50, size=2)
        T_gt = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])
    elif transform_type == TransformModel.SIMILARITY:
        scale = rng.uniform(0.85, 1.15)
        angle = rng.uniform(-np.pi / 6, np.pi / 6)
        tx, ty = rng.uniform(-40, 40, size=2)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        T_gt = np.array([
            [scale * cos_a, -scale * sin_a, tx],
            [scale * sin_a,  scale * cos_a, ty],
            [0.0,            0.0,           1.0]
        ])
    elif transform_type == TransformModel.AFFINE:
        theta = rng.uniform(-0.3, 0.3)
        sx, sy = rng.uniform(0.9, 1.1), rng.uniform(0.9, 1.1)
        shear = rng.uniform(-0.1, 0.1)
        tx, ty = rng.uniform(-50, 50, size=2)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        S = np.array([[sx, shear], [0.0, sy]])
        A = R @ S
        T_gt = np.array([[A[0, 0], A[0, 1], tx], [A[1, 0], A[1, 1], ty], [0.0, 0.0, 1.0]])
    else:
        pts1 = np.array([[100, 100], [w - 100, 100], [w - 100, h - 100], [100, h - 100]], dtype=np.float64)
        perturb = rng.uniform(-40, 40, size=(4, 2))
        pts2 = pts1 + perturb
        T_gt = _fit_homography(pts1, pts2)

    src_inliers = rng.uniform([100, 100], [w - 100, h - 100], size=(num_inliers, 2))
    ref_inliers = apply_transformation(src_inliers, T_gt)
    ref_inliers += rng.normal(0, noise_sigma, size=ref_inliers.shape)

    src_outliers = rng.uniform([50, 50], [w - 50, h - 50], size=(num_outliers, 2))
    ref_outliers = rng.uniform([50, 50], [w - 50, h - 50], size=(num_outliers, 2))

    total_pts = num_inliers + num_outliers
    source_points = np.vstack([src_inliers, src_outliers])
    reference_points = np.vstack([ref_inliers, ref_outliers])
    gt_mask = np.array([True] * num_inliers + [False] * num_outliers, dtype=bool)

    shuffle_idx = rng.permutation(total_pts)
    return source_points[shuffle_idx], reference_points[shuffle_idx], gt_mask[shuffle_idx], T_gt


def run_unit_tests():
    """Executes unit tests for all transformation models."""
    print("=" * 65)
    print("Running LunaX Geometric Verification Test Suite")
    print("=" * 65)

    test_models = [
        TransformModel.TRANSLATION,
        TransformModel.SIMILARITY,
        TransformModel.AFFINE,
        TransformModel.HOMOGRAPHY
    ]

    for model in test_models:
        src, dst, gt_mask, T_gt = generate_synthetic_correspondences(
            num_inliers=120,
            num_outliers=60,
            noise_sigma=0.4,
            transform_type=model,
            seed=101
        )

        cfg = GeometricVerificationConfig(
            model=model,
            reprojection_threshold=2.5,
            confidence=0.999,
            max_iterations=1500,
            random_seed=42
        )

        res = verify_matches(src, dst, config=cfg)

        tp = np.sum(res.inlier_mask & gt_mask)
        fp = np.sum(res.inlier_mask & ~gt_mask)
        fn = np.sum(~res.inlier_mask & gt_mask)

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)

        print(f"\n[MODEL: {model.value.upper()}]")
        print(f" -> Verified Inliers:    {res.diagnostics.inlier_count} / Expected: {np.sum(gt_mask)}")
        print(f" -> Inlier Precision:    {precision * 100:.2f}%")
        print(f" -> Inlier Recall:       {recall * 100:.2f}%")
        print(f" -> Reprojection RMSE:   {res.diagnostics.error_stats.rmse:.4f} px")

        assert res.is_valid
        assert precision > 0.95
        assert recall > 0.95

    print("\n" + "=" * 65)
    print("ALL GEOMETRIC VERIFICATION TESTS PASSED (100% Vectorized & Deterministic)")
    print("=" * 65)


if __name__ == "__main__":
    run_unit_tests()
