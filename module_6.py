# --- START OF FILE module_6.py ---

"""
LunaX - Lunar Surface Exploration & Autonav Pipeline
Module 06: Transformation Model Estimation and Selection

Given verified feature correspondences, this module determines the most appropriate
geometric transformation model (Translation, Similarity, Affine, or Projective).
It prioritizes simpler models to prevent over-fitting, evaluates fit quality using
information criteria (like BIC), and prevents degenerate or numerically unstable models.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


# =====================================================================
# 1. Configuration & Data Structures
# =====================================================================

@dataclass
class ModelEvaluationConfig:
    """Configuration for model evaluation and selection."""
    min_points_translation: int = 1
    min_points_similarity: int = 2
    min_points_affine: int = 3
    min_points_projective: int = 4
    
    # Tolerances for numerical stability
    min_scale: float = 0.05
    max_scale: float = 20.0
    min_determinant: float = 1e-5
    
    # Noise floor for Information Criterion (prevents log(0))
    noise_floor_variance: float = 0.01 
    inlier_threshold: float = 3.0
    max_condition_number: float = 1e10
    selection_tolerance: float = 1e-6


@dataclass
class TransformationDiagnostics:
    """Diagnostics and evaluation metrics for a single transformation model."""
    model_name: str
    dof: int
    is_stable: bool
    rmse: float
    median_error: float
    max_error: float
    std_error: float
    percentile_95_error: float
    information_criterion: float  # e.g., BIC score (lower is better)
    stability_reason: str = "OK"
    inlier_consistency: float = 0.0
    residual_distribution: Dict[str, float] = field(default_factory=dict)
    selection_reason: str = ""
    candidate_models: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "dof": self.dof,
            "is_stable": self.is_stable,
            "rmse": round(self.rmse, 6),
            "median_error": round(self.median_error, 6),
            "max_error": round(self.max_error, 6),
            "std_error": round(self.std_error, 6),
            "percentile_95_error": round(self.percentile_95_error, 6),
            "information_criterion": round(self.information_criterion, 4),
            "stability_reason": self.stability_reason,
            "inlier_consistency": round(self.inlier_consistency, 6),
            "residual_distribution": {k: round(v, 6) for k, v in self.residual_distribution.items()},
            "selection_reason": self.selection_reason,
            "candidate_models": self.candidate_models,
        }


# =====================================================================
# 2. Numerical Utilities & Projections
# =====================================================================

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


def _apply_transformation(points: np.ndarray, transformation: np.ndarray) -> np.ndarray:
    """Projective transformation for (N, 2) points."""
    transformation = np.asarray(transformation, dtype=np.float64)
    pts_h = np.hstack([points, np.ones((len(points), 1), dtype=np.float64)])
    proj_h = (transformation @ pts_h.T).T
    
    # Avoid division by zero
    w = proj_h[:, 2]
    # Invalid projections must remain invalid; silently changing a zero
    # denominator into a tiny number can make a bad homography look usable.
    if np.any(np.abs(w) < 1e-12):
        return np.full((len(points), 2), np.nan, dtype=np.float64)
    
    return proj_h[:, :2] / w[:, np.newaxis]


def _check_stability(T: np.ndarray, model_name: str, config: ModelEvaluationConfig) -> Tuple[bool, str]:
    """Checks if a transformation matrix represents a physically plausible/stable warp."""
    if T is not None:
        T = np.asarray(T, dtype=np.float64)
    if T is None or not np.all(np.isfinite(T)):
        return False, "Matrix contains NaN or Inf"
    if np.asarray(T).shape != (3, 3):
        return False, "Expected a standardized 3x3 homogeneous matrix"
    
    # Ensure it's a normalized homogeneous matrix (bottom right is ~1.0)
    if abs(T[2, 2]) < 1e-12:
        return False, "Degenerate scale (T[2,2] is 0)"
    
    T_norm = T / T[2, 2]
    
    try:
        condition = np.linalg.cond(T_norm)
    except np.linalg.LinAlgError:
        return False, "Could not compute matrix condition number"
    if not np.isfinite(condition) or condition > config.max_condition_number:
        return False, f"Numerically ill-conditioned matrix ({condition:.2e})"

    # Check 2x2 determinant for scale and collapse.
    det_2x2 = np.linalg.det(T_norm[:2, :2])
    if abs(det_2x2) < config.min_determinant:
        return False, f"Determinant too small ({det_2x2:.2e})"
    
    # Calculate approximate scale
    approx_scale = np.sqrt(abs(det_2x2))
    if approx_scale < config.min_scale or approx_scale > config.max_scale:
        return False, f"Scale out of bounds ({approx_scale:.3f})"
    
    if model_name == "projective":
        # Check if the perspective components are excessively large (causes flipped horizons)
        perspective_mag = np.linalg.norm(T_norm[2, :2])
        if perspective_mag > 0.05:  # Empirical threshold for typical camera matching
            return False, f"Unstable perspective distortion ({perspective_mag:.4f})"
            
    return True, "OK"


def _as_point_array(points: np.ndarray, name: str) -> np.ndarray:
    """Validate correspondence coordinates without modifying their ordering."""
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def _has_sufficient_spread(points: np.ndarray, minimum_rank: int = 2) -> bool:
    """Reject repeated/collinear samples before fitting affine or projective maps."""
    if len(points) == 0:
        return False
    centered = points - np.mean(points, axis=0)
    scale = max(float(np.linalg.norm(centered, ord=np.inf)), 1.0)
    return np.linalg.matrix_rank(centered, tol=1e-10 * scale) >= minimum_rank


# =====================================================================
# 3. Model Estimation Functions
# =====================================================================

def estimate_translation(source_points: np.ndarray, reference_points: np.ndarray) -> Tuple[Optional[np.ndarray], TransformationDiagnostics]:
    """Estimates a 2 DoF Translation model."""
    source_points = _as_point_array(source_points, "source_points")
    reference_points = _as_point_array(reference_points, "reference_points")
    if len(source_points) != len(reference_points):
        raise ValueError("Source and reference point counts must match.")
    if len(source_points) < 1:
        return None, _empty_diagnostics("translation", 2)
    
    t = np.mean(reference_points - source_points, axis=0)
    T = np.eye(3, dtype=np.float64)
    T[0, 2] = t[0]
    T[1, 2] = t[1]
    
    diagnostics = evaluate_transformation(source_points, reference_points, T, "translation", 2)
    return T, diagnostics


def estimate_similarity(source_points: np.ndarray, reference_points: np.ndarray) -> Tuple[Optional[np.ndarray], TransformationDiagnostics]:
    """Estimates a 4 DoF Similarity model (Rotation, Scale, Translation)."""
    source_points = _as_point_array(source_points, "source_points")
    reference_points = _as_point_array(reference_points, "reference_points")
    if len(source_points) != len(reference_points):
        raise ValueError("Source and reference point counts must match.")
    if len(source_points) < 2:
        return None, _empty_diagnostics("similarity", 4)
    
    x, y = source_points[:, 0], source_points[:, 1]
    u, v = reference_points[:, 0], reference_points[:, 1]
    num_pts = len(source_points)

    A = np.zeros((2 * num_pts, 4), dtype=np.float64)
    A[0::2, 0] = x; A[0::2, 1] = -y; A[0::2, 2] = 1.0
    A[1::2, 0] = y; A[1::2, 1] = x;  A[1::2, 3] = 1.0

    b = np.empty(2 * num_pts, dtype=np.float64)
    b[0::2] = u; b[1::2] = v

    try:
        sol, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
        if rank < 4:
            raise np.linalg.LinAlgError("Rank deficient")
        a_val, b_val, tx, ty = sol
        T = np.array([
            [a_val, -b_val, tx],
            [b_val,  a_val, ty],
            [0.0,    0.0,   1.0]
        ], dtype=np.float64)
    except (np.linalg.LinAlgError, ValueError):
        return None, _empty_diagnostics("similarity", 4)
    
    diagnostics = evaluate_transformation(source_points, reference_points, T, "similarity", 4)
    return T, diagnostics


def estimate_affine(source_points: np.ndarray, reference_points: np.ndarray) -> Tuple[Optional[np.ndarray], TransformationDiagnostics]:
    """Estimates a 6 DoF Affine model."""
    source_points = _as_point_array(source_points, "source_points")
    reference_points = _as_point_array(reference_points, "reference_points")
    if len(source_points) != len(reference_points):
        raise ValueError("Source and reference point counts must match.")
    if len(source_points) < 3:
        return None, _empty_diagnostics("affine", 6)
    if not _has_sufficient_spread(source_points) or not _has_sufficient_spread(reference_points):
        diag = _empty_diagnostics("affine", 6)
        diag.stability_reason = "Degenerate (collinear or repeated) correspondences"
        return None, diag
    
    src_norm, T_src = _normalize_points(source_points)
    dst_norm, T_dst = _normalize_points(reference_points)
    
    x, y = src_norm[:, 0], src_norm[:, 1]
    u, v = dst_norm[:, 0], dst_norm[:, 1]
    num_pts = len(source_points)

    A = np.zeros((2 * num_pts, 6), dtype=np.float64)
    A[0::2, 0] = x; A[0::2, 1] = y; A[0::2, 2] = 1.0
    A[1::2, 3] = x; A[1::2, 4] = y; A[1::2, 5] = 1.0

    b = np.empty(2 * num_pts, dtype=np.float64)
    b[0::2] = u; b[1::2] = v

    try:
        sol, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
        if rank < 6:
            raise np.linalg.LinAlgError("Rank deficient")
        M_norm = np.array([
            [sol[0], sol[1], sol[2]],
            [sol[3], sol[4], sol[5]],
            [0.0,    0.0,    1.0]
        ], dtype=np.float64)
        T = np.linalg.inv(T_dst) @ M_norm @ T_src
        T = T / T[2, 2]
    except (np.linalg.LinAlgError, ValueError):
        return None, _empty_diagnostics("affine", 6)
    
    diagnostics = evaluate_transformation(source_points, reference_points, T, "affine", 6)
    return T, diagnostics


def estimate_projective(source_points: np.ndarray, reference_points: np.ndarray) -> Tuple[Optional[np.ndarray], TransformationDiagnostics]:
    """Estimates an 8 DoF Projective/Homography model via DLT."""
    source_points = _as_point_array(source_points, "source_points")
    reference_points = _as_point_array(reference_points, "reference_points")
    if len(source_points) != len(reference_points):
        raise ValueError("Source and reference point counts must match.")
    if len(source_points) < 4:
        return None, _empty_diagnostics("projective", 8)
    if not _has_sufficient_spread(source_points) or not _has_sufficient_spread(reference_points):
        diag = _empty_diagnostics("projective", 8)
        diag.stability_reason = "Degenerate (collinear or repeated) correspondences"
        return None, diag
    
    src_norm, T_src = _normalize_points(source_points)
    dst_norm, T_dst = _normalize_points(reference_points)
    
    x, y = src_norm[:, 0], src_norm[:, 1]
    u, v = dst_norm[:, 0], dst_norm[:, 1]
    num_pts = len(source_points)

    A = np.zeros((2 * num_pts, 9), dtype=np.float64)
    A[0::2, 0] = -x; A[0::2, 1] = -y; A[0::2, 2] = -1.0
    A[0::2, 6] = u * x; A[0::2, 7] = u * y; A[0::2, 8] = u
    A[1::2, 3] = -x; A[1::2, 4] = -y; A[1::2, 5] = -1.0
    A[1::2, 6] = v * x; A[1::2, 7] = v * y; A[1::2, 8] = v

    try:
        _, _, Vh = np.linalg.svd(A)
        H_norm = Vh[-1].reshape(3, 3)
        if np.abs(np.linalg.det(H_norm)) < 1e-12:
            raise np.linalg.LinAlgError("Degenerate homography")
        T = np.linalg.inv(T_dst) @ H_norm @ T_src
        T = T / T[2, 2]
    except (np.linalg.LinAlgError, ValueError):
        return None, _empty_diagnostics("projective", 8)
    
    diagnostics = evaluate_transformation(source_points, reference_points, T, "projective", 8)
    return T, diagnostics


# =====================================================================
# 4. Evaluation and Model Selection
# =====================================================================

def _empty_diagnostics(model_name: str, dof: int) -> TransformationDiagnostics:
    return TransformationDiagnostics(
        model_name=model_name, dof=dof, is_stable=False,
        rmse=float('inf'), median_error=float('inf'), max_error=float('inf'),
        std_error=float('inf'), percentile_95_error=float('inf'),
        information_criterion=float('inf'), stability_reason="Estimation failed"
    )

def evaluate_transformation(
    source_points: np.ndarray,
    reference_points: np.ndarray,
    transformation: np.ndarray,
    model_name: str = "unknown",
    dof: int = 0,
    config: Optional[ModelEvaluationConfig] = None
) -> TransformationDiagnostics:
    """Evaluates transformation quality and computes model selection criteria."""
    source_points = _as_point_array(source_points, "source_points")
    reference_points = _as_point_array(reference_points, "reference_points")
    if len(source_points) != len(reference_points):
        raise ValueError("Source and reference point counts must match.")
    if config is None:
        config = ModelEvaluationConfig()

    transformation = np.asarray(transformation, dtype=np.float64) if transformation is not None else None
    is_stable, reason = _check_stability(transformation, model_name, config)
    
    if not is_stable or len(source_points) == 0:
        diag = _empty_diagnostics(model_name, dof)
        diag.stability_reason = reason
        return diag

    projected = _apply_transformation(source_points, transformation)
    if not np.all(np.isfinite(projected)):
        diag = _empty_diagnostics(model_name, dof)
        diag.stability_reason = "Projection has an undefined homogeneous denominator"
        return diag
    errors = np.linalg.norm(reference_points - projected, axis=1)
    
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    n = len(source_points)
    
    # Calculate BIC (Bayesian Information Criterion)
    # Using Gaussian error assumption: BIC = n * ln(MSE + noise_floor) + k * ln(n)
    # We add a noise floor to prevent ln(0) when matching is perfectly exact (e.g. synthetic data)
    mse = (rmse ** 2) + config.noise_floor_variance
    bic = n * math.log(mse) + dof * math.log(n)
    
    return TransformationDiagnostics(
        model_name=model_name,
        dof=dof,
        is_stable=is_stable,
        rmse=rmse,
        median_error=float(np.median(errors)),
        max_error=float(np.max(errors)),
        std_error=float(np.std(errors)),
        percentile_95_error=float(np.percentile(errors, 95)),
        information_criterion=bic,
        stability_reason=reason,
        inlier_consistency=float(np.mean(errors <= config.inlier_threshold)),
        residual_distribution={
            "min": float(np.min(errors)), "q25": float(np.percentile(errors, 25)),
            "median": float(np.median(errors)), "q75": float(np.percentile(errors, 75)),
            "p95": float(np.percentile(errors, 95)), "max": float(np.max(errors)),
        },
    )


def select_best_model(
    source_points: np.ndarray,
    reference_points: np.ndarray,
    config: Optional[ModelEvaluationConfig] = None
) -> Tuple[str, Optional[np.ndarray], TransformationDiagnostics]:
    """
    Evaluates multiple candidate models on VERIFIED correspondences and selects 
    the best one based on penalized complexity (Information Criterion).
    """
    config = config or ModelEvaluationConfig()
    
    pts_src = _as_point_array(source_points, "source_points")
    pts_dst = _as_point_array(reference_points, "reference_points")
    
    if len(pts_src) != len(pts_dst):
        raise ValueError("Source and reference point counts must match.")
        
    estimators = [
        (estimate_translation, "translation"),
        (estimate_similarity, "similarity"),
        (estimate_affine, "affine"),
        (estimate_projective, "projective"),
    ]
    
    best_model_name = "none"
    best_transformation = None
    best_diagnostics = _empty_diagnostics(best_model_name, 0)
    best_ic = float('inf')
    
    report_table = []
    candidate_models: Dict[str, Dict[str, Any]] = {}
    
    for estimator_func, name in estimators:
        T, diag = estimator_func(pts_src, pts_dst)
        # Public estimators use their default evaluation config; selection must
        # honour caller-supplied thresholds and numerical tolerances.
        if T is not None:
            diag = evaluate_transformation(pts_src, pts_dst, T, name, diag.dof, config)
        
        status = "STABLE" if diag.is_stable else f"UNSTABLE ({diag.stability_reason})"
        report_table.append(
            f"{name.ljust(12)} | DoF: {diag.dof} | IC: {diag.information_criterion:10.2f} | "
            f"RMSE: {diag.rmse:8.4f} | MedErr: {diag.median_error:8.4f} | {status}"
        )
        candidate_models[name] = diag.to_dict()
        
        if diag.is_stable and (diag.information_criterion < best_ic - config.selection_tolerance):
            best_ic = diag.information_criterion
            best_model_name = name
            best_transformation = T
            best_diagnostics = diag
            
    if best_transformation is not None:
        best_diagnostics.selection_reason = (
            f"Selected: lowest BIC ({best_ic:.3f}) among stable candidates; "
            "BIC penalizes additional degrees of freedom."
        )
    else:
        best_diagnostics.selection_reason = "No stable non-degenerate candidate model was available."
    best_diagnostics.candidate_models = candidate_models

    # Notebook-friendly comparison table; callers also receive all metrics in diagnostics.
    print("\n--- Model Selection Comparison Table ---")
    for line in report_table:
        print(line)
    print(f"-> SELECTED MODEL: {best_model_name.upper()}")
    print("----------------------------------------\n")
            
    return best_model_name, best_transformation, best_diagnostics


# =====================================================================
# 5. Testing & Execution
# =====================================================================

def test_synthetic_selection():
    """Unit test validating that model selection penalizes complexity appropriately."""
    print("Running Synthetic Model Selection Tests...")
    np.random.seed(42)
    n_pts = 50
    src = np.random.uniform(0, 1000, (n_pts, 2))
    
    def test_case(true_model_name: str, T_gt: np.ndarray, noise_std: float = 0.5):
        dst = _apply_transformation(src, T_gt)
        dst += np.random.normal(0, noise_std, dst.shape)
        
        selected_name, T_est, diag = select_best_model(src, dst)
        
        print(f"Target: {true_model_name.upper()} | Selected: {selected_name.upper()}")
        assert selected_name == true_model_name, f"Expected {true_model_name}, got {selected_name}"

    # 1. Translation
    T_trans = np.array([[1, 0, 50], [0, 1, -30], [0, 0, 1]], dtype=np.float64)
    test_case("translation", T_trans)
    
    # 2. Similarity
    theta = np.radians(15)
    s = 1.2
    T_sim = np.array([
        [s*np.cos(theta), -s*np.sin(theta), 20],
        [s*np.sin(theta),  s*np.cos(theta), 40],
        [0, 0, 1]
    ], dtype=np.float64)
    test_case("similarity", T_sim)
    
    # 3. Affine
    T_aff = np.array([
        [1.1, 0.2, 10],
        [-0.1, 0.9, -15],
        [0, 0, 1]
    ], dtype=np.float64)
    test_case("affine", T_aff)
    
    # 4. Projective
    T_proj = np.array([
        [1.05, 0.1, 15],
        [-0.05, 0.95, -10],
        [0.0001, 0.0002, 1]
    ], dtype=np.float64)
    test_case("projective", T_proj)
    print("All synthetic tests passed!\n")


def run_real_images(path1: str, path2: str):
    """Executes evaluation on the real image pair provided by the user."""
    print(f"Processing real images:\n1: {path1}\n2: {path2}\n")
    img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        print("ERROR: Could not read one or both images. Check file paths.")
        return
    
    # Extract features and match to simulate 'verified correspondences' input
    sift = cv2.SIFT_create(nfeatures=1500)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
            
    if len(good) < 10:
        print("Not enough matches found.")
        return
        
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)
    
    # Perform strict geometric verification to isolate INLIERS (simulate Module 05)
    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.USAC_MAGSAC, 3.0)
    mask = mask.ravel().astype(bool)
    
    inliers_src = src_pts[mask]
    inliers_dst = dst_pts[mask]
    
    print(f"Extracted {len(inliers_src)} verified inliers. Handing over to Model Selection (Module 06).")
    
    selected_model, T, diag = select_best_model(inliers_src, inliers_dst)
    
    print("Diagnostics Summary:")
    print(json.dumps(diag.to_dict(), indent=2))
    print("\nFinal Selected Transformation Matrix:")
    print(T)


if __name__ == "__main__":
    test_synthetic_selection()

# --- END OF FILE module_6.py ---
