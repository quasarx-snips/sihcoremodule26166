"""LunaX Module 11: complete callable SIH26166 image-registration pipeline."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from module_1_2 import ImagePreprocessor
from module_3 import SiftDetector
from module_4 import match_feature_sets, visualize_matches
from module_5 import GeometricVerificationConfig, verify_matches, visualize_inliers_outliers
from module_6 import select_best_model
from module_7 import register_image
from module_8 import RefinementConfig, refine_correspondences
from module_9 import select_spatially_distributed_matches
from module_10 import evaluate_registration, print_registration_report


@dataclass
class PipelineConfig:
    verbose: bool = True
    save_outputs: bool = False
    output_dir: Optional[str] = None
    clahe_clip_limit: float = 2.5
    clahe_tile_grid_size: tuple = (8, 8)
    sift_features: int = 3000
    matching: Dict[str, Any] = field(default_factory=lambda: {"ratio": 0.75, "mutual_consistency": True})
    verification: Dict[str, Any] = field(default_factory=lambda: {"reprojection_threshold": 3.0, "min_inliers": 4, "random_seed": 42})
    spatial: Dict[str, Any] = field(default_factory=lambda: {"rows": 4, "cols": 4, "max_matches_per_cell": 10})
    refinement: Dict[str, Any] = field(default_factory=lambda: {"enabled": True, "max_refinement_count": 500})
    evaluation_thresholds: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegistrationResult:
    success: bool
    source_image: Optional[np.ndarray] = None
    reference_image: Optional[np.ndarray] = None
    processed_source: Optional[np.ndarray] = None
    processed_reference: Optional[np.ndarray] = None
    source_features: Any = None
    reference_features: Any = None
    candidate_matches: Any = None
    verified_matches: Any = None
    inlier_mask: Optional[np.ndarray] = None
    selected_model: Optional[str] = None
    transformation: Optional[np.ndarray] = None
    registered_image: Optional[np.ndarray] = None
    refined_correspondences: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


def _cfg(config: Optional[Any]) -> PipelineConfig:
    if config is None: return PipelineConfig()
    if isinstance(config, PipelineConfig): return config
    if isinstance(config, dict): return PipelineConfig(**config)
    raise TypeError("config must be None, PipelineConfig, or a dictionary")


def _json_safe(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if hasattr(value, "to_dict"): return _json_safe(value.to_dict())
    if hasattr(value, "__dataclass_fields__"): return _json_safe(asdict(value))
    if isinstance(value, dict): return {str(k): _json_safe(v) for k, v in value.items() if k not in {"error_histogram", "residual_vector_visualization"}}
    if isinstance(value, (list, tuple)): return [_json_safe(v) for v in value]
    return value


def _save(result: RegistrationResult, output_dir: str, match_vis, inlier_vis, registration_metadata) -> None:
    target = Path(output_dir); target.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target / "registered.png"), result.registered_image)
    cv2.imwrite(str(target / "candidate_matches.png"), match_vis)
    cv2.imwrite(str(target / "verified_inliers.png"), inlier_vis)
    cv2.imwrite(str(target / "overlay.png"), registration_metadata["overlay"])
    (target / "metrics.json").write_text(json.dumps(_json_safe({"metrics": result.metrics, "diagnostics": result.diagnostics}), indent=2))
    result.diagnostics["output_dir"] = str(target)


def run_lunax_from_arrays(source_image, reference_image, config=None):
    """Run Modules 1-10 on arrays. Failures are returned explicitly, never hidden."""
    cfg = _cfg(config); result = RegistrationResult(False); timings: Dict[str, float] = {}; started = time.perf_counter()
    try:
        source, reference = np.asarray(source_image), np.asarray(reference_image)
        if source.size == 0 or reference.size == 0: raise ValueError("source and reference images must be non-empty")
        result.source_image, result.reference_image = source, reference
        t = time.perf_counter(); pre = ImagePreprocessor(cfg.clahe_clip_limit, cfg.clahe_tile_grid_size)
        source_norm, reference_norm = pre.normalize(source), pre.normalize(reference)
        result.processed_source, result.processed_reference = pre.enhance(source_norm), pre.enhance(reference_norm); timings["preprocessing"] = time.perf_counter() - t
        t = time.perf_counter(); detector = SiftDetector(n_features=cfg.sift_features)
        source_features, source_desc = detector.detect(result.processed_source); reference_features, reference_desc = detector.detect(result.processed_reference)
        result.source_features, result.reference_features = source_features, reference_features; timings["feature_extraction"] = time.perf_counter() - t
        if not len(source_desc) or not len(reference_desc): raise RuntimeError("Feature extraction produced zero descriptors")
        t = time.perf_counter(); matches = match_feature_sets((source_features, source_desc), (reference_features, reference_desc), cfg.matching); result.candidate_matches = matches; timings["feature_matching"] = time.perf_counter() - t
        if not len(matches.source_points): raise RuntimeError("Feature matching produced zero accepted matches")
        t = time.perf_counter(); verification = verify_matches(matches.source_points, matches.reference_points, GeometricVerificationConfig(**cfg.verification)); timings["geometric_verification"] = time.perf_counter() - t
        result.verified_matches, result.inlier_mask = verification, verification.inlier_mask
        if not verification.is_valid or verification.transformation is None or not np.any(verification.inlier_mask): raise RuntimeError(f"Geometric verification failed: {verification.diagnostics.summary()}")
        verified_src, verified_ref = verification.source_inliers, verification.reference_inliers
        t = time.perf_counter(); model, transform, model_diag = select_best_model(verified_src, verified_ref); timings["model_selection"] = time.perf_counter() - t
        if transform is None: raise RuntimeError("No stable transformation model was selected")
        result.selected_model, result.transformation = model, transform
        t = time.perf_counter(); scores = -matches.distances[verification.inlier_mask]
        selected, spatial_diag = select_spatially_distributed_matches(verified_src, verified_ref, scores, source.shape, cfg.spatial); timings["spatial_selection"] = time.perf_counter() - t
        if not len(selected): raise RuntimeError("Spatial selection produced zero matches")
        spatial_src, spatial_ref = verified_src[selected], verified_ref[selected]
        t = time.perf_counter(); ref_cfg = {**cfg.refinement, "transformation": transform}; refined_src, refined_ref, refinement_diag = refine_correspondences(result.processed_source, result.processed_reference, spatial_src, spatial_ref, RefinementConfig(**ref_cfg)); timings["subpixel_refinement"] = time.perf_counter() - t
        result.refined_correspondences = {"source_points": refined_src, "reference_points": refined_ref, **refinement_diag}
        t = time.perf_counter(); registered, registration_metadata = register_image(source_norm, reference_norm, transform, model); timings["registration"] = time.perf_counter() - t; result.registered_image = registered
        t = time.perf_counter(); metrics = evaluate_registration(refined_src, refined_ref, transform, refinement_diag["valid_mask"], reference.shape, cfg.evaluation_thresholds); timings["evaluation"] = time.perf_counter() - t; result.metrics = metrics
        result.diagnostics = {"timings_seconds": timings, "total_seconds": time.perf_counter() - started, "verification": verification.diagnostics.to_dict(), "model_selection": model_diag.to_dict(), "spatial_selection": spatial_diag, "registration": {k: v for k, v in registration_metadata.items() if k not in {"overlay", "difference_map", "visualization"}}}
        match_vis = visualize_matches(source_norm, reference_norm, matches); inlier_vis = visualize_inliers_outliers(source_norm, reference_norm, matches.source_points, matches.reference_points, verification.inlier_mask)
        if cfg.save_outputs:
            if not cfg.output_dir: raise ValueError("output_dir is required when save_outputs=True")
            _save(result, cfg.output_dir, match_vis, inlier_vis, registration_metadata)
        result.success = True
        if cfg.verbose: print_lunax_pipeline_report(result)
    except Exception as exc:
        timings["total_seconds"] = time.perf_counter() - started; result.error = f"{type(exc).__name__}: {exc}"; result.diagnostics["timings_seconds"] = timings
        if cfg.verbose: print(f"LunaX pipeline FAILED: {result.error}")
    return result


def run_lunax_registration(source_path, reference_path, config=None):
    """Load two images from caller-provided paths and run the complete pipeline."""
    source, reference = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE), cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
    if source is None or reference is None:
        return RegistrationResult(False, error=f"Could not load source={source_path!s} or reference={reference_path!s}")
    return run_lunax_from_arrays(source, reference, config)


def print_lunax_pipeline_report(result):
    print("\n## LunaX Registration Pipeline")
    print("Status:", "SUCCESS" if result.success else "FAILED")
    if result.error: print("Error:", result.error); return
    print("Selected Model:", result.selected_model)
    print_registration_report(result.metrics)
    print("Stage timings (s):", ", ".join(f"{k}={v:.3f}" for k, v in result.diagnostics["timings_seconds"].items()))


def run_notebook_demo(source_path, reference_path, output_dir=None):
    """Notebook-friendly lunar-image demonstration with caller-supplied paths.

    Example cell: ``result = run_notebook_demo(source_path, reference_path)``.
    The report exposes candidate matches, rejected outliers, selected geometry,
    registered imagery, and image-based quantitative metrics.
    """
    config = {"verbose": True, "save_outputs": output_dir is not None, "output_dir": str(output_dir) if output_dir else None}
    return run_lunax_registration(source_path, reference_path, config)


def test_synthetic_end_to_end():
    """Notebook/terminal demo: transformed terrain-like image -> rejected outliers -> registration metrics."""
    rng = np.random.default_rng(12); source = cv2.GaussianBlur(rng.integers(0, 256, (360, 420), dtype=np.uint8), (0, 0), 1.0)
    for _ in range(25): cv2.circle(source, tuple(rng.integers(25, 335, 2)), int(rng.integers(4, 18)), int(rng.integers(80, 255)), 1)
    transform = np.array([[0.99, -0.05, 18.], [0.05, 0.99, -11.], [0., 0., 1.]])
    reference = cv2.warpAffine(source, transform[:2], (source.shape[1], source.shape[0]))
    result = run_lunax_from_arrays(source, reference, {"verbose": False, "verification": {"reprojection_threshold": 3.0, "min_inliers": 4, "random_seed": 42}})
    assert result.success, result.error
    assert result.verified_matches.diagnostics.outlier_count >= 0 and result.metrics["inlier_error_statistics"]["rmse"] < 3.0
    return result


if __name__ == "__main__":
    demo = test_synthetic_end_to_end(); print_lunax_pipeline_report(demo)
