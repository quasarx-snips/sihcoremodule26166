"""LunaX end-to-end lunar correspondence, verification, and registration API."""
from __future__ import annotations
import json, time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import cv2
import numpy as np
from .preprocessing import ImagePreprocessor
from .features import FeatureStore, SiftDetector, TerrainFeatureExtractor
from .matching import match_feature_sets, visualize_matches
from .geometry import GeometricVerificationConfig, verify_matches, visualize_inliers_outliers
from .registration import register_image
from .refinement import RefinementConfig, refine_correspondences
from .metrics import calculate_spatial_coverage, evaluate_registration

@dataclass
class PipelineConfig:
    verbose: bool = True; save_outputs: bool = False; output_dir: Optional[str] = None
    clahe_clip_limit: float = 2.5; clahe_tile_grid_size: tuple = (8, 8)
    sift_features: int = 3000; sift_contrast_threshold: float = 0.01; sift_edge_threshold: float = 15.0; root_sift: bool = False
    use_terrain_landmarks: bool = True; crater_model_path: Optional[str] = None
    matching: Dict[str, Any] = field(default_factory=lambda: {"ratio": .75, "mutual_consistency": True})
    verification: Dict[str, Any] = field(default_factory=lambda: {"model":"auto", "reprojection_threshold":3., "confidence":.999, "max_iterations":2000, "min_inliers":4, "random_seed":42})
    refinement: Dict[str, Any] = field(default_factory=lambda: {"enabled": True, "max_refinement_count": 500})
    evaluation_thresholds: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RegistrationResult:
    success: bool; source_image: Optional[np.ndarray] = None; reference_image: Optional[np.ndarray] = None
    processed_source: Optional[np.ndarray] = None; processed_reference: Optional[np.ndarray] = None
    source_features: Any = None; reference_features: Any = None; candidate_matches: Any = None; verified_matches: Any = None
    inlier_mask: Optional[np.ndarray] = None; selected_model: Optional[str] = None; transformation: Optional[np.ndarray] = None
    registered_image: Optional[np.ndarray] = None; refined_correspondences: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict); diagnostics: Dict[str, Any] = field(default_factory=dict); error: Optional[str] = None

def _cfg(config: Optional[Any]) -> PipelineConfig:
    if config is None: return PipelineConfig()
    if isinstance(config, PipelineConfig): return config
    if isinstance(config, dict): return PipelineConfig(**config)
    raise TypeError("config must be None, PipelineConfig, or a dictionary")

def _json_safe(value: Any) -> Any:
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if hasattr(value, "to_dict"): return _json_safe(value.to_dict())
    if hasattr(value, "__dataclass_fields__"): return _json_safe(asdict(value))
    if isinstance(value, dict): return {str(k): _json_safe(v) for k,v in value.items() if k not in {"error_histogram","residual_vector_visualization"}}
    if isinstance(value, (list, tuple)): return [_json_safe(v) for v in value]
    return value

def _write(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image): raise IOError(f"Could not write output image: {path}")

def _save(result: RegistrationResult, root_name: str, candidate: np.ndarray, verified: np.ndarray, registration: Dict[str, Any], source_desc: np.ndarray, reference_desc: np.ndarray) -> None:
    root = Path(root_name)
    # One flat run directory keeps every artifact visible while debugging.
    files = {"enhanced_source":root/"source_enhanced.png", "enhanced_reference":root/"reference_enhanced.png", "candidate_matches":root/"candidate_matches.png", "verified_matches":root/"inliers_outliers.png", "registered":root/"registered_source.png", "overlay":root/"overlay.png", "difference":root/"difference.png", "registration_panel":root/"registration_panel.png"}
    images = {"enhanced_source":result.processed_source, "enhanced_reference":result.processed_reference, "candidate_matches":candidate, "verified_matches":verified, "registered":result.registered_image, "overlay":registration["overlay"], "difference":registration["difference_map"], "registration_panel":registration["visualization"]}
    for key, image in images.items(): _write(files[key], image)
    root.mkdir(parents=True, exist_ok=True)
    FeatureStore.save_features(root / "source_features.json", result.source_features, result.processed_source.shape)
    FeatureStore.save_features(root / "reference_features.json", result.reference_features, result.processed_reference.shape)
    FeatureStore.save_descriptors(root / "source_descriptors.npy", source_desc); FeatureStore.save_descriptors(root / "reference_descriptors.npy", reference_desc)
    result.diagnostics["output_dir"] = str(root); result.diagnostics["artifact_paths"] = {key:str(path) for key,path in files.items()}
    report = root / "metrics.json"
    result.diagnostics["artifact_paths"]["report"] = str(report)
    report.write_text(json.dumps(_json_safe({"metrics":result.metrics,"diagnostics":result.diagnostics}), indent=2), encoding="utf-8")

def _confidence(metrics: Dict[str, Any]) -> float:
    """Evidence score; it is deliberately not presented as a calibrated probability."""
    ratio, count = float(metrics["inlier_ratio"]), min(float(metrics["verified_inliers"])/50., 1.)
    coverage = float(metrics["spatial_coverage"]["coverage_percentage"])/100.
    rmse = float(metrics["inlier_error_statistics"]["rmse"]); error = np.exp(-rmse/3.) if np.isfinite(rmse) else 0.
    return float(100*np.clip(.40*ratio + .25*count + .20*coverage + .15*error, 0, 1))

def _extract(enhanced: np.ndarray, cfg: PipelineConfig, label: str):
    sift = SiftDetector(cfg.sift_features, cfg.sift_contrast_threshold, cfg.sift_edge_threshold, cfg.root_sift)
    if not cfg.use_terrain_landmarks: return sift.detect(enhanced)
    extractor = TerrainFeatureExtractor(sift_detector=sift, max_total_features=cfg.sift_features, onnx_crater_model_path=cfg.crater_model_path)
    _, features, descriptors = extractor.extract_array(enhanced, label)
    return features, descriptors

def run_lunax_from_arrays(source_image: Any, reference_image: Any, config: Optional[Any] = None) -> RegistrationResult:
    """Run preprocessing, multi-feature extraction, matching, RANSAC and registration on CPU."""
    cfg, result, timings, started = _cfg(config), RegistrationResult(False), {}, time.perf_counter()
    try:
        pre = ImagePreprocessor(cfg.clahe_clip_limit, cfg.clahe_tile_grid_size)
        t=time.perf_counter(); source_norm, reference_norm = pre.normalize(source_image), pre.normalize(reference_image)
        result.source_image, result.reference_image = source_norm, reference_norm; result.processed_source, result.processed_reference = pre.enhance(source_norm), pre.enhance(reference_norm); timings["preprocessing"]=time.perf_counter()-t
        t=time.perf_counter(); result.source_features, source_desc = _extract(result.processed_source, cfg, "source"); result.reference_features, reference_desc = _extract(result.processed_reference, cfg, "reference"); timings["feature_extraction"]=time.perf_counter()-t
        if not len(source_desc) or not len(reference_desc): raise RuntimeError("Insufficient features: no descriptors were generated")
        t=time.perf_counter(); matches=match_feature_sets((result.source_features,source_desc),(result.reference_features,reference_desc),cfg.matching); result.candidate_matches=matches; timings["feature_matching"]=time.perf_counter()-t
        if len(matches.source_points)<2: raise RuntimeError("Insufficient correspondences for geometric verification")
        t=time.perf_counter(); verification=verify_matches(matches.source_points,matches.reference_points,GeometricVerificationConfig(**cfg.verification)); timings["geometric_verification"]=time.perf_counter()-t
        result.verified_matches, result.inlier_mask = verification, verification.inlier_mask
        if not verification.is_valid or verification.transformation is None: raise RuntimeError("No geometrically consistent transformation found")
        # Use the RANSAC-refit model consistently; do not replace it after verification.
        result.selected_model, result.transformation = verification.diagnostics.model_name, verification.transformation
        in_src, in_ref = matches.source_points[verification.inlier_mask], matches.reference_points[verification.inlier_mask]
        t=time.perf_counter(); refined_src, refined_ref, refine_diag=refine_correspondences(result.processed_source,result.processed_reference,in_src,in_ref,RefinementConfig(**{**cfg.refinement,"transformation":result.transformation})); timings["local_refinement"]=time.perf_counter()-t
        result.refined_correspondences={"source_points":refined_src,"reference_points":refined_ref,**refine_diag}
        t=time.perf_counter(); result.registered_image, registration=register_image(source_norm,reference_norm,result.transformation,result.selected_model); timings["registration"]=time.perf_counter()-t
        t=time.perf_counter(); result.metrics=evaluate_registration(matches.source_points,matches.reference_points,result.transformation,verification.inlier_mask,reference_norm.shape,cfg.evaluation_thresholds); timings["evaluation"]=time.perf_counter()-t
        result.metrics["spatial_coverage"] = calculate_spatial_coverage(in_src, source_norm.shape); result.metrics["transformation_model"] = result.selected_model; result.metrics["transformation_matrix"] = result.transformation.tolist(); result.metrics["correspondence_confidence"] = _confidence(result.metrics); result.metrics["status"] = "VALID CORRESPONDENCE" if result.metrics["passed"] else "NO RELIABLE CORRESPONDENCE"
        result.diagnostics={"execution":"CPU","timings_seconds":timings,"total_seconds":time.perf_counter()-started,"verification":verification.diagnostics.to_dict(),"feature_counts":{"source":len(result.source_features),"reference":len(result.reference_features)},"registration":{k:v for k,v in registration.items() if k not in {"overlay","difference_map","visualization"}}}
        candidate, verified = visualize_matches(source_norm,reference_norm,matches), visualize_inliers_outliers(source_norm,reference_norm,matches.source_points,matches.reference_points,verification.inlier_mask)
        if cfg.save_outputs:
            if not cfg.output_dir: raise ValueError("output_dir is required when save_outputs=True")
            _save(result,cfg.output_dir,candidate,verified,registration,source_desc,reference_desc)
        result.success=True
        if cfg.verbose: print_lunax_pipeline_report(result)
    except Exception as exc:
        result.error=f"{type(exc).__name__}: {exc}"; timings["total_seconds"]=time.perf_counter()-started; result.diagnostics.setdefault("timings_seconds",timings)
        if cfg.verbose: print(f"LunaX pipeline FAILED: {result.error}")
    return result

def run_lunax_registration(source_path: Any, reference_path: Any, config: Optional[Any] = None) -> RegistrationResult:
    source, reference=cv2.imread(str(source_path),cv2.IMREAD_UNCHANGED),cv2.imread(str(reference_path),cv2.IMREAD_UNCHANGED)
    if source is None or reference is None: return RegistrationResult(False,error=f"Could not load source={source_path!s} or reference={reference_path!s}")
    return run_lunax_from_arrays(source,reference,config)

def print_lunax_pipeline_report(result: RegistrationResult) -> None:
    print("\n================ LUNAX RESULT ================")
    if not result.success: print("Status:",result.error or "FAILED"); print("================================================"); return
    m,s=result.metrics,result.metrics["inlier_error_statistics"]
    print(f"Image A features       : {len(result.source_features)}\nImage B features       : {len(result.reference_features)}\n\nCandidate matches      : {m['candidate_matches']}\nVerified inliers       : {m['verified_inliers']}\nRejected outliers      : {m['outliers']}\n\nInlier ratio           : {m['inlier_ratio']*100:.1f} %\nTransformation         : {result.selected_model.upper()}\n\nMean reprojection      : {s['mean']:.2f} px\nMedian reprojection    : {s['median']:.2f} px\nRMSE                   : {s['rmse']:.2f} px\n95th percentile error  : {s['p95']:.2f} px\n\nSpatial coverage       : {m['spatial_coverage']['coverage_percentage']:.1f} %\nCorrespondence         : {m['status']}\nConfidence             : {m['correspondence_confidence']:.1f} %\nExecution              : {result.diagnostics['execution']}")
    if "artifact_paths" in result.diagnostics: print("Registration output    :",result.diagnostics["artifact_paths"]["registered"])
    print("================================================")

def run_notebook_demo(source_path: Any, reference_path: Any, output_dir: Optional[Any] = None) -> RegistrationResult:
    return run_lunax_registration(source_path,reference_path,{"verbose":True,"save_outputs":output_dir is not None,"output_dir":str(output_dir) if output_dir else None})
