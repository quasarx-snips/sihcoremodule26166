"""LunaX Module 4 -- descriptor matching.

This module deliberately stops at descriptor correspondences.  Estimating a
homography, pose, or any other geometric model belongs to the next stage.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


DEFAULT_CONFIG: Dict[str, Any] = {
    "method": "BF",              # "BF" or "FLANN"
    "ratio": 0.75,
    "knn_k": 2,
    "mutual_consistency": False,
    "match_same_feature_type": True,
    "unique_train_matches": True,
    "allow_singleton_feature_types": ("crater",),
    "norm": None,                # inferred: L2 for float/SIFT, Hamming for uint8
    "flann_trees": 5,
    "flann_checks": 50,
}


@dataclass
class MatchResult:
    """Descriptor correspondences, with indices retained in every record."""
    candidate_matches: List[cv2.DMatch] = field(default_factory=list)
    accepted_matches: List[cv2.DMatch] = field(default_factory=list)
    source_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2), np.float32))
    reference_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2), np.float32))
    distances: np.ndarray = field(default_factory=lambda: np.empty((0,), np.float32))
    number_raw_matches: int = 0
    number_filtered_matches: int = 0
    match_records: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def raw_matches(self) -> int:
        return self.number_raw_matches

    @property
    def filtered_matches(self) -> int:
        return self.number_filtered_matches

    @property
    def matches(self) -> List[cv2.DMatch]:
        """The descriptor matches that passed configured filters."""
        return self.accepted_matches


def _as_descriptors(descriptors: Any) -> np.ndarray:
    """Normalize descriptors while retaining OpenCV-supported dtype."""
    if descriptors is None:
        return np.empty((0, 0), dtype=np.float32)
    array = np.asarray(descriptors)
    if array.size == 0:
        width = array.shape[1] if array.ndim == 2 else 0
        return np.empty((0, width), dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("Descriptors must be a two-dimensional array.")
    if array.dtype not in (np.float32, np.uint8):
        array = array.astype(np.float32)
    return np.ascontiguousarray(array)


def _norm_for(descriptors: np.ndarray, configured: Any = None) -> int:
    if configured is not None:
        if isinstance(configured, str):
            name = configured.upper()
            if name in ("L2", "NORM_L2"): return cv2.NORM_L2
            if name in ("HAMMING", "NORM_HAMMING"): return cv2.NORM_HAMMING
            raise ValueError("norm must be L2 or Hamming")
        return int(configured)
    return cv2.NORM_HAMMING if descriptors.dtype == np.uint8 else cv2.NORM_L2


def match_descriptors(source_descriptors: Any, reference_descriptors: Any,
                      method: str = "BF", config: Optional[Mapping[str, Any]] = None) -> List[List[cv2.DMatch]]:
    """Return KNN candidate matches. Empty inputs produce an empty list."""
    options = {**DEFAULT_CONFIG, **(config or {})}
    source, reference = _as_descriptors(source_descriptors), _as_descriptors(reference_descriptors)
    if not len(source) or not len(reference):
        return []
    if source.shape[1] != reference.shape[1]:
        raise ValueError("Source and reference descriptor dimensions must match.")

    backend = method.upper()
    k = max(1, min(int(options["knn_k"]), len(reference)))
    if backend == "BF":
        matcher = cv2.BFMatcher(_norm_for(source, options.get("norm")), crossCheck=False)
    elif backend == "FLANN":
        # FLANN's KD-tree works on float descriptors; LSH supports binary ones.
        if source.dtype == np.uint8:
            index_params = dict(algorithm=6, table_number=12, key_size=20, multi_probe_level=2)
        else:
            source, reference = source.astype(np.float32), reference.astype(np.float32)
            index_params = dict(algorithm=1, trees=int(options["flann_trees"]))
        matcher = cv2.FlannBasedMatcher(index_params, dict(checks=int(options["flann_checks"])))
    else:
        raise ValueError("method must be 'BF' or 'FLANN'.")
    return matcher.knnMatch(source, reference, k=k)


def ratio_test(matches: Iterable[Sequence[cv2.DMatch]], ratio: float = 0.75) -> List[cv2.DMatch]:
    """Apply Lowe's nearest-neighbour distance ratio test."""
    if not 0 < ratio <= 1:
        raise ValueError("ratio must be in (0, 1].")
    good: List[cv2.DMatch] = []
    for neighbours in matches:
        if not neighbours:
            continue
        if len(neighbours) == 1:
            # No ambiguity estimate exists, so do not silently treat it as good.
            continue
        best, runner_up = neighbours[0], neighbours[1]
        if best.distance < ratio * runner_up.distance:
            good.append(best)
    return good


def mutual_consistency_filter(source_descriptors: Any, reference_descriptors: Any,
                              matches: Iterable[Any]) -> List[cv2.DMatch]:
    """Keep matches whose nearest neighbour is reciprocal in descriptor space."""
    source, reference = _as_descriptors(source_descriptors), _as_descriptors(reference_descriptors)
    flat = _flatten_matches(matches)
    if not flat or not len(source) or not len(reference):
        return []
    reverse = match_descriptors(reference, source, method="BF", config={"knn_k": 1})
    reverse_pairs = {(group[0].queryIdx, group[0].trainIdx) for group in reverse if group}
    return [m for m in flat if (m.trainIdx, m.queryIdx) in reverse_pairs]


def _flatten_matches(matches: Iterable[Any]) -> List[cv2.DMatch]:
    flattened: List[cv2.DMatch] = []
    for item in matches:
        if isinstance(item, cv2.DMatch):
            flattened.append(item)
        elif item:  # permit direct use of KNN output
            flattened.append(item[0])
    return flattened


def _point(keypoint: Any) -> Tuple[float, float]:
    if hasattr(keypoint, "pt"):
        return float(keypoint.pt[0]), float(keypoint.pt[1])
    if hasattr(keypoint, "x") and hasattr(keypoint, "y"):
        return float(keypoint.x), float(keypoint.y)
    if isinstance(keypoint, Mapping):
        return float(keypoint["x"]), float(keypoint["y"])
    return float(keypoint[0]), float(keypoint[1])


def matches_to_correspondences(source_keypoints: Sequence[Any], reference_keypoints: Sequence[Any],
                               matches: Iterable[Any]) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Convert matches into point arrays and index-preserving plain records."""
    valid: List[cv2.DMatch] = []
    for m in _flatten_matches(matches):
        if 0 <= m.queryIdx < len(source_keypoints) and 0 <= m.trainIdx < len(reference_keypoints):
            valid.append(m)
    source_points = np.asarray([_point(source_keypoints[m.queryIdx]) for m in valid], dtype=np.float32).reshape(-1, 2)
    reference_points = np.asarray([_point(reference_keypoints[m.trainIdx]) for m in valid], dtype=np.float32).reshape(-1, 2)
    records = [{"source_index": int(m.queryIdx), "reference_index": int(m.trainIdx), "distance": float(m.distance)} for m in valid]
    return source_points, reference_points, records


def _feature_parts(feature_set: Any) -> Tuple[Sequence[Any], Any]:
    """Read Module 03-style data or a conventional FeatureSet object/dict."""
    if isinstance(feature_set, tuple) and len(feature_set) == 2:
        keypoints, descriptors = feature_set
    elif isinstance(feature_set, Mapping):
        descriptors = feature_set.get("descriptors")
        keypoints = feature_set.get("keypoints", feature_set.get("features", []))
    else:
        descriptors = getattr(feature_set, "descriptors", None)
        keypoints = getattr(feature_set, "keypoints", getattr(feature_set, "features", []))
    if descriptors is None:
        raise ValueError("FeatureSet must provide a 'descriptors' array.")
    # Module 03 returns all terrain features, while only SIFT records map to descriptors.
    indexed = [p for p in keypoints if getattr(p, "descriptor_index", None) is not None]
    if indexed:
        indexed.sort(key=lambda p: p.descriptor_index)
        keypoints = indexed
    return keypoints, descriptors


def _feature_type(keypoint: Any) -> str:
    """Return a matching class, with untyped keypoints sharing one class."""
    if isinstance(keypoint, Mapping):
        return str(keypoint.get("feature_type", "__untyped__"))
    return str(getattr(keypoint, "feature_type", "__untyped__"))


def _typed_candidate_matches(source_descriptors: Any, reference_descriptors: Any,
                             source_keypoints: Sequence[Any], reference_keypoints: Sequence[Any],
                             options: Mapping[str, Any]) -> List[List[cv2.DMatch]]:
    """Run KNN matching per feature class when terrain labels are available."""
    source = _as_descriptors(source_descriptors)
    reference = _as_descriptors(reference_descriptors)
    if not options["match_same_feature_type"]:
        return match_descriptors(source, reference, options["method"], options)
    source_groups: Dict[str, List[int]] = {}
    reference_groups: Dict[str, List[int]] = {}
    for index, keypoint in enumerate(source_keypoints):
        source_groups.setdefault(_feature_type(keypoint), []).append(index)
    for index, keypoint in enumerate(reference_keypoints):
        reference_groups.setdefault(_feature_type(keypoint), []).append(index)
    candidates: List[List[cv2.DMatch]] = []
    for feature_class, source_indices in source_groups.items():
        reference_indices = reference_groups.get(feature_class, [])
        if not reference_indices:
            continue
        local = match_descriptors(source[source_indices], reference[reference_indices], options["method"], options)
        for neighbours in local:
            candidates.append([
                cv2.DMatch(source_indices[match.queryIdx], reference_indices[match.trainIdx], 0, match.distance)
                for match in neighbours
            ])
    return candidates


def _unique_train_matches(matches: Iterable[cv2.DMatch]) -> List[cv2.DMatch]:
    """Keep the lowest-distance match for each source and reference index."""
    selected: List[cv2.DMatch] = []
    used_source, used_reference = set(), set()
    for match in sorted(matches, key=lambda item: item.distance):
        if match.queryIdx not in used_source and match.trainIdx not in used_reference:
            selected.append(match)
            used_source.add(match.queryIdx)
            used_reference.add(match.trainIdx)
    return selected


def match_feature_sets(source_features: Any, reference_features: Any,
                       config: Optional[Mapping[str, Any]] = None) -> MatchResult:
    """Match two FeatureSets and report descriptor-only correspondences."""
    options = {**DEFAULT_CONFIG, **(config or {})}
    source_keypoints, source_descriptors = _feature_parts(source_features)
    reference_keypoints, reference_descriptors = _feature_parts(reference_features)
    raw = _typed_candidate_matches(source_descriptors, reference_descriptors,
                                  source_keypoints, reference_keypoints, options)
    filtered = ratio_test(raw, float(options["ratio"])) if int(options["knn_k"]) >= 2 else _flatten_matches(raw)
    # A rare feature class (notably a large crater) can have exactly one
    # reference candidate. Lowe's test cannot score that case; retain it only
    # for explicitly configured classes and let one-to-one selection decide.
    singleton_types = set(options.get("allow_singleton_feature_types", ()))
    if singleton_types and int(options["knn_k"]) >= 2:
        filtered.extend(
            neighbours[0] for neighbours in raw
            if len(neighbours) == 1 and _feature_type(source_keypoints[neighbours[0].queryIdx]) in singleton_types
        )
    if options["mutual_consistency"]:
        filtered = mutual_consistency_filter(source_descriptors, reference_descriptors, filtered)
    if options["unique_train_matches"]:
        filtered = _unique_train_matches(filtered)
    source_points, reference_points, records = matches_to_correspondences(source_keypoints, reference_keypoints, filtered)
    distances = np.asarray([r["distance"] for r in records], dtype=np.float32)
    result = MatchResult(
        candidate_matches=_flatten_matches(raw),
        accepted_matches=filtered,
        source_points=source_points,
        reference_points=reference_points,
        distances=distances,
        number_raw_matches=len(raw),
        number_filtered_matches=len(filtered),
        match_records=records,
    )
    retained = 100.0 * result.number_filtered_matches / result.number_raw_matches if result.number_raw_matches else 0.0
    print(f"raw matches: {result.number_raw_matches}")
    print(f"filtered matches: {result.number_filtered_matches}")
    print(f"percentage retained: {retained:.1f}%")
    return result


def _display_image(image: Any) -> np.ndarray:
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_COLOR)
    array = np.asarray(image)
    if array.ndim == 2:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    return array.copy()


def visualize_matches(source: Any, reference: Any, match_result: MatchResult,
                      max_matches: int = 200) -> np.ndarray:
    """Return a match image; it neither opens a GUI window nor writes a file."""
    src, ref = _display_image(source), _display_image(reference)
    height, width = max(src.shape[0], ref.shape[0]), src.shape[1] + ref.shape[1]
    canvas = np.zeros((height, width, 3), dtype=src.dtype)
    canvas[:src.shape[0], :src.shape[1]] = src
    canvas[:ref.shape[0], src.shape[1]:] = ref
    for i, (sp, rp) in enumerate(zip(match_result.source_points[:max_matches], match_result.reference_points[:max_matches])):
        color = tuple(int(v) for v in np.random.default_rng(i).integers(80, 256, 3))
        a, b = tuple(np.round(sp).astype(int)), tuple(np.round(rp + [src.shape[1], 0]).astype(int))
        cv2.line(canvas, a, b, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, a, 3, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, b, 3, color, 1, cv2.LINE_AA)
    return canvas


def visualize_ground_truth_matches(
    source: Any,
    reference: Any,
    match_result: MatchResult,
    source_to_reference: np.ndarray,
    tolerance_px: float = 4.0,
    max_matches: int = 300,
    gap_px: int = 24,
    title: str = "Ground-truth correspondence check",
) -> np.ndarray:
    """Render correct (green) and wrong (red) matches using a *known* transform.

    This is a diagnostic visualizer for synthetic data only.  It does not fit
    or verify a geometric model and therefore does not affect Module 04 match
    filtering. ``source_to_reference`` may be a 2x3 affine or 3x3 homography.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)  # Render PNGs without an OpenCV/desktop GUI.
    import matplotlib.pyplot as plt  # Optional dependency, used only for this report.
    from matplotlib.lines import Line2D

    if tolerance_px <= 0:
        raise ValueError("tolerance_px must be positive.")
    if gap_px < 0:
        raise ValueError("gap_px must be non-negative.")
    src, ref = _display_image(source), _display_image(reference)
    matrix = np.asarray(source_to_reference, dtype=np.float64)
    if matrix.shape == (2, 3):
        matrix = np.vstack((matrix, [0.0, 0.0, 1.0]))
    if matrix.shape != (3, 3):
        raise ValueError("source_to_reference must be a 2x3 affine or 3x3 homography.")

    points = match_result.source_points[:max_matches].astype(np.float64)
    homogeneous = np.column_stack((points, np.ones(len(points))))
    expected = homogeneous @ matrix.T
    expected = expected[:, :2] / expected[:, 2:3] if len(expected) else np.empty((0, 2))
    errors = np.linalg.norm(expected - match_result.reference_points[:max_matches], axis=1)
    correct = errors <= tolerance_px

    # Matplotlib provides titles and a compact legend matching the requested report.
    figure, axis = plt.subplots(figsize=(18, 9), dpi=100)
    height = max(src.shape[0], ref.shape[0])
    joined = np.full((height, src.shape[1] + gap_px + ref.shape[1], 3), 255, dtype=np.uint8)
    joined[:src.shape[0], :src.shape[1]] = src
    reference_x = src.shape[1] + gap_px
    joined[:ref.shape[0], reference_x:reference_x + ref.shape[1]] = ref
    axis.imshow(joined, cmap=None)
    for src_pt, ref_pt, is_correct in zip(points, match_result.reference_points[:max_matches], correct):
        color = "#00e539" if is_correct else "#f51b27"
        axis.plot((src_pt[0], ref_pt[0] + reference_x), (src_pt[1], ref_pt[1]), color=color, linewidth=0.65, alpha=0.78)
        axis.scatter((src_pt[0], ref_pt[0] + reference_x), (src_pt[1], ref_pt[1]), c=color, s=10, linewidths=0)
    correct_count, wrong_count = int(correct.sum()), int((~correct).sum())
    axis.set_title(f"{title} (tol {tolerance_px:g}px)")
    axis.axis("off")
    axis.legend(handles=[
        Line2D([0], [0], color="#00e539", label=f"correct ({correct_count})"),
        Line2D([0], [0], color="#f51b27", label=f"wrong ({wrong_count})"),
    ], loc="lower center", ncol=2, framealpha=0.9, fontsize=10)
    figure.tight_layout()
    figure.canvas.draw()
    rgba = np.asarray(figure.canvas.buffer_rgba())
    rendered = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR).copy()
    plt.close(figure)
    return rendered


def run_sample_14_synthetic_demo(
    image_path: Union[str, Path],
    output_path: Union[str, Path],
    tolerance_px: float = 4.0,
    max_matches: int = 300,
    config: Optional[Mapping[str, Any]] = None,
    random_seed: Optional[int] = 41,
) -> Tuple[MatchResult, np.ndarray]:
    """Rotate an image, extract Module 03 features for both, then match them.

    ``random_seed`` keeps the randomly selected rotation reproducible. Set it
    to ``None`` to choose a fresh angle on each run.
    """
    source = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if source is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    height, width = source.shape[:2]
    angle_degrees = float(np.random.default_rng(random_seed).uniform(-12.0, 12.0))
    transform = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle_degrees, 1.0)
    reference = cv2.warpAffine(source, transform, (width, height), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT101)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rotated_path = destination.with_name(f"{Path(image_path).stem}_rotated_{angle_degrees:+.2f}deg.png")
    if not cv2.imwrite(str(rotated_path), reference):
        raise OSError(f"Could not write rotated test image: {rotated_path}")

    # Use Module 03 end-to-end for each image. Its default cap is 3,000.
    from .features import TerrainFeatureExtractor
    extractor = TerrainFeatureExtractor(max_total_features=3000)
    source_enhanced, source_feature_records, source_descriptors = extractor.extract(image_path)
    reference_enhanced, reference_feature_records, reference_descriptors = extractor.extract(rotated_path)
    demo_config = {"method": "BF", "ratio": 0.75, **(config or {})}
    result = match_feature_sets(
        (source_feature_records, source_descriptors),
        (reference_feature_records, reference_descriptors),
        demo_config,
    )
    rendered = visualize_ground_truth_matches(
        source_enhanced, reference_enhanced, result, transform, tolerance_px=tolerance_px,
        max_matches=max_matches,
        title=f"{Path(image_path).stem}_synthetic: ground-truth check",
    )
    if not cv2.imwrite(str(destination), rendered):
        raise OSError(f"Could not write visualization: {destination}")
    print(f"random rotation: {angle_degrees:+.2f} degrees")
    print(f"rotated test image: {rotated_path}")
    print(f"saved visualization: {destination}")
    return result, rendered


def _apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    table = np.clip((np.arange(256, dtype=np.float32) / 255.0) ** gamma * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(image, table)


def _ground_truth_metrics(result: MatchResult, transform: np.ndarray, tolerance_px: float) -> Dict[str, Any]:
    """Score matches against a transform known only to a synthetic benchmark."""
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape == (2, 3):
        matrix = np.vstack((matrix, [0.0, 0.0, 1.0]))
    points = result.source_points.astype(np.float64)
    projected = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    projected = projected[:, :2] / projected[:, 2:3] if len(projected) else np.empty((0, 2))
    correct = np.linalg.norm(projected - result.reference_points, axis=1) <= tolerance_px
    count = int(correct.sum())
    total = len(correct)
    return {
        "correct_matches": count,
        "incorrect_matches": int(total - count),
        "precision_percent": round(100.0 * count / total, 2) if total else 0.0,
    }


def _save_robustness_dashboard(panels: Mapping[str, Path], report: Mapping[str, Mapping[str, Any]],
                               output_path: Path) -> None:
    """Create a single judge-facing board containing every stress condition."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(20, 10), dpi=120)
    for axis, (name, panel_path) in zip(axes.flat, panels.items()):
        panel = cv2.imread(str(panel_path), cv2.IMREAD_COLOR)
        if panel is None:
            raise FileNotFoundError(f"Could not read generated panel: {panel_path}")
        axis.imshow(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
        metrics = report[name]
        axis.set_title(f"{name.replace('_', ' ').title()} — "
                       f"{metrics['correct_matches']} correct / {metrics['incorrect_matches']} wrong "
                       f"({metrics['precision_percent']:.1f}% precision)", fontsize=11)
        axis.axis("off")
    figure.suptitle("LunaX descriptor matching robustness benchmark", fontsize=18, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def run_robustness_benchmark(
    image_path: Union[str, Path],
    output_dir: Union[str, Path],
    tolerance_px: float = 4.0,
    max_visualized_matches: int = 300,
    descriptor_backend: str = "SUPERPOINT",
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Benchmark Module 03 -> Module 04 under common harsh image conditions.

    Each variant has a known affine transform, used strictly for evaluation;
    no geometric verification is passed back into the descriptor matcher.
    """
    source = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if source is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    height, width = source.shape[:2]
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    rotation_zoom = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), 14.0, 0.82).astype(np.float32)
    rotated = cv2.warpAffine(source, rotation_zoom, (width, height), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REFLECT101)
    lighting = _apply_gamma(cv2.convertScaleAbs(source, alpha=0.58, beta=18), 1.35)
    combined = cv2.GaussianBlur(_apply_gamma(cv2.convertScaleAbs(rotated, alpha=0.58, beta=18), 1.35), (9, 9), 2.2)
    variants = {
        "lighting": (lighting, identity),
        "rotation_zoom": (rotated, rotation_zoom),
        "blur": (cv2.GaussianBlur(source, (9, 9), 2.2), identity),
        "combined_harsh": (combined, rotation_zoom),
    }
    from .features import TerrainFeatureExtractor
    extractor = TerrainFeatureExtractor(max_total_features=3000, descriptor_backend=descriptor_backend)
    _, source_features, source_descriptors = extractor.extract(image_path)
    match_config = {"method": "BF", "ratio": 0.78, "mutual_consistency": True, **(config or {})}
    report: Dict[str, Dict[str, Any]] = {"descriptor_backend": descriptor_backend.upper()}
    panels: Dict[str, Path] = {}
    for name, (variant, transform) in variants.items():
        variant_path = output / f"{Path(image_path).stem}_{name}.png"
        cv2.imwrite(str(variant_path), variant)
        _, reference_features, reference_descriptors = extractor.extract(variant_path)
        result = match_feature_sets((source_features, source_descriptors),
                                    (reference_features, reference_descriptors), match_config)
        metrics = _ground_truth_metrics(result, transform, tolerance_px)
        report[name] = {
            "raw_matches": result.number_raw_matches,
            "filtered_matches": result.number_filtered_matches,
            **metrics,
        }
        panel_path = output / f"{name}_correspondences.png"
        panel = visualize_ground_truth_matches(
            source, variant, result, transform, tolerance_px=tolerance_px,
            max_matches=max_visualized_matches,
            title=f"{name}: ground-truth correspondence check",
        )
        if not cv2.imwrite(str(panel_path), panel):
            raise OSError(f"Could not write correspondence panel: {panel_path}")
        panels[name] = panel_path
        report[name]["correspondence_visualization"] = str(panel_path)
        print(f"[Robustness] {name}: {metrics['correct_matches']}/{result.number_filtered_matches} correct "
              f"({metrics['precision_percent']:.2f}% precision)")
    dashboard_path = output / "all_conditions_comparison.png"
    _save_robustness_dashboard(panels, report, dashboard_path)
    report["comparison_visualization"] = str(dashboard_path)
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"saved robustness report: {report_path}")
    return report


if __name__ == "__main__":
    print("Import this module and call a demo with explicit image and output paths.")
