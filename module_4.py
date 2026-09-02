"""
LunaX Module 4 — Feature / Descriptor Matching
==============================================

Turns two Module-3 terrain feature sets into a list of *candidate*
correspondences.  This module deliberately stops before geometric
verification: RANSAC / transform fitting lives in Module 5.

PIPELINE FLOW:
    1. Attach descriptors to every feature that has none (craters, ridges,
       texture points get a SIFT descriptor computed at their location/scale)
    2. Group features by type (crater<->crater, ridge<->ridge, ...) — optional
    3. KNN descriptor matching (BFMatcher baseline, FLANN optional)
    4. Lowe ratio test
    5. Mutual (cross-check) consistency A->B and B->A
    6. Optional scale-ratio consistency (robust, relative to the median)
    7. One-to-one enforcement and scoring
    8. FeatureMatch records with ids, coordinates, scores + diagnostics
    9. Visualization and JSON export

USAGE:
    from module_3 import TerrainFeatureExtractor
    from module_4 import (TerrainFeatureSet, attach_descriptors,
                          match_terrain_features, visualize_matches)

    extractor = TerrainFeatureExtractor()
    img_a, feats_a, desc_a = extractor.extract("a.png")
    img_b, feats_b, desc_b = extractor.extract("b.png")
    set_a = attach_descriptors(TerrainFeatureSet(feats_a, desc_a, img_a.shape), img_a)
    set_b = attach_descriptors(TerrainFeatureSet(feats_b, desc_b, img_b.shape), img_b)
    result = match_terrain_features(set_a, set_b)
    fig = visualize_matches(img_a, img_b, result)

CLI:
    python module_4.py demo <image_a> [image_b] [output_dir]
    python module_4.py match <feat_a.json> <desc_a.npy> <feat_b.json> <desc_b.npy> [output_dir]
    python module_4.py test

Dependencies:
    - opencv-python (cv2)
    - numpy
    - matplotlib (visualization only)

Author: LunaX Team
Version: 4.0
Date: 2026-09-02
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from module_3 import TerrainFeature, ImagePreprocessor, TerrainFeatureExtractor, FeatureStore

ImageArray = np.ndarray
FEATURE_TYPES: Tuple[str, ...] = ("crater", "ridge", "texture", "sift")
SEMANTIC_TYPES: Tuple[str, ...] = ("crater", "ridge", "texture")


# ============================================================================
# 0. CONFIGURATION
# ============================================================================

def create_matching_config(**overrides: Any) -> Dict[str, Any]:
    """Default Module-4 configuration. Override any key via kwargs.

    Keys:
        matcher:              "BF" | "FLANN"
        knn:                  neighbours requested per query (>=2 for ratio test)
        ratio:                Lowe ratio threshold (0 < ratio < 1)
        type_ratio:           per-feature-type ratio overrides, e.g. {"crater": 0.85}
                              (semantic features are self-similar, so a looser
                              ratio keeps candidates for Module 5 to verify)
        mutual:               require A->B and B->A agreement
        unique_target:        keep only the best match per target feature
        type_constraint:      only match features of the same type
        match_types:          feature types to match (subset of FEATURE_TYPES)
        scale_constraint:     enforce robust scale-ratio consistency
        scale_log_tolerance:  allowed |log(scale_b/scale_a) - median| (0.69 ~ 2x)
        min_score:            drop candidates with score below this
        max_matches:          cap on returned candidates (best-first), None = all
        semantic_desc_min_size / semantic_desc_max_size:
                              clamp for the SIFT patch size used on semantic features
        semantic_orientation: "gradient" assigns a dominant gradient orientation to
                              features without one (rotation invariance); "none" keeps 0
    """
    cfg: Dict[str, Any] = {
        "matcher": "BF",
        "knn": 2,
        "ratio": 0.75,
        "type_ratio": {"crater": 0.85, "texture": 0.85, "ridge": 0.8},
        "mutual": True,
        "unique_target": True,
        "type_constraint": True,
        "match_types": list(FEATURE_TYPES),
        "scale_constraint": True,
        "scale_log_tolerance": 0.69,
        "min_score": 0.0,
        "max_matches": None,
        "semantic_desc_min_size": 8.0,
        "semantic_desc_max_size": 96.0,
        "semantic_orientation": "gradient",
    }
    unknown = set(overrides) - set(cfg)
    if unknown:
        raise KeyError(f"Unknown matching config keys: {sorted(unknown)}")
    cfg.update(overrides)
    if not (0.0 < cfg["ratio"] < 1.0):
        raise ValueError("ratio must be in (0, 1)")
    if any(not (0.0 < r < 1.0) for r in cfg["type_ratio"].values()):
        raise ValueError("type_ratio values must be in (0, 1)")
    if cfg["semantic_orientation"] not in ("gradient", "none"):
        raise ValueError("semantic_orientation must be 'gradient' or 'none'")
    if cfg["knn"] < 1:
        raise ValueError("knn must be >= 1")
    if cfg["matcher"].upper() not in ("BF", "FLANN"):
        raise ValueError("matcher must be 'BF' or 'FLANN'")
    return cfg


# ============================================================================
# 1. DATA STRUCTURES
# ============================================================================

@dataclass
class TerrainFeatureSet:
    """Module-3 output for one image, bundled for matching.

    Attributes:
        features:    list of TerrainFeature (any type)
        descriptors: (N_desc, D) array; feature.descriptor_index indexes rows
        image_shape: (H, W) of the image the features were extracted from
    """

    features: List[TerrainFeature]
    descriptors: np.ndarray
    image_shape: Tuple[int, int]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.descriptors = _as_descriptor_array(self.descriptors)
        self.image_shape = tuple(int(v) for v in self.image_shape[:2])  # type: ignore[assignment]
        for f in self.features:
            if f.descriptor_index is not None and not (0 <= f.descriptor_index < len(self.descriptors)):
                raise ValueError(
                    f"descriptor_index {f.descriptor_index} out of range for "
                    f"{len(self.descriptors)} descriptors"
                )

    def __len__(self) -> int:
        return len(self.features)

    def count_by_type(self) -> Dict[str, int]:
        counts = {t: 0 for t in FEATURE_TYPES}
        for f in self.features:
            counts[f.feature_type] = counts.get(f.feature_type, 0) + 1
        return counts

    def indices_of_type(self, feature_type: str) -> List[int]:
        return [i for i, f in enumerate(self.features) if f.feature_type == feature_type]

    def has_descriptor(self, idx: int) -> bool:
        return self.features[idx].descriptor_index is not None

    def descriptor_for(self, idx: int) -> Optional[np.ndarray]:
        di = self.features[idx].descriptor_index
        return None if di is None else self.descriptors[di]

    def points(self) -> np.ndarray:
        return np.array([[f.x, f.y] for f in self.features], dtype=np.float32).reshape(-1, 2)

    @classmethod
    def from_module3_outputs(
        cls, features_json: Union[str, Path], descriptors_npy: Union[str, Path]
    ) -> "TerrainFeatureSet":
        """Load the JSON + .npy pair written by module_3.FeatureStore."""
        payload = FeatureStore.load_features(features_json)
        descriptors = FeatureStore.load_descriptors(descriptors_npy)
        features = [TerrainFeature(**f) for f in payload["features"]]
        return cls(features, descriptors, tuple(payload["image_shape"]), payload.get("metadata", {}))

    @classmethod
    def from_extractor(
        cls, extractor: TerrainFeatureExtractor, image_path: Union[str, Path]
    ) -> Tuple["TerrainFeatureSet", ImageArray]:
        enhanced, features, descriptors = extractor.extract(image_path)
        fs = cls(features, descriptors, enhanced.shape, {"source_image": str(image_path)})
        return fs, enhanced


@dataclass
class FeatureMatch:
    """One candidate correspondence between feature `id_a` in A and `id_b` in B.

    Attributes:
        id_a, id_b:        indices into TerrainFeatureSet.features
        x_a, y_a, x_b, y_b: pixel coordinates in image A / image B
        feature_type:      type of the matched pair (both sides share it under
                           type_constraint; otherwise "<type_a>-><type_b>")
        distance:          descriptor distance to the best neighbour
        ratio:             distance / second-best distance (1.0 if unavailable)
        score:             1 - ratio, in (0, 1]; higher is more distinctive
        scale_ratio:       scale_b / scale_a (None if either side lacks scale)
        mutual:            passed the cross-check test
    """

    id_a: int
    id_b: int
    x_a: float
    y_a: float
    x_b: float
    y_b: float
    feature_type: str
    distance: float
    ratio: float
    score: float
    scale_ratio: Optional[float] = None
    mutual: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def point_a(self) -> Tuple[float, float]:
        return (self.x_a, self.y_a)

    @property
    def point_b(self) -> Tuple[float, float]:
        return (self.x_b, self.y_b)


@dataclass
class MatchResult:
    """Container returned by match_terrain_features (input to Module 5)."""

    matches: List[FeatureMatch]
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    image_shape_a: Optional[Tuple[int, int]] = None
    image_shape_b: Optional[Tuple[int, int]] = None

    def __len__(self) -> int:
        return len(self.matches)

    def __iter__(self):
        return iter(self.matches)

    @property
    def points_a(self) -> np.ndarray:
        return np.array([[m.x_a, m.y_a] for m in self.matches], dtype=np.float32).reshape(-1, 2)

    @property
    def points_b(self) -> np.ndarray:
        return np.array([[m.x_b, m.y_b] for m in self.matches], dtype=np.float32).reshape(-1, 2)

    @property
    def scores(self) -> np.ndarray:
        return np.array([m.score for m in self.matches], dtype=np.float32)

    def count_by_type(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for m in self.matches:
            counts[m.feature_type] = counts.get(m.feature_type, 0) + 1
        return counts

    def top(self, n: int) -> "MatchResult":
        best = sorted(self.matches, key=lambda m: m.score, reverse=True)[:n]
        return MatchResult(best, dict(self.diagnostics), dict(self.config),
                           self.image_shape_a, self.image_shape_b)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_count": len(self.matches),
            "image_shape_a": list(self.image_shape_a) if self.image_shape_a else None,
            "image_shape_b": list(self.image_shape_b) if self.image_shape_b else None,
            "config": self.config,
            "diagnostics": self.diagnostics,
            "matches": [m.to_dict() for m in self.matches],
        }

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=_json_default))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "MatchResult":
        d = json.loads(Path(path).read_text())
        return cls(
            [FeatureMatch(**m) for m in d["matches"]],
            d.get("diagnostics", {}),
            d.get("config", {}),
            tuple(d["image_shape_a"]) if d.get("image_shape_a") else None,
            tuple(d["image_shape_b"]) if d.get("image_shape_b") else None,
        )


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _as_descriptor_array(descriptors: Optional[np.ndarray], dim: int = 128) -> np.ndarray:
    if descriptors is None or np.size(descriptors) == 0:
        return np.empty((0, dim), dtype=np.float32)
    arr = np.asarray(descriptors)
    if arr.ndim != 2:
        raise ValueError(f"descriptors must be 2-D (N, D), got shape {arr.shape}")
    if arr.dtype == np.uint8:
        return arr
    return arr.astype(np.float32, copy=False)


# ============================================================================
# 2. DESCRIPTOR MATCHING PRIMITIVES
# ============================================================================

def _is_binary(descriptors: np.ndarray) -> bool:
    return descriptors.dtype == np.uint8


def _make_matcher(method: str, binary: bool) -> cv2.DescriptorMatcher:
    method = method.upper()
    if method == "BF":
        return cv2.BFMatcher(cv2.NORM_HAMMING if binary else cv2.NORM_L2, crossCheck=False)
    if method == "FLANN":
        if binary:
            index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)  # LSH
        else:
            index_params = dict(algorithm=1, trees=5)  # KD-tree
        return cv2.FlannBasedMatcher(index_params, dict(checks=64))
    raise ValueError(f"Unknown matcher method '{method}' (expected 'BF' or 'FLANN')")


def match_descriptors(
    descriptors_a: np.ndarray,
    descriptors_b: np.ndarray,
    method: str = "BF",
    k: int = 2,
) -> List[List[cv2.DMatch]]:
    """KNN-match descriptor rows of A against B.

    Args:
        descriptors_a: (Na, D) query descriptors (float32 -> L2, uint8 -> Hamming)
        descriptors_b: (Nb, D) train descriptors
        method:        "BF" (exact, baseline) or "FLANN" (approximate)
        k:             neighbours per query (2 for the Lowe ratio test)

    Returns:
        List with one entry per query; each entry is a list of up to k
        cv2.DMatch sorted by ascending distance.  Empty inputs yield [].
    """
    a = _as_descriptor_array(descriptors_a)
    b = _as_descriptor_array(descriptors_b)
    if len(a) == 0 or len(b) == 0:
        return []
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"descriptor dims differ: {a.shape[1]} vs {b.shape[1]}")
    if _is_binary(a) != _is_binary(b):
        raise ValueError("cannot match binary descriptors against float descriptors")

    k_eff = min(k, len(b))
    matcher = _make_matcher(method, _is_binary(a))
    raw = matcher.knnMatch(a, b, k=k_eff)
    # FLANN may return fewer than k for some queries; normalize to lists.
    return [sorted(list(group), key=lambda m: m.distance) for group in raw]


def ratio_test(knn_matches: Iterable[Sequence[cv2.DMatch]], ratio: float = 0.75) -> List[cv2.DMatch]:
    """Lowe's ratio test: keep the best neighbour when it is clearly better
    than the second best (d1 < ratio * d2).  Queries with a single neighbour
    are kept only when d2 is unavailable *and* d1 == 0 (exact duplicate)."""
    good: List[cv2.DMatch] = []
    for group in knn_matches:
        if len(group) == 0:
            continue
        if len(group) == 1:
            if group[0].distance == 0.0:
                good.append(group[0])
            continue
        m, n = group[0], group[1]
        if n.distance <= 0.0:
            continue  # two identical neighbours -> ambiguous
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def mutual_consistency(
    matches_ab: Iterable[cv2.DMatch], matches_ba: Iterable[cv2.DMatch]
) -> List[cv2.DMatch]:
    """Cross-check: keep A->B matches whose B->A counterpart points back.

    `matches_ab[i]` maps query i (in A) to train j (in B); `matches_ba`
    maps query j (in B) to train i (in A).  Returned DMatch objects are
    from `matches_ab`.
    """
    back: Dict[int, Tuple[int, float]] = {}
    for m in matches_ba:
        prev = back.get(m.queryIdx)
        if prev is None or m.distance < prev[1]:
            back[m.queryIdx] = (m.trainIdx, m.distance)
    mutual: List[cv2.DMatch] = []
    for m in matches_ab:
        entry = back.get(m.trainIdx)
        if entry is not None and entry[0] == m.queryIdx:
            mutual.append(m)
    return mutual


def enforce_unique_targets(matches: Iterable[cv2.DMatch]) -> List[cv2.DMatch]:
    """Keep only the closest query for every train index (one-to-one)."""
    best: Dict[int, cv2.DMatch] = {}
    for m in matches:
        cur = best.get(m.trainIdx)
        if cur is None or m.distance < cur.distance:
            best[m.trainIdx] = m
    return sorted(best.values(), key=lambda m: m.queryIdx)


def _ratio_lookup(knn: List[List[cv2.DMatch]]) -> Dict[int, float]:
    """queryIdx -> d1/d2 for every query with >= 2 neighbours."""
    out: Dict[int, float] = {}
    for group in knn:
        if len(group) >= 2 and group[1].distance > 0:
            out[group[0].queryIdx] = float(group[0].distance / group[1].distance)
        elif len(group) >= 1:
            out[group[0].queryIdx] = 1.0
    return out


def match_descriptor_sets(
    descriptors_a: np.ndarray,
    descriptors_b: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[List[cv2.DMatch], Dict[str, float], Dict[str, int]]:
    """Full descriptor-only pipeline: KNN -> ratio -> mutual -> unique.

    Returns (matches, ratio_by_query, stage_counts).  Works for any
    descriptor array (SIFT float32 today, learned/binary descriptors later).
    """
    cfg = create_matching_config(**(config or {}))
    stages: Dict[str, int] = {}

    knn_ab = match_descriptors(descriptors_a, descriptors_b, cfg["matcher"], cfg["knn"])
    stages["knn"] = sum(1 for g in knn_ab if g)
    ratios = _ratio_lookup(knn_ab)

    good_ab = ratio_test(knn_ab, cfg["ratio"]) if cfg["knn"] >= 2 else [g[0] for g in knn_ab if g]
    stages["ratio"] = len(good_ab)

    if cfg["mutual"]:
        knn_ba = match_descriptors(descriptors_b, descriptors_a, cfg["matcher"], cfg["knn"])
        good_ba = ratio_test(knn_ba, cfg["ratio"]) if cfg["knn"] >= 2 else [g[0] for g in knn_ba if g]
        good_ab = mutual_consistency(good_ab, good_ba)
        stages["mutual"] = len(good_ab)

    if cfg["unique_target"]:
        good_ab = enforce_unique_targets(good_ab)
        stages["unique"] = len(good_ab)

    return good_ab, ratios, stages


# ============================================================================
# 3. DESCRIPTORS FOR SEMANTIC FEATURES
# ============================================================================

def _semantic_keypoint(f: TerrainFeature, min_size: float, max_size: float) -> cv2.KeyPoint:
    if f.feature_type == "crater":
        size = 2.0 * (f.scale or min_size)          # scale = radius -> diameter
    elif f.feature_type == "ridge":
        size = 0.5 * (f.scale or min_size)          # scale = length -> half length
    else:
        size = float(f.scale or min_size)           # texture block size
    size = float(np.clip(size, min_size, max_size))
    angle = -1.0 if f.orientation is None else float(f.orientation % 360.0)
    return cv2.KeyPoint(float(f.x), float(f.y), size, angle)


def dominant_orientations(image: ImageArray, keypoints: Sequence[cv2.KeyPoint], n_bins: int = 36) -> np.ndarray:
    """SIFT-style dominant gradient orientation (degrees, [0, 360)) for each
    keypoint: magnitude-weighted, Gaussian-windowed orientation histogram
    over a radius of ~1.5 * keypoint scale.  Used to make descriptors of
    semantic features (craters, texture points) rotation invariant."""
    if len(keypoints) == 0:
        return np.empty(0, dtype=np.float32)
    gray = image.astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    ang = np.degrees(np.arctan2(gy, gx)) % 360.0
    h, w = gray.shape[:2]
    out = np.zeros(len(keypoints), dtype=np.float32)
    bin_width = 360.0 / n_bins
    for i, kp in enumerate(keypoints):
        sigma = max(1.5 * kp.size / 2.0, 1.5)
        r = int(round(3 * sigma))
        x0, y0 = int(round(kp.pt[0])), int(round(kp.pt[1]))
        xs, xe = max(0, x0 - r), min(w, x0 + r + 1)
        ys, ye = max(0, y0 - r), min(h, y0 + r + 1)
        if xe <= xs or ye <= ys:
            continue
        yy, xx = np.mgrid[ys:ye, xs:xe]
        weight = np.exp(-((xx - kp.pt[0]) ** 2 + (yy - kp.pt[1]) ** 2) / (2 * sigma ** 2)) * mag[ys:ye, xs:xe]
        bins = (ang[ys:ye, xs:xe] / bin_width).astype(int) % n_bins
        hist = np.bincount(bins.ravel(), weights=weight.ravel(), minlength=n_bins)
        hist = (np.roll(hist, 1) + hist + np.roll(hist, -1)) / 3.0  # circular smoothing
        peak = int(np.argmax(hist))
        # parabolic interpolation of the peak
        l, c, rr = hist[(peak - 1) % n_bins], hist[peak], hist[(peak + 1) % n_bins]
        denom = l - 2 * c + rr
        offset = 0.0 if abs(denom) < 1e-12 else 0.5 * (l - rr) / denom
        out[i] = ((peak + 0.5 + offset) * bin_width) % 360.0
    return out


def attach_descriptors(
    feature_set: TerrainFeatureSet,
    image: ImageArray,
    config: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
) -> TerrainFeatureSet:
    """Compute SIFT descriptors at semantic feature locations so craters,
    ridges and texture points become matchable.  Returns a new set; the
    original descriptor rows (SIFT) keep their indices, new rows are appended.

    Features whose local patch cannot be computed (e.g. extreme sizes) keep
    descriptor_index=None and are excluded from descriptor matching.
    """
    cfg = create_matching_config(**(config or {}))
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype != np.uint8:
        image = ImagePreprocessor().normalize(image)

    features = [TerrainFeature(**f.to_dict()) for f in feature_set.features]
    todo = [i for i, f in enumerate(features)
            if f.feature_type in SEMANTIC_TYPES and (overwrite or f.descriptor_index is None)]
    if not todo:
        return TerrainFeatureSet(features, feature_set.descriptors.copy(), feature_set.image_shape,
                                 dict(feature_set.metadata))

    kps = [_semantic_keypoint(features[i], cfg["semantic_desc_min_size"], cfg["semantic_desc_max_size"])
           for i in todo]
    if cfg["semantic_orientation"] == "gradient":
        need = [j for j, kp in enumerate(kps) if kp.angle < 0]
        if need:
            angles = dominant_orientations(image, [kps[j] for j in need])
            for j, a in zip(need, angles):
                kps[j].angle = float(a)
                features[todo[j]].extra["descriptor_orientation"] = float(a)
    sift = cv2.SIFT_create()
    kps_out, desc = sift.compute(image, kps)
    if desc is None or len(kps_out) == 0:
        return TerrainFeatureSet(features, feature_set.descriptors.copy(), feature_set.image_shape,
                                 dict(feature_set.metadata))

    # OpenCV returns keypoints in order but may drop some; map back by position.
    if len(kps_out) == len(kps):
        kept_positions = list(range(len(todo)))
    else:
        lookup = {(round(k.pt[0], 2), round(k.pt[1], 2)): j for j, k in enumerate(kps)}
        kept_positions = [lookup[(round(k.pt[0], 2), round(k.pt[1], 2))] for k in kps_out]

    base = feature_set.descriptors
    if len(base) and base.shape[1] != desc.shape[1]:
        raise ValueError("semantic descriptor dimension does not match existing descriptors")
    merged = np.vstack([base, desc.astype(np.float32)]) if len(base) else desc.astype(np.float32)

    offset = len(base)
    for row, pos in enumerate(kept_positions):
        features[todo[pos]].descriptor_index = offset + row
        features[todo[pos]].extra["descriptor_source"] = "sift_at_semantic"

    meta = dict(feature_set.metadata)
    meta["semantic_descriptors_attached"] = len(kept_positions)
    return TerrainFeatureSet(features, merged, feature_set.image_shape, meta)


# ============================================================================
# 4. TERRAIN FEATURE MATCHING
# ============================================================================

def _scale_consistency_mask(scale_ratios: np.ndarray, log_tol: float) -> np.ndarray:
    """Robust filter: keep matches whose log scale-ratio sits within `log_tol`
    of the median.  Matches with unknown scale (NaN) are always kept."""
    valid = np.isfinite(scale_ratios) & (scale_ratios > 0)
    keep = np.ones(len(scale_ratios), dtype=bool)
    if valid.sum() < 3:
        return keep
    logs = np.log(scale_ratios[valid])
    med = np.median(logs)
    keep[valid] = np.abs(logs - med) <= log_tol
    return keep


def match_terrain_features(
    features_a: TerrainFeatureSet,
    features_b: TerrainFeatureSet,
    config: Optional[Dict[str, Any]] = None,
) -> MatchResult:
    """Produce candidate correspondences between two terrain feature sets.

    Only features carrying a descriptor participate; run attach_descriptors()
    first if you want craters / ridges / texture points to be matched.  With
    `type_constraint` (default) a crater can only match a crater, etc.

    Returns a MatchResult (no geometric verification — see Module 5).
    """
    cfg = create_matching_config(**(config or {}))
    t0 = time.time()
    diag: Dict[str, Any] = {
        "features_a": len(features_a), "features_b": len(features_b),
        "features_a_by_type": features_a.count_by_type(),
        "features_b_by_type": features_b.count_by_type(),
        "with_descriptor_a": sum(1 for f in features_a.features if f.descriptor_index is not None),
        "with_descriptor_b": sum(1 for f in features_b.features if f.descriptor_index is not None),
        "groups": {},
    }

    types = [t for t in cfg["match_types"] if t in FEATURE_TYPES]
    if cfg["type_constraint"]:
        groups = [(t, features_a.indices_of_type(t), features_b.indices_of_type(t)) for t in types]
    else:
        ia = [i for i, f in enumerate(features_a.features) if f.feature_type in types]
        ib = [i for i, f in enumerate(features_b.features) if f.feature_type in types]
        groups = [("all", ia, ib)]

    matches: List[FeatureMatch] = []
    for name, ia, ib in groups:
        ia = [i for i in ia if features_a.has_descriptor(i)]
        ib = [i for i in ib if features_b.has_descriptor(i)]
        gdiag: Dict[str, Any] = {"candidates_a": len(ia), "candidates_b": len(ib)}
        if not ia or not ib:
            gdiag["stages"] = {}
            gdiag["matches"] = 0
            diag["groups"][name] = gdiag
            continue

        da = np.stack([features_a.descriptor_for(i) for i in ia])
        db = np.stack([features_b.descriptor_for(i) for i in ib])
        group_cfg = dict(cfg)
        group_cfg["ratio"] = cfg["type_ratio"].get(name, cfg["ratio"])
        gdiag["ratio"] = group_cfg["ratio"]
        good, ratios, stages = match_descriptor_sets(da, db, group_cfg)

        group_matches: List[FeatureMatch] = []
        for m in good:
            fa = features_a.features[ia[m.queryIdx]]
            fb = features_b.features[ib[m.trainIdx]]
            r = ratios.get(m.queryIdx, 1.0)
            sr = (fb.scale / fa.scale) if (fa.scale and fb.scale and fa.scale > 0) else None
            group_matches.append(FeatureMatch(
                id_a=ia[m.queryIdx], id_b=ib[m.trainIdx],
                x_a=fa.x, y_a=fa.y, x_b=fb.x, y_b=fb.y,
                feature_type=fa.feature_type if fa.feature_type == fb.feature_type
                else f"{fa.feature_type}->{fb.feature_type}",
                distance=float(m.distance), ratio=float(r), score=float(1.0 - r),
                scale_ratio=None if sr is None else float(sr),
                mutual=bool(cfg["mutual"]),
            ))

        if cfg["scale_constraint"] and group_matches:
            srs = np.array([np.nan if gm.scale_ratio is None else gm.scale_ratio for gm in group_matches])
            keep = _scale_consistency_mask(srs, cfg["scale_log_tolerance"])
            group_matches = [gm for gm, k in zip(group_matches, keep) if k]
            stages["scale"] = len(group_matches)

        if cfg["min_score"] > 0:
            group_matches = [gm for gm in group_matches if gm.score >= cfg["min_score"]]
            stages["min_score"] = len(group_matches)

        gdiag["stages"] = stages
        gdiag["matches"] = len(group_matches)
        diag["groups"][name] = gdiag
        matches.extend(group_matches)

    matches.sort(key=lambda m: m.score, reverse=True)
    if cfg["max_matches"] is not None:
        matches = matches[: int(cfg["max_matches"])]

    srs = np.array([m.scale_ratio for m in matches if m.scale_ratio], dtype=float)
    diag.update({
        "total_matches": len(matches),
        "matches_by_type": _count_types(matches),
        "score_mean": float(np.mean([m.score for m in matches])) if matches else 0.0,
        "score_median": float(np.median([m.score for m in matches])) if matches else 0.0,
        "median_scale_ratio": float(np.median(srs)) if len(srs) else None,
        "elapsed_s": round(time.time() - t0, 4),
    })
    return MatchResult(matches, diag, cfg, features_a.image_shape, features_b.image_shape)


def _count_types(matches: Iterable[FeatureMatch]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for m in matches:
        out[m.feature_type] = out.get(m.feature_type, 0) + 1
    return out


# ============================================================================
# 5. SYNTHETIC PAIRS & GROUND-TRUTH EVALUATION
# ============================================================================

def make_synthetic_pair(
    image: ImageArray,
    rotation_deg: float = 15.0,
    scale: float = 0.9,
    translation: Tuple[float, float] = (20.0, -10.0),
    brightness_gain: float = 1.0,
    brightness_offset: float = 0.0,
    gamma: float = 1.0,
    noise_sigma: float = 0.0,
    seed: int = 0,
) -> Tuple[ImageArray, np.ndarray]:
    """Warp `image` with a known similarity transform + photometric change.

    Returns (image_b, M) where M is the 2x3 affine mapping A -> B pixels.
    """
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), rotation_deg, scale)
    M[:, 2] += np.asarray(translation, dtype=np.float64)
    warped = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    out = warped.astype(np.float32) / 255.0
    if gamma != 1.0:
        out = np.power(np.clip(out, 0, 1), gamma)
    out = out * brightness_gain + brightness_offset / 255.0
    if noise_sigma > 0:
        rng = np.random.default_rng(seed)
        out = out + rng.normal(0.0, noise_sigma / 255.0, out.shape).astype(np.float32)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8), M


def apply_affine(points: np.ndarray, M: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    ones = np.ones((len(pts), 1))
    return (np.hstack([pts, ones]) @ M.T).astype(np.float32)


def evaluate_against_ground_truth(
    result: MatchResult, M_ab: np.ndarray, tolerance_px: float = 3.0
) -> Dict[str, Any]:
    """Score candidate matches against a known A->B transform.

    This is *evaluation only* (no filtering) so Module 4 stays RANSAC-free.
    """
    if len(result) == 0:
        return {"n": 0, "correct": 0, "precision": 0.0, "median_error_px": None,
                "tolerance_px": tolerance_px, "by_type": {}}
    proj = apply_affine(result.points_a, M_ab)
    err = np.linalg.norm(proj - result.points_b, axis=1)
    correct = err <= tolerance_px
    by_type: Dict[str, Dict[str, float]] = {}
    for m, c in zip(result.matches, correct):
        d = by_type.setdefault(m.feature_type, {"n": 0, "correct": 0})
        d["n"] += 1
        d["correct"] += int(c)
    for d in by_type.values():
        d["precision"] = d["correct"] / d["n"] if d["n"] else 0.0
    return {
        "n": int(len(err)),
        "correct": int(correct.sum()),
        "precision": float(correct.mean()),
        "median_error_px": float(np.median(err)),
        "mean_error_px": float(np.mean(err)),
        "tolerance_px": tolerance_px,
        "by_type": by_type,
    }


def compute_repeatability(
    features_a: TerrainFeatureSet,
    features_b: TerrainFeatureSet,
    M_ab: np.ndarray,
    tolerance_px: float = 3.0,
) -> Dict[str, Dict[str, float]]:
    """Per type: how many features of A have a same-type feature of B within
    `tolerance_px` after applying the known transform.  This is the upper
    bound on achievable matches and separates detector repeatability
    (Module 3) from matching quality (Module 4)."""
    out: Dict[str, Dict[str, float]] = {}
    for t in FEATURE_TYPES:
        ia, ib = features_a.indices_of_type(t), features_b.indices_of_type(t)
        if not ia or not ib:
            out[t] = {"n_a": len(ia), "n_b": len(ib), "repeatable": 0, "repeatability": 0.0}
            continue
        pa = apply_affine(np.array([[features_a.features[i].x, features_a.features[i].y] for i in ia]), M_ab)
        pb = np.array([[features_b.features[i].x, features_b.features[i].y] for i in ib], dtype=np.float32)
        d = np.sqrt(((pa[:, None, :] - pb[None, :, :]) ** 2).sum(-1))
        rep = int((d.min(axis=1) <= tolerance_px).sum())
        out[t] = {"n_a": len(ia), "n_b": len(ib), "repeatable": rep, "repeatability": rep / len(ia)}
    return out


# ============================================================================
# 6. VISUALIZATION
# ============================================================================

COLOR_MAP_RGB: Dict[str, Tuple[float, float, float]] = {
    "crater": (1.0, 0.2, 0.2),
    "ridge": (0.2, 0.4, 1.0),
    "texture": (1.0, 0.9, 0.1),
    "sift": (1.0, 0.2, 1.0),
}


def _side_by_side(image_a: ImageArray, image_b: ImageArray) -> Tuple[np.ndarray, int]:
    def to_rgb(img: ImageArray) -> np.ndarray:
        if img.dtype != np.uint8:
            img = ImagePreprocessor().normalize(img)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else img[..., ::-1]

    a, b = to_rgb(image_a), to_rgb(image_b)
    h = max(a.shape[0], b.shape[0])
    canvas = np.zeros((h, a.shape[1] + b.shape[1], 3), dtype=np.uint8)
    canvas[: a.shape[0], : a.shape[1]] = a
    canvas[: b.shape[0], a.shape[1]:] = b
    return canvas, a.shape[1]


def visualize_matches(
    image_a: ImageArray,
    image_b: ImageArray,
    matches: Union[MatchResult, Sequence[FeatureMatch]],
    max_draw: int = 300,
    title: Optional[str] = None,
    correct_mask: Optional[np.ndarray] = None,
    figsize: Tuple[float, float] = (16, 8),
    line_alpha: float = 0.6,
):
    """Side-by-side match plot. Lines are coloured by feature type, or
    green/red if `correct_mask` (from ground-truth evaluation) is given.
    Draws the `max_draw` highest-scoring matches.  Returns a matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    mlist = list(matches.matches if isinstance(matches, MatchResult) else matches)
    order = np.argsort([-m.score for m in mlist])[:max_draw]
    canvas, offset = _side_by_side(image_a, image_b)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(canvas)
    ax.set_axis_off()
    counts = _count_types(mlist)
    for idx in order:
        m = mlist[idx]
        if correct_mask is not None:
            color = (0.1, 0.9, 0.2) if correct_mask[idx] else (1.0, 0.1, 0.1)
        else:
            color = COLOR_MAP_RGB.get(m.feature_type, (1.0, 1.0, 1.0))
        ax.plot([m.x_a, m.x_b + offset], [m.y_a, m.y_b], color=color, linewidth=0.8, alpha=line_alpha)
        ax.plot(m.x_a, m.y_a, "o", color=color, markersize=3, alpha=0.9)
        ax.plot(m.x_b + offset, m.y_b, "o", color=color, markersize=3, alpha=0.9)

    if correct_mask is not None:
        handles = [Line2D([0], [0], color=(0.1, 0.9, 0.2), label=f"correct ({int(np.sum(correct_mask))})"),
                   Line2D([0], [0], color=(1.0, 0.1, 0.1), label=f"wrong ({int(len(correct_mask) - np.sum(correct_mask))})")]
    else:
        handles = [Line2D([0], [0], color=COLOR_MAP_RGB.get(t, (1, 1, 1)), label=f"{t} ({n})")
                   for t, n in sorted(counts.items())]
    ax.legend(handles=handles, loc="lower center", ncol=max(1, len(handles)), fontsize=9, framealpha=0.8)
    ax.set_title(title or f"{len(mlist)} candidate matches (showing {len(order)})")
    fig.tight_layout()
    return fig


# ============================================================================
# 7. SELF TESTS (synthetic)
# ============================================================================

def _synthetic_descriptors(n: int, dim: int = 128, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, dim)).astype(np.float32)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def _synthetic_terrain_image(size: int = 512, n_blobs: int = 60, seed: int = 1) -> ImageArray:
    """Crater-like blobs + ridges on a smooth background; deterministic."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    img = 110 + 30 * np.sin(xx / 90.0) * np.cos(yy / 70.0)
    for _ in range(n_blobs):
        cx, cy = rng.uniform(20, size - 20, 2)
        r = rng.uniform(6, 30)
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        rim = np.exp(-((d - r) ** 2) / (2 * (0.25 * r) ** 2))
        bowl = np.exp(-(d ** 2) / (2 * (0.6 * r) ** 2))
        img += 45 * rim * (1 + 0.6 * np.sign(xx - cx)) - 35 * bowl
    for _ in range(6):
        x1, y1, x2, y2 = rng.uniform(0, size, 4)
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), 200, 2)
    img += rng.normal(0, 3, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def run_self_tests(verbose: bool = True) -> Dict[str, bool]:
    """Synthetic checks that exercise every public function."""
    results: Dict[str, bool] = {}

    def check(name: str, cond: bool) -> None:
        results[name] = bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # --- 1. descriptor primitives on synthetic descriptors -----------------
    base = _synthetic_descriptors(200)
    perm = np.random.default_rng(3).permutation(200)
    noisy = base[perm] + 0.02 * np.random.default_rng(4).normal(size=base.shape).astype(np.float32)
    knn = match_descriptors(base, noisy, "BF", k=2)
    check("knn returns one group per query", len(knn) == 200 and all(len(g) == 2 for g in knn))
    good = ratio_test(knn, 0.75)
    inv = np.argsort(perm)
    check("ratio test recovers permutation", len(good) >= 190 and all(inv[m.queryIdx] == m.trainIdx for m in good))
    knn_flann = match_descriptors(base, noisy, "FLANN", k=2)
    good_flann = ratio_test(knn_flann, 0.75)
    check("FLANN agrees with BF", len(good_flann) >= 0.9 * len(good))

    back = ratio_test(match_descriptors(noisy, base, "BF", 2), 0.75)
    mutual = mutual_consistency(good, back)
    check("mutual consistency keeps true pairs", len(mutual) >= 0.95 * len(good))

    # inject distractors: rows in B with no partner in A must not survive mutual+unique
    distract = np.vstack([noisy, _synthetic_descriptors(50, seed=9)])
    good_d = ratio_test(match_descriptors(base, distract, "BF", 2), 0.75)
    back_d = ratio_test(match_descriptors(distract, base, "BF", 2), 0.75)
    mutual_d = enforce_unique_targets(mutual_consistency(good_d, back_d))
    check("distractors rejected", all(m.trainIdx < 200 for m in mutual_d) and len(mutual_d) >= 180)
    check("empty input handled", match_descriptors(np.empty((0, 128)), base) == [])

    # --- 2. full terrain pipeline on a synthetic image pair ---------------
    img_a = _synthetic_terrain_image()
    img_b, M = make_synthetic_pair(img_a, rotation_deg=20, scale=0.85, translation=(15, -8),
                                   brightness_gain=1.3, gamma=0.8, noise_sigma=4)
    extractor = TerrainFeatureExtractor(onnx_crater_model_path="__missing__.onnx", max_total_features=None)
    set_a, ea = extract_feature_set(img_a, extractor)
    set_b, eb = extract_feature_set(img_b, extractor)
    check("semantic descriptors attached", all(f.descriptor_index is not None for f in set_a.features))

    result = match_terrain_features(set_a, set_b)
    ev = evaluate_against_ground_truth(result, M, tolerance_px=4.0)
    check("candidate matches produced", len(result) >= 30)
    check(f"precision vs ground truth >= 0.6 (got {ev['precision']:.2f}, n={ev['n']})", ev["precision"] >= 0.6)
    check("type constraint respected", all("->" not in m.feature_type for m in result.matches))
    check("one-to-one targets", len({m.id_b for m in result.matches}) == len(result))

    sift_only = match_terrain_features(set_a, set_b, {"match_types": ["sift"]})
    check("match_types filter", all(m.feature_type == "sift" for m in sift_only.matches))

    loose = match_terrain_features(set_a, set_b, {"mutual": False, "scale_constraint": False})
    check("mutual+scale filters reduce candidates", len(loose) >= len(result))

    tmp = Path("_module4_selftest.json")
    result.save(tmp)
    reloaded = MatchResult.load(tmp)
    tmp.unlink(missing_ok=True)
    check("JSON round trip", len(reloaded) == len(result) and reloaded.matches[0] == result.matches[0])

    fig = visualize_matches(ea, eb, result, max_draw=100)
    check("visualize_matches returns figure", fig is not None)
    import matplotlib.pyplot as plt
    plt.close(fig)

    if verbose:
        print(f"\n{sum(results.values())}/{len(results)} checks passed")
    return results


def _extract_from_array(
    extractor: TerrainFeatureExtractor, image: ImageArray
) -> Tuple[List[TerrainFeature], np.ndarray]:
    """Run the Module-3 detectors on an in-memory image (no disk I/O)."""
    enhanced = extractor.preprocessor.enhance(extractor.preprocessor.normalize(image))
    feats = (extractor.crater_detector.detect(enhanced)
             + extractor.ridge_detector.detect(enhanced)
             + extractor.texture_detector.detect(enhanced))
    sift_feats, desc = extractor.sift_detector.detect(enhanced)
    feats += sift_feats
    return extractor._cap_features(feats, desc)


def extract_feature_set(
    image: ImageArray,
    extractor: Optional[TerrainFeatureExtractor] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[TerrainFeatureSet, ImageArray]:
    """Convenience: Module-3 extraction on an array + semantic descriptors.
    Returns (feature_set, enhanced_image)."""
    extractor = extractor or TerrainFeatureExtractor()
    enhanced = extractor.preprocessor.enhance(extractor.preprocessor.normalize(image))
    feats, desc = _extract_from_array(extractor, image)
    fs = attach_descriptors(TerrainFeatureSet(feats, desc, enhanced.shape), enhanced, config)
    return fs, enhanced


# ============================================================================
# 8. DEMO / CLI
# ============================================================================

def run_demo(
    image_a_path: Union[str, Path],
    image_b_path: Optional[Union[str, Path]] = None,
    output_dir: Union[str, Path] = "module4_outputs",
    config: Optional[Dict[str, Any]] = None,
    onnx_model_path: Union[str, Path] = "models/crater_unet.onnx",
    max_features: Optional[int] = 3000,
) -> Dict[str, Any]:
    """Match two lunar images (or one image against a synthetic transform of
    itself) and write JSON + PNG outputs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_a_path).stem

    pre = ImagePreprocessor()
    raw_a = pre.normalize(pre.load(image_a_path))
    M: Optional[np.ndarray] = None
    if image_b_path is None:
        raw_b, M = make_synthetic_pair(raw_a, rotation_deg=25, scale=0.8, translation=(30, -20),
                                       brightness_gain=1.25, gamma=0.75, noise_sigma=5)
        stem += "_synthetic"
        print("[Module 4] synthetic pair: rot=25deg scale=0.8 shift=(30,-20) gain=1.25 gamma=0.75")
    else:
        raw_b = pre.normalize(pre.load(image_b_path))

    extractor = TerrainFeatureExtractor(onnx_crater_model_path=onnx_model_path, max_total_features=max_features)
    set_a, enh_a = extract_feature_set(raw_a, extractor, config)
    set_b, enh_b = extract_feature_set(raw_b, extractor, config)
    print(f"[Module 4] features A: {set_a.count_by_type()}")
    print(f"[Module 4] features B: {set_b.count_by_type()}")

    result = match_terrain_features(set_a, set_b, config)
    d = result.diagnostics
    print(f"[Module 4] {d['total_matches']} candidate matches {d['matches_by_type']} "
          f"in {d['elapsed_s']}s; median scale ratio={d['median_scale_ratio']}")
    for g, gd in d["groups"].items():
        print(f"           {g:8s} A={gd['candidates_a']:5d} B={gd['candidates_b']:5d} stages={gd['stages']}")

    outputs: Dict[str, Any] = {"result": result}
    correct_mask = None
    if M is not None:
        ev = evaluate_against_ground_truth(result, M, tolerance_px=4.0)
        ev["repeatability"] = compute_repeatability(set_a, set_b, M, tolerance_px=4.0)
        result.diagnostics["ground_truth"] = ev
        proj = apply_affine(result.points_a, M)
        correct_mask = np.linalg.norm(proj - result.points_b, axis=1) <= 4.0
        print(f"[Module 4] ground truth: precision={ev['precision']:.3f} "
              f"({ev['correct']}/{ev['n']}), median err={ev['median_error_px']:.2f}px")
        for t, td in ev["by_type"].items():
            print(f"           {t:8s} {td['correct']:4d}/{td['n']:4d} = {td['precision']:.2f}")
        print("[Module 4] detector repeatability (upper bound on matches):")
        for t, rd in ev["repeatability"].items():
            print(f"           {t:8s} {rd['repeatable']:4d}/{rd['n_a']:4d} = {rd['repeatability']:.2f}")

    json_path = output_dir / f"{stem}_matches.json"
    result.save(json_path)
    outputs["matches_json"] = json_path

    fig = visualize_matches(enh_a, enh_b, result, max_draw=300, title=f"{stem}: {len(result)} candidates by type")
    png = output_dir / f"{stem}_matches.png"
    fig.savefig(png, dpi=110)
    plt.close(fig)
    outputs["matches_png"] = png
    if correct_mask is not None:
        fig = visualize_matches(enh_a, enh_b, result, max_draw=300, correct_mask=correct_mask,
                                title=f"{stem}: ground-truth check (tol 4px)")
        png_gt = output_dir / f"{stem}_matches_gt.png"
        fig.savefig(png_gt, dpi=110)
        plt.close(fig)
        outputs["matches_gt_png"] = png_gt

    print("[Module 4] saved:", ", ".join(str(v) for k, v in outputs.items() if k != "result"))
    return outputs


def main(argv: Optional[List[str]] = None) -> int:
    import sys
    args = list(sys.argv[1:] if argv is None else argv)
    usage = (
        "Usage:\n"
        "  python module_4.py demo <image_a> [image_b] [output_dir]\n"
        "  python module_4.py match <feat_a.json> <desc_a.npy> <feat_b.json> <desc_b.npy> [output_dir]\n"
        "  python module_4.py test"
    )
    if not args:
        print(usage)
        return 1
    cmd = args[0]
    if cmd == "test":
        return 0 if all(run_self_tests().values()) else 1
    if cmd == "demo":
        if len(args) < 2:
            print(usage)
            return 1
        rest = args[2:]
        img_b = rest.pop(0) if rest and Path(rest[0]).is_file() else None
        out = rest[0] if rest else "module4_outputs"
        run_demo(args[1], img_b, out)
        return 0
    if cmd == "match":
        if len(args) < 5:
            print(usage)
            return 1
        set_a = TerrainFeatureSet.from_module3_outputs(args[1], args[2])
        set_b = TerrainFeatureSet.from_module3_outputs(args[3], args[4])
        out = Path(args[5] if len(args) > 5 else "module4_outputs")
        out.mkdir(parents=True, exist_ok=True)
        result = match_terrain_features(set_a, set_b)
        path = out / f"{Path(args[1]).stem}__{Path(args[3]).stem}_matches.json"
        result.save(path)
        print(f"[Module 4] {len(result)} candidate matches {result.count_by_type()} -> {path}")
        return 0
    print(usage)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
