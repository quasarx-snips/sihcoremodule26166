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
    python module_4.py stress <image> [output_json]
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
        semantic_scales:      patch-size multipliers for semantic descriptors; several
                              scales make fixed-size features (texture blocks, ridges)
                              matchable across zoom / anisotropic stretch
        root_sift:            Hellinger (RootSIFT) normalisation of float histogram
                              descriptors before matching (more illumination robust)
        affine_tilts:         ASIFT-style view simulation: for every tilt t the image
                              is compressed by t along several directions and extra
                              descriptors are computed for every feature, giving
                              invariance to stretch / shear / off-nadir viewing.
                              [] disables.  Applied by simulate_affine_descriptors().
        affine_angle_step:    base angular step (degrees) between simulated tilt
                              directions; the effective step is step / t
        photometric_normalize: percentile contrast stretch before extraction (see
                              photometric_normalize()); lets under-/over-exposed
                              frames reach the detectors with a usable dynamic range
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
        "semantic_scales": [0.5, 1.0, 2.0],
        "root_sift": True,
        "affine_tilts": [2.0],
        "affine_angle_step": 120.0,
        "photometric_normalize": True,
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
    if not cfg["semantic_scales"] or any(s <= 0 for s in cfg["semantic_scales"]):
        raise ValueError("semantic_scales must be a non-empty list of positive multipliers")
    if any(t <= 1.0 for t in cfg["affine_tilts"]):
        raise ValueError("affine_tilts must be > 1")
    if cfg["affine_angle_step"] <= 0:
        raise ValueError("affine_angle_step must be positive")
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

    def descriptor_indices_for(self, idx: int) -> List[int]:
        """All descriptor rows describing feature `idx` (several when semantic
        descriptors were computed at multiple scales)."""
        f = self.features[idx]
        extra = f.extra.get("descriptor_indices")
        if extra:
            return [int(i) for i in extra]
        return [] if f.descriptor_index is None else [int(f.descriptor_index)]

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


def root_sift(descriptors: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Hellinger kernel normalisation: L1-normalise then sqrt.  Euclidean
    distance on the result equals the Hellinger distance between the SIFT
    histograms, which is markedly more robust to illumination-driven
    gradient-magnitude changes.  Non-float / signed descriptors pass through."""
    if descriptors.size == 0 or descriptors.dtype == np.uint8 or np.any(descriptors < 0):
        return descriptors
    d = descriptors.astype(np.float32)
    d = d / (np.abs(d).sum(axis=1, keepdims=True) + eps)
    return np.sqrt(d)


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


def _owner_ratio_test(
    knn: List[List[cv2.DMatch]], owner_b: np.ndarray, ratio: float
) -> Dict[int, Tuple[int, float, float]]:
    """Ratio test where several descriptor rows may describe the same feature
    (multi-scale descriptors).  The second-best distance is taken from the
    nearest row belonging to a *different* feature, so a feature competing
    against its own other scales is not rejected.  Returns
    queryRow -> (trainRow, distance, ratio)."""
    out: Dict[int, Tuple[int, float, float]] = {}
    for group in knn:
        if not group:
            continue
        best = group[0]
        own = owner_b[best.trainIdx]
        second = next((m for m in group[1:] if owner_b[m.trainIdx] != own), None)
        if second is None:
            last = group[-1].distance
            if len(group) > 1 and last > 0.0:
                out[best.queryIdx] = (best.trainIdx, float(best.distance), float(best.distance / last))
            elif best.distance == 0.0:
                out[best.queryIdx] = (best.trainIdx, 0.0, 0.0)
            continue
        if second.distance <= 0.0:
            continue
        r = best.distance / second.distance
        if r < ratio:
            out[best.queryIdx] = (best.trainIdx, float(best.distance), float(r))
    return out


def _best_per_owner(
    rows: Dict[int, Tuple[int, float, float]], owner_q: np.ndarray, owner_t: np.ndarray
) -> Dict[int, Tuple[int, float, float]]:
    """Collapse row-level matches to feature level: queryFeature -> (trainFeature, dist, ratio)."""
    best: Dict[int, Tuple[int, float, float]] = {}
    for q_row, (t_row, dist, r) in rows.items():
        q, t = int(owner_q[q_row]), int(owner_t[t_row])
        cur = best.get(q)
        if cur is None or dist < cur[1]:
            best[q] = (t, dist, r)
    return best


def match_feature_rows(
    descriptors_a: np.ndarray,
    descriptors_b: np.ndarray,
    owner_a: np.ndarray,
    owner_b: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[int, Tuple[int, float, float]], Dict[str, int]]:
    """Feature-level matching when each feature may own several descriptor
    rows.  Pipeline: KNN -> owner-aware ratio -> collapse to features ->
    mutual (feature level) -> unique target.  Returns
    (featureA -> (featureB, distance, ratio), stage_counts)."""
    cfg = create_matching_config(**(config or {}))
    owner_a = np.asarray(owner_a, dtype=int)
    owner_b = np.asarray(owner_b, dtype=int)
    rows_per_a = max(1, int(np.bincount(owner_a).max())) if len(owner_a) else 1
    rows_per_b = max(1, int(np.bincount(owner_b).max())) if len(owner_b) else 1
    if cfg["root_sift"]:
        descriptors_a, descriptors_b = root_sift(descriptors_a), root_sift(descriptors_b)
    stages: Dict[str, int] = {}

    k_ab = max(cfg["knn"], rows_per_b + 1)
    knn_ab = match_descriptors(descriptors_a, descriptors_b, cfg["matcher"], k_ab)
    stages["knn"] = len({int(owner_a[g[0].queryIdx]) for g in knn_ab if g})
    fwd = _best_per_owner(_owner_ratio_test(knn_ab, owner_b, cfg["ratio"]), owner_a, owner_b)
    stages["ratio"] = len(fwd)

    if cfg["mutual"]:
        k_ba = max(cfg["knn"], rows_per_a + 1)
        knn_ba = match_descriptors(descriptors_b, descriptors_a, cfg["matcher"], k_ba)
        bwd = _best_per_owner(_owner_ratio_test(knn_ba, owner_a, cfg["ratio"]), owner_b, owner_a)
        fwd = {a: v for a, v in fwd.items() if bwd.get(v[0], (-1,))[0] == a}
        stages["mutual"] = len(fwd)

    if cfg["unique_target"]:
        best_for_b: Dict[int, Tuple[int, Tuple[int, float, float]]] = {}
        for a, v in fwd.items():
            cur = best_for_b.get(v[0])
            if cur is None or v[1] < cur[1][1]:
                best_for_b[v[0]] = (a, v)
        fwd = {a: v for a, v in best_for_b.values()}
        stages["unique"] = len(fwd)
    return fwd, stages


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

    for i in todo:
        features[i].extra.pop("descriptor_indices", None)
    min_size, max_size = cfg["semantic_desc_min_size"], cfg["semantic_desc_max_size"]
    base_kps = [_semantic_keypoint(features[i], min_size, max_size) for i in todo]
    if cfg["semantic_orientation"] == "gradient":
        need = [j for j, kp in enumerate(base_kps) if kp.angle < 0]
        if need:
            angles = dominant_orientations(image, [base_kps[j] for j in need])
            for j, a in zip(need, angles):
                base_kps[j].angle = float(a)
                features[todo[j]].extra["descriptor_orientation"] = float(a)

    # one keypoint per (feature, scale multiplier); the multiplier closest to 1 is primary
    scales = sorted(float(s) for s in cfg["semantic_scales"])
    primary_scale = min(scales, key=lambda s: abs(np.log(s)))
    kps: List[cv2.KeyPoint] = []
    kp_owner: List[Tuple[int, float]] = []
    for pos, kp in enumerate(base_kps):
        for s in scales:
            size = float(np.clip(kp.size * s, min_size, max_size * max(1.0, s)))
            kps.append(cv2.KeyPoint(kp.pt[0], kp.pt[1], size, kp.angle))
            kp_owner.append((pos, s))

    sift = cv2.SIFT_create()
    kps_out, desc = sift.compute(image, kps)
    if desc is None or len(kps_out) == 0:
        return TerrainFeatureSet(features, feature_set.descriptors.copy(), feature_set.image_shape,
                                 dict(feature_set.metadata))

    # OpenCV keeps order but may drop keypoints; map back by (position, size).
    if len(kps_out) == len(kps):
        kept = list(range(len(kps)))
    else:
        lookup = {(round(k.pt[0], 2), round(k.pt[1], 2), round(k.size, 2)): j for j, k in enumerate(kps)}
        kept = [lookup[(round(k.pt[0], 2), round(k.pt[1], 2), round(k.size, 2))] for k in kps_out]

    base = feature_set.descriptors
    if len(base) and base.shape[1] != desc.shape[1]:
        raise ValueError("semantic descriptor dimension does not match existing descriptors")
    merged = np.vstack([base, desc.astype(np.float32)]) if len(base) else desc.astype(np.float32)

    offset = len(base)
    attached = set()
    for row, j in enumerate(kept):
        pos, s = kp_owner[j]
        f = features[todo[pos]]
        rows = f.extra.setdefault("descriptor_indices", [])
        rows.append(offset + row)
        if s == primary_scale or f.descriptor_index is None:
            f.descriptor_index = offset + row
        f.extra["descriptor_source"] = "sift_at_semantic"
        attached.add(pos)

    meta = dict(feature_set.metadata)
    meta["semantic_descriptors_attached"] = len(attached)
    meta["semantic_scales"] = scales
    return TerrainFeatureSet(features, merged, feature_set.image_shape, meta)


def affine_simulation_views(tilts: Sequence[float], angle_step: float) -> List[Tuple[float, float]]:
    """(tilt, direction_deg) pairs covering the affine half-sphere, ASIFT style:
    directions are spaced angle_step / t apart over [0, 180)."""
    views: List[Tuple[float, float]] = []
    for t in tilts:
        step = angle_step / t
        phi = 0.0
        while phi < 180.0 - 1e-6:
            views.append((float(t), float(phi)))
            phi += step
    return views


def simulate_affine_descriptors(
    feature_set: TerrainFeatureSet,
    image: ImageArray,
    config: Optional[Dict[str, Any]] = None,
) -> TerrainFeatureSet:
    """Add descriptors computed on tilted copies of the image to every feature.

    For each (tilt t, direction phi) the image is rotated by phi and squeezed
    by t along y (the camera looking at the terrain obliquely), feature
    positions are mapped through the same affine map, a dominant orientation
    is re-estimated in the warped image and a SIFT descriptor is appended as an
    extra row for the feature.  Matching against a plain image then covers
    relative tilts up to ~t, i.e. stretch, shear and perspective foreshortening
    that a single upright SIFT patch cannot absorb.  Returns a new set.
    """
    cfg = create_matching_config(**(config or {}))
    views = affine_simulation_views(cfg["affine_tilts"], cfg["affine_angle_step"])
    if not views or len(feature_set) == 0:
        return TerrainFeatureSet([TerrainFeature(**f.to_dict()) for f in feature_set.features],
                                 feature_set.descriptors.copy(), feature_set.image_shape, dict(feature_set.metadata))
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype != np.uint8:
        image = ImagePreprocessor().normalize(image)

    features = [TerrainFeature(**f.to_dict()) for f in feature_set.features]
    idx = [i for i, f in enumerate(features) if f.descriptor_index is not None]
    if not idx:
        return TerrainFeatureSet(features, feature_set.descriptors.copy(), feature_set.image_shape,
                                 dict(feature_set.metadata))
    for i in idx:
        rows = features[i].extra.setdefault("descriptor_indices", [])
        if not rows:
            rows.append(int(features[i].descriptor_index))

    min_size, max_size = cfg["semantic_desc_min_size"], cfg["semantic_desc_max_size"]
    pts = np.array([[features[i].x, features[i].y] for i in idx], dtype=np.float64)
    sizes = np.array([
        _semantic_keypoint(features[i], min_size, max_size).size if features[i].feature_type in SEMANTIC_TYPES
        else float(features[i].scale or min_size) for i in idx
    ])
    h, w = image.shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)

    sift = cv2.SIFT_create()
    new_rows: List[np.ndarray] = []
    row_owner: List[int] = []
    for t, phi in views:
        th = np.radians(phi)
        A = np.array([[1.0, 0.0], [0.0, 1.0 / t]]) @ np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        wc = corners @ A.T
        shift = -wc.min(axis=0)
        out_w, out_h = int(np.ceil(wc.max(axis=0)[0] + shift[0])), int(np.ceil(wc.max(axis=0)[1] + shift[1]))
        M = np.hstack([A, shift[:, None]])
        warped = cv2.warpAffine(image, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        # anti-alias along the squeezed axis as ASIFT does
        warped = cv2.GaussianBlur(warped, (0, 0), sigmaX=0.01, sigmaY=0.8 * np.sqrt(t * t - 1))

        wpts = pts @ A.T + shift
        wsizes = np.clip(sizes / np.sqrt(t), min_size, max_size)
        kps = [cv2.KeyPoint(float(x), float(y), float(s), -1.0) for (x, y), s in zip(wpts, wsizes)]
        angles = dominant_orientations(warped, kps)
        for kp, a in zip(kps, angles):
            kp.angle = float(a)
        kps_out, desc = sift.compute(warped, kps)
        if desc is None or len(kps_out) == 0:
            continue
        if len(kps_out) == len(kps):
            kept = list(range(len(kps)))
        else:
            lookup = {(round(k.pt[0], 2), round(k.pt[1], 2)): j for j, k in enumerate(kps)}
            kept = [lookup[(round(k.pt[0], 2), round(k.pt[1], 2))] for k in kps_out]
        new_rows.append(desc.astype(np.float32))
        row_owner.extend(idx[j] for j in kept)

    if not new_rows:
        return TerrainFeatureSet(features, feature_set.descriptors.copy(), feature_set.image_shape,
                                 dict(feature_set.metadata))
    base = feature_set.descriptors
    extra = np.vstack(new_rows)
    if len(base) and base.shape[1] != extra.shape[1]:
        raise ValueError("simulated descriptor dimension does not match existing descriptors")
    merged = np.vstack([base, extra]) if len(base) else extra
    offset = len(base)
    for row, owner in enumerate(row_owner):
        features[owner].extra["descriptor_indices"].append(offset + row)

    meta = dict(feature_set.metadata)
    meta["affine_views"] = views
    meta["affine_descriptor_rows"] = len(row_owner)
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

        rows_a = [(i, d) for i in ia for d in features_a.descriptor_indices_for(i)]
        rows_b = [(i, d) for i in ib for d in features_b.descriptor_indices_for(i)]
        da = features_a.descriptors[[d for _, d in rows_a]]
        db = features_b.descriptors[[d for _, d in rows_b]]
        owner_a = np.array([i for i, _ in rows_a])
        owner_b = np.array([i for i, _ in rows_b])
        group_cfg = dict(cfg)
        group_cfg["ratio"] = cfg["type_ratio"].get(name, cfg["ratio"])
        gdiag["ratio"] = group_cfg["ratio"]
        gdiag["descriptor_rows_a"], gdiag["descriptor_rows_b"] = len(rows_a), len(rows_b)
        good, stages = match_feature_rows(da, db, owner_a, owner_b, group_cfg)

        group_matches: List[FeatureMatch] = []
        for id_a, (id_b, dist, r) in good.items():
            fa = features_a.features[id_a]
            fb = features_b.features[id_b]
            sr = (fb.scale / fa.scale) if (fa.scale and fb.scale and fa.scale > 0) else None
            group_matches.append(FeatureMatch(
                id_a=int(id_a), id_b=int(id_b),
                x_a=fa.x, y_a=fa.y, x_b=fb.x, y_b=fb.y,
                feature_type=fa.feature_type if fa.feature_type == fb.feature_type
                else f"{fa.feature_type}->{fb.feature_type}",
                distance=float(dist), ratio=float(r), score=float(1.0 - r),
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

def build_homography(
    shape: Tuple[int, ...],
    rotation_deg: float = 0.0,
    scale: float = 1.0,
    stretch: Tuple[float, float] = (1.0, 1.0),
    shear_deg: float = 0.0,
    tilt_deg: Tuple[float, float] = (0.0, 0.0),
    translation: Tuple[float, float] = (0.0, 0.0),
    flip: bool = False,
) -> np.ndarray:
    """3x3 homography about the image centre.

    Composition (applied to a centred point, in order): flip -> anisotropic
    stretch (sx, sy) -> shear -> rotation*scale -> perspective tilt about the
    x / y axes (camera off-nadir look) -> translation.  All angles in degrees.
    """
    h, w = shape[0], shape[1]
    cx, cy = w / 2.0, h / 2.0
    T_in = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)

    F = np.diag([-1.0 if flip else 1.0, 1.0, 1.0])
    S = np.diag([float(stretch[0]), float(stretch[1]), 1.0])
    Sh = np.array([[1, np.tan(np.radians(shear_deg)), 0], [0, 1, 0], [0, 0, 1]])
    th = np.radians(rotation_deg)
    R = scale * np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1 / scale]])

    # perspective: rotate the image plane about x / y, project with focal ~ image size
    f = float(max(h, w))
    ax, ay = np.radians(tilt_deg[0]), np.radians(tilt_deg[1])
    Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    K = np.array([[f, 0, 0], [0, f, 0], [0, 0, 1]])
    K_inv = np.linalg.inv(K)
    P = K @ (Ry @ Rx) @ K_inv

    H0 = P @ R @ Sh @ S @ F @ T_in
    c = H0 @ np.array([cx, cy, 1.0])
    c = c[:2] / c[2]
    T_out = np.array([[1, 0, cx + translation[0] - c[0]],
                      [0, 1, cy + translation[1] - c[1]],
                      [0, 0, 1]], dtype=np.float64)
    H = T_out @ H0
    return H / H[2, 2]


def apply_illumination(
    image: ImageArray,
    gain: float = 1.0,
    offset: float = 0.0,
    gamma: float = 1.0,
    gradient: Tuple[float, float] = (0.0, 0.0),
    shadow_count: int = 0,
    shadow_strength: float = 0.7,
    shadow_radius: Tuple[float, float] = (0.08, 0.25),
    blur_sigma: float = 0.0,
    noise_sigma: float = 0.0,
    seed: int = 0,
) -> ImageArray:
    """Photometric stress: gamma / gain / offset, a directional brightness
    ramp (`gradient` = fractional change across width / height, i.e. sun
    angle), soft multiplicative shadow blobs, blur (resolution loss) and noise.
    """
    rng = np.random.default_rng(seed)
    h, w = image.shape[:2]
    out = image.astype(np.float32) / 255.0
    if gamma != 1.0:
        out = np.power(np.clip(out, 0, 1), gamma)
    out = out * gain + offset / 255.0
    if gradient != (0.0, 0.0):
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        ramp = 1.0 + gradient[0] * (xx / w - 0.5) + gradient[1] * (yy / h - 0.5)
        out = out * ramp
    if shadow_count > 0:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        mask = np.ones((h, w), dtype=np.float32)
        for _ in range(shadow_count):
            cx, cy = rng.uniform(0, w), rng.uniform(0, h)
            rx = rng.uniform(*shadow_radius) * w
            ry = rng.uniform(*shadow_radius) * h
            blob = np.exp(-(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2))
            mask *= 1.0 - shadow_strength * blob
        out = out * mask
    if blur_sigma > 0:
        out = cv2.GaussianBlur(out, (0, 0), blur_sigma)
    if noise_sigma > 0:
        out = out + rng.normal(0.0, noise_sigma / 255.0, out.shape).astype(np.float32)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def make_synthetic_pair(
    image: ImageArray,
    rotation_deg: float = 15.0,
    scale: float = 0.9,
    translation: Tuple[float, float] = (20.0, -10.0),
    stretch: Tuple[float, float] = (1.0, 1.0),
    shear_deg: float = 0.0,
    tilt_deg: Tuple[float, float] = (0.0, 0.0),
    flip: bool = False,
    brightness_gain: float = 1.0,
    brightness_offset: float = 0.0,
    gamma: float = 1.0,
    gradient: Tuple[float, float] = (0.0, 0.0),
    shadow_count: int = 0,
    shadow_strength: float = 0.7,
    blur_sigma: float = 0.0,
    noise_sigma: float = 0.0,
    seed: int = 0,
) -> Tuple[ImageArray, np.ndarray]:
    """Warp `image` with a known geometric transform (similarity, affine
    stretch/shear, perspective tilt) plus photometric stress.

    Returns (image_b, H) where H is the 3x3 homography mapping A -> B pixels.
    """
    h, w = image.shape[:2]
    H = build_homography(image.shape, rotation_deg, scale, stretch, shear_deg, tilt_deg, translation, flip)
    warped = cv2.warpPerspective(image, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    out = apply_illumination(warped, brightness_gain, brightness_offset, gamma, gradient,
                             shadow_count, shadow_strength, blur_sigma=blur_sigma,
                             noise_sigma=noise_sigma, seed=seed)
    return out, H


def apply_transform(points: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Map (N,2) points through a 2x3 affine or 3x3 homography."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    ones = np.ones((len(pts), 1))
    hom = np.hstack([pts, ones])
    if M.shape[0] == 2:
        return (hom @ M.T).astype(np.float32)
    proj = hom @ M.T
    return (proj[:, :2] / proj[:, 2:3]).astype(np.float32)


apply_affine = apply_transform


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


STRESS_CASES: Dict[str, Dict[str, Any]] = {
    "baseline":        dict(rotation_deg=25, scale=0.8, translation=(30, -20), brightness_gain=1.25, gamma=0.75, noise_sigma=5),
    "rot_90":          dict(rotation_deg=90),
    "rot_170":         dict(rotation_deg=170, scale=0.95),
    "scale_0.5":       dict(rotation_deg=10, scale=0.5),
    "scale_1.8":       dict(rotation_deg=-15, scale=1.8),
    "stretch_x1.5":    dict(rotation_deg=10, stretch=(1.5, 1.0)),
    "stretch_y0.6":    dict(rotation_deg=-30, stretch=(1.0, 0.6)),
    "shear_25":        dict(rotation_deg=-10, shear_deg=25),
    "tilt_35":         dict(rotation_deg=20, tilt_deg=(35, 0)),
    "tilt_both_25":    dict(rotation_deg=45, scale=0.85, tilt_deg=(25, 25)),
    "sun_gradient":    dict(rotation_deg=30, gradient=(1.2, -0.6), brightness_gain=0.7, gamma=1.6),
    "shadows":         dict(rotation_deg=-40, shadow_count=6, shadow_strength=0.85, brightness_gain=1.4, gamma=0.6),
    "low_light_noise": dict(rotation_deg=60, scale=0.9, brightness_gain=0.35, noise_sigma=12),
    "overexposed":     dict(rotation_deg=-75, brightness_gain=1.3, brightness_offset=20, gamma=0.7),
    "blur_lowres":     dict(rotation_deg=15, scale=0.7, blur_sigma=2.0, noise_sigma=6),
    "everything":      dict(rotation_deg=130, scale=0.7, stretch=(1.3, 0.9), shear_deg=12, tilt_deg=(20, -15),
                            gradient=(0.8, 0.4), shadow_count=4, brightness_gain=1.3, gamma=0.7,
                            blur_sigma=1.0, noise_sigma=8),
}


def run_stress_suite(
    image: ImageArray,
    extractor: Optional[TerrainFeatureExtractor] = None,
    cases: Optional[Dict[str, Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None,
    tolerance_px: float = 4.0,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Match `image` against warped/relit copies of itself under each stress
    case and score against the exact ground-truth homography.  Reports, per
    case: candidates, precision, median error, per-type correct counts and
    detector repeatability (the upper bound Module 3 allows)."""
    extractor = extractor or TerrainFeatureExtractor()
    cases = cases or STRESS_CASES
    set_a, _ = extract_feature_set(image, extractor, config)
    report: Dict[str, Dict[str, Any]] = {}
    if verbose:
        print(f"{'case':17s} {'n':>5s} {'prec':>6s} {'med_px':>7s}  per-type correct/n (repeatable)")
    for name, params in cases.items():
        img_b, H = make_synthetic_pair(image, **params)
        set_b, _ = extract_feature_set(img_b, extractor, config)
        result = match_terrain_features(set_a, set_b, config)
        ev = evaluate_against_ground_truth(result, H, tolerance_px)
        rep = compute_repeatability(set_a, set_b, H, tolerance_px)
        report[name] = {"params": params, "n": ev["n"], "correct": ev["correct"], "precision": ev["precision"],
                        "median_error_px": ev["median_error_px"], "by_type": ev["by_type"], "repeatability": rep}
        if verbose:
            per = " ".join(f"{t}:{ev['by_type'].get(t, {}).get('correct', 0)}/{ev['by_type'].get(t, {}).get('n', 0)}"
                           f"({rep[t]['repeatable']})" for t in FEATURE_TYPES)
            med = f"{ev['median_error_px']:.2f}" if ev["median_error_px"] is not None else "  n/a"
            print(f"{name:17s} {ev['n']:5d} {ev['precision']:6.3f} {med:>7s}  {per}")
    return report


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

    rs = root_sift(np.abs(base) * 100)
    check("root_sift is unit L2", np.allclose(np.linalg.norm(rs, axis=1), 1.0, atol=1e-4))

    # owner-aware ratio: a feature competing against its own second scale must survive
    owner_b = np.array([0, 0, 1])
    knn_own = [[cv2.DMatch(0, 0, 1.0), cv2.DMatch(0, 1, 1.05), cv2.DMatch(0, 2, 3.0)]]
    check("owner-aware ratio ignores same-feature rows", 0 in _owner_ratio_test(knn_own, owner_b, 0.75))
    check("plain ratio would reject it", ratio_test(knn_own, 0.75) == [])

    # multi-row matching collapses to one match per feature, one-to-one
    da = np.vstack([base[:50], base[:50] + 0.01])            # 2 rows per feature in A
    db = np.vstack([base[:50] + 0.005, base[:50] - 0.005])   # 2 rows per feature in B
    fwd, _ = match_feature_rows(da, db, np.tile(np.arange(50), 2), np.tile(np.arange(50), 2),
                                {"root_sift": False})
    check("multi-row matching recovers identity", len(fwd) >= 48 and all(v[0] == a for a, v in fwd.items()))

    # homography helpers: centre stays fixed, translation applied, perspective is non-affine
    H = build_homography((200, 300), rotation_deg=40, scale=0.7, stretch=(1.3, 0.8), shear_deg=15,
                         tilt_deg=(30, -20), translation=(7, -3))
    centre = apply_transform(np.array([[150.0, 100.0]]), H)[0]
    check("homography keeps centre (+translation)", np.allclose(centre, [157.0, 97.0], atol=1e-6))
    check("homography is perspective", abs(H[2, 0]) + abs(H[2, 1]) > 1e-6)

    # --- 2. full terrain pipeline on a synthetic image pair ---------------
    img_a = _synthetic_terrain_image()
    img_b, M = make_synthetic_pair(img_a, rotation_deg=20, scale=0.85, translation=(15, -8),
                                   brightness_gain=1.3, gamma=0.8, noise_sigma=4)
    extractor = TerrainFeatureExtractor(onnx_crater_model_path="__missing__.onnx", max_total_features=None)
    set_a, ea = extract_feature_set(img_a, extractor)
    set_b, eb = extract_feature_set(img_b, extractor)
    check("semantic descriptors attached", all(f.descriptor_index is not None for f in set_a.features))
    n_views = len(affine_simulation_views([2.0], 120.0))
    check("multi-scale + affine rows attached",
          all(len(set_a.descriptor_indices_for(i)) >= 1 + n_views for i in range(len(set_a))))
    plain, _ = extract_feature_set(img_a, extractor, {"affine_tilts": [], "semantic_scales": [1.0]})
    check("affine/multi-scale can be disabled",
          all(len(plain.descriptor_indices_for(i)) == 1 for i in range(len(plain))))

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

    # --- 3. stress: twist / stretch / perspective / harsh lighting -------------
    stress = {k: STRESS_CASES[k] for k in ("rot_170", "stretch_y0.6", "shear_25", "tilt_both_25",
                                            "shadows", "low_light_noise", "everything")}
    report = run_stress_suite(img_a, extractor, stress, verbose=verbose)
    for name, r in report.items():
        check(f"stress {name}: >=20 candidates, precision >=0.6 (n={r['n']}, p={r['precision']:.2f})",
              r["n"] >= 20 and r["precision"] >= 0.6)

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


def photometric_normalize(
    image: ImageArray,
    low_pct: float = 0.5,
    high_pct: float = 99.5,
) -> ImageArray:
    """Make exposure-mismatched frames comparable before feature extraction:
    stretch the [low_pct, high_pct] intensity range to 0..255 so under-exposed
    and washed-out frames reach the detectors with a usable dynamic range.
    Percentiles ignore zero-fill borders and saturated pixels.  Returns uint8."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = image.astype(np.float32)
    valid = img[(img > 0) & (img < 255)]
    if valid.size < 0.05 * img.size:
        valid = img.ravel()
    lo, hi = np.percentile(valid, [low_pct, high_pct])
    if hi - lo < 1e-6:
        return ImagePreprocessor().normalize(image)
    return (np.clip((img - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def extract_feature_set(
    image: ImageArray,
    extractor: Optional[TerrainFeatureExtractor] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[TerrainFeatureSet, ImageArray]:
    """Convenience: Module-3 extraction on an array + semantic descriptors
    (+ optional photometric normalisation, multi-scale and affine-simulated
    descriptor rows).  Returns (feature_set, enhanced_image)."""
    cfg = create_matching_config(**(config or {}))
    extractor = extractor or TerrainFeatureExtractor()
    if cfg["photometric_normalize"]:
        image = photometric_normalize(image)
    enhanced = extractor.preprocessor.enhance(extractor.preprocessor.normalize(image))
    feats, desc = _extract_from_array(extractor, image)
    fs = attach_descriptors(TerrainFeatureSet(feats, desc, enhanced.shape), enhanced, config)
    fs = simulate_affine_descriptors(fs, enhanced, config)
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
        "  python module_4.py stress <image> [output_json]\n"
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
    if cmd == "stress":
        if len(args) < 2:
            print(usage)
            return 1
        pre = ImagePreprocessor()
        image = pre.normalize(pre.load(args[1]))
        extractor = TerrainFeatureExtractor(onnx_crater_model_path="models/crater_unet.onnx", max_total_features=3000)
        report = run_stress_suite(image, extractor)
        if len(args) > 2:
            Path(args[2]).parent.mkdir(parents=True, exist_ok=True)
            with open(args[2], "w") as fh:
                json.dump(report, fh, indent=2, default=_json_default)
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
