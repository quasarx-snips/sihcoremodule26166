"""Synthetic tests for Module 4 (feature / descriptor matching)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from module_3 import TerrainFeature, TerrainFeatureExtractor  # noqa: E402
from module_4 import (  # noqa: E402
    MatchResult,
    TerrainFeatureSet,
    _synthetic_descriptors,
    _synthetic_terrain_image,
    compute_repeatability,
    create_matching_config,
    dominant_orientations,
    enforce_unique_targets,
    evaluate_against_ground_truth,
    extract_feature_set,
    make_synthetic_pair,
    match_descriptors,
    match_terrain_features,
    mutual_consistency,
    ratio_test,
    visualize_matches,
)


# ----------------------------------------------------------------------------
# descriptor primitives
# ----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def desc_pair():
    base = _synthetic_descriptors(200, seed=0)
    perm = np.random.default_rng(3).permutation(200)
    noisy = base[perm] + 0.02 * np.random.default_rng(4).normal(size=base.shape).astype(np.float32)
    return base, noisy, perm


@pytest.mark.parametrize("method", ["BF", "FLANN"])
def test_match_descriptors_knn_shape(desc_pair, method):
    base, noisy, _ = desc_pair
    knn = match_descriptors(base, noisy, method=method, k=2)
    assert len(knn) == 200
    assert all(len(g) == 2 and g[0].distance <= g[1].distance for g in knn)


def test_match_descriptors_empty_and_errors(desc_pair):
    base, _, _ = desc_pair
    assert match_descriptors(np.empty((0, 128), np.float32), base) == []
    assert match_descriptors(base, None) == []
    with pytest.raises(ValueError):
        match_descriptors(base, np.zeros((5, 64), np.float32))
    with pytest.raises(ValueError):
        match_descriptors(base, np.zeros((5, 128), np.uint8))


def test_binary_descriptors_use_hamming():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, size=(50, 32), dtype=np.uint8)
    b = a.copy()
    b[:, 0] ^= 1  # flip one bit per row
    knn = match_descriptors(a, b, "BF", k=1)
    assert all(g[0].trainIdx == i and g[0].distance == 1.0 for i, g in enumerate(knn))


def test_ratio_test_recovers_permutation(desc_pair):
    base, noisy, perm = desc_pair
    inv = np.argsort(perm)
    good = ratio_test(match_descriptors(base, noisy), ratio=0.75)
    assert len(good) >= 190
    assert all(inv[m.queryIdx] == m.trainIdx for m in good)


def test_ratio_test_rejects_ambiguous():
    d = np.ones((3, 8), np.float32)
    knn = match_descriptors(d, d, k=2)  # every neighbour is identical
    assert ratio_test(knn) == []


def test_ratio_test_strictness_monotonic(desc_pair):
    base, noisy, _ = desc_pair
    knn = match_descriptors(base, noisy)
    assert len(ratio_test(knn, 0.5)) <= len(ratio_test(knn, 0.75)) <= len(ratio_test(knn, 0.95))


def test_mutual_consistency_and_unique(desc_pair):
    base, noisy, _ = desc_pair
    distract = np.vstack([noisy, _synthetic_descriptors(50, seed=9)])
    ab = ratio_test(match_descriptors(base, distract))
    ba = ratio_test(match_descriptors(distract, base))
    mutual = mutual_consistency(ab, ba)
    assert 180 <= len(mutual) <= len(ab)
    assert all(m.trainIdx < 200 for m in mutual)
    uniq = enforce_unique_targets(mutual)
    assert len({m.trainIdx for m in uniq}) == len(uniq)


def test_mutual_consistency_rejects_non_reciprocal():
    import cv2
    ab = [cv2.DMatch(0, 5, 0.1), cv2.DMatch(1, 6, 0.1)]
    ba = [cv2.DMatch(5, 0, 0.1), cv2.DMatch(6, 3, 0.1)]
    out = mutual_consistency(ab, ba)
    assert [(m.queryIdx, m.trainIdx) for m in out] == [(0, 5)]


# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------

def test_config_validation():
    cfg = create_matching_config(ratio=0.8, matcher="FLANN")
    assert cfg["ratio"] == 0.8 and cfg["matcher"] == "FLANN"
    with pytest.raises(KeyError):
        create_matching_config(not_a_key=1)
    with pytest.raises(ValueError):
        create_matching_config(ratio=1.5)
    with pytest.raises(ValueError):
        create_matching_config(matcher="ANN")


# ----------------------------------------------------------------------------
# terrain feature sets
# ----------------------------------------------------------------------------

def _tiny_set():
    feats = [
        TerrainFeature("sift", 10, 10, 4.0, 0.0, 0.5, 0),
        TerrainFeature("crater", 50, 50, 6.0, None, 0.9),
    ]
    return TerrainFeatureSet(feats, np.zeros((1, 128), np.float32), (100, 100))


def test_feature_set_validation():
    fs = _tiny_set()
    assert len(fs) == 2 and fs.count_by_type()["crater"] == 1
    assert fs.has_descriptor(0) and not fs.has_descriptor(1)
    with pytest.raises(ValueError):
        TerrainFeatureSet([TerrainFeature("sift", 0, 0, descriptor_index=3)], np.zeros((1, 128)), (10, 10))


def test_feature_set_from_module3_outputs(tmp_path):
    from module_3 import FeatureStore
    fs = _tiny_set()
    FeatureStore.save_features(tmp_path / "f.json", fs.features, fs.image_shape)
    FeatureStore.save_descriptors(tmp_path / "d.npy", fs.descriptors)
    loaded = TerrainFeatureSet.from_module3_outputs(tmp_path / "f.json", tmp_path / "d.npy")
    assert len(loaded) == 2 and loaded.image_shape == (100, 100)
    assert loaded.features[1].feature_type == "crater"


def test_dominant_orientation_rotates_with_image():
    import cv2
    img = np.zeros((256, 256), np.uint8)
    img[:, 128:] = 200  # vertical step edge -> gradient points along +x (0 deg)
    img = cv2.GaussianBlur(img, (0, 0), 3)
    kp = [cv2.KeyPoint(128.0, 128.0, 20.0, -1)]
    a0 = dominant_orientations(img, kp)[0]
    assert min(a0, 360 - a0) < 10
    rot = cv2.warpAffine(img, cv2.getRotationMatrix2D((128, 128), 40, 1.0), (256, 256),
                         borderMode=cv2.BORDER_REPLICATE)
    a1 = dominant_orientations(rot, kp)[0]
    diff = (a0 - a1 + 180) % 360 - 180
    assert abs(abs(diff) - 40) < 10


# ----------------------------------------------------------------------------
# end-to-end on a synthetic terrain pair (no ONNX model needed)
# ----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_sets():
    img_a = _synthetic_terrain_image()
    img_b, M = make_synthetic_pair(img_a, rotation_deg=20, scale=0.85, translation=(15, -8),
                                   brightness_gain=1.3, gamma=0.8, noise_sigma=4)
    extractor = TerrainFeatureExtractor(onnx_crater_model_path="__missing__.onnx", max_total_features=None)
    set_a, enh_a = extract_feature_set(img_a, extractor)
    set_b, enh_b = extract_feature_set(img_b, extractor)
    return set_a, set_b, enh_a, enh_b, M


def test_attach_descriptors_covers_semantic_features(synthetic_sets):
    set_a, *_ = synthetic_sets
    assert all(f.descriptor_index is not None for f in set_a.features)
    assert set_a.descriptors.shape[0] >= len(set_a.features)
    # SIFT rows keep their original indices
    sift_rows = [f.descriptor_index for f in set_a.features if f.feature_type == "sift"]
    assert sift_rows == list(range(len(sift_rows)))


def test_match_terrain_features_ground_truth(synthetic_sets):
    set_a, set_b, _, _, M = synthetic_sets
    result = match_terrain_features(set_a, set_b)
    ev = evaluate_against_ground_truth(result, M, tolerance_px=4.0)
    assert len(result) >= 30
    assert ev["precision"] >= 0.6
    assert ev["median_error_px"] < 2.0
    assert all("->" not in m.feature_type for m in result.matches)
    assert len({m.id_b for m in result.matches}) == len(result)
    assert result.diagnostics["total_matches"] == len(result)
    assert set(result.diagnostics["groups"]) == {"crater", "ridge", "texture", "sift"}
    assert result.points_a.shape == result.points_b.shape == (len(result), 2)


def test_scale_ratio_tracks_synthetic_scale(synthetic_sets):
    set_a, set_b, _, _, _ = synthetic_sets
    result = match_terrain_features(set_a, set_b, {"match_types": ["sift"]})
    med = result.diagnostics["median_scale_ratio"]
    assert med is not None and 0.7 < med < 1.0  # true scale 0.85


def test_config_filters(synthetic_sets):
    set_a, set_b, *_ = synthetic_sets
    strict = match_terrain_features(set_a, set_b)
    loose = match_terrain_features(set_a, set_b, {"mutual": False, "scale_constraint": False})
    assert len(loose) >= len(strict)
    sift_only = match_terrain_features(set_a, set_b, {"match_types": ["sift"]})
    assert sift_only.matches and all(m.feature_type == "sift" for m in sift_only.matches)
    capped = match_terrain_features(set_a, set_b, {"max_matches": 10})
    assert len(capped) == 10 and capped.scores.tolist() == sorted(capped.scores, reverse=True)
    untyped = match_terrain_features(set_a, set_b, {"type_constraint": False})
    assert list(untyped.diagnostics["groups"]) == ["all"]


def test_repeatability_reports_upper_bound(synthetic_sets):
    set_a, set_b, _, _, M = synthetic_sets
    rep = compute_repeatability(set_a, set_b, M, tolerance_px=4.0)
    assert rep["sift"]["repeatable"] > 0
    result = match_terrain_features(set_a, set_b, {"match_types": ["sift"]})
    ev = evaluate_against_ground_truth(result, M, 4.0)
    assert ev["correct"] <= rep["sift"]["repeatable"]


def test_match_result_json_roundtrip(synthetic_sets, tmp_path):
    set_a, set_b, *_ = synthetic_sets
    result = match_terrain_features(set_a, set_b)
    path = tmp_path / "m.json"
    result.save(path)
    loaded = MatchResult.load(path)
    assert len(loaded) == len(result)
    assert loaded.matches[0] == result.matches[0]
    assert loaded.image_shape_a == result.image_shape_a


def test_visualize_matches_returns_figure(synthetic_sets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    set_a, set_b, enh_a, enh_b, M = synthetic_sets
    result = match_terrain_features(set_a, set_b)
    fig = visualize_matches(enh_a, enh_b, result, max_draw=50)
    assert fig is not None and len(fig.axes) == 1
    plt.close(fig)
    mask = np.zeros(len(result), bool)
    fig = visualize_matches(enh_a, enh_b, result, correct_mask=mask)
    plt.close(fig)


def test_empty_sets():
    empty = TerrainFeatureSet([], np.empty((0, 128), np.float32), (10, 10))
    result = match_terrain_features(empty, empty)
    assert len(result) == 0
    assert evaluate_against_ground_truth(result, np.eye(2, 3))["n"] == 0
