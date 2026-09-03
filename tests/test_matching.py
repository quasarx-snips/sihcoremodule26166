"""Synthetic and image-backed tests for LunaX Module 04."""
from types import SimpleNamespace

import cv2
import numpy as np

from lunax.matching import (match_feature_sets, match_descriptors, matches_to_correspondences,
                      ratio_test, visualize_matches)


def test_synthetic_matching_retains_indices_and_points():
    rng = np.random.default_rng(4)
    descriptors = rng.normal(size=(12, 128)).astype(np.float32)
    source = SimpleNamespace(keypoints=[(float(i), float(i + 1)) for i in range(12)], descriptors=descriptors)
    reference = SimpleNamespace(keypoints=[(float(i + 10), float(i + 20)) for i in range(12)], descriptors=descriptors.copy())

    result = match_feature_sets(source, reference, {"ratio": 0.8, "mutual_consistency": True})

    assert result.number_raw_matches == 12
    assert result.number_filtered_matches == 12
    assert result.source_points.dtype == np.float32
    assert np.array_equal(result.source_points, np.asarray(source.keypoints, np.float32))
    assert [r["source_index"] for r in result.match_records] == list(range(12))
    assert [r["reference_index"] for r in result.match_records] == list(range(12))


def test_empty_descriptors_and_ratio_with_one_neighbour_are_safe():
    assert match_descriptors(None, np.empty((2, 128), np.float32)) == []
    assert ratio_test([]) == []
    source, reference, records = matches_to_correspondences([], [], [])
    assert source.shape == reference.shape == (0, 2)
    assert records == []


def test_module_3_feature_tuple_uses_descriptor_indexed_keypoints_only():
    descriptors = np.eye(3, dtype=np.float32)
    # Module 03 returns all terrain records first, then SIFT records carrying
    # descriptor_index values. Matching must not use the terrain records.
    source_features = [SimpleNamespace(x=-1.0, y=-1.0, descriptor_index=None),
                       SimpleNamespace(x=10.0, y=20.0, descriptor_index=0),
                       SimpleNamespace(x=30.0, y=40.0, descriptor_index=1),
                       SimpleNamespace(x=50.0, y=60.0, descriptor_index=2)]
    reference_features = [SimpleNamespace(x=-2.0, y=-2.0, descriptor_index=None),
                          SimpleNamespace(x=15.0, y=25.0, descriptor_index=0),
                          SimpleNamespace(x=35.0, y=45.0, descriptor_index=1),
                          SimpleNamespace(x=55.0, y=65.0, descriptor_index=2)]
    result = match_feature_sets((source_features, descriptors), (reference_features, descriptors), {"ratio": 0.8})

    assert np.array_equal(result.source_points, np.array([[10, 20], [30, 40], [50, 60]], np.float32))
    assert np.array_equal(result.reference_points, np.array([[15, 25], [35, 45], [55, 65]], np.float32))


def test_real_image_sift_matching_and_visualization():
    image = np.zeros((240, 240), np.uint8)
    cv2.circle(image, (80, 80), 25, 255, 3)
    cv2.rectangle(image, (130, 120), (190, 180), 180, 3)
    cv2.line(image, (30, 200), (205, 35), 220, 3)
    transformed = cv2.warpAffine(image, np.float32([[1, 0, 8], [0, 1, 5]]), (240, 240))
    sift = cv2.SIFT_create()
    source_kp, source_desc = sift.detectAndCompute(image, None)
    reference_kp, reference_desc = sift.detectAndCompute(transformed, None)

    result = match_feature_sets(SimpleNamespace(keypoints=source_kp, descriptors=source_desc),
                                SimpleNamespace(keypoints=reference_kp, descriptors=reference_desc),
                                {"method": "FLANN", "ratio": 0.8})
    visualization = visualize_matches(image, transformed, result)

    assert result.number_raw_matches > 0
    assert result.number_filtered_matches > 0
    assert visualization.shape == (240, 480, 3)
