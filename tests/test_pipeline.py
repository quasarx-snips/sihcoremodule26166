"""Portable synthetic checks for the complete LunaX pipeline."""
import cv2
import numpy as np

from lunax.preprocessing import ImagePreprocessor
from lunax.pipeline import run_lunax_from_arrays
from lunax.geometry import GeometricVerificationConfig, verify_matches


def test_preprocessing_handles_float_and_colour():
    image = np.dstack([np.linspace(0, 1, 64, dtype=np.float32)] * 3)
    normalized = ImagePreprocessor().normalize(image)
    assert normalized.dtype == np.uint8 and normalized.shape == (1, 64)
    assert ImagePreprocessor().enhance(normalized).shape == normalized.shape


def test_ransac_rejects_injected_outliers():
    rng = np.random.default_rng(2)
    source = rng.uniform(10, 190, (30, 2)); reference = source + [7, -4]
    reference[-8:] = rng.uniform(0, 200, (8, 2))
    result = verify_matches(source, reference, GeometricVerificationConfig(model="translation", reprojection_threshold=.1, min_inliers=15, random_seed=3))
    assert result.is_valid and result.inlier_mask.sum() == 22


def test_end_to_end_known_similarity_transform():
    rng = np.random.default_rng(9)
    source = cv2.GaussianBlur(rng.integers(0, 255, (260, 300), dtype=np.uint8), (0, 0), 1.1)
    for x, y in rng.integers([20, 20], [280, 240], (25, 2)):
        cv2.circle(source, (int(x), int(y)), 5, 230, 1)
    transform = np.array([[.99, -.04, 12.], [.04, .99, -8.], [0., 0., 1.]])
    reference = cv2.warpAffine(source, transform[:2], (300, 260))
    result = run_lunax_from_arrays(source, reference, {"verbose": False, "use_terrain_landmarks": False})
    assert result.success, result.error
    assert result.metrics["verified_inliers"] >= 4
    assert result.metrics["inlier_error_statistics"]["rmse"] < 3
