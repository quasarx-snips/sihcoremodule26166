"""Regression tests for Module 05 geometric verification."""

import numpy as np

from lunax.geometry import estimate_geometry


def test_auto_prefers_similarity_for_rotation_zoom_with_outlier():
    source = np.array([[0, 0], [30, 0], [0, 40], [30, 40], [15, 20], [50, 10]], dtype=np.float64)
    angle = np.deg2rad(12.0)
    scale = 0.85
    transform = np.array([
        [scale * np.cos(angle), -scale * np.sin(angle), 8.0],
        [scale * np.sin(angle),  scale * np.cos(angle), -5.0],
        [0.0, 0.0, 1.0],
    ])
    reference = (np.c_[source, np.ones(len(source))] @ transform.T)[:, :2]
    reference[-1] = [300.0, -100.0]  # Descriptor outlier.

    estimated, inliers, diagnostics = estimate_geometry(
        source, reference, config={"model": "auto", "reprojection_threshold": 1.0, "min_inliers": 4, "random_seed": 7}
    )

    assert diagnostics.is_valid
    assert diagnostics.model_name == "similarity"
    assert inliers.sum() == 5
    assert np.allclose(estimated, transform, atol=1e-6)
