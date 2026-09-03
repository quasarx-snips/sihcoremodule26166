"""LunaX Module 07: image registration and warping.

Transforms always follow the Module 06 convention: a homogeneous matrix maps
SOURCE coordinates to REFERENCE coordinates.  This module deliberately only
warps and provides local visual diagnostics; it does not score global accuracy.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import cv2
import numpy as np


def _matrix_3x3(transformation: Any) -> np.ndarray:
    """Return a finite source-to-reference homogeneous transformation."""
    matrix = np.asarray(transformation, dtype=np.float64)
    if matrix.shape == (2, 3):
        matrix = np.vstack((matrix, (0.0, 0.0, 1.0)))
    if matrix.shape != (3, 3):
        raise ValueError("transformation must have shape (2, 3) or (3, 3)")
    if not np.all(np.isfinite(matrix)) or abs(matrix[2, 2]) < 1e-12:
        raise ValueError("transformation must be finite and non-degenerate")
    return matrix / matrix[2, 2]


def _output_size(output_shape: Any) -> Tuple[int, int]:
    if len(output_shape) < 2:
        raise ValueError("output_shape must provide at least (height, width)")
    height, width = int(output_shape[0]), int(output_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("output_shape dimensions must be positive")
    return width, height  # OpenCV uses (width, height)


def _is_projective(matrix: np.ndarray, model: str) -> bool:
    normalized = model.lower()
    if normalized not in {"auto", "translation", "similarity", "affine", "projective", "homography"}:
        raise ValueError("model must be auto, translation, similarity, affine, projective, or homography")
    if normalized in {"projective", "homography"}:
        return True
    if normalized in {"translation", "similarity", "affine"}:
        return False
    return not np.allclose(matrix[2], (0.0, 0.0, 1.0), rtol=0.0, atol=1e-10)


def _cv_image(image: np.ndarray) -> Tuple[np.ndarray, np.dtype]:
    """Convert unsupported integer/bool arrays safely for OpenCV warping."""
    array = np.asarray(image)
    if array.ndim not in (2, 3) or array.size == 0:
        raise ValueError("image must be a non-empty grayscale or multi-channel array")
    if array.dtype in (np.dtype("uint8"), np.dtype("uint16"), np.dtype("float32")):
        return array, array.dtype
    return array.astype(np.float32), array.dtype


def _restore_dtype(image: np.ndarray, original_dtype: np.dtype) -> np.ndarray:
    if original_dtype == image.dtype:
        return image
    if np.issubdtype(original_dtype, np.bool_):
        return image > 0.5
    if np.issubdtype(original_dtype, np.integer):
        limits = np.iinfo(original_dtype)
        return np.clip(np.rint(image), limits.min, limits.max).astype(original_dtype)
    return image.astype(original_dtype)


def warp_image(source, transformation, output_shape, model="auto"):
    """Warp SOURCE into the reference coordinate frame of ``output_shape``.

    ``transformation`` maps source pixels to destination/reference pixels.
    Affine matrices use ``cv2.warpAffine``; projective matrices use
    ``cv2.warpPerspective``. Output dimensions are exactly (height, width).
    """
    matrix = _matrix_3x3(transformation)
    source_cv, original_dtype = _cv_image(source)
    size = _output_size(output_shape)
    interpolation = cv2.INTER_NEAREST if source_cv.dtype == np.bool_ else cv2.INTER_LINEAR
    if _is_projective(matrix, model):
        warped = cv2.warpPerspective(source_cv, matrix, size, flags=interpolation,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    else:
        warped = cv2.warpAffine(source_cv, matrix[:2], size, flags=interpolation,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return _restore_dtype(warped, original_dtype)


def _to_float(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if np.issubdtype(array.dtype, np.integer):
        return array.astype(np.float32) / max(float(np.iinfo(array.dtype).max), 1.0)
    return array.astype(np.float32)


def _match_channels(reference: np.ndarray, source: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a gray/RGB pair to compatible channels without changing size."""
    ref, src = np.asarray(reference), np.asarray(source)
    if ref.shape[:2] != src.shape[:2]:
        raise ValueError("reference and registered_source must have the same spatial shape")
    if ref.ndim == src.ndim and ref.shape == src.shape:
        return ref, src
    if ref.ndim == 2 and src.ndim == 3:
        return cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR), src
    if ref.ndim == 3 and src.ndim == 2:
        return ref, cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    raise ValueError("images must be grayscale or multi-channel arrays")


def create_overlay(reference, registered_source, alpha=0.5):
    """Return a dtype-safe alpha overlay in the reference coordinate system."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    ref, reg = _match_channels(np.asarray(reference), np.asarray(registered_source))
    blended = (1.0 - alpha) * _to_float(ref) + alpha * _to_float(reg)
    if np.issubdtype(ref.dtype, np.integer):
        return np.clip(np.rint(blended * np.iinfo(ref.dtype).max), 0,
                       np.iinfo(ref.dtype).max).astype(ref.dtype)
    return blended.astype(ref.dtype)


def create_difference_map(reference, registered_source):
    """Return a uint8 absolute-difference visualization (black = agreement)."""
    ref, reg = _match_channels(np.asarray(reference), np.asarray(registered_source))
    difference = np.abs(_to_float(ref) - _to_float(reg))
    if difference.ndim == 3:
        difference = np.mean(difference, axis=2)
    return np.clip(np.rint(difference * 255.0), 0, 255).astype(np.uint8)


def registration_visualization(reference, registered_source):
    """Create a compact 2x2 visual: reference, registration, overlay, difference."""
    ref, reg = _match_channels(np.asarray(reference), np.asarray(registered_source))
    overlay = create_overlay(ref, reg)
    difference = create_difference_map(ref, reg)

    def color(image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image

    panels = [color(ref), color(reg), color(overlay), cv2.cvtColor(difference, cv2.COLOR_GRAY2BGR)]
    panels = [np.clip(np.rint(_to_float(panel) * 255.0), 0, 255).astype(np.uint8)
              if panel.dtype != np.uint8 else panel for panel in panels]
    return np.vstack((np.hstack((panels[0], panels[1])), np.hstack((panels[2], panels[3]))))


def register_image(source, reference, transformation, model="auto"):
    """Register SOURCE to REFERENCE and return local visual diagnostics.

    The metadata intentionally contains only warp settings and diagnostic images,
    not a final global accuracy score.
    """
    reference_array = np.asarray(reference)
    registered = warp_image(source, transformation, reference_array.shape, model=model)
    matrix = _matrix_3x3(transformation)
    resolved_model = "projective" if _is_projective(matrix, model) else "affine"
    metadata: Dict[str, Any] = {
        "model": resolved_model,
        "coordinate_convention": "source_to_reference",
        "transformation": matrix.tolist(),
        "source_shape": tuple(np.asarray(source).shape),
        "reference_shape": tuple(reference_array.shape),
        "output_shape": tuple(registered.shape),
        "overlay": create_overlay(reference_array, registered),
        "difference_map": create_difference_map(reference_array, registered),
        "visualization": registration_visualization(reference_array, registered),
    }
    return registered, metadata


def test_synthetic_registration() -> Dict[str, Any]:
    """Synthetic known-transform check, usable from a VS Code terminal."""
    source = np.zeros((160, 200), dtype=np.uint8)
    cv2.circle(source, (55, 60), 22, 220, -1)
    cv2.rectangle(source, (120, 85), (175, 125), 150, -1)
    transform = np.array([[1.0, 0.08, 18.0], [-0.04, 1.0, 22.0],
                          [0.00025, -0.00015, 1.0]], dtype=np.float64)
    reference = warp_image(source, transform, source.shape, model="projective")
    registered, metadata = register_image(source, reference, transform, model="auto")
    # Same warp settings and a known transform should reproduce the reference.
    max_difference = int(np.max(create_difference_map(reference, registered)))
    assert max_difference == 0, f"Known-transform recovery failed (max diff {max_difference})"
    metadata["synthetic_max_difference"] = max_difference
    return metadata


if __name__ == "__main__":
    result = test_synthetic_registration()
    print("Synthetic registration passed; max difference:", result["synthetic_max_difference"])
