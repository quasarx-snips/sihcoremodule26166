"""LunaX Module 09: deterministic spatially distributed match selection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class SpatialGrid:
    image_shape: tuple
    rows: int
    cols: int
    cell_height: float
    cell_width: float


@dataclass
class SpatialSelectionConfig:
    rows: int = 4
    cols: int = 4
    max_matches_per_cell: int = 10


def create_spatial_grid(image_shape, rows=4, cols=4):
    """Create a source-image grid; image_shape follows numpy's (height, width, ...)."""
    if len(image_shape) < 2 or rows <= 0 or cols <= 0:
        raise ValueError("image_shape must provide height/width and rows/cols must be positive")
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    return SpatialGrid((height, width), int(rows), int(cols), height / rows, width / cols)


def assign_points_to_grid(points, grid):
    """Return (row, col) per point, or (-1, -1) for points outside the image."""
    if not isinstance(grid, SpatialGrid):
        raise TypeError("grid must be created by create_spatial_grid")
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    cells = np.full((len(array), 2), -1, dtype=np.int32)
    height, width = grid.image_shape
    valid = np.isfinite(array).all(axis=1) & (array[:, 0] >= 0) & (array[:, 0] < width) & (array[:, 1] >= 0) & (array[:, 1] < height)
    cells[valid, 0] = np.minimum((array[valid, 1] / grid.cell_height).astype(int), grid.rows - 1)
    cells[valid, 1] = np.minimum((array[valid, 0] / grid.cell_width).astype(int), grid.cols - 1)
    return cells


def calculate_spatial_coverage(points, image_shape):
    """Return occupied-cell coverage and a row-major count matrix (default 4x4 grid)."""
    grid = create_spatial_grid(image_shape)
    cells = assign_points_to_grid(points, grid)
    counts = np.zeros((grid.rows, grid.cols), dtype=np.int32)
    for row, col in cells:
        if row >= 0:
            counts[row, col] += 1
    occupied = int(np.count_nonzero(counts))
    return {"grid_rows": grid.rows, "grid_cols": grid.cols, "occupied_cells": occupied,
            "total_cells": grid.rows * grid.cols, "coverage_percentage": 100.0 * occupied / counts.size,
            "points_per_cell": counts, "outside_image_count": int(np.sum(cells[:, 0] < 0))}


def select_spatially_distributed_matches(source_points, reference_points, scores, image_shape, config=None):
    """Keep highest score matches per source-grid cell, retaining original array indices."""
    cfg = SpatialSelectionConfig(**config) if isinstance(config, dict) else (config or SpatialSelectionConfig())
    if cfg.max_matches_per_cell < 1:
        raise ValueError("max_matches_per_cell must be at least one")
    source = np.asarray(source_points, dtype=np.float64)
    reference = np.asarray(reference_points, dtype=np.float64)
    quality = np.asarray(scores, dtype=np.float64).reshape(-1)
    if source.ndim != 2 or source.shape[1] != 2 or reference.shape != source.shape or len(quality) != len(source):
        raise ValueError("source_points/reference_points must be (N,2) and scores must have N entries")
    grid = create_spatial_grid(image_shape, cfg.rows, cfg.cols)
    cells = assign_points_to_grid(source, grid)
    selected = []
    for row in range(grid.rows):
        for col in range(grid.cols):
            members = np.flatnonzero((cells[:, 0] == row) & (cells[:, 1] == col) & np.isfinite(quality))
            # lexsort makes ties deterministic by retaining the lower original index first.
            ranked = members[np.lexsort((members, -quality[members]))]
            selected.extend(ranked[:cfg.max_matches_per_cell].tolist())
    selected_indices = np.asarray(sorted(selected), dtype=np.int64)  # original ordering is convenient downstream
    before = calculate_spatial_coverage(source, image_shape)
    after = calculate_spatial_coverage(source[selected_indices], image_shape)
    diagnostics: Dict[str, Any] = {"grid": grid, "cell_indices": cells, "selected_source_points": source[selected_indices],
        "selected_reference_points": reference[selected_indices], "selected_scores": quality[selected_indices],
        "candidate_count": int(len(source)), "selected_count": int(len(selected_indices)),
        "coverage_before": before, "coverage_after": after}
    return selected_indices, diagnostics


def visualize_match_distribution(image, points, grid):
    """Return BGR visualization of points and configured grid boundaries."""
    canvas = np.asarray(image).copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    if canvas.ndim != 3:
        raise ValueError("image must be grayscale or color")
    for row in range(1, grid.rows):
        y = round(row * grid.cell_height); cv2.line(canvas, (0, y), (canvas.shape[1] - 1, y), (0, 255, 255), 1)
    for col in range(1, grid.cols):
        x = round(col * grid.cell_width); cv2.line(canvas, (x, 0), (x, canvas.shape[0] - 1), (0, 255, 255), 1)
    for x, y in np.asarray(points, dtype=np.float64):
        if np.isfinite(x) and np.isfinite(y): cv2.circle(canvas, (round(x), round(y)), 3, (0, 0, 255), -1)
    return canvas


def test_spatial_selection():
    """Clustered, uniform, and sparse deterministic-selection checks."""
    shape = (100, 100)
    clustered = np.array([[5. + i, 5.] for i in range(20)])
    uniform = np.array([[12., 12.], [62., 12.], [12., 62.], [62., 62.]])
    sparse = np.array([[50., 50.]])
    for points, expected in ((clustered, 1), (uniform, 4), (sparse, 1)):
        indices, diag = select_spatially_distributed_matches(points, points, np.arange(len(points)), shape,
            {"rows": 2, "cols": 2, "max_matches_per_cell": 1})
        assert len(indices) == expected and diag["selected_count"] == expected
    return "Spatial selection tests passed"


if __name__ == "__main__": print(test_spatial_selection())
