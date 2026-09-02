"""
LunaX Module 1 — Image Preprocessing Unit
==========================================

A lightweight, production-grade image preprocessing module designed exclusively
for loading, normalizing, and applying CLAHE (Contrast Limited Adaptive Histogram 
Equalization) to lunar imagery.

Dependencies:
    - opencv-python (cv2)
    - numpy
"""

from pathlib import Path
from typing import Tuple, Union
import cv2
import numpy as np

ImageArray = np.ndarray

class ImagePreprocessor:
    """Loads lunar grayscale images, normalizes to uint8, and applies CLAHE."""

    def __init__(
        self,
        clip_limit: float = 2.5,
        tile_grid_size: Tuple[int, int] = (8, 8),
    ) -> None:
        """
        Initialize preprocessor with CLAHE parameters.
        
        Args:
            clip_limit: CLAHE clipping limit (2.5 is standard for lunar imagery)
            tile_grid_size: Size of CLAHE tile grid
        """
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def load(self, image_path: Union[str, Path]) -> ImageArray:
        """Load grayscale image from disk."""
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")
        return img

    def normalize(self, img: ImageArray) -> ImageArray:
        """Scale to full uint8 range regardless of source bit depth."""
        if img.dtype != np.uint8:
            img_min, img_max = float(img.min()), float(img.max())
            if img_max - img_min < 1e-6:
                img = np.zeros_like(img, dtype=np.uint8)
            else:
                img = ((img - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
        return img

    def enhance(self, img: ImageArray) -> ImageArray:
        """Apply CLAHE contrast enhancement."""
        return self.clahe.apply(img)

    def process(self, image_path: Union[str, Path]) -> Tuple[ImageArray, ImageArray]:
        """
        Load and preprocess image.
        
        Returns:
            Tuple of (normalized_raw, enhanced) grayscale images
        """
        raw = self.normalize(self.load(image_path))
        enhanced = self.enhance(raw)
        return raw, enhanced