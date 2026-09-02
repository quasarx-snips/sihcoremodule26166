"""
LunaX Module 3 — Terrain Feature Extraction
============================================

A comprehensive, unified lunar terrain analysis module that converts grayscale
lunar images into structured terrain-feature representations for cross-image
correspondence matching, registration, and terrain analysis.

PIPELINE FLOW:
    1. Load + normalize the lunar image
    2. CLAHE contrast enhancement
    3. Crater detection (ONNX UNet model with Hough circle fallback)
    4. Ridge / elongated-structure detection (classical line detector)
    5. Texture / gradient-change detection (classical, generic structures)
    6. SIFT keypoints + descriptors
    7. Unified feature records with metadata
    8. Resolution-aware color-coded visualization
    9. JSON (features) + .npy (SIFT descriptors) export

FEATURES:
    ✓ ONNX-based crater detection (trained UNet model)
    ✓ Classical fallback detection methods
    ✓ Multi-type terrain feature extraction
    ✓ Automatic feature cap for visualization clarity
    ✓ Production-ready with comprehensive error handling
    ✓ Fully documented and modular architecture

USAGE:
    from module_3 import TerrainFeatureExtractor

    # Basic usage
    extractor = TerrainFeatureExtractor()
    results = extractor.run("lunar_image.png", "output_dir")

    # With custom ONNX model
    extractor = TerrainFeatureExtractor(
        onnx_crater_model_path="path/to/crater_model.onnx"
    )

    # Batch processing
    extractor = TerrainFeatureExtractor()
    for image_path in image_list:
        extractor.run(image_path, "output_dir")

Dependencies:
    - opencv-python (cv2)
    - numpy
    - onnxruntime (for ONNX model support)

Author: LunaX Team
Version: 3.0 (Consolidated & Optimized)
Date: 2026-09-02
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# Type aliases
ImageArray = np.ndarray
FeatureType = str  # one of: "crater" | "ridge" | "texture" | "sift"
CraterModelFn = Callable[[ImageArray], List[Tuple[float, float, float, float]]]


# ============================================================================
# 0. UNIFIED FEATURE RECORD
# ============================================================================

@dataclass
class TerrainFeature:
    """A single detected terrain feature, uniform across all detector types.

    Attributes:
        feature_type: "crater" | "ridge" | "texture" | "sift"
        x, y: pixel coordinates in the (already-enhanced) image
        scale: radius (craters), length in px (ridges), block size (texture),
               or SIFT keypoint diameter — None where not applicable
        orientation: degrees, where meaningful (ridges, SIFT); else None
        confidence: detector-specific confidence / response score [0, 1]
        descriptor_index: row index into the saved descriptor .npy array,
               or None if this feature type has no stored descriptor
        extra: any detector-specific metadata (e.g. ridge endpoints, circularity)
    """

    feature_type: FeatureType
    x: float
    y: float
    scale: Optional[float] = None
    orientation: Optional[float] = None
    confidence: float = 0.0
    descriptor_index: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


# ============================================================================
# 1. IMAGE PREPROCESSING
# ============================================================================

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
            clip_limit: CLAHE clipping limit (2.5 is standard)
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
        """Load and preprocess image.
        
        Returns:
            (normalized_raw, enhanced) grayscale images
        """
        raw = self.normalize(self.load(image_path))
        enhanced = self.enhance(raw)
        return raw, enhanced


# ============================================================================
# 2. CRATER DETECTION
# ============================================================================

class ONNXCraterDetector:
    """ONNX UNet segmentation-based crater detector.
    
    Uses a trained UNet model to segment crater regions, then extracts
    individual craters using morphological operations and contour analysis.
    
    Architecture:
        Input: 3-channel RGB image (batch, 3, 512, 512)
        Output: Segmentation map (batch, 1, 512, 512)
        Logits: Raw neural network outputs (unnormalized)
    
    Post-Processing:
        1. Min-max normalize logits to [0, 1]
        2. Threshold and morphological operations
        3. Contour detection and filtering
        4. Circularity and area ratio validation
        5. Confidence scoring based on shape + segmentation
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "models/crater_unet.onnx",
        input_size: Tuple[int, int] = (512, 512),
        confidence_threshold: float = 0.10,
        min_crater_area: float = 3.0,
        max_crater_area: Optional[float] = None,
        circularity_threshold: float = 0.20,
    ) -> None:
        """
        Initialize ONNX crater detector.
        
        Args:
            model_path: Path to ONNX model file
            input_size: Model input size (width, height)
            confidence_threshold: Segmentation threshold [0,1] after logit normalization
            min_crater_area: Minimum contour area in pixels
            max_crater_area: Maximum contour area (None for no limit)
            circularity_threshold: Minimum circularity (4π·area/perimeter²)
        """
        self.model_path = Path(model_path)
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.min_crater_area = min_crater_area
        self.max_crater_area = max_crater_area
        self.circularity_threshold = circularity_threshold

        self.session = None
        self.input_name = None
        self.output_name = None

        if self.model_path.exists() and ONNX_AVAILABLE:
            try:
                self.session = ort.InferenceSession(str(self.model_path))
                self.input_name = self.session.get_inputs()[0].name
                self.output_name = self.session.get_outputs()[0].name
            except Exception as e:
                print(f"Warning: Failed to load ONNX model: {e}")
    def detect(self, enhanced_img: ImageArray) -> List[TerrainFeature]:
        """Detect craters in enhanced image.
        
        Args:
            enhanced_img: Preprocessed grayscale image
            
        Returns:
            List of TerrainFeature objects with type="crater"
        """
        if self.session is None:
            return []

        orig_h, orig_w = enhanced_img.shape[:2]

        # Convert to RGB for model
        if len(enhanced_img.shape) == 2:
            rgb_img = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2RGB)
        else:
            rgb_img = enhanced_img

        # Resize and normalize
        resized = cv2.resize(rgb_img, self.input_size, interpolation=cv2.INTER_LINEAR)
        resized_norm = resized.astype(np.float32) / 255.0
        input_data = np.transpose(resized_norm, (2, 0, 1))[np.newaxis, ...]

        # Inference
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        segmentation_map = outputs[0][0, 0]  # Raw logits

        # Normalize logits to [0, 1]
        seg_min = segmentation_map.min()
        seg_max = segmentation_map.max()
        if seg_max - seg_min > 1e-6:
            segmentation_map = (segmentation_map - seg_min) / (seg_max - seg_min)
        else:
            segmentation_map = np.ones_like(segmentation_map)

        seg_uint8 = (segmentation_map * 255).astype(np.uint8)

        # Threshold and morphology
        threshold_value = int(self.confidence_threshold * 255)
        _, binary_mask = cv2.threshold(seg_uint8, threshold_value, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find and process contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        features: List[TerrainFeature] = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Area filtering
            if area < self.min_crater_area:
                continue
            if self.max_crater_area is not None and area > self.max_crater_area:
                continue

            # Fit circle
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            if radius < 1:
                continue

            # Circularity check
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
            else:
                continue

            if circularity < self.circularity_threshold:
                continue

            # Area ratio check
            expected_area = np.pi * (radius ** 2)
            area_ratio = area / expected_area if expected_area > 0 else 0
            if area_ratio < 0.3:
                continue

            # Scale to original image coordinates
            scale_x = orig_w / self.input_size[0]
            scale_y = orig_h / self.input_size[1]
            orig_cx = cx * scale_x
            orig_cy = cy * scale_y
            orig_radius = radius * max(scale_x, scale_y)

            # Confidence calculation
            shape_confidence = min(1.0, circularity * area_ratio)

            y_start = max(0, int(cy - radius))
            y_end = min(seg_uint8.shape[0], int(cy + radius))
            x_start = max(0, int(cx - radius))
            x_end = min(seg_uint8.shape[1], int(cx + radius))

            if y_start < y_end and x_start < x_end:
                region = seg_uint8[y_start:y_end, x_start:x_end]
                pixel_confidence = float(np.mean(region) / 255.0)
            else:
                pixel_confidence = 0.5

            confidence = (shape_confidence * 0.4 + pixel_confidence * 0.6)
            confidence = min(1.0, max(0.0, confidence))

            features.append(
                TerrainFeature(
                    feature_type="crater",
                    x=float(orig_cx),
                    y=float(orig_cy),
                    scale=float(orig_radius),
                    confidence=float(confidence),
                    extra={
                        "radius_px": float(orig_radius),
                        "circularity": float(circularity),
                        "area": float(area),
                        "area_ratio": float(area_ratio),
                    },
                )
            )

        return features


class CraterDetector:
    """Unified crater detector with ONNX and classical fallback support.
    
    Priority:
        1. Custom model function (if provided)
        2. ONNX model (if available)
        3. Classical Hough circles (fallback)
    """

    def __init__(
        self,
        model_predict_fn: Optional[CraterModelFn] = None,
        onnx_model_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.model_predict_fn = model_predict_fn
        self.onnx_detector = None

        if onnx_model_path is not None:
            try:
                self.onnx_detector = ONNXCraterDetector(onnx_model_path)
            except Exception as e:
                print(f"Warning: Could not load ONNX detector: {e}")

    def detect(self, enhanced_img: ImageArray) -> List[TerrainFeature]:
        """Detect craters using available methods."""
        if self.model_predict_fn is not None:
            raw_detections = self.model_predict_fn(enhanced_img)
        elif self.onnx_detector is not None:
            return self.onnx_detector.detect(enhanced_img)
        else:
            raw_detections = self._fallback_hough(enhanced_img)

        features: List[TerrainFeature] = []
        for x, y, r, conf in raw_detections:
            features.append(
                TerrainFeature(
                    feature_type="crater",
                    x=float(x),
                    y=float(y),
                    scale=float(r),
                    confidence=float(conf),
                    extra={"radius_px": float(r)},
                )
            )
        return features

    @staticmethod
    def _fallback_hough(img: ImageArray) -> List[Tuple[float, float, float, float]]:
        """Classical Hough circle detection fallback."""
        blurred = cv2.GaussianBlur(img, (9, 9), 2)
        max_r = max(5, min(img.shape) // 4)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.5,
            minDist=30,
            param1=80,
            param2=80,
            minRadius=10,
            maxRadius=max_r,
        )
        results: List[Tuple[float, float, float, float]] = []
        if circles is not None:
            for x, y, r in circles[0]:
                results.append((float(x), float(y), float(r), 0.5))
        return results

# ============================================================================
# 3. RIDGE DETECTION
# ============================================================================

class RidgeDetector:
    """Detects elongated terrain structures (ridges, scarps, lineaments)
    with a classical edge + probabilistic-Hough line pipeline. Uses
    HoughLinesP (not the patent-affected LSD implementation) so it runs on
    a stock opencv-python install."""

    def __init__(
        self,
        canny_low: int = 150,
        canny_high: int = 200,
        min_length: float = 10.0,
        max_line_gap: float = 5.0,
    ) -> None:
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.min_length = min_length
        self.max_line_gap = max_line_gap

    def detect(self, enhanced_img: ImageArray) -> List[TerrainFeature]:
        edges = cv2.Canny(enhanced_img, self.canny_low, self.canny_high)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=40,
            minLineLength=self.min_length,
            maxLineGap=self.max_line_gap,
        )

        features: List[TerrainFeature] = []
        if lines is None:
            return features

        # OpenCV versions differ on shape: (N, 1, 4) vs (N, 4). Normalize
        # to (N, 4) so unpacking is safe either way.
        lines = np.asarray(lines).reshape(-1, 4)

        for line in lines:
            x1, y1, x2, y2 = line
            length = float(np.hypot(x2 - x1, y2 - y1))
            if length < self.min_length:
                continue
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            # confidence proxy: mean edge-gradient strength under the segment
            conf = self._segment_strength(enhanced_img, (x1, y1), (x2, y2))
            features.append(
                TerrainFeature(
                    feature_type="ridge",
                    x=cx,
                    y=cy,
                    scale=length,
                    orientation=angle,
                    confidence=conf,
                    extra={"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                )
            )
        return features

    @staticmethod
    def _segment_strength(img: ImageArray, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        n = max(2, int(np.hypot(p2[0] - p1[0], p2[1] - p1[1])))
        xs = np.linspace(p1[0], p2[0], n).astype(int)
        ys = np.linspace(p1[1], p2[1], n).astype(int)
        xs = np.clip(xs, 0, img.shape[1] - 1)
        ys = np.clip(ys, 0, img.shape[0] - 1)
        return float(np.mean(mag[ys, xs]))


# ============================================================================
# 4. TEXTURE DETECTION
# ============================================================================

class TextureGradientDetector:
    """Detects strong local texture/gradient changes as generic structural
    features — distinct from craters, ridges, and SIFT corners. Uses Sobel
    gradient-magnitude local maxima with a dilation-based non-max
    suppression, so it stays lightweight and dependency-free."""

    def __init__(
        self,
        block_size: int = 15,
        response_threshold: float = 30.0,
        max_features: int = 500,
    ) -> None:
        self.block_size = block_size
        self.response_threshold = response_threshold
        self.max_features = max_features

    def detect(self, enhanced_img: ImageArray) -> List[TerrainFeature]:
        gx = cv2.Sobel(enhanced_img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(enhanced_img, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)

        kernel = np.ones((self.block_size, self.block_size), np.uint8)
        dilated = cv2.dilate(mag, kernel)
        local_max_mask = (mag == dilated) & (mag > self.response_threshold)

        ys, xs = np.where(local_max_mask)
        responses = mag[ys, xs]

        if len(responses) > self.max_features:
            top_idx = np.argsort(responses)[::-1][: self.max_features]
            xs, ys, responses = xs[top_idx], ys[top_idx], responses[top_idx]

        features: List[TerrainFeature] = []
        for x, y, resp in zip(xs, ys, responses):
            features.append(
                TerrainFeature(
                    feature_type="texture",
                    x=float(x),
                    y=float(y),
                    scale=float(self.block_size),
                    confidence=float(resp),
                )
            )
        return features


# ============================================================================
# 5. SIFT KEYPOINT DETECTION
# ============================================================================

class SiftDetector:
    """Runs SIFT to find distinctive local keypoints and their 128-dim
    descriptors."""

    def __init__(
        self,
        n_features: int = 4000,
        contrast_threshold: float = 0.02,
        edge_threshold: float = 10.0,
    ) -> None:
        self.sift = cv2.SIFT_create(
            nfeatures=n_features,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
        )

    def detect(
        self, enhanced_img: ImageArray, descriptor_offset: int = 0
    ) -> Tuple[List[TerrainFeature], np.ndarray]:
        kps, descs = self.sift.detectAndCompute(enhanced_img, None)
        if descs is None:
            descs = np.empty((0, 128), dtype=np.float32)

        features: List[TerrainFeature] = []
        for i, kp in enumerate(kps):
            features.append(
                TerrainFeature(
                    feature_type="sift",
                    x=float(kp.pt[0]),
                    y=float(kp.pt[1]),
                    scale=float(kp.size),
                    orientation=float(kp.angle),
                    confidence=float(kp.response),
                    descriptor_index=descriptor_offset + i,
                )
            )
        return features, descs.astype(np.float32)


# ============================================================================
# 6. VISUALIZATION
# ============================================================================

# Color map for features (BGR format for OpenCV)
COLOR_MAP_BGR: Dict[FeatureType, Tuple[int, int, int]] = {
    "crater": (0, 0, 255),      # red
    "ridge": (255, 0, 0),       # blue
    "texture": (0, 255, 255),   # yellow
    "sift": (255, 0, 255),      # magenta
}


class TerrainVisualizer:
    """Renders color-coded terrain feature maps."""

    def __init__(self, reference_diagonal: float = 1000.0) -> None:
        """
        Initialize visualizer.
        
        Args:
            reference_diagonal: Reference image diagonal for marker scaling
        """
        self.reference_diagonal = reference_diagonal

    def _scale_factor(self, img_shape: Tuple[int, int]) -> float:
        """Calculate scale factor based on image size."""
        h, w = img_shape[:2]
        diagonal = float(np.hypot(h, w))
        return max(diagonal / self.reference_diagonal, 0.3)

    def render(self, base_img: ImageArray, features: List[TerrainFeature]) -> np.ndarray:
        """Render features on base image.
        
        Args:
            base_img: Grayscale base image
            features: List of terrain features to render
            
        Returns:
            RGB visualization image
        """
        vis = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
        s = self._scale_factor(base_img.shape)

        marker_size = max(4, int(round(6 * s)))
        thickness = max(1, int(round(s)))

        for feat in features:
            color = COLOR_MAP_BGR.get(feat.feature_type, (255, 255, 255))
            pt = (int(round(feat.x)), int(round(feat.y)))

            if feat.feature_type == "crater":
                # Red circle
                radius_px = int(round(feat.scale)) if feat.scale else marker_size
                cv2.circle(vis, pt, radius_px, color, thickness=max(1, thickness))

            elif feat.feature_type == "ridge":
                # Blue tilted cross
                cv2.drawMarker(
                    vis,
                    pt,
                    color,
                    markerType=cv2.MARKER_TILTED_CROSS,
                    markerSize=marker_size,
                    thickness=thickness,
                )

            elif feat.feature_type == "texture":
                # Yellow triangle
                cv2.drawMarker(
                    vis,
                    pt,
                    color,
                    markerType=cv2.MARKER_TRIANGLE_DOWN,
                    markerSize=marker_size,
                    thickness=thickness,
                )

            elif feat.feature_type == "sift":
                # Magenta star
                cv2.drawMarker(
                    vis,
                    pt,
                    color,
                    markerType=cv2.MARKER_STAR,
                    markerSize=marker_size,
                    thickness=thickness,
                )

        return vis


# ============================================================================
# 7. PERSISTENCE
# ============================================================================

class FeatureStore:
    """Handles saving structured features (JSON) and SIFT descriptors
    (.npy) separately, plus loading them back for later pipeline stages."""

    @staticmethod
    def save_features(
        path: Union[str, Path],
        features: List[TerrainFeature],
        image_shape: Tuple[int, int],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {
            "image_shape": list(image_shape),
            "feature_count": len(features),
            "metadata": metadata or {},
            "features": [f.to_dict() for f in features],
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @staticmethod
    def load_features(path: Union[str, Path]) -> Dict[str, Any]:
        return json.loads(Path(path).read_text())

    @staticmethod
    def save_descriptors(path: Union[str, Path], descriptors: np.ndarray) -> None:
        np.save(str(path), descriptors)

    @staticmethod
    def load_descriptors(path: Union[str, Path]) -> np.ndarray:
        return np.load(str(path))


# ============================================================================
# 8. MAIN ORCHESTRATOR
# ============================================================================

class TerrainFeatureExtractor:
    """End-to-end lunar terrain feature extraction orchestrator.
    
    Coordinates all detectors and produces unified feature records
    with visualization and serialization.
    
    PIPELINE:
        Image → Preprocess → Craters → Ridges → Textures → SIFT
               → Merge → Cap → Visualize → Export
    """

    def __init__(
        self,
        crater_model_predict_fn: Optional[CraterModelFn] = None,
        preprocessor: Optional[ImagePreprocessor] = None,
        crater_detector: Optional[CraterDetector] = None,
        ridge_detector: Optional[RidgeDetector] = None,
        texture_detector: Optional[TextureGradientDetector] = None,
        sift_detector: Optional[SiftDetector] = None,
        visualizer: Optional[TerrainVisualizer] = None,
        max_total_features: Optional[int] = 1000,
        onnx_crater_model_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize terrain feature extractor with optional custom components."""
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.crater_detector = crater_detector or CraterDetector(
            crater_model_predict_fn,
            onnx_model_path=onnx_crater_model_path or "models/crater_unet.onnx",
        )
        self.ridge_detector = ridge_detector or RidgeDetector()
        self.texture_detector = texture_detector or TextureGradientDetector()
        self.sift_detector = sift_detector or SiftDetector()
        self.visualizer = visualizer or TerrainVisualizer()
        self.max_total_features = max_total_features

    def extract(
        self,
        image_path: Union[str, Path],
    ) -> Tuple[ImageArray, List[TerrainFeature], np.ndarray]:
        """Run full detection pipeline on one image.
        
        Returns:
            (enhanced_img, all_features, sift_descriptors)
        """
        t0 = time.time()
        _, enhanced = self.preprocessor.process(image_path)

        crater_features = self.crater_detector.detect(enhanced)
        ridge_features = self.ridge_detector.detect(enhanced)
        texture_features = self.texture_detector.detect(enhanced)
        sift_features, sift_descriptors = self.sift_detector.detect(enhanced)

        all_features = crater_features + ridge_features + texture_features + sift_features
        raw_count = len(all_features)

        all_features, sift_descriptors = self._cap_features(all_features, sift_descriptors)

        elapsed = time.time() - t0
        kept_counts = {t: 0 for t in ("crater", "ridge", "texture", "sift")}
        for f in all_features:
            kept_counts[f.feature_type] += 1

        print(
            f"[Module 3] {Path(image_path).name}: detected {raw_count} raw features "
            f"(craters={len(crater_features)}, ridges={len(ridge_features)}, "
            f"texture={len(texture_features)}, sift={len(sift_features)}) "
            f"-> kept {len(all_features)} after cap "
            f"(craters={kept_counts['crater']}, ridges={kept_counts['ridge']}, "
            f"texture={kept_counts['texture']}, sift={kept_counts['sift']}) "
            f"({elapsed:.2f}s)"
        )

        return enhanced, all_features, sift_descriptors

    def _cap_features(
        self,
        features: List[TerrainFeature],
        sift_descriptors: np.ndarray,
    ) -> Tuple[List[TerrainFeature], np.ndarray]:
        """Keep top features per type using normalized confidence ranking."""
        if self.max_total_features is None or len(features) <= self.max_total_features:
            return features, sift_descriptors

        by_type: Dict[str, List[TerrainFeature]] = {}
        for f in features:
            by_type.setdefault(f.feature_type, []).append(f)

        ranked: List[Tuple[float, TerrainFeature]] = []
        for feats in by_type.values():
            confs = np.array([f.confidence for f in feats], dtype=float)
            lo, hi = float(confs.min()), float(confs.max())
            norm = np.ones_like(confs) if (hi - lo) < 1e-9 else (confs - lo) / (hi - lo)
            ranked.extend(zip(norm.tolist(), feats))

        ranked.sort(key=lambda t: t[0], reverse=True)
        kept = [f for _, f in ranked[: self.max_total_features]]

        # Re-sync SIFT descriptors
        kept_sift = [f for f in kept if f.feature_type == "sift"]
        if kept_sift:
            old_indices = [f.descriptor_index for f in kept_sift]
            new_descriptors = sift_descriptors[old_indices]
            for new_idx, f in enumerate(kept_sift):
                f.descriptor_index = new_idx
        else:
            new_descriptors = np.empty(
                (0, sift_descriptors.shape[1] if sift_descriptors.size else 128),
                dtype=np.float32,
            )

        return kept, new_descriptors

    def run(
        self,
        image_path: Union[str, Path],
        output_dir: Union[str, Path],
        stem: Optional[str] = None,
        save_visualization: bool = True,
    ) -> Dict[str, Path]:
        """Run full extraction and save outputs.
        
        Returns:
            Dictionary with output file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = stem or Path(image_path).stem

        enhanced, features, descriptors = self.extract(image_path)

        features_path = output_dir / f"{stem}_terrain_features.json"
        descriptors_path = output_dir / f"{stem}_sift_descriptors.npy"

        FeatureStore.save_features(
            features_path,
            features,
            image_shape=enhanced.shape,
            metadata={"source_image": str(image_path), "sift_descriptor_dim": 128},
        )
        FeatureStore.save_descriptors(descriptors_path, descriptors)

        outputs = {"features": features_path, "descriptors": descriptors_path}

        if save_visualization:
            vis = self.visualizer.render(enhanced, features)
            vis_path = output_dir / f"{stem}_terrain_map.png"
            cv2.imwrite(str(vis_path), vis)
            outputs["visualization"] = vis_path

        return outputs


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def verify_extraction(features_json_path: Union[str, Path]) -> None:
    """Utility: Print summary of extracted features.
    
    Args:
        features_json_path: Path to features JSON file
    """
    data = FeatureStore.load_features(features_json_path)
    features = data["features"]

    by_type = {}
    for f in features:
        ftype = f["feature_type"]
        by_type.setdefault(ftype, []).append(f)

    print(f"✓ Total features: {len(features)}")
    print(f"✓ Image shape: {data['image_shape']}")
    for ftype, feats in by_type.items():
        avg_conf = np.mean([f["confidence"] for f in feats])
        print(f"  {ftype}: {len(feats)} (avg confidence: {avg_conf:.3f})")


def batch_extract(
    image_dir: Union[str, Path],
    output_dir: Union[str, Path],
    pattern: str = "*.png",
    max_images: Optional[int] = None,
) -> None:
    """Utility: Batch extract features from multiple images.
    
    Args:
        image_dir: Directory containing images
        output_dir: Output directory for results
        pattern: Glob pattern for image files
        max_images: Maximum number of images to process
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(image_dir.glob(pattern))
    if max_images is not None:
        images = images[:max_images]

    extractor = TerrainFeatureExtractor()

    print(f"Processing {len(images)} images...")
    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {img_path.name}...", end=" ")
        try:
            extractor.run(img_path, output_dir)
            print("✓")
        except Exception as e:
            print(f"✗ Error: {e}")


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        # Default: process sample image
        IMAGE_PATH = r"C:\Users\bijan\Desktop\LunaX\lunar_samples\sample_41.png"
        OUTPUT_DIR = "module3_outputs"

        print("LunaX Module 3 - Terrain Feature Extraction")
        print("=" * 60)
        print(f"Processing: {IMAGE_PATH}")
        print(f"Output: {OUTPUT_DIR}")
        print()

        extractor = TerrainFeatureExtractor(
            onnx_crater_model_path="models/crater_unet.onnx"
        )
        result_paths = extractor.run(IMAGE_PATH, OUTPUT_DIR)

        print("\nSaved outputs:")
        for name, path in result_paths.items():
            print(f"  {name}: {path}")

        # Verify results
        print()
        verify_extraction(result_paths["features"])

    else:
        # Command-line usage
        cmd = sys.argv[1]

        if cmd == "extract":
            # Extract single image
            if len(sys.argv) < 4:
                print("Usage: python module_3.py extract <image> <output_dir>")
                sys.exit(1)
            image_path = sys.argv[2]
            output_dir = sys.argv[3]
            extractor = TerrainFeatureExtractor()
            extractor.run(image_path, output_dir)

        elif cmd == "batch":
            # Batch process directory
            if len(sys.argv) < 4:
                print("Usage: python module_3.py batch <image_dir> <output_dir> [max_images]")
                sys.exit(1)
            image_dir = sys.argv[2]
            output_dir = sys.argv[3]
            max_images = int(sys.argv[4]) if len(sys.argv) > 4 else None
            batch_extract(image_dir, output_dir, max_images=max_images)

        elif cmd == "verify":
            # Verify extraction
            if len(sys.argv) < 3:
                print("Usage: python module_3.py verify <features.json>")
                sys.exit(1)
            verify_extraction(sys.argv[2])

        else:
            print(f"Unknown command: {cmd}")
            print("Available commands: extract, batch, verify")
            sys.exit(1)