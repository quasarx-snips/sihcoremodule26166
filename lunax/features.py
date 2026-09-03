"""
LunaX Module 3 — Terrain Feature Extraction
============================================

A comprehensive lunar terrain analysis module that converts preprocessed 
grayscale lunar images into structured terrain-feature representations.

PIPELINE FLOW:
    1. Load + CLAHE enhance (via Module 1)
    2. Crater detection (ONNX UNet model with Hough circle fallback)
    3. Ridge / elongated-structure detection (classical line detector)
    4. Texture / gradient-change detection (classical, generic structures)
    5. SIFT keypoints + descriptors
    6. Unified feature records with metadata
    7. Resolution-aware color-coded visualization
    8. JSON (features) + .npy (SIFT descriptors) export

Dependencies:
    - opencv-python (cv2)
    - numpy
    - onnxruntime (for ONNX model support)
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# --- IMPORT MODULE 1 ---
from .preprocessing import ImagePreprocessor

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

ImageArray = np.ndarray
FeatureType = str  # "crater" | "ridge" | "texture" | "sift" | "superpoint"
CraterModelFn = Callable[[ImageArray], List[Tuple[float, float, float, float]]]


# ============================================================================
# 0. UNIFIED FEATURE RECORD
# ============================================================================

@dataclass
class TerrainFeature:
    feature_type: FeatureType
    x: float
    y: float
    scale: Optional[float] = None
    orientation: Optional[float] = None
    confidence: float = 0.0
    descriptor_index: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# 1. CRATER DETECTION
# ============================================================================

class ONNXCraterDetector:
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        input_size: Tuple[int, int] = (512, 512),
        confidence_threshold: float = 0.10,
        min_crater_area: float = 3.0,
        max_crater_area: Optional[float] = None,
        circularity_threshold: float = 0.20,
    ) -> None:
        self.model_path = Path(model_path) if model_path is not None else None
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.min_crater_area = min_crater_area
        self.max_crater_area = max_crater_area
        self.circularity_threshold = circularity_threshold
        self.session = None

        if self.model_path is not None and self.model_path.exists() and ONNX_AVAILABLE:
            try:
                self.session = ort.InferenceSession(str(self.model_path))
            except Exception as e:
                print(f"Warning: Failed to load ONNX model: {e}")

    def detect(self, enhanced_img: ImageArray) -> List[TerrainFeature]:
        if self.session is None:
            return []

        orig_h, orig_w = enhanced_img.shape[:2]
        rgb_img = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2RGB) if len(enhanced_img.shape) == 2 else enhanced_img
        resized = cv2.resize(rgb_img, self.input_size, interpolation=cv2.INTER_LINEAR)
        input_data = np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))[np.newaxis, ...]

        outputs = self.session.run(None, {self.session.get_inputs()[0].name: input_data})
        segmentation_map = outputs[0][0, 0]

        seg_min, seg_max = segmentation_map.min(), segmentation_map.max()
        if seg_max - seg_min > 1e-6:
            segmentation_map = (segmentation_map - seg_min) / (seg_max - seg_min)
        
        seg_uint8 = (segmentation_map * 255).astype(np.uint8)
        _, binary_mask = cv2.threshold(seg_uint8, int(self.confidence_threshold * 255), 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        features: List[TerrainFeature] = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_crater_area or (self.max_crater_area and area > self.max_crater_area):
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            if radius < 1: continue

            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            if circularity < self.circularity_threshold: continue

            expected_area = np.pi * (radius ** 2)
            area_ratio = area / expected_area if expected_area > 0 else 0
            if area_ratio < 0.3: continue

            scale_x, scale_y = orig_w / self.input_size[0], orig_h / self.input_size[1]
            orig_cx, orig_cy = cx * scale_x, cy * scale_y
            orig_radius = radius * max(scale_x, scale_y)

            shape_confidence = min(1.0, circularity * area_ratio)
            y_start, y_end = max(0, int(cy - radius)), min(seg_uint8.shape[0], int(cy + radius))
            x_start, x_end = max(0, int(cx - radius)), min(seg_uint8.shape[1], int(cx + radius))
            
            pixel_confidence = float(np.mean(seg_uint8[y_start:y_end, x_start:x_end]) / 255.0) if y_start < y_end and x_start < x_end else 0.5
            confidence = min(1.0, max(0.0, (shape_confidence * 0.4 + pixel_confidence * 0.6)))

            features.append(TerrainFeature(
                feature_type="crater", x=float(orig_cx), y=float(orig_cy), scale=float(orig_radius),
                confidence=float(confidence), extra={"radius_px": float(orig_radius), "circularity": float(circularity), "area": float(area)}
            ))
        return features


class CraterDetector:
    def __init__(self, model_predict_fn: Optional[CraterModelFn] = None, onnx_model_path: Optional[Union[str, Path]] = None) -> None:
        self.model_predict_fn = model_predict_fn
        self.onnx_detector = ONNXCraterDetector(onnx_model_path) if onnx_model_path else None

    def detect(self, enhanced_img: ImageArray) -> List[TerrainFeature]:
        if self.model_predict_fn:
            raw_detections = self.model_predict_fn(enhanced_img)
        elif self.onnx_detector and self.onnx_detector.session:
            onnx_craters = self.onnx_detector.detect(enhanced_img)
            if onnx_craters: return onnx_craters
            raw_detections = self._fallback_hough(enhanced_img)
        else:
            raw_detections = self._fallback_hough(enhanced_img)

        return [TerrainFeature(feature_type="crater", x=float(x), y=float(y), scale=float(r), confidence=float(conf), extra={"radius_px": float(r)}) 
                for x, y, r, conf in raw_detections]

    @staticmethod
    def _fallback_hough(img: ImageArray) -> List[Tuple[float, float, float, float]]:
        blurred = cv2.GaussianBlur(img, (9, 9), 2)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.5, minDist=30, param1=80, param2=80, minRadius=10, maxRadius=max(5, min(img.shape) // 4))
        return [(float(x), float(y), float(r), 0.5) for x, y, r in circles[0]] if circles is not None else []


# ============================================================================
# 2. RIDGE & TEXTURE DETECTION
# ============================================================================

class RidgeDetector:
    def __init__(self, canny_low: int = 150, canny_high: int = 200, min_length: float = 10.0, max_line_gap: float = 5.0) -> None:
        self.canny_low, self.canny_high, self.min_length, self.max_line_gap = canny_low, canny_high, min_length, max_line_gap

    def detect(self, enhanced_img: ImageArray) -> List[TerrainFeature]:
        edges = cv2.Canny(enhanced_img, self.canny_low, self.canny_high)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=40, minLineLength=self.min_length, maxLineGap=self.max_line_gap)
        if lines is None: return []
        
        features = []
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            length = float(np.hypot(x2 - x1, y2 - y1))
            if length < self.min_length: continue
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            features.append(TerrainFeature(
                feature_type="ridge", x=cx, y=cy, scale=length, orientation=float(np.degrees(np.arctan2(y2 - y1, x2 - x1))),
                confidence=self._segment_strength(enhanced_img, (x1, y1), (x2, y2)), extra={"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}
            ))
        return features

    @staticmethod
    def _segment_strength(img: ImageArray, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        mag = cv2.magnitude(cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3))
        n = max(2, int(np.hypot(p2[0] - p1[0], p2[1] - p1[1])))
        xs = np.clip(np.linspace(p1[0], p2[0], n).astype(int), 0, img.shape[1] - 1)
        ys = np.clip(np.linspace(p1[1], p2[1], n).astype(int), 0, img.shape[0] - 1)
        return float(np.mean(mag[ys, xs]))


class TextureGradientDetector:
    def __init__(self, block_size: int = 15, response_threshold: float = 30.0, max_features: int = 500) -> None:
        self.block_size, self.response_threshold, self.max_features = block_size, response_threshold, max_features

    def detect(self, enhanced_img: ImageArray) -> List[TerrainFeature]:
        mag = cv2.magnitude(cv2.Sobel(enhanced_img, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(enhanced_img, cv2.CV_32F, 0, 1, ksize=3))
        dilated = cv2.dilate(mag, np.ones((self.block_size, self.block_size), np.uint8))
        local_max_mask = (mag == dilated) & (mag > self.response_threshold)
        ys, xs = np.where(local_max_mask)
        responses = mag[ys, xs]
        
        if len(responses) > self.max_features:
            top_idx = np.argsort(responses)[::-1][:self.max_features]
            xs, ys, responses = xs[top_idx], ys[top_idx], responses[top_idx]
            
        return [TerrainFeature(feature_type="texture", x=float(x), y=float(y), scale=float(self.block_size), confidence=float(resp)) 
                for x, y, resp in zip(xs, ys, responses)]


# ============================================================================
# 3. SIFT DETECTION & VISUALIZATION
# ============================================================================

class SiftDetector:
    def __init__(self, n_features: int = 6000, contrast_threshold: float = 0.01, edge_threshold: float = 15.0,
                 root_sift: bool = False) -> None:
        self.sift = cv2.SIFT_create(nfeatures=n_features, contrastThreshold=contrast_threshold, edgeThreshold=edge_threshold)
        self.root_sift = root_sift

    def _normalize_descriptors(self, descriptors: Optional[np.ndarray]) -> np.ndarray:
        """Apply RootSIFT's L1 + square-root normalization when enabled."""
        if descriptors is None:
            return np.empty((0, 128), dtype=np.float32)
        descriptors = descriptors.astype(np.float32, copy=False)
        if not self.root_sift or not len(descriptors):
            return descriptors
        return np.sqrt(descriptors / np.maximum(descriptors.sum(axis=1, keepdims=True), 1e-12))

    def detect(self, enhanced_img: ImageArray, descriptor_offset: int = 0) -> Tuple[List[TerrainFeature], np.ndarray]:
        kps, descs = self.sift.detectAndCompute(enhanced_img, None)
        descs = self._normalize_descriptors(descs)
        return [TerrainFeature(feature_type="sift", x=float(kp.pt[0]), y=float(kp.pt[1]), scale=float(kp.size), 
                               orientation=float(kp.angle), confidence=float(kp.response), descriptor_index=descriptor_offset + i) 
                for i, kp in enumerate(kps)], descs

    def describe_features(self, enhanced_img: ImageArray, features: List[TerrainFeature]) -> np.ndarray:
        """Compute one SIFT descriptor for every retained terrain feature.

        Module 04 can therefore match detected craters, ridges, and texture
        landmarks directly, instead of matching only incidental SIFT points
        around them.  Features too close to an image border may have no valid
        descriptor and retain ``descriptor_index=None``.
        """
        for feature in features:
            feature.descriptor_index = None
        if not features:
            return np.empty((0, 128), dtype=np.float32)
        keypoints = []
        for index, feature in enumerate(features):
            size = max(3.0, float(feature.scale) if feature.scale else 8.0)
            angle = float(feature.orientation) if feature.orientation is not None else -1.0
            keypoints.append(cv2.KeyPoint(float(feature.x), float(feature.y), size, angle,
                                          float(feature.confidence), 0, index))
        described_keypoints, descriptors = self.sift.compute(enhanced_img, keypoints)
        if descriptors is None:
            return np.empty((0, 128), dtype=np.float32)
        descriptors = self._normalize_descriptors(descriptors)
        for descriptor_index, keypoint in enumerate(described_keypoints):
            original_index = keypoint.class_id
            if 0 <= original_index < len(features):
                features[original_index].descriptor_index = descriptor_index
        return descriptors


COLOR_MAP_BGR = {"crater": (0, 0, 255), "ridge": (255, 0, 0), "texture": (0, 255, 255), "sift": (255, 0, 255)}

class TerrainVisualizer:
    def __init__(self, reference_diagonal: float = 1000.0) -> None:
        self.reference_diagonal = reference_diagonal

    def render(self, base_img: ImageArray, features: List[TerrainFeature]) -> np.ndarray:
        vis = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
        s = max(float(np.hypot(*base_img.shape[:2])) / self.reference_diagonal, 0.3)
        marker_size, thickness = max(4, int(round(6 * s))), max(1, int(round(s)))

        for feat in features:
            pt = (int(round(feat.x)), int(round(feat.y)))
            color = COLOR_MAP_BGR.get(feat.feature_type, (255, 255, 255))
            if feat.feature_type == "crater":
                cv2.circle(vis, pt, int(round(feat.scale)) if feat.scale else marker_size, color, thickness=max(1, thickness))
            else:
                marker_type = cv2.MARKER_TILTED_CROSS if feat.feature_type == "ridge" else (cv2.MARKER_TRIANGLE_DOWN if feat.feature_type == "texture" else cv2.MARKER_STAR)
                cv2.drawMarker(vis, pt, color, markerType=marker_type, markerSize=marker_size, thickness=thickness)
        return vis


# ============================================================================
# 4. PERSISTENCE & ORCHESTRATOR
# ============================================================================

class FeatureStore:
    @staticmethod
    def save_features(path: Union[str, Path], features: List[TerrainFeature], image_shape: Tuple[int, int], metadata: Optional[Dict[str, Any]] = None) -> None:
        Path(path).write_text(json.dumps({"image_shape": list(image_shape), "feature_count": len(features), "metadata": metadata or {}, "features": [f.to_dict() for f in features]}, indent=2))

    @staticmethod
    def save_descriptors(path: Union[str, Path], descriptors: np.ndarray) -> None:
        np.save(str(path), descriptors)


class TerrainFeatureExtractor:
    def __init__(
        self,
        preprocessor: Optional[ImagePreprocessor] = None,
        crater_detector: Optional[CraterDetector] = None,
        ridge_detector: Optional[RidgeDetector] = None,
        texture_detector: Optional[TextureGradientDetector] = None,
        sift_detector: Optional[SiftDetector] = None,
        visualizer: Optional[TerrainVisualizer] = None,
        max_total_features: Optional[int] = 3000,
        onnx_crater_model_path: Optional[Union[str, Path]] = None,
    ) -> None:
        # Uses Module 1's ImagePreprocessor by default
        self.preprocessor = preprocessor or ImagePreprocessor()
        # ONNX is optional: without a caller-supplied path, CraterDetector uses
        # its classical Hough fallback instead of assuming a repository layout.
        self.crater_detector = crater_detector or CraterDetector(onnx_model_path=onnx_crater_model_path)
        self.ridge_detector = ridge_detector or RidgeDetector()
        self.texture_detector = texture_detector or TextureGradientDetector()
        self.sift_detector = sift_detector or SiftDetector()
        self.visualizer = visualizer or TerrainVisualizer()
        self.max_total_features = max_total_features

    def extract(self, image_path: Union[str, Path]) -> Tuple[ImageArray, List[TerrainFeature], np.ndarray]:
        _, enhanced = self.preprocessor.process(image_path)
        return self.extract_array(enhanced, label=Path(image_path).name)

    def extract_array(self, enhanced: ImageArray, label: str = "array") -> Tuple[ImageArray, List[TerrainFeature], np.ndarray]:
        """Extract terrain landmarks from a preprocessed array.

        This is the array counterpart of :meth:`extract`, allowing the high-level
        pipeline to preprocess once and retain both normalized and enhanced data.
        """
        t0 = time.time()
        enhanced = self.preprocessor.normalize(enhanced)
        crater_features = self.crater_detector.detect(enhanced)
        ridge_features = self.ridge_detector.detect(enhanced)
        texture_features = self.texture_detector.detect(enhanced)
        descriptor_features, descriptors = self.sift_detector.detect(enhanced)

        all_features = crater_features + ridge_features + texture_features + descriptor_features
        raw_count = len(all_features)
        all_features, descriptors = self._cap_features(all_features, descriptors)
        # SIFT describes semantic terrain detections as well as local points.
        descriptors = self.sift_detector.describe_features(enhanced, all_features)

        elapsed = time.time() - t0
        print(f"[Module 3] {label}: detected {raw_count} raw features -> kept {len(all_features)} after cap; "
              f"described {len(descriptors)} with SIFT ({elapsed:.2f}s)")
        return enhanced, all_features, descriptors

    def _cap_features(self, features: List[TerrainFeature], sift_descriptors: np.ndarray) -> Tuple[List[TerrainFeature], np.ndarray]:
        if self.max_total_features is None or len(features) <= self.max_total_features:
            return features, sift_descriptors

        by_type: Dict[str, List[TerrainFeature]] = {}
        for f in features: by_type.setdefault(f.feature_type, []).append(f)

        ranked = []
        for feats in by_type.values():
            confs = np.array([f.confidence for f in feats], dtype=float)
            lo, hi = float(confs.min()), float(confs.max())
            norm = np.ones_like(confs) if (hi - lo) < 1e-9 else (confs - lo) / (hi - lo)
            ranked.extend(zip(norm.tolist(), feats))

        ranked.sort(key=lambda t: t[0], reverse=True)
        kept = [f for _, f in ranked[:self.max_total_features]]

        kept_described = [f for f in kept if f.descriptor_index is not None]
        if kept_described:
            new_descriptors = sift_descriptors[[f.descriptor_index for f in kept_described]]
            for new_idx, f in enumerate(kept_described): f.descriptor_index = new_idx
        else:
            descriptor_width = sift_descriptors.shape[1] if sift_descriptors.ndim == 2 else 128
            new_descriptors = np.empty((0, descriptor_width), dtype=np.float32)

        return kept, new_descriptors

    def run(self, image_path: Union[str, Path], output_dir: Union[str, Path], stem: Optional[str] = None, save_visualization: bool = True) -> Dict[str, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = stem or Path(image_path).stem

        enhanced, features, descriptors = self.extract(image_path)

        features_path = output_dir / f"{stem}_terrain_features.json"
        descriptors_path = output_dir / f"{stem}_sift_descriptors.npy"

        FeatureStore.save_features(features_path, features, image_shape=enhanced.shape, metadata={"source_image": str(image_path), "sift_descriptor_dim": 128})
        FeatureStore.save_descriptors(descriptors_path, descriptors)

        outputs = {"features": features_path, "descriptors": descriptors_path}
        if save_visualization:
            vis_path = output_dir / f"{stem}_terrain_map.png"
            cv2.imwrite(str(vis_path), self.visualizer.render(enhanced, features))
            outputs["visualization"] = vis_path

        return outputs


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "extract":
        TerrainFeatureExtractor().run(sys.argv[2], sys.argv[3])
    else:
        print("Usage: python -m lunax.features extract <image> <output_dir>")
