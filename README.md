# LunaX

LunaX is a classical computer-vision prototype for SIH26166: finding and verifying correspondences between lunar optical observations with varying scale and illumination.

It follows a chain of evidence: grayscale normalization and CLAHE, SIFT local descriptors plus crater/ridge/texture landmarks, ratio and mutual descriptor filtering, RANSAC geometric verification, then image registration. A visual descriptor is only a candidate; a common geometric transform is the proof used to accept it.

## Run

```powershell
python run_lunax.py --image-a lunar_samples/sample_13.png --image-b lunar_samples/sample_14.png --output-dir outputs/demo
```

The report contains actual candidate/inlier counts, inlier ratio, error statistics, transformation, spatial coverage, and an interpretable `Correspondence Confidence` evidence score. The score combines inlier ratio, count, coverage, and reprojection error; it is not a calibrated probability.

All artifacts for a run are saved together in the selected output directory: images, feature JSON/NPY files, and `metrics.json`.

## API

```python
from lunax import run_lunax_registration
result = run_lunax_registration("image_a.png", "image_b.png", {"save_outputs": True, "output_dir": "outputs/run"})
```

The returned result includes preprocessed images, terrain features, candidate matches, verified mask, transformation, registered image, metrics, diagnostics, and output paths.

## Limitations

Crater detections are supporting landmarks, not an exhaustive crater catalogue. Illumination robustness is practical (CLAHE, gradients, local descriptors, and geometric verification), not absolute sun-angle invariance. The included pipeline runs on CPU; tiling, CUDA, and multi-GPU execution remain future performance work. Results depend on overlap and image quality.
