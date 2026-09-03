"""Execute all harsh-sample registrations and write consolidated reports."""
from __future__ import annotations

import contextlib
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np

from lunax import PipelineConfig, run_lunax_registration


VARIANTS = ("lighting", "crop", "rotation", "combined")
SAMPLES = range(2, 51)


def _result_record(sample: int, variant: str, result, seconds: float) -> dict:
    if not result.success:
        return {"sample": sample, "testcase": variant, "success": False, "error": result.error, "runtime_seconds": seconds}
    metrics, stats = result.metrics, result.metrics["inlier_error_statistics"]
    return {"sample": sample, "testcase": variant, "success": True, "status": metrics["status"],
            "candidate_matches": metrics["candidate_matches"], "verified_inliers": metrics["verified_inliers"],
            "outliers": metrics["outliers"], "inlier_ratio": metrics["inlier_ratio"],
            "rmse_px": stats["rmse"], "median_error_px": stats["median"],
            "spatial_coverage_percent": metrics["spatial_coverage"]["coverage_percentage"],
            "confidence_percent": metrics["correspondence_confidence"],
            "transformation_model": metrics["transformation_model"], "runtime_seconds": seconds}


def _draw_dashboard(records: list[dict], output: Path) -> None:
    left, cell_w, cell_h, top = 105, 315, 49, 100
    canvas = np.full((top + 49 * cell_h + 58, left + 4 * cell_w, 3), 248, np.uint8)
    cv2.putText(canvas, "LUNAX | HARSH VARIANT TEST REPORT", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Each cell: status | inliers/candidates | confidence | RMSE", (20, 58), cv2.FONT_HERSHEY_SIMPLEX, .45, (80, 80, 80), 1, cv2.LINE_AA)
    lookup = {(r["sample"], r["testcase"]): r for r in records}
    for col, testcase in enumerate(VARIANTS):
        x = left + col * cell_w; cv2.rectangle(canvas, (x, top - 29), (x + cell_w, top), (60, 60, 60), -1)
        cv2.putText(canvas, testcase.upper(), (x + 12, top - 9), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
    for row, sample in enumerate(SAMPLES):
        y = top + row * cell_h; cv2.putText(canvas, f"sample_{sample}", (9, y + 30), cv2.FONT_HERSHEY_SIMPLEX, .43, (40, 40, 40), 1, cv2.LINE_AA)
        for col, testcase in enumerate(VARIANTS):
            x, record = left + col * cell_w, lookup[(sample, testcase)]
            valid = record.get("status") == "VALID CORRESPONDENCE"
            color = (212, 243, 212) if valid else ((215, 215, 255) if not record["success"] else (225, 233, 250))
            cv2.rectangle(canvas, (x, y), (x + cell_w - 1, y + cell_h - 1), color, -1); cv2.rectangle(canvas, (x, y), (x + cell_w - 1, y + cell_h - 1), (190, 190, 190), 1)
            if record["success"]:
                headline = "VALID" if valid else "NO MATCH"
                detail = f"{record['verified_inliers']}/{record['candidate_matches']}   confidence {record['confidence_percent']:.0f}%   RMSE {record['rmse_px']:.1f}px"
            else: headline, detail = "ERROR", record["error"][:42]
            cv2.putText(canvas, headline, (x + 7, y + 18), cv2.FONT_HERSHEY_SIMPLEX, .37, (25, 100, 25) if valid else (80, 35, 35), 1, cv2.LINE_AA)
            cv2.putText(canvas, detail, (x + 7, y + 38), cv2.FONT_HERSHEY_SIMPLEX, .33, (45, 45, 45), 1, cv2.LINE_AA)
    completed = [r for r in records if r["success"]]; valid = sum(r.get("status") == "VALID CORRESPONDENCE" for r in completed)
    cv2.putText(canvas, f"196 runs | completed {len(completed)} | valid correspondences {valid}", (15, canvas.shape[0] - 23), cv2.FONT_HERSHEY_SIMPLEX, .48, (30, 30, 30), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(output), canvas): raise IOError(f"Could not write {output}")


def main() -> int:
    report_dir, originals, variants = Path("tests/test1"), Path("lunar_samples"), Path("outputs/harsh_test_images")
    report_dir.mkdir(parents=True, exist_ok=True)
    # SIFT baseline only makes the full 196-case regression practical while
    # retaining preprocessing, matching, RANSAC, metrics, and registration.
    cfg = PipelineConfig(verbose=False, save_outputs=False, sift_features=1200, use_terrain_landmarks=False,
        verification={"model":"auto", "reprojection_threshold":3.0, "confidence":.999, "max_iterations":400, "min_inliers":4, "random_seed":42})
    records, started = [], time.perf_counter()
    for sample in SAMPLES:
        for testcase in VARIANTS:
            begun = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_lunax_registration(originals / f"sample_{sample}.png", variants / f"sample_{sample}_{testcase}.png", cfg)
            records.append(_result_record(sample, testcase, result, time.perf_counter() - begun))
            print(f"[{len(records):03d}/196] sample_{sample} / {testcase}", flush=True)
    completed = [r for r in records if r["success"]]
    payload = {"report_type":"LunaX harsh-variant regression", "test_configuration":{"feature_backend":"SIFT baseline", "sift_features":1200, "ransac_max_iterations":400, "testcases":list(VARIANTS)}, "summary":{"total_runs":196, "completed_runs":len(completed), "valid_correspondences":sum(r.get("status") == "VALID CORRESPONDENCE" for r in completed), "total_runtime_seconds":time.perf_counter() - started}, "results":records}
    json_path, image_path = report_dir / "harsh_test_report.json", report_dir / "harsh_test_report.png"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8"); _draw_dashboard(records, image_path)
    print(f"Created: {image_path} and {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
