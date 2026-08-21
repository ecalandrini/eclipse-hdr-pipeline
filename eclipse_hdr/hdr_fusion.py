"""Multi-scale Laplacian pyramid HDR exposure fusion (Mertens et al.) with parameter reporting."""

import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import tifffile


def fuse_and_export_hdr(
    aligned_masters: list[np.ndarray],
    master_names: list[str],
    output_path: Path,
    contrast_w: float = 0.8,
    sat_w: float = 1.0,
    exp_w: float = 0.8,
) -> None:
    """Fuses bracketed astronomical masters using OpenCV Mertens fusion and reports all parameters.

    Args:
        aligned_masters: List of (H, W, 3) float32 RGB arrays in [0.0, 1.0].
        master_names: Names/exposure labels of the input master brackets.
        output_path: Path to the target 16-bit TIFF.
        contrast_w: Mertens contrast weight (Laplacian gradient magnitude).
        sat_w: Mertens saturation weight (color vibrancy).
        exp_w: Mertens exposure weight (Gaussian midtone curve centered at 0.5).
    """
    num_masters = len(aligned_masters)
    if num_masters == 0:
        raise ValueError("No aligned master images provided for fusion.")

    # 1. Report Algorithm Configuration & Parameter Summary
    print("\n" + "=" * 60, flush=True)
    print("           OPENCV MERTENS HDR FUSION CONFIGURATION           ", flush=True)
    print("=" * 60, flush=True)
    print(f"  * Input Exposure Brackets   : {num_masters} frames", flush=True)
    print(f"  * Contrast Weight (W_c)     : {contrast_w:.3f} (Laplacian edge response)", flush=True)
    print(f"  * Saturation Weight (W_s)   : {sat_w:.3f} (Color channel spread)", flush=True)
    print(f"  * Exposure Weight (W_e)     : {exp_w:.3f} (Gaussian midtone bell curve)", flush=True)
    print("-" * 60, flush=True)

    # 2. Estimate sky pedestal and report per-bracket photometric metrics
    bg_levels = [float(np.percentile(img, 0.5)) for img in aligned_masters]
    min_bg = min(bg_levels)
    print(f"  * Estimated Global Sky Pedestal: {min_bg:.6f}", flush=True)

    cleaned_frames = [np.maximum(0.0, img - min_bg) for img in aligned_masters]
    global_max = max(float(np.percentile(f, 99.99)) for f in cleaned_frames) or 1.0

    print("\n  Bracket Metrics (Input to Laplacian Pyramid):", flush=True)
    input_frames_32f = []
    for name, f in zip(master_names, cleaned_frames):
        norm = np.clip(f / global_max, 0.0, 1.0).astype(np.float32)
        mean_sig = float(np.mean(norm))
        peak_sig = float(np.max(norm))
        print(f"    - {name:22s} | Mean: {mean_sig:.5f} | Peak: {peak_sig:.4f}", flush=True)
        input_frames_32f.append(norm)

    # 3. Execute OpenCV Mertens Pyramid Merge
    print("\n  * Processing Laplacian and Gaussian multi-scale pyramids...", flush=True)
    merge_mertens = cv2.createMergeMertens(
        contrast_weight=contrast_w,
        saturation_weight=sat_w,
        exposure_weight=exp_w,
    )

    fusion_rgb = merge_mertens.process(input_frames_32f)
    fusion_rgb = np.nan_to_num(fusion_rgb, nan=0.0, posinf=1.0, neginf=0.0)
    fusion_rgb = np.clip(fusion_rgb, 0.0, 1.0)

    # Normalize fusion dynamic range
    f_max = float(np.percentile(fusion_rgb, 99.99)) or 1.0
    fusion_rgb = np.clip(fusion_rgb / f_max, 0.0, 1.0)

    # 4. Save 16-Bit Master TIFF
    fusion_16u = (fusion_rgb * 65535.0).astype(np.uint16)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tifffile.imwrite(
        str(output_path),
        fusion_16u,
        photometric="rgb",
        compression=None,
    )
    print(f"\n  [Exported 16-bit Master TIFF] -> {output_path.resolve()}", flush=True)

    # 5. Export JSON Parameter Log Sidecar
    params_json_path = output_path.with_suffix(".json")
    params_record = {
        "algorithm": "OpenCV MergeMertens Multi-Scale Exposure Fusion",
        "parameters": {
            "contrast_weight": contrast_w,
            "saturation_weight": sat_w,
            "exposure_weight": exp_w,
        },
        "normalization": {
            "sky_pedestal": min_bg,
            "global_max": global_max,
            "fusion_p99_max": f_max,
        },
        "input_brackets": master_names,
    }
    with open(params_json_path, "w", encoding="utf-8") as f:
        json.dump(params_record, f, indent=2)
    print(f"  [Saved Parameter Report]      -> {params_json_path.resolve()}", flush=True)

    # 6. Save Non-Linear Preview JPG (Asinh Stretch)
    preview_jpg = output_path.with_suffix(".jpg")
    stretch_factor = 20.0
    stretched = np.arcsinh(fusion_rgb * stretch_factor) / np.arcsinh(stretch_factor)
    preview_8u = (np.clip(stretched, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(preview_8u, mode="RGB").save(preview_jpg, quality=95)
    print(f"  [Saved Preview JPG]           -> {preview_jpg.resolve()}", flush=True)
    print("=" * 60 + "\n", flush=True)