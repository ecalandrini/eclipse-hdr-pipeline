"""Multi-scale Laplacian pyramid HDR exposure fusion preserving exposure hierarchy."""

from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import tifffile


def fuse_and_export_hdr(
    aligned_masters: list[np.ndarray],
    output_path: Path,
    contrast_w: float = 0.8,
    sat_w: float = 1.0,
    exp_w: float = 0.8,
) -> None:
    """Fuses bracketed astronomical masters preserving dynamic range and outer streamers."""
    num_masters = len(aligned_masters)
    if num_masters == 0:
        raise ValueError("No aligned master images provided for fusion.")

    print(
        f"  * Preparing {num_masters} exposure brackets for Mertens fusion...",
        flush=True,
    )

    # 1. Estimate global background pedestal
    bg_levels = [float(np.percentile(img, 0.5)) for img in aligned_masters]
    min_bg = min(bg_levels)

    # 2. Subtract background WITHOUT individual per-frame dynamic clipping
    cleaned_frames = [np.maximum(0.0, img - min_bg) for img in aligned_masters]

    # Global maximum across all frames to keep physical exposure hierarchy intact
    global_max = max(float(np.percentile(f, 99.99)) for f in cleaned_frames) or 1.0

    input_frames_32f = []
    for f in cleaned_frames:
        # Scale to [0.0, 1.0] using global max so short exposures stay small and long exposures span higher
        norm = np.clip(f / global_max, 0.0, 1.0).astype(np.float32)
        input_frames_32f.append(norm)

    print("  * Running OpenCV Mertens multi-scale pyramid fusion...", flush=True)
    merge_mertens = cv2.createMergeMertens(
        contrast_weight=contrast_w,
        saturation_weight=sat_w,
        exposure_weight=exp_w,
    )

    fusion_rgb = merge_mertens.process(input_frames_32f)
    fusion_rgb = np.nan_to_num(fusion_rgb, nan=0.0, posinf=1.0, neginf=0.0)
    fusion_rgb = np.clip(fusion_rgb, 0.0, 1.0)

    # Normalize fusion highlights
    f_max = float(np.percentile(fusion_rgb, 99.99)) or 1.0
    fusion_rgb = np.clip(fusion_rgb / f_max, 0.0, 1.0)

    # 3. Export 16-bit Master TIFF
    fusion_16u = (fusion_rgb * 65535.0).astype(np.uint16)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tifffile.imwrite(
        str(output_path),
        fusion_16u,
        photometric="rgb",
        compression=None,
    )
    print(f"  [Saved 16-bit TIFF] -> {output_path.resolve()}", flush=True)

    # 4. Export preview JPG using Asinh stretch to reveal outer coronal streamers
    preview_jpg = output_path.with_suffix(".jpg")

    # Asinh stretch
    stretch_factor = 20.0
    stretched = np.arcsinh(fusion_rgb * stretch_factor) / np.arcsinh(stretch_factor)
    preview_8u = (np.clip(stretched, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(preview_8u, mode="RGB").save(preview_jpg, quality=95)
    print(f"  [Saved Preview JPG] -> {preview_jpg.resolve()}", flush=True)
