"""Multi-scale Laplacian pyramid HDR exposure fusion (Mertens et al.) optimized for astrophotography."""

from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import tifffile


def fuse_and_export_hdr(
    aligned_masters: list[np.ndarray],
    output_path: Path,
    contrast_w: float = 1.5,
    sat_w: float = 1.0,
    exp_w: float = 0.0,
) -> None:
    """Fuses linear bracketed astronomical masters into a 16-bit TIFF and an auto-stretched preview JPG."""
    num_masters = len(aligned_masters)
    if num_masters == 0:
        raise ValueError("No aligned master images provided for fusion.")

    print(
        f"  * Calibrating and scaling {num_masters} exposure brackets for Mertens fusion...",
        flush=True,
    )

    # 1. Determine the global background level across all masters
    bg_levels = [float(np.percentile(img, 1.0)) for img in aligned_masters]
    min_bg = min(bg_levels)
    print(f"  * Detected background pedestal: {min_bg:.5f}", flush=True)

    # 2. Subtract pedestal and find global peak across the entire bracket series
    cleaned_frames = []
    for img in aligned_masters:
        sub = np.maximum(0.0, img - min_bg)
        cleaned_frames.append(sub)

    global_max = max(float(np.max(f)) for f in cleaned_frames) or 1.0
    print(
        f"  * Dynamic range peak after background subtraction: {global_max:.5f}",
        flush=True,
    )

    # 3. Normalize brackets so the highest dynamic range fills [0.0, 1.0]
    input_frames_32f = []
    input_frames_8u = []
    for f in cleaned_frames:
        norm = np.clip(f / global_max, 0.0, 1.0).astype(np.float32)
        bgr_32f = cv2.cvtColor(norm, cv2.COLOR_RGB2BGR)
        bgr_8u = (norm * 255.0).astype(np.uint8)

        input_frames_32f.append(bgr_32f)
        input_frames_8u.append(bgr_8u)

    print("  * Running Mertens Laplacian exposure fusion...", flush=True)
    # Using exp_weight=0.0 relies strictly on local contrast and saturation gradients,
    # preventing faint linear astronomical signals from being suppressed.
    merge_mertens = cv2.createMergeMertens(
        contrast_weight=contrast_w,
        saturation_weight=sat_w,
        exposure_weight=exp_w,
    )

    try:
        fusion_bgr = merge_mertens.process(input_frames_8u)
    except Exception:
        fusion_bgr = merge_mertens.process(input_frames_32f)

    # Convert BGR back to RGB
    fusion_rgb = cv2.cvtColor(fusion_bgr, cv2.COLOR_BGR2RGB)
    fusion_rgb = np.clip(fusion_rgb, 0.0, 1.0)

    # Normalize fusion output to maximize dynamic range
    fusion_max = float(np.max(fusion_rgb)) or 1.0
    fusion_rgb /= fusion_max

    # 4. Export 16-bit Master TIFF
    fusion_16u = (np.clip(fusion_rgb, 0.0, 1.0) * 65535.0).astype(np.uint16)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tifffile.imwrite(
        str(output_path),
        fusion_16u,
        photometric="rgb",
        compression=None,
    )
    print(f"  [Saved 16-bit TIFF] -> {output_path.resolve()}", flush=True)

    # 5. Export Tonemapped Preview JPEG (Astronomical Midtone Transfer Function)
    preview_jpg = output_path.with_suffix(".jpg")
    m = 0.05  # Midtone balance (lower value brightens faint streamers)
    stretched = np.where(
        fusion_rgb <= 0.0,
        0.0,
        ((m - 1.0) * fusion_rgb) / (((2.0 * m - 1.0) * fusion_rgb) - m),
    )
    preview_8u = (np.clip(stretched, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(preview_8u, mode="RGB").save(preview_jpg, quality=95)
    print(f"  [Saved Preview JPG] -> {preview_jpg.resolve()}", flush=True)
