"""Multi-scale Laplacian pyramid HDR exposure fusion with lunar disk cleaning."""

from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import tifffile


def fuse_and_export_hdr(
    aligned_masters: list[np.ndarray],
    output_path: Path,
    contrast_w: float = 1.0,
    sat_w: float = 1.2,
    exp_w: float = 0.5,
    lunar_mask_margin_px: float = 5.0,
) -> None:
    """Fuses linear bracketed astronomical masters into a 16-bit TIFF and an auto-stretched preview JPG."""
    num_masters = len(aligned_masters)
    if num_masters == 0:
        raise ValueError("No aligned master images provided for fusion.")

    print(
        f"  * Preparing {num_masters} exposure brackets for Mertens fusion...",
        flush=True,
    )

    # 1. Background pedestal subtraction
    bg_levels = [float(np.percentile(img, 0.5)) for img in aligned_masters]
    min_bg = min(bg_levels)

    cleaned_frames_32f = []
    for img in aligned_masters:
        sub = np.maximum(0.0, img - min_bg)
        p99 = float(np.percentile(sub, 99.95)) or 1.0
        norm = np.clip(sub / p99, 0.0, 1.0).astype(np.float32)
        cleaned_frames_32f.append(norm)

    print("  * Running OpenCV Mertens multi-scale exposure fusion...", flush=True)
    merge_mertens = cv2.createMergeMertens(
        contrast_weight=contrast_w,
        saturation_weight=sat_w,
        exposure_weight=exp_w,
    )

    fusion_rgb = merge_mertens.process(cleaned_frames_32f)
    fusion_rgb = np.nan_to_num(fusion_rgb, nan=0.0, posinf=1.0, neginf=0.0)
    fusion_rgb = np.clip(fusion_rgb, 0.0, 1.0)

    # 2. Lunar Disk Masking: eliminate the internal drifting moon ghost
    h, w, _ = fusion_rgb.shape
    cx, cy = w / 2.0, h / 2.0
    r_est = min(h, w) * 0.22 - lunar_mask_margin_px

    y_g, x_g = np.ogrid[:h, :w]
    r_dist = np.hypot(x_g - cx, y_g - cy)
    # Smooth 3-pixel cosine/linear roll-off
    lunar_mask = np.clip((r_dist - (r_est - 2.0)) / 3.0, 0.0, 1.0)[:, :, np.newaxis]
    fusion_rgb *= lunar_mask

    # Normalize highlight dynamic range
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

    # 4. Export preview JPG
    preview_jpg = output_path.with_suffix(".jpg")
    m = 0.08
    stretched = np.where(
        fusion_rgb <= 0.0,
        0.0,
        ((m - 1.0) * fusion_rgb) / (((2.0 * m - 1.0) * fusion_rgb) - m),
    )
    preview_8u = (np.clip(stretched, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(preview_8u, mode="RGB").save(preview_jpg, quality=95)
    print(f"  [Saved Preview JPG] -> {preview_jpg.resolve()}", flush=True)
