"""Artifact-free multi-scale coronal enhancement using logarithmic bandpass filtering."""

from pathlib import Path
from astropy.io import fits
import cv2
import numpy as np
from PIL import Image
import tifffile


def detect_solar_center(img_rgb: np.ndarray) -> tuple[float, float, float]:
    """Estimates the solar center (cx, cy) and approximate lunar radius in pixel coordinates."""
    lum = 0.299 * img_rgb[:, :, 0] + 0.587 * img_rgb[:, :, 1] + 0.114 * img_rgb[:, :, 2]
    h, w = lum.shape

    blur = cv2.GaussianBlur((np.clip(lum, 0.0, 1.0) * 255.0).astype(np.uint8), (9, 9), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(c)
        if 0.1 * min(h, w) < radius < 0.45 * min(h, w):
            return float(cx), float(cy), float(radius)

    return float(w / 2.0), float(h / 2.0), min(h, w) * 0.22


def multiscale_coronal_enhancement(
    img_rgb: np.ndarray,
    center: tuple[float, float],
    r_lunar: float,
    compression_gamma: float = 0.5,
    fine_detail_boost: float = 1.4,
    medium_detail_boost: float = 1.2,
) -> np.ndarray:
    """Enhances fine magnetic streamers and loops across the whole field without circular artifacts."""
    h, w, c = img_rgb.shape
    cx, cy = center

    # 1. Lunar silhouette protection mask
    y_idx, x_idx = np.ogrid[:h, :w]
    dist_from_moon = np.hypot(x_idx - cx, y_idx - cy).astype(np.float32)
    moon_mask = np.clip((dist_from_moon - (r_lunar * 0.98)) / (0.04 * r_lunar), 0.0, 1.0)
    moon_mask = np.repeat(moon_mask[:, :, np.newaxis], c, axis=2)

    # 2. Smooth non-linear dynamic range compression
    base_compressed = np.power(np.clip(img_rgb, 0.0, 1.0), compression_gamma)

    # 3. Multi-Scale Frequency Decomposition
    blur_fine = cv2.GaussianBlur(base_compressed, (0, 0), sigmaX=2.0)
    high_freq = base_compressed - blur_fine

    blur_med = cv2.GaussianBlur(base_compressed, (0, 0), sigmaX=8.0)
    blur_coarse = cv2.GaussianBlur(base_compressed, (0, 0), sigmaX=32.0)
    med_freq = blur_med - blur_coarse

    # 4. Detail injection
    enhanced = (
        base_compressed
        + (fine_detail_boost * high_freq * moon_mask)
        + (medium_detail_boost * med_freq * moon_mask)
    )

    enhanced = enhanced * moon_mask

    # 5. Global normalisation
    p99 = float(np.percentile(enhanced[enhanced > 0], 99.9)) or 1.0
    return np.clip(enhanced / p99, 0.0, 1.0)


def process_coronal_features(
    input_master_path: Path,
    output_dir: Path,
    sharpen_amount: float = 1.4,
) -> None:
    """Post-processing pipeline for total eclipse HDR composites."""
    print("\n" + "=" * 65, flush=True)
    print("       POST-PROCESSING: MULTI-SCALE CORONAL DETAIL EXTRACTION     ", flush=True)
    print("=" * 65, flush=True)
    print(f"  * Input File            : {input_master_path.resolve()}", flush=True)

    if input_master_path.suffix.lower() in (".fit", ".fits"):
        with fits.open(input_master_path) as h:
            data = h[0].data.astype(np.float32)
        if data.ndim == 3:
            img = np.transpose(data, (1, 2, 0)) if data.shape[0] == 3 else data
        elif data.ndim == 2:
            img = np.repeat(data[:, :, np.newaxis], 3, axis=2)
        else:
            raise ValueError(f"Unsupported FITS shape: {data.shape}")
        p99 = float(np.percentile(img[img > 0], 99.9)) or 1.0
        img = np.clip(img / p99, 0.0, 1.0)
    else:
        img_raw = tifffile.imread(str(input_master_path)).astype(np.float32)
        if img_raw.ndim == 2:
            img_raw = np.repeat(img_raw[:, :, np.newaxis], 3, axis=2)
        img = img_raw / 65535.0 if img_raw.max() > 255.0 else img_raw / 255.0

    cx, cy, r_lunar = detect_solar_center(img)
    print(f"  * Detected Lunar Center : ({cx:.2f}, {cy:.2f})", flush=True)
    print(f"  * Lunar Limb Radius     : {r_lunar:.2f} px", flush=True)
    print(f"  * Sharpening Multiplier : {sharpen_amount:.2f}", flush=True)
    print("-" * 65, flush=True)

    enhanced = multiscale_coronal_enhancement(
        img_rgb=img,
        center=(cx, cy),
        r_lunar=r_lunar,
        compression_gamma=0.5,
        fine_detail_boost=sharpen_amount,
        medium_detail_boost=sharpen_amount * 0.85,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    out_tiff = output_dir / f"{input_master_path.stem}_Enhanced.tif"
    tifffile.imwrite(str(out_tiff), (enhanced * 65535.0).astype(np.uint16), photometric="rgb")
    print(f"  [Exported 16-bit Enhanced TIFF] -> {out_tiff.resolve()}", flush=True)

    out_jpg = output_dir / f"{input_master_path.stem}_Enhanced.jpg"
    preview_8u = (enhanced * 255.0).astype(np.uint8)
    Image.fromarray(preview_8u, mode="RGB").save(out_jpg, quality=95)
    print(f"  [Exported Enhanced JPG Preview] -> {out_jpg.resolve()}", flush=True)
    print("=" * 65 + "\n", flush=True)