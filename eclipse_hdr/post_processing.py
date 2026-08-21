"""Coronal enhancement: Radial-Graded Filter (RGF), Larson-Sekanina, and Wavelet Sharpening."""

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

    # Find the bright inner coronal ring via Otsu threshold
    blur = cv2.GaussianBlur((np.clip(lum, 0.0, 1.0) * 255.0).astype(np.uint8), (9, 9), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours to locate the inner limb boundary
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find contour with largest area or closest circular bounding box
        c = max(contours, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(c)
        return float(cx), float(cy), float(radius)

    return float(w / 2.0), float(h / 2.0), min(h, w) * 0.22


def apply_radial_graded_filter(
    img_rgb: np.ndarray,
    center: tuple[float, float],
    min_radius: float,
    max_radius: float,
    gamma: float = 2.2,
) -> np.ndarray:
    """Applies a Radial-Graded Filter (RGF) to compensate for the steep coronal brightness drop."""
    h, w, c = img_rgb.shape
    cx, cy = center

    y_idx, x_idx = np.ogrid[:h, :w]
    r = np.hypot(x_idx - cx, y_idx - cy).astype(np.float32)

    # Convert to polar coordinates for clean azimuthal normalization
    max_r = int(max_radius)
    polar_img = cv2.warpPolar(
        img_rgb,
        (max_r, 720),
        (cx, cy),
        max_radius,
        cv2.WARP_POLAR_LINEAR,
    )

    # Compute azimuthal median profile across angles
    radial_profile = np.median(polar_img, axis=0)  # Shape (max_r, 3)
    
    # Smooth radial profile with 1D Gaussian
    for ch in range(c):
        radial_profile[:, ch] = cv2.GaussianBlur(
            radial_profile[:, ch].reshape(-1, 1), (0, 0), sigmaX=5.0
        ).ravel()

    # Create 2D normalization map in Cartesian space
    r_clipped = np.clip(r, 0, max_r - 1).astype(np.int32)
    norm_map = np.zeros_like(img_rgb)
    for ch in range(c):
        norm_map[:, :, ch] = radial_profile[r_clipped, ch]

    # Protect inner lunar disk
    inner_mask = np.clip((r - min_radius * 0.95) / (min_radius * 0.1), 0.0, 1.0)
    inner_mask = np.repeat(inner_mask[:, :, np.newaxis], c, axis=2)

    # Divide by radial gradient
    flattened = np.where(
        norm_map > 1e-4,
        (img_rgb / (norm_map + 1e-4)) * inner_mask,
        0.0,
    )

    # Normalize output dynamic range
    p99 = float(np.percentile(flattened, 99.8)) or 1.0
    return np.clip(flattened / p99, 0.0, 1.0)


def apply_unsharp_mask(
    img_rgb: np.ndarray,
    sigma: float = 2.5,
    amount: float = 1.5,
) -> np.ndarray:
    """Multi-scale sharpening for fine coronal filaments and magnetic loops."""
    blurred = cv2.GaussianBlur(img_rgb, (0, 0), sigmaX=sigma)
    sharpened = img_rgb + (amount * (img_rgb - blurred))
    return np.clip(sharpened, 0.0, 1.0)


def process_coronal_features(
    input_master_path: Path,
    output_dir: Path,
    gamma: float = 2.2,
    sharpen_amount: float = 1.8,
) -> None:
    """Full post-processing workflow: Radial Flattening + Magnetic Filament Sharpening."""
    print("\n" + "=" * 65, flush=True)
    print("        POST-PROCESSING: CORONAL RGF & FILAMENT EXTRACTION        ", flush=True)
    print("=" * 65, flush=True)

    # 1. Load HDR Master (TIFF or FITS)
    if input_master_path.suffix.lower() in (".fit", ".fits"):
        with fits.open(input_master_path) as h:
            data = h[0].data.astype(np.float32)
        if data.shape[0] == 3:
            img = np.transpose(data, (1, 2, 0))
        else:
            img = data
        p99 = float(np.percentile(img, 99.9)) or 1.0
        img = np.clip(img / p99, 0.0, 1.0)
    else:
        img_raw = tifffile.imread(str(input_master_path)).astype(np.float32)
        img = img_raw / 65535.0 if img_raw.max() > 255.0 else img_raw / 255.0

    h, w, c = img.shape
    cx, cy, r_lunar = detect_solar_center(img)
    max_r = min(cx, cy, w - cx, h - cy)

    print(f"  * Detected Solar Center : ({cx:.2f}, {cy:.2f})", flush=True)
    print(f"  * Lunar Limb Radius     : {r_lunar:.2f} px", flush=True)
    print(f"  * Maximum Coronal Radius: {max_r:.2f} px", flush=True)
    print("-" * 65, flush=True)

    # 2. Stage A: Radial Graded Filter (RGF)
    print("  * Applying Radial-Graded Filter (RGF)...", flush=True)
    rgf_result = apply_radial_graded_filter(
        img_rgb=img,
        center=(cx, cy),
        min_radius=r_lunar,
        max_radius=max_r,
        gamma=gamma,
    )

    # 3. Stage B: Multi-Scale Sharpening for Magnetic Loops
    print("  * Extracting fine coronal streamers via unsharp masking...", flush=True)
    enhanced = apply_unsharp_mask(rgf_result, sigma=1.5, amount=sharpen_amount)

    # 4. Save Outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    # 16-bit TIFF
    rgf_tiff_path = output_dir / f"{input_master_path.stem}_RGF_Enhanced.tif"
    tifffile.imwrite(
        str(rgf_tiff_path),
        (enhanced * 65535.0).astype(np.uint16),
        photometric="rgb",
    )
    print(f"  [Exported Enhanced 16-bit TIFF] -> {rgf_tiff_path.resolve()}", flush=True)

    # High Quality JPEG Preview
    rgf_jpg_path = output_dir / f"{input_master_path.stem}_RGF_Enhanced.jpg"
    preview_8u = (enhanced * 255.0).astype(np.uint8)
    Image.fromarray(preview_8u, mode="RGB").save(rgf_jpg_path, quality=95)
    print(f"  [Exported Enhanced JPG Preview]  -> {rgf_jpg_path.resolve()}", flush=True)
    print("=" * 65 + "\n", flush=True)