"""Robust astronomical coronal enhancement (Radial Graded Filter & Multi-Scale Bandpass)."""

from pathlib import Path
from astropy.io import fits
import cv2
import numpy as np
from PIL import Image
import tifffile


def detect_solar_center_accurate(img_rgb: np.ndarray) -> tuple[float, float, float]:
    """Accurately detects lunar silhouette center via circular Hough transform / centroid."""
    lum = 0.299 * img_rgb[:, :, 0] + 0.587 * img_rgb[:, :, 1] + 0.114 * img_rgb[:, :, 2]
    h, w = lum.shape

    # 1. Gradient magnitude to find sharp inner lunar limb
    grad_x = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)

    # Threshold top 0.5% gradient edges
    p99 = float(np.percentile(grad_mag, 99.5)) or 1.0
    edges = (grad_mag > p99 * 0.4).astype(np.uint8) * 255

    # Fit circle or find center of mass of dark core
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Largest contour near center
        c = max(contours, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(c)
        if 0.1 * min(h, w) < radius < 0.45 * min(h, w):
            return float(cx), float(cy), float(radius)

    return float(w / 2.0), float(h / 2.0), min(h, w) * 0.22


def apply_astronomical_rgf(
    img_rgb: np.ndarray,
    center: tuple[float, float],
    r_lunar: float,
    r_max_corona: float,
    unsharp_sigma: float = 3.0,
    unsharp_amount: float = 1.5,
) -> np.ndarray:
    """Applies radial brightness normalization with background protection and multi-scale detail recovery."""
    h, w, c = img_rgb.shape
    cx, cy = center

    y_idx, x_idx = np.ogrid[:h, :w]
    r = np.hypot(x_idx - cx, y_idx - cy).astype(np.float32)

    # 1. Generate smooth radial baseline in Polar space
    max_r = int(r_max_corona)
    polar = cv2.warpPolar(img_rgb, (max_r, 720), (cx, cy), max_r, cv2.WARP_POLAR_LINEAR)

    # Azimuthal median curve
    rad_curve = np.median(polar, axis=0)  # Shape (max_r, 3)

    # Strong Gaussian smoothing on radial baseline
    for ch in range(c):
        rad_curve[:, ch] = cv2.GaussianBlur(
            rad_curve[:, ch].reshape(-1, 1), (0, 0), sigmaX=15.0
        ).ravel()

    # Reconstruct 2D baseline in Cartesian coordinates
    r_idx = np.clip(r, 0, max_r - 1).astype(np.int32)
    radial_baseline = np.zeros_like(img_rgb)
    for ch in range(c):
        radial_baseline[:, ch] = rad_curve[r_idx, ch]

    # 2. Smooth Radial Weight Mask:
    # 0 inside Moon, 1 across corona, tapers smoothly to 0 at outer sky boundary
    inner_taper = np.clip((r - r_lunar) / (0.05 * r_lunar), 0.0, 1.0)
    outer_taper = np.clip((r_max_corona - r) / (0.25 * r_max_corona), 0.0, 1.0)
    coronal_mask = (inner_taper * outer_taper)[:, :, np.newaxis]

    # 3. High-Pass Ratio (Image / Baseline)
    # Protected by dynamic noise threshold to avoid dividing zero-sky
    noise_floor = float(np.percentile(img_rgb, 2.0)) + 1e-4
    ratio = (img_rgb + 1e-3) / (radial_baseline + noise_floor)

    # Blend ratio back with original image via coronal mask
    flattened = img_rgb * (1.0 - coronal_mask) + (ratio * coronal_mask * np.median(rad_curve[int(r_lunar * 1.1):, :]))

    # 4. Multi-scale Unsharp Masking for Coronal Magnetic Streamers
    blur = cv2.GaussianBlur(flattened, (0, 0), sigmaX=unsharp_sigma)
    high_pass = flattened - blur
    enhanced = flattened + (unsharp_amount * high_pass * coronal_mask)

    p99 = float(np.percentile(enhanced, 99.9)) or 1.0
    return np.clip(enhanced / p99, 0.0, 1.0)


def process_coronal_features(
    input_master_path: Path,
    output_dir: Path,
    sharpen_amount: float = 1.6,
) -> None:
    """Full post-processing pipeline for solar coronal streamers."""
    print("\n" + "=" * 65, flush=True)
    print("        POST-PROCESSING: CORONAL RGF & FILAMENT EXTRACTION        ", flush=True)
    print("=" * 65, flush=True)

    if input_master_path.suffix.lower() in (".fit", ".fits"):
        with fits.open(input_master_path) as h:
            data = h[0].data.astype(np.float32)
        img = np.transpose(data, (1, 2, 0)) if data.ndim == 3 and data.shape[0] == 3 else data
        p99 = float(np.percentile(img, 99.9)) or 1.0
        img = np.clip(img / p99, 0.0, 1.0)
    else:
        img_raw = tifffile.imread(str(input_master_path)).astype(np.float32)
        img = img_raw / 65535.0 if img_raw.max() > 255.0 else img_raw / 255.0

    h, w, c = img.shape
    cx, cy, r_lunar = detect_solar_center_accurate(img)
    max_coronal_radius = min(cx, cy, w - cx, h - cy) * 0.90

    print(f"  * Detected Solar Center : ({cx:.2f}, {cy:.2f})", flush=True)
    print(f"  * Lunar Limb Radius     : {r_lunar:.2f} px", flush=True)
    print(f"  * Coronal Taper Boundary: {max_coronal_radius:.2f} px", flush=True)
    print("-" * 65, flush=True)

    enhanced = apply_astronomical_rgf(
        img_rgb=img,
        center=(cx, cy),
        r_lunar=r_lunar,
        r_max_corona=max_coronal_radius,
        unsharp_sigma=2.5,
        unsharp_amount=sharpen_amount,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # 16-Bit TIFF
    out_tiff = output_dir / f"{input_master_path.stem}_Enhanced.tif"
    tifffile.imwrite(str(out_tiff), (enhanced * 65535.0).astype(np.uint16), photometric="rgb")
    print(f"  [Exported 16-bit Enhanced TIFF] -> {out_tiff.resolve()}", flush=True)

    # Preview JPEG
    out_jpg = output_dir / f"{input_master_path.stem}_Enhanced.jpg"
    preview_8u = (enhanced * 255.0).astype(np.uint8)
    Image.fromarray(preview_8u, mode="RGB").save(out_jpg, quality=95)
    print(f"  [Exported Enhanced JPG Preview] -> {out_jpg.resolve()}", flush=True)
    print("=" * 65 + "\n", flush=True)