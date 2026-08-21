"""Miloslav Druckmüller-style Adaptive Contrast Enhancement (ACF) in Polar Space."""

from pathlib import Path
from astropy.io import fits
import cv2
import numpy as np
from PIL import Image
import tifffile


def load_master_for_enhancement(input_path: Path) -> tuple[np.ndarray, bool]:
    """Loads master array and detects dynamic range format."""
    ext = input_path.suffix.lower()
    is_linear = False

    if ext in (".fit", ".fits"):
        is_linear = True
        with fits.open(input_path, memmap=False) as hdul:
            data = hdul[0].data.astype(np.float32)
        if data.ndim == 3:
            img = np.transpose(data, (1, 2, 0)) if data.shape[0] == 3 else data
        elif data.ndim == 2:
            img = np.repeat(data[:, :, np.newaxis], 3, axis=2)
        else:
            raise ValueError(f"Unsupported FITS shape: {data.shape}")

        # Protect positive values
        p99 = float(np.percentile(img[img > 0], 99.95)) if np.any(img > 0) else 1.0
        img = np.clip(img / max(p99, 1e-6), 0.0, 1.0)
    else:
        raw = tifffile.imread(str(input_path))
        if raw.ndim == 2:
            raw = np.repeat(raw[:, :, np.newaxis], 3, axis=2)
        elif raw.ndim == 3 and raw.shape[2] > 3:
            raw = raw[:, :, :3]

        if raw.dtype == np.uint16:
            img = raw.astype(np.float32) / 65535.0
        elif raw.dtype == np.uint8:
            img = raw.astype(np.float32) / 255.0
        else:
            img = raw.astype(np.float32)
            if "linear" in input_path.stem.lower():
                is_linear = True

        p_max = float(np.percentile(img, 99.99)) or 1.0
        img = np.clip(img / p_max, 0.0, 1.0)

    return img, is_linear


def detect_solar_center_accurate(img_rgb: np.ndarray) -> tuple[float, float, float]:
    """Detects solar center (cx, cy) and lunar radius in pixel coordinates."""
    lum = 0.299 * img_rgb[:, :, 0] + 0.587 * img_rgb[:, :, 1] + 0.114 * img_rgb[:, :, 2]
    h, w = lum.shape

    grad_x = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)

    p99 = float(np.percentile(grad_mag, 99.5)) or 1.0
    edges = (grad_mag > p99 * 0.4).astype(np.uint8) * 255

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(c)
        if 0.1 * min(h, w) < radius < 0.45 * min(h, w):
            return float(cx), float(cy), float(radius)

    return float(w / 2.0), float(h / 2.0), min(h, w) * 0.22


def apply_druckmuller_acf(
    img_rgb: np.ndarray,
    center: tuple[float, float],
    r_lunar: float,
    boost: float = 1.8,
) -> np.ndarray:
    """Adaptive Contrast Enhancement (ACF) in polar coordinates.
    
    Transforms the frame to polar (r, theta), computes azimuthal moving-window
    mean and standard deviation, standardizes each radial ring to isolate
    tangential magnetic structures, and maps back to Cartesian coordinates.
    """
    h, w, c = img_rgb.shape
    cx, cy = center
    max_r = int(min(cx, cy, w - cx, h - cy) * 0.98)
    n_theta = 1440  # High angular resolution

    # 1. Non-linear log pre-stretch so noise floor doesn't dominate outer bands
    log_img = np.log10(np.maximum(img_rgb, 1e-5))
    
    # 2. Warp to Polar Coordinates (r, theta)
    polar_img = cv2.warpPolar(
        log_img,
        (max_r, n_theta),
        (cx, cy),
        max_r,
        cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS,
    )

    # 3. For each color channel, compute moving angular baseline and deviation
    polar_enhanced = np.zeros_like(polar_img)

    for ch in range(c):
        plane = polar_img[:, :, ch]

        # Low-pass filter along theta (angular direction) using large kernel
        # Wrap borders along theta for 360-degree continuity
        plane_padded = np.vstack([plane[-90:, :], plane, plane[:90, :]])
        mu = cv2.GaussianBlur(plane_padded, (1, 101), sigmaX=0, sigmaY=25.0)[90:-90, :]

        # High-pass angular residual
        diff = plane - mu

        # Standard deviation along theta
        diff_sq = cv2.GaussianBlur(
            np.vstack([diff[-90:, :] ** 2, diff ** 2, diff[:90, :] ** 2]),
            (1, 101),
            sigmaX=0,
            sigmaY=25.0,
        )[90:-90, :]
        sigma = np.sqrt(np.maximum(diff_sq, 1e-4))

        # Normalized adaptive contrast (Druckmüller quotient)
        norm_plane = diff / (sigma + 0.05)

        # Re-scale back to displayable dynamic range
        polar_enhanced[:, :, ch] = norm_plane

    # 4. Warp back to Cartesian coordinates
    enhanced_cartesian = cv2.warpPolar(
        polar_enhanced,
        (w, h),
        (cx, cy),
        max_r,
        cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP,
    )

    # 5. Lunar limb protection & blending with original frame
    y_idx, x_idx = np.ogrid[:h, :w]
    dist_r = np.hypot(x_idx - cx, y_idx - cy).astype(np.float32)

    # Mask: 0 inside Moon, 1 in corona, smoothly tapers to 0 at extreme frame edge
    coronal_mask = np.clip((dist_r - r_lunar) / (0.04 * r_lunar), 0.0, 1.0)
    outer_mask = np.clip((max_r - dist_r) / (0.15 * max_r), 0.0, 1.0)
    blend_mask = (coronal_mask * outer_mask)[:, :, np.newaxis]

    # Combine normalized high-pass details with compressed base
    base_curved = np.arcsinh(img_rgb * 30.0) / np.arcsinh(30.0)
    
    # Scale enhanced details to visible range
    p99_enh = float(np.percentile(np.abs(enhanced_cartesian), 99.5)) or 1.0
    norm_details = np.clip(enhanced_cartesian / (p99_enh * 1.5), -1.0, 1.0)

    final_rgb = base_curved + (boost * norm_details * blend_mask * 0.4)
    final_rgb = final_rgb * coronal_mask[:, :, np.newaxis]

    p99_final = float(np.percentile(final_rgb[final_rgb > 0], 99.9)) or 1.0
    return np.clip(final_rgb / p99_final, 0.0, 1.0).astype(np.float32)


def process_coronal_features(
    input_master_path: Path,
    output_dir: Path,
    algorithm: str = "druckmuller",
    sharpen_amount: float = 1.8,
) -> None:
    """Full coronal feature enhancement pipeline."""
    print("\n" + "=" * 65, flush=True)
    print("       CORONAL FEATURE ENHANCEMENT (DRUCKMÜLLER ACF)              ", flush=True)
    print("=" * 65, flush=True)
    print(f"  * Input File            : {input_master_path.resolve()}", flush=True)
    print(f"  * Algorithm             : DRUCKMÜLLER POLAR ACF", flush=True)
    print(f"  * Boost Intensity       : {sharpen_amount:.2f}", flush=True)

    img, is_linear = load_master_for_enhancement(input_master_path)
    cx, cy, r_lunar = detect_solar_center_accurate(img)
    print(f"  * Solar Centroid        : ({cx:.2f}, {cy:.2f})", flush=True)
    print(f"  * Lunar Radius          : {r_lunar:.2f} px", flush=True)
    print("-" * 65, flush=True)

    print("  * Computing Polar Adaptive Contrast Normalization...", flush=True)
    enhanced = apply_druckmuller_acf(
        img_rgb=img,
        center=(cx, cy),
        r_lunar=r_lunar,
        boost=sharpen_amount,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # 16-Bit Master TIFF
    out_tiff = output_dir / f"{input_master_path.stem}_Druckmuller_Enhanced.tif"
    tifffile.imwrite(
        str(out_tiff),
        (enhanced * 65535.0).astype(np.uint16),
        photometric="rgb",
    )
    print(f"  [Exported 16-bit Enhanced TIFF] -> {out_tiff.resolve()}", flush=True)

    # Preview JPG
    out_jpg = output_dir / f"{input_master_path.stem}_Druckmuller_Enhanced.jpg"
    preview_8u = (enhanced * 255.0).astype(np.uint8)
    Image.fromarray(preview_8u, mode="RGB").save(out_jpg, quality=95)
    print(f"  [Exported Enhanced JPG Preview] -> {out_jpg.resolve()}", flush=True)
    print("=" * 65 + "\n", flush=True)