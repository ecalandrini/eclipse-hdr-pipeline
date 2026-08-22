"""Coronal enhancement module using Ray-Casting Limb Fitting and Druckmüller Polar ACF."""

from pathlib import Path
from astropy.io import fits
import cv2
import numpy as np
from PIL import Image
import tifffile


def load_master_for_enhancement(input_path: Path) -> tuple[np.ndarray, bool]:
    """Loads FITS/TIFF master and normalizes to [0.0, 1.0] float32 RGB."""
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

        p99 = float(np.percentile(img[img > 0], 99.95)) if np.any(img > 0) else 1.0
        img = np.clip(img / max(p99, 1e-6), 0.0, 1.0)
    else:
        raw = tifffile.imread(str(input_path)).astype(np.float32)
        if raw.ndim == 2:
            raw = np.repeat(raw[:, :, np.newaxis], 3, axis=2)
        elif raw.ndim == 3 and raw.shape[2] > 3:
            raw = raw[:, :, :3]

        if raw.dtype == np.uint16:
            img = raw / 65535.0
        elif raw.dtype == np.uint8:
            img = raw / 255.0
        else:
            img = raw
            if "linear" in input_path.stem.lower():
                is_linear = True

        p_max = float(np.percentile(img[img > 0], 99.95)) if np.any(img > 0) else 1.0
        img = np.clip(img / max(p_max, 1e-6), 0.0, 1.0)

    return img, is_linear


def solve_center_and_limb_raycast(
    img_rgb: np.ndarray,
    n_rays: int = 360,
    r_min: int = 100,
    r_max: int = 380,
) -> tuple[float, float, float]:
    """Deterministically finds sub-pixel lunar center and radius via radial gradient ray-casting."""
    lum = 0.299 * img_rgb[..., 0] + 0.587 * img_rgb[..., 1] + 0.114 * img_rgb[..., 2]
    h, w = lum.shape
    cx_init, cy_init = float(w / 2.0), float(h / 2.0)

    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    radii = np.arange(r_min, r_max, 1.0, dtype=np.float32)

    limb_xs, limb_ys = [], []

    for theta in angles:
        ray_x = cx_init + radii * np.cos(theta)
        ray_y = cy_init + radii * np.sin(theta)

        valid = (ray_x >= 0) & (ray_x < w - 1) & (ray_y >= 0) & (ray_y < h - 1)
        if not np.all(valid):
            continue

        profile = cv2.remap(
            lum.astype(np.float32),
            ray_x.astype(np.float32).reshape(1, -1),
            ray_y.astype(np.float32).reshape(1, -1),
            interpolation=cv2.INTER_LINEAR,
        ).ravel()

        d_profile = np.gradient(profile)
        peak_idx = int(np.argmax(d_profile))
        r_peak = radii[peak_idx]

        limb_xs.append(cx_init + r_peak * np.cos(theta))
        limb_ys.append(cy_init + r_peak * np.sin(theta))

    xs = np.array(limb_xs, dtype=np.float64)
    ys = np.array(limb_ys, dtype=np.float64)

    # Kåsa Algebraic Least Squares Circle Fit
    A = np.column_stack([2.0 * xs, 2.0 * ys, np.ones_like(xs)])
    b = xs**2 + ys**2
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    cx = float(sol[0])
    cy = float(sol[1])
    r_lunar = float(np.sqrt(sol[2] + cx**2 + cy**2))

    return cx, cy, r_lunar


def apply_druckmuller_polar_acf(
    img_rgb: np.ndarray,
    center: tuple[float, float],
    r_lunar: float,
    asinh_beta: float = 20.0,
    boost: float = 1.2,
) -> np.ndarray:
    """Miloslav Druckmüller Polar ACF with Background Pedestal Removal and Luminance SNR Gating."""
    h, w, c = img_rgb.shape
    cx, cy = center
    max_r = int(min(cx, cy, w - cx, h - cy) * 0.95)
    n_theta = 1440

    # 1. Background Pedestal Subtraction
    # Sample corner pixels to find the true sky background floor
    border_px = np.concatenate(
        [
            img_rgb[:25, :, :].reshape(-1, c),
            img_rgb[-25:, :, :].reshape(-1, c),
            img_rgb[:, :25, :].reshape(-1, c),
            img_rgb[:, -25:, :].reshape(-1, c),
        ],
        axis=0,
    )
    bg_pedestal = np.median(border_px, axis=0)
    bg_std = np.std(border_px, axis=0)

    # Clean flux above background
    clean_flux = np.maximum(0.0, img_rgb - bg_pedestal)

    # Normalize clean flux
    p99 = np.percentile(clean_flux, 99.9, axis=(0, 1)) + 1e-6
    clean_norm = np.clip(clean_flux / p99, 0.0, 1.0)

    # 2. Perceptual Dynamic Range Compression
    stretched = np.arcsinh(clean_norm * asinh_beta) / np.arcsinh(asinh_beta)

    # 3. Polar Warp
    polar_img = cv2.warpPolar(
        stretched,
        (max_r, n_theta),
        (cx, cy),
        max_r,
        cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS,
    )

    # 4. 2D Low-Pass Background Filter in Polar Space
    polar_high_pass = np.zeros_like(polar_img)
    pad = 120

    for ch in range(c):
        plane = polar_img[:, :, ch]
        padded = np.vstack([plane[-pad:, :], plane, plane[:pad, :]])

        # 2D Gaussian Kernel (sigmaX=3.0px across radius, sigmaY=40.0px across angle)
        mu_2d = cv2.GaussianBlur(padded, (21, 141), sigmaX=3.0, sigmaY=40.0)[
            pad:-pad, :
        ]
        diff = plane - mu_2d

        # Soft-threshold noise in faint zones
        noise_level = float(np.std(diff[:, int(r_lunar * 1.5) :]))
        diff_clean = np.sign(diff) * np.maximum(0.0, np.abs(diff) - 1.2 * noise_level)
        polar_high_pass[:, :, ch] = diff_clean

    # 5. Inverse Polar Transform to Cartesian Space
    cartesian_streamers = cv2.warpPolar(
        polar_high_pass,
        (w, h),
        (cx, cy),
        max_r,
        cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP,
    )

    # 6. Strict Geometric & Luminance SNR Mask
    y_idx, x_idx = np.ogrid[:h, :w]
    dist_map = np.hypot(x_idx - cx, y_idx - cy)

    # Moon silhouette gate: 0 inside Moon, 1 at limb
    moon_gate = np.clip((dist_map - r_lunar) / (0.04 * r_lunar), 0.0, 1.0)

    # Luminance SNR gate: Only inject streamers where base signal exceeds 3*sigma of sky noise
    lum_base = (
        0.299 * clean_norm[..., 0]
        + 0.587 * clean_norm[..., 1]
        + 0.114 * clean_norm[..., 2]
    )
    snr_gate = np.clip((lum_base - 0.005) / 0.02, 0.0, 1.0)[:, :, None]

    # Radial decay window
    r_effective = np.maximum(0.0, dist_map - r_lunar)
    radial_window = np.exp(-0.5 * (r_effective / 220.0) ** 2)[:, :, None]

    total_detail_mask = moon_gate[:, :, None] * snr_gate * radial_window

    # Combine Base + Streamers
    final_composite = stretched + (boost * cartesian_streamers * total_detail_mask)
    final_composite = np.clip(final_composite * moon_gate[:, :, None], 0.0, 1.0)

    return final_composite.astype(np.float32)


def process_coronal_features(
    input_master_path: Path,
    output_dir: Path,
    algorithm: str = "druckmuller",
    sharpen_amount: float = 1.4,
) -> None:
    """Full coronal feature enhancement pipeline."""
    print("\n" + "=" * 65, flush=True)
    print(
        "       CORONAL FEATURE ENHANCEMENT PIPELINE                      ", flush=True
    )
    print("=" * 65, flush=True)
    print(f"  * Input File            : {input_master_path.resolve()}", flush=True)
    print(f"  * Algorithm             : {algorithm.upper()}", flush=True)

    img, is_linear = load_master_for_enhancement(input_master_path)
    cx, cy, r_lunar = solve_center_and_limb_raycast(img)
    print(f"  * Solved Lunar Centroid : ({cx:.2f}, {cy:.2f})", flush=True)
    print(f"  * Physical Limb Radius  : {r_lunar:.2f} px", flush=True)
    print("-" * 65, flush=True)

    enhanced = apply_druckmuller_polar_acf(
        img_rgb=img,
        center=(cx, cy),
        r_lunar=r_lunar,
        asinh_beta=20.0,
        boost=sharpen_amount,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # 16-Bit Master TIFF
    out_tiff = (
        output_dir / f"{input_master_path.stem}_{algorithm.capitalize()}_Enhanced.tif"
    )
    tifffile.imwrite(
        str(out_tiff),
        (enhanced * 65535.0).astype(np.uint16),
        photometric="rgb",
    )
    print(f"  [Exported 16-bit Enhanced TIFF] -> {out_tiff.resolve()}", flush=True)

    # Preview JPG
    out_jpg = (
        output_dir / f"{input_master_path.stem}_{algorithm.capitalize()}_Enhanced.jpg"
    )
    preview_8u = (enhanced * 255.0).astype(np.uint8)
    Image.fromarray(preview_8u, mode="RGB").save(out_jpg, quality=95)
    print(f"  [Exported Enhanced JPG Preview] -> {out_jpg.resolve()}", flush=True)
    print("=" * 65 + "\n", flush=True)
