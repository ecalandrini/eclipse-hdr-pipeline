"""Benchmark and visual comparison of 4 solar center and lunar limb detection algorithms."""

from pathlib import Path
import sys
from astropy.io import fits
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tifffile


def load_normalized_image(input_path: Path) -> np.ndarray:
    """Loads FITS/TIFF and normalizes to [0.0, 1.0] float32 RGB."""
    ext = input_path.suffix.lower()
    if ext in (".fit", ".fits"):
        with fits.open(input_path, memmap=False) as h:
            d = h[0].data.astype(np.float32)
        img = np.transpose(d, (1, 2, 0)) if d.ndim == 3 and d.shape[0] == 3 else d
        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, axis=2)
    else:
        raw = tifffile.imread(str(input_path)).astype(np.float32)
        if raw.ndim == 2:
            raw = np.repeat(raw[:, :, None], 3, axis=2)
        elif raw.ndim == 3 and raw.shape[2] > 3:
            raw = raw[:, :, :3]
        img = (
            raw / 65535.0
            if raw.dtype == np.uint16 or raw.max() > 255.0
            else raw / 255.0
        )

    p99 = float(np.percentile(img[img > 0], 99.95)) if np.any(img > 0) else 1.0
    return np.clip(img / max(p99, 1e-5), 0.0, 1.0)


# -------------------------------------------------------------------------
# ALGORITHM 1: Circular Hough Transform
# -------------------------------------------------------------------------
def algo_1_hough_transform(lum: np.ndarray) -> tuple[float, float, float, str]:
    """Uses accumulator voting on gradient vectors to find circular features."""
    h, w = lum.shape
    gray_8u = (np.clip(lum, 0, 1) * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(gray_8u, (9, 9), 2.0)

    # Search around plausible eclipse radii (150px - 320px)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=200,
        param1=80,
        param2=25,
        minRadius=150,
        maxRadius=320,
    )

    if circles is not None and len(circles[0]) > 0:
        # Pick the circle closest to the image center
        candidates = circles[0]
        dist_to_center = [np.hypot(c[0] - w / 2, c[1] - h / 2) for c in candidates]
        best_idx = int(np.argmin(dist_to_center))
        cx, cy, r = candidates[best_idx]
        return float(cx), float(cy), float(r), "Success"

    return float(w / 2.0), float(h / 2.0), 220.0, "Failed to converge (Defaulted)"


# -------------------------------------------------------------------------
# ALGORITHM 2: Inner Ring Image Moments
# -------------------------------------------------------------------------
def algo_2_image_moments(lum: np.ndarray) -> tuple[float, float, float, str]:
    """Computes brightness mass moments on thresholded inner coronal ring."""
    h, w = lum.shape
    blur = cv2.GaussianBlur((np.clip(lum, 0, 1) * 255).astype(np.uint8), (15, 15), 0)
    thresh_val = int(np.percentile(blur, 94.0))
    _, binary = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY)

    m = cv2.moments(binary)
    if m["m00"] > 0:
        cx = float(m["m10"] / m["m00"])
        cy = float(m["m01"] / m["m00"])
        # Estimate radius from equivalent circular area of inner ring
        area = float(np.count_nonzero(binary))
        r_est = float(np.sqrt(area / np.pi))
        return cx, cy, r_est, "Success"

    return float(w / 2.0), float(h / 2.0), 220.0, "Failed (Zero mass)"


# -------------------------------------------------------------------------
# ALGORITHM 3: Contour FitEllipse / MinEnclosingCircle
# -------------------------------------------------------------------------
def algo_3_contour_fitting(lum: np.ndarray) -> tuple[float, float, float, str]:
    """Fits an algebraic ellipse directly to the largest gradient contour."""
    h, w = lum.shape
    gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)
    gmag = cv2.magnitude(gx, gy)

    p99 = float(np.percentile(gmag, 99.5)) or 1.0
    edges = (gmag > p99 * 0.4).astype(np.uint8) * 255

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    valid_c = [c for c in contours if len(c) >= 5]

    if valid_c:
        # Choose the contour closest to frame center with substantial perimeter
        best_c = max(valid_c, key=cv2.contourArea)
        (cx, cy), (d1, d2), _ = cv2.fitEllipse(best_c)
        r = float((d1 + d2) / 4.0)
        return float(cx), float(cy), r, "Success"

    return float(w / 2.0), float(h / 2.0), 220.0, "Failed (No valid contour)"


# -------------------------------------------------------------------------
# ALGORITHM 4: Radial Ray-Casting & Least-Squares Circle Fit
# -------------------------------------------------------------------------
def algo_4_radial_raycasting(
    lum: np.ndarray,
    n_rays: int = 360,
    r_min: int = 100,
    r_max: int = 380,
) -> tuple[float, float, float, str]:
    """Casts radial rays outward from approximate center, finds max dI/dr peak, fits circle."""
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

    if len(limb_xs) < 20:
        return cx_init, cy_init, 220.0, "Failed (Insufficient limb points)"

    xs = np.array(limb_xs, dtype=np.float64)
    ys = np.array(limb_ys, dtype=np.float64)

    # Kåsa Algebraic Least Squares Circle Fit
    A = np.column_stack([2.0 * xs, 2.0 * ys, np.ones_like(xs)])
    b = xs**2 + ys**2
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    cx = float(sol[0])
    cy = float(sol[1])
    r_lunar = float(np.sqrt(sol[2] + cx**2 + cy**2))

    return cx, cy, r_lunar, "Success"


# -------------------------------------------------------------------------
# Execution & Visual Benchmark
# -------------------------------------------------------------------------
def run_benchmark(image_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    img_rgb = load_normalized_image(image_path)
    lum = 0.299 * img_rgb[..., 0] + 0.587 * img_rgb[..., 1] + 0.114 * img_rgb[..., 2]
    h, w = lum.shape

    print("\n" + "=" * 80)
    print("        SOLAR CENTER & LUNAR LIMB DETECTION ALGORITHM BENCHMARK        ")
    print("=" * 80)
    print(f"Target Master Image : {image_path.name} ({w} x {h} px)")
    print("-" * 80)

    algorithms = [
        ("1. Circular Hough Transform", algo_1_hough_transform),
        ("2. Inner Ring Moments", algo_2_image_moments),
        ("3. Contour FitEllipse", algo_3_contour_fitting),
        ("4. Radial Ray-Casting Least-Squares", algo_4_radial_raycasting),
    ]

    results = []
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()

    # Crop window around center for clear visual inspection
    zoom_size = 500
    crop_x1 = max(0, int(w / 2 - zoom_size))
    crop_x2 = min(w, int(w / 2 + zoom_size))
    crop_y1 = max(0, int(h / 2 - zoom_size))
    crop_y2 = min(h, int(h / 2 + zoom_size))

    for idx, (name, fn) in enumerate(algorithms):
        cx, cy, r, status = fn(lum)
        results.append((name, cx, cy, r, status))

        # Render visual panel
        panel = img_rgb.copy()
        cv2.circle(
            panel, (int(cx), int(cy)), int(r), (0, 1.0, 0), 2
        )  # Green circle = Lunar limb
        cv2.drawMarker(
            panel, (int(cx), int(cy)), (1.0, 0, 0), cv2.MARKER_CROSS, 25, 2
        )  # Red cross = Center

        zoomed = panel[crop_y1:crop_y2, crop_x1:crop_x2]
        axes[idx].imshow(zoomed)
        axes[idx].set_title(
            f"{name}\nCenter: ({cx:.1f}, {cy:.1f}) | R: {r:.1f}px | Status: {status}",
            fontsize=11,
        )
        axes[idx].axis("off")

    # Print Summary Table
    print(
        f"{'Algorithm':<38} | {'Center (cx, cy)':<18} | {'Radius R (px)':<14} | {'Status'}"
    )
    print("-" * 80)
    for name, cx, cy, r, status in results:
        print(f"{name:<38} | ({cx:6.1f}, {cy:6.1f})   | {r:6.1f} px      | {status}")
    print("=" * 80 + "\n")

    # Save Side-by-Side Plot
    fig.tight_layout()
    comparison_png = output_dir / f"{image_path.stem}_algorithm_comparison.png"
    fig.savefig(comparison_png, dpi=180)
    plt.close(fig)
    print(
        f"[Benchmark Result] Saved 4-Panel Comparison -> {comparison_png.resolve()}\n"
    )


if __name__ == "__main__":
    target = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("Eclipse_HDR_Master_Artistic_HDR.tif")
    )
    out = Path("workspace/benchmarks")
    run_benchmark(target, out)
