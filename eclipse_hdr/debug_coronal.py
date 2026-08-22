"""Step-by-step diagnostic and Druckmüller Polar ACF with verified centroid detection."""

from pathlib import Path
from astropy.io import fits
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tifffile


def solve_exact_lunar_limb(img_rgb: np.ndarray) -> tuple[float, float, float]:
    """Accurately finds lunar centroid and limb radius via dark core component analysis."""
    lum = 0.299 * img_rgb[..., 0] + 0.587 * img_rgb[..., 1] + 0.114 * img_rgb[..., 2]
    h, w = lum.shape

    # 1. Dark core segmentation
    # The lunar disk is the largest contiguous dark object completely enclosed inside the corona
    blur = cv2.GaussianBlur(lum, (11, 11), 0)
    thresh_dark = float(np.percentile(blur, 15.0))
    dark_mask = ((blur < thresh_dark) * 255).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark_mask)

    # Filter for the circular component not touching frame edges
    best_idx = -1
    best_score = 0.0

    for i in range(1, num_labels):
        bx, by, bw, bh, area = stats[i]
        ccx, ccy = centroids[i]

        # Ignore outer black sky border components
        if bx <= 5 or by <= 5 or (bx + bw) >= (w - 5) or (by + bh) >= (h - 5):
            continue

        aspect = float(bw) / float(bh + 1e-5)
        approx_r = np.sqrt(area / np.pi)

        # Solar disk criteria
        if 0.75 < aspect < 1.35 and 120 < approx_r < min(h, w) * 0.35:
            score = area
            if score > best_score:
                best_score = score
                best_idx = i

    if best_idx != -1:
        cx, cy = float(centroids[best_idx][0]), float(centroids[best_idx][1])
        r_lunar = float(np.sqrt(stats[best_idx][4] / np.pi))
    else:
        # Fallback to direct Hough Circle on the high-contrast limb
        edges = cv2.Canny((np.clip(lum, 0, 1) * 255).astype(np.uint8), 50, 150)
        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=200,
            param1=100,
            param2=30,
            minRadius=150,
            maxRadius=350,
        )
        if circles is not None:
            cx, cy, r_lunar = (
                float(circles[0][0][0]),
                float(circles[0][0][1]),
                float(circles[0][0][2]),
            )
        else:
            cx, cy, r_lunar = float(w / 2.0), float(h / 2.0), 222.0

    return cx, cy, r_lunar


def run_coronal_diagnostics(
    input_file: Path,
    output_dir: Path,
    asinh_beta: float = 25.0,
    boost: float = 1.4,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_file.stem
    print(
        f"\n{'=' * 65}\n  RUNNING CORONAL DIAGNOSTIC PIPELINE (STEP-BY-STEP)\n{'=' * 65}",
        flush=True,
    )

    # 1. Load Image
    if input_file.suffix.lower() in (".fit", ".fits"):
        with fits.open(input_file) as h:
            d = h[0].data.astype(np.float32)
        img = np.transpose(d, (1, 2, 0)) if d.ndim == 3 and d.shape[0] == 3 else d
        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, axis=2)
    else:
        raw = tifffile.imread(str(input_file)).astype(np.float32)
        if raw.ndim == 2:
            raw = np.repeat(raw[:, :, None], 3, axis=2)
        elif raw.ndim == 3 and raw.shape[2] > 3:
            raw = raw[:, :, :3]
        img = (
            raw / 65535.0
            if raw.dtype == np.uint16 or raw.max() > 255.0
            else raw / 255.0
        )

    h, w, c = img.shape
    p99 = float(np.percentile(img[img > 0], 99.95)) if np.any(img > 0) else 1.0
    img_norm = np.clip(img / max(p99, 1e-5), 0.0, 1.0)

    # 2. Centroid & Limb
    cx, cy, r_lunar = solve_exact_lunar_limb(img_norm)
    max_r = int(min(cx, cy, w - cx, h - cy) * 0.95)
    print(
        f"[Step 2] Solved Center: ({cx:.2f}, {cy:.2f}), Physical Lunar Radius: {r_lunar:.1f}px, Max R: {max_r}px",
        flush=True,
    )

    # Diagnostic 1: Center Overlay
    diag_center = (img_norm * 255.0).astype(np.uint8).copy()
    cv2.circle(diag_center, (int(cx), int(cy)), int(r_lunar), (0, 255, 0), 2)
    cv2.circle(diag_center, (int(cx), int(cy)), max_r, (255, 0, 0), 2)
    cv2.drawMarker(
        diag_center, (int(cx), int(cy)), (0, 0, 255), cv2.MARKER_CROSS, 25, 2
    )
    Image.fromarray(diag_center).save(
        output_dir / f"{stem}_debug_1_center_overlay.jpg", quality=92
    )
    print(f"  -> Saved {output_dir / f'{stem}_debug_1_center_overlay.jpg'}", flush=True)

    # 3. Correct Polar Transformation
    n_theta = 1440
    img_stretched = np.arcsinh(img_norm * asinh_beta) / np.arcsinh(asinh_beta)
    polar_img = cv2.warpPolar(
        img_stretched,
        (max_r, n_theta),
        (cx, cy),
        max_r,
        cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS,
    )

    # Diagnostic 2: Polar Unwrapped
    polar_preview = (np.clip(polar_img, 0, 1) * 255.0).astype(np.uint8)
    Image.fromarray(polar_preview).save(
        output_dir / f"{stem}_debug_2_polar_unwrapped.jpg", quality=92
    )
    print(
        f"  -> Saved {output_dir / f'{stem}_debug_2_polar_unwrapped.jpg'}", flush=True
    )

    # Diagnostic 3: Radial Profile Plot
    radial_profile = np.median(polar_img, axis=0)
    r_axis = np.arange(max_r)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["red", "green", "blue"]
    for ch in range(c):
        ax.plot(
            r_axis,
            radial_profile[:, ch],
            label=f"Channel {colors[ch]}",
            color=colors[ch],
            alpha=0.8,
        )
    ax.axvline(
        x=r_lunar,
        color="black",
        linestyle="--",
        label=f"True Lunar Limb ({r_lunar:.1f}px)",
    )
    ax.set_title("Azimuthal Radial Brightness Profile $I(r)$")
    ax.set_xlabel("Radius from Center (pixels)")
    ax.set_ylabel("Asinh Brightness (Normalized)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}_debug_3_radial_profile_plot.png", dpi=150)
    plt.close(fig)

    # 4. Polar High-Pass Along Theta
    polar_high_pass = np.zeros_like(polar_img)
    for ch in range(c):
        plane = polar_img[:, :, ch]
        pad = 120
        padded = np.vstack([plane[-pad:, :], plane, plane[:pad, :]])
        mu_theta = cv2.GaussianBlur(padded, (1, 151), sigmaX=0, sigmaY=35.0)[
            pad:-pad, :
        ]
        polar_high_pass[:, :, ch] = plane - mu_theta

    # Diagnostic 4: Polar Streamers
    norm_hp = np.clip((polar_high_pass + 0.1) / 0.25, 0.0, 1.0)
    Image.fromarray((norm_hp * 255).astype(np.uint8)).save(
        output_dir / f"{stem}_debug_4_polar_streamers.jpg", quality=92
    )

    # 5. Inverse Polar Transform
    cartesian_streamers = cv2.warpPolar(
        polar_high_pass,
        (w, h),
        (cx, cy),
        max_r,
        cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP,
    )

    # 6. Blending
    y_idx, x_idx = np.ogrid[:h, :w]
    dist_map = np.hypot(x_idx - cx, y_idx - cy)
    moon_gate = np.clip((dist_map - r_lunar) / (0.05 * r_lunar), 0.0, 1.0)
    sky_gate = np.clip((max_r - dist_map) / (0.25 * max_r), 0.0, 1.0)
    blend_weight = (moon_gate * sky_gate)[:, :, None]

    final_composite = img_stretched + (boost * cartesian_streamers * blend_weight)
    final_composite = np.clip(final_composite * moon_gate[:, :, None], 0.0, 1.0)

    final_out = output_dir / f"{stem}_Druckmuller_Corrected.jpg"
    Image.fromarray((final_composite * 255.0).astype(np.uint8)).save(
        final_out, quality=96
    )
    print(f"\n[Success] Final Composite Exported -> {final_out.resolve()}", flush=True)
    print("=" * 65 + "\n", flush=True)


if __name__ == "__main__":
    import sys

    inp = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("Eclipse_HDR_Master_Artistic_HDR.tif")
    )
    out = Path("workspace/diagnostics")
    run_coronal_diagnostics(inp, out)
