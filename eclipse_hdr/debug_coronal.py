"""Restored step-by-step diagnostic with original centroid detection and physical radius."""

from pathlib import Path
from astropy.io import fits
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tifffile


def solve_solar_center_and_limb(img_rgb: np.ndarray) -> tuple[float, float, float]:
    """Fits an exact algebraic least-squares circle directly to the inner lunar limb."""
    lum = 0.299 * img_rgb[..., 0] + 0.587 * img_rgb[..., 1] + 0.114 * img_rgb[..., 2]
    h, w = lum.shape

    # 1. High-pass gradient magnitude to isolate sharp limb transition
    grad_x = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)

    # 2. Extract edge points strictly along the sharp inner boundary
    p_edge = float(np.percentile(grad_mag, 99.4))
    edge_pts = np.argwhere(grad_mag > p_edge)  # Array of (y, x) coords

    # 3. Initial centroid estimate
    cy_init = float(np.mean(edge_pts[:, 0]))
    cx_init = float(np.mean(edge_pts[:, 1]))

    # Filter out far-field outliers: keep only points within annular zone of the limb
    r_dists = np.hypot(edge_pts[:, 1] - cx_init, edge_pts[:, 0] - cy_init)
    r_med = float(np.median(r_dists))
    valid_pts = edge_pts[np.abs(r_dists - r_med) < 30.0]

    # 4. Kåsa / Pratt Algebraic Least Squares Circle Fit:
    # Solves: [2*x, 2*y, 1] @ [cx, cy, C]^T = x^2 + y^2
    xs = valid_pts[:, 1].astype(np.float64)
    ys = valid_pts[:, 0].astype(np.float64)

    A = np.column_stack([2.0 * xs, 2.0 * ys, np.ones_like(xs)])
    b = xs**2 + ys**2
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    cx = float(sol[0])
    cy = float(sol[1])
    r_lunar = float(np.sqrt(sol[2] + cx**2 + cy**2))

    return cx, cy, r_lunar


def run_coronal_diagnostics(
    input_file: Path,
    output_dir: Path,
    asinh_beta: float = 20.0,
    boost: float = 1.4,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_file.stem
    print(
        f"\n{'=' * 65}\n  RUNNING CORONAL DIAGNOSTIC PIPELINE (RESTORED CENTROID)\n{'=' * 65}",
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

    # 2. Restored Centroid & Physical Limb
    cx, cy, r_lunar = solve_solar_center_and_limb(img_norm)
    max_r = int(min(cx, cy, w - cx, h - cy) * 0.95)
    print(
        f"[Step 2] Restored Center: ({cx:.2f}, {cy:.2f}), Physical Limb Radius: {r_lunar:.1f}px, Max R: {max_r}px",
        flush=True,
    )

    # Diagnostic 1: Center & Radius Overlay
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
        x=r_lunar, color="black", linestyle="--", label=f"Lunar Limb ({r_lunar:.1f}px)"
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
    print(
        f"  -> Saved {output_dir / f'{stem}_debug_4_polar_streamers.jpg'}", flush=True
    )

    # 5. Inverse Polar Transform to Cartesian
    cartesian_streamers = cv2.warpPolar(
        polar_high_pass,
        (w, h),
        (cx, cy),
        max_r,
        cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP,
    )

    # 6. Smooth Blending
    y_idx, x_idx = np.ogrid[:h, :w]
    dist_map = np.hypot(x_idx - cx, y_idx - cy)
    moon_gate = np.clip((dist_map - r_lunar) / (0.05 * r_lunar), 0.0, 1.0)
    sky_gate = np.clip((max_r - dist_map) / (0.25 * max_r), 0.0, 1.0)
    blend_weight = (moon_gate * sky_gate)[:, :, None]

    final_composite = img_stretched + (boost * cartesian_streamers * blend_weight)
    final_composite = np.clip(final_composite * moon_gate[:, :, None], 0.0, 1.0)

    # Export Final
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
