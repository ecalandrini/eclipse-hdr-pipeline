"""Step-by-step diagnostic and Druckmüller Polar ACF with intermediate inspection plots."""

from pathlib import Path
from astropy.io import fits
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tifffile


def run_coronal_diagnostics(
    input_file: Path,
    output_dir: Path,
    asinh_beta: float = 30.0,
    boost: float = 1.5,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_file.stem
    print(f"\n{'=' * 65}\n  RUNNING CORONAL DIAGNOSTIC PIPELINE (STEP-BY-STEP)\n{'=' * 65}", flush=True)

    # -------------------------------------------------------------
    # STEP 1: Load Image and Estimate Statistics
    # -------------------------------------------------------------
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
        img = raw / 65535.0 if raw.dtype == np.uint16 or raw.max() > 255.0 else raw / 255.0

    h, w, c = img.shape
    p99 = float(np.percentile(img[img > 0], 99.95)) if np.any(img > 0) else 1.0
    img_norm = np.clip(img / max(p99, 1e-5), 0.0, 1.0)
    print(f"[Step 1] Image Loaded: {w}x{h} px, Max Val: {img.max():.4f}, 99.95%: {p99:.4f}", flush=True)

    # -------------------------------------------------------------
    # STEP 2: Find Solar / Lunar Center and Radii
    # -------------------------------------------------------------
    lum = 0.299 * img_norm[..., 0] + 0.587 * img_norm[..., 1] + 0.114 * img_norm[..., 2]
    
    # Gradient magnitude to detect inner lunar limb
    gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)
    gmag = cv2.magnitude(gx, gy)
    
    p_edge = float(np.percentile(gmag, 99.2))
    edges = (gmag > p_edge * 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cx, cy, r_lunar = float(w / 2.0), float(h / 2.0), min(h, w) * 0.22
    if contours:
        c_largest = max(contours, key=cv2.contourArea)
        (ccx, ccy), cr = cv2.minEnclosingCircle(c_largest)
        if 0.12 * min(h, w) < cr < 0.40 * min(h, w):
            cx, cy, r_lunar = float(ccx), float(ccy), float(cr)

    max_r = int(min(cx, cy, w - cx, h - cy) * 0.95)
    print(f"[Step 2] Center: ({cx:.1f}, {cy:.1f}), Lunar R: {r_lunar:.1f}px, Max R: {max_r}px", flush=True)

    # Save Step 2 Diagnostic: Center Overlay
    diag_center = (img_norm * 255.0).astype(np.uint8).copy()
    cv2.circle(diag_center, (int(cx), int(cy)), int(r_lunar), (0, 255, 0), 2)
    cv2.circle(diag_center, (int(cx), int(cy)), max_r, (255, 0, 0), 2)
    cv2.drawMarker(diag_center, (int(cx), int(cy)), (0, 0, 255), cv2.MARKER_CROSS, 25, 2)
    Image.fromarray(diag_center).save(output_dir / f"{stem}_debug_1_center_overlay.jpg", quality=92)
    print(f"  -> Saved {output_dir / f'{stem}_debug_1_center_overlay.jpg'}", flush=True)

    # -------------------------------------------------------------
    # STEP 3: Correct Polar Coordinate Transformation
    # Note: cv2.warpPolar dsize is (width=max_r, height=n_theta)
    # Output array has shape: (n_theta, max_r, 3)
    # Axis 0 = Theta (0 to 360 deg, Y-axis)
    # Axis 1 = Radius (0 to max_r px, X-axis)
    # -------------------------------------------------------------
    n_theta = 1440
    # Compress dynamic range with asinh before polar transform so outer faint signal isn't lost
    img_stretched = np.arcsinh(img_norm * asinh_beta) / np.arcsinh(asinh_beta)

    polar_img = cv2.warpPolar(
        img_stretched,
        (max_r, n_theta),
        (cx, cy),
        max_r,
        cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS,
    )
    # Shape: (1440, max_r, 3)
    print(f"[Step 3] Polar Image Shape: {polar_img.shape} (Theta x Radius)", flush=True)

    # Save Step 3 Diagnostic: Unwrapped Polar Image
    polar_preview = (np.clip(polar_img, 0, 1) * 255.0).astype(np.uint8)
    Image.fromarray(polar_preview).save(output_dir / f"{stem}_debug_2_polar_unwrapped.jpg", quality=92)
    print(f"  -> Saved {output_dir / f'{stem}_debug_2_polar_unwrapped.jpg'}", flush=True)

    # -------------------------------------------------------------
    # STEP 4: Compute & Plot Radial Profiles
    # -------------------------------------------------------------
    # Compute median along Theta axis (axis 0) for each radius (axis 1)
    radial_profile = np.median(polar_img, axis=0)  # Shape (max_r, 3)
    r_axis = np.arange(max_r)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["red", "green", "blue"]
    for ch in range(c):
        ax.plot(r_axis, radial_profile[:, ch], label=f"Channel {colors[ch]}", color=colors[ch], alpha=0.8)
    ax.axvline(x=r_lunar, color="black", linestyle="--", label=f"Lunar Limb ({r_lunar:.1f}px)")
    ax.set_title("Azimuthal Radial Brightness Profile $I(r)$")
    ax.set_xlabel("Radius from Center (pixels)")
    ax.set_ylabel("Asinh Brightness (Normalized)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    plot_path = output_dir / f"{stem}_debug_3_radial_profile_plot.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  -> Saved Radial Profile Plot: {plot_path}", flush=True)

    # -------------------------------------------------------------
    # STEP 5: High-Pass Filtering along Theta (Azimuthal Direction)
    # -------------------------------------------------------------
    polar_high_pass = np.zeros_like(polar_img)

    for ch in range(c):
        plane = polar_img[:, :, ch]
        
        # Periodic padding along Theta (axis 0)
        pad = 120
        padded = np.vstack([plane[-pad:, :], plane, plane[:pad, :]])
        
        # Smooth only along the theta axis (vertical axis 0)
        # ksize: (1, 151) -> width=1 (radius untouched), height=151 (angular smoothing)
        mu_theta = cv2.GaussianBlur(padded, (1, 151), sigmaX=0, sigmaY=35.0)[pad:-pad, :]
        
        # Subtract low-pass background to reveal angular streamers
        diff = plane - mu_theta
        polar_high_pass[:, :, ch] = diff

    # Save Step 5 Diagnostic: Polar High-Pass Details
    norm_hp = np.clip((polar_high_pass + 0.1) / 0.25, 0.0, 1.0)
    Image.fromarray((norm_hp * 255).astype(np.uint8)).save(output_dir / f"{stem}_debug_4_polar_streamers.jpg", quality=92)
    print(f"  -> Saved {output_dir / f'{stem}_debug_4_polar_streamers.jpg'}", flush=True)

    # -------------------------------------------------------------
    # STEP 6: Warp High-Pass Streamers back to Cartesian Space
    # -------------------------------------------------------------
    cartesian_streamers = cv2.warpPolar(
        polar_high_pass,
        (w, h),
        (cx, cy),
        max_r,
        cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP,
    )

    # Radial Mask to preserve lunar interior and sky
    y_idx, x_idx = np.ogrid[:h, :w]
    dist_map = np.hypot(x_idx - cx, y_idx - cy)
    
    # Smooth masks
    moon_mask = np.clip((dist_map - r_lunar * 0.98) / (0.05 * r_lunar), 0.0, 1.0)
    sky_mask = np.clip((max_r - dist_map) / (0.20 * max_r), 0.0, 1.0)
    coronal_gate = (moon_mask * sky_mask)[:, :, None]

    # Combine
    final_composite = img_stretched + (boost * cartesian_streamers * coronal_gate)
    final_composite = np.clip(final_composite * moon_mask[:, :, None], 0.0, 1.0)

    # Save Final Result
    final_out = output_dir / f"{stem}_Druckmuller_Corrected.jpg"
    Image.fromarray((final_composite * 255.0).astype(np.uint8)).save(final_out, quality=96)
    print(f"\n[Finished] Final Composite Exported -> {final_out.resolve()}", flush=True)
    print("=" * 65 + "\n", flush=True)


if __name__ == "__main__":
    import sys
    inp = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Eclipse_HDR_Master_Artistic_HDR.tif")
    out = Path("workspace/diagnostics")
    run_coronal_diagnostics(inp, out)