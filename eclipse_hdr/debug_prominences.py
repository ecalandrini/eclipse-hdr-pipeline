"""Limb Chromaticity Analyzer and Prominence Debugger."""

from pathlib import Path
from astropy.io import fits
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tifffile


def load_normalized_rgb(file_path: Path) -> np.ndarray:
    """Loads FITS/TIFF/JPG and normalizes to [0.0, 1.0] float32 RGB."""
    ext = file_path.suffix.lower()
    if ext in (".fit", ".fits"):
        with fits.open(file_path, memmap=False) as h:
            d = h[0].data.astype(np.float32)
        img = np.transpose(d, (1, 2, 0)) if d.ndim == 3 and d.shape[0] == 3 else d
        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, axis=2)
    else:
        raw = tifffile.imread(str(file_path)).astype(np.float32)
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
    return np.clip(img / max(p99, 1e-5), 0.0, 1.0).astype(np.float32)


def debug_and_extract_prominences(
    short_exp_path: Path,
    hdr_master_path: Path,
    output_dir: Path,
    cx: float = 2492.4,
    cy: float = 1612.9,
    r_lunar: float = 222.7,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    short_img = load_normalized_rgb(short_exp_path)
    hdr_img = load_normalized_rgb(hdr_master_path)
    h, w, _ = short_img.shape

    print(
        f"\n{'=' * 70}\n       CHROMOSPHERIC PROMINENCE EXTRACTION DEBUGGER\n{'=' * 70}"
    )
    print(f"  * Short Exposure Image : {short_exp_path.name}")
    print(
        f"  * Geometry             : Center=({cx:.1f}, {cy:.1f}), R_limb={r_lunar:.1f}px"
    )

    # 1. Annular Mask around the limb (where prominences can physically exist)
    y_idx, x_idx = np.ogrid[:h, :w]
    dist_map = np.hypot(x_idx - cx, y_idx - cy)
    annulus_mask = (dist_map >= r_lunar - 15.0) & (dist_map <= r_lunar + 60.0)

    # 2. Extract Limb Pixels and Inspect Statistics
    r_vals = short_img[..., 0][annulus_mask]
    g_vals = short_img[..., 1][annulus_mask]
    b_vals = short_img[..., 2][annulus_mask]

    print(
        f"  * Annular Limb Max Values -> R: {r_vals.max():.4f}, G: {g_vals.max():.4f}, B: {b_vals.max():.4f}"
    )
    print(
        f"  * Annular Limb Medians    -> R: {np.median(r_vals):.4f}, G: {np.median(g_vals):.4f}, B: {np.median(b_vals):.4f}"
    )

    # Plot RGB Distribution along the limb
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(r_vals, bins=100, color="red", alpha=0.5, label="Red Channel")
    ax.hist(g_vals, bins=100, color="green", alpha=0.5, label="Green Channel")
    ax.hist(b_vals, bins=100, color="blue", alpha=0.5, label="Blue Channel")
    ax.set_title("Limb Annulus RGB Intensity Distribution")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    plot_out = output_dir / f"{short_exp_path.stem}_debug_rgb_histogram.png"
    fig.savefig(plot_out, dpi=150)
    plt.close(fig)
    print(f"  -> Saved RGB Histogram Plot : {plot_out.name}")

    # -------------------------------------------------------------
    # METHOD A: Normalized Chromatic Excess (Self-Calibrating)
    # -------------------------------------------------------------
    # Evaluates R / (G + 1e-4) relative to the median limb background ratio
    r_over_g = (short_img[..., 0] + 1e-4) / (short_img[..., 1] + 1e-4)
    med_rg = float(np.median(r_over_g[annulus_mask]))

    # Prominence mask: pixels with R/G ratio > 1.2 * median ratio AND minimum brightness
    prom_mask_a = (r_over_g > med_rg * 1.25) & annulus_mask & (short_img[..., 0] > 0.05)

    layer_a = np.zeros_like(short_img)
    layer_a[prom_mask_a] = short_img[prom_mask_a]

    # Save Method A
    Image.fromarray((np.clip(layer_a, 0, 1) * 255).astype(np.uint8)).save(
        output_dir / f"{short_exp_path.stem}_method_a_rg_ratio.jpg", quality=95
    )

    # -------------------------------------------------------------
    # METHOD B: CIELAB a* Channel (Direct Chromatic Redness)
    # -------------------------------------------------------------
    img_8u = (np.clip(short_img, 0, 1) * 255).astype(np.uint8)
    lab = cv2.cvtColor(img_8u, cv2.COLOR_RGB2LAB)
    a_channel = lab[..., 1]  # 128 is neutral, >128 is Red/Magenta

    # Find red threshold on limb
    a_limb = a_channel[annulus_mask]
    thresh_a = int(np.percentile(a_limb, 90.0))
    prom_mask_b = (a_channel > max(132, thresh_a)) & annulus_mask

    layer_b = np.zeros_like(short_img)
    layer_b[prom_mask_b] = short_img[prom_mask_b]

    # Save Method B
    Image.fromarray((np.clip(layer_b, 0, 1) * 255).astype(np.uint8)).save(
        output_dir / f"{short_exp_path.stem}_method_b_cielab_redness.jpg", quality=95
    )

    # -------------------------------------------------------------
    # METHOD C: Direct Short-Exposure Chromosphere Cutout (Best Quality)
    # -------------------------------------------------------------
    # Instead of artificial synthesis, grab the physical chromosphere directly
    # from the short exposure and blend it into the HDR composite
    smooth_annulus = np.clip((dist_map - (r_lunar - 3.0)) / 3.0, 0.0, 1.0) * np.clip(
        ((r_lunar + 25.0) - dist_map) / 8.0, 0.0, 1.0
    )

    # Layer short exposure over HDR master where short exposure is bright
    short_lum = (
        0.299 * short_img[..., 0]
        + 0.587 * short_img[..., 1]
        + 0.114 * short_img[..., 2]
    )
    prom_weight = np.clip((short_lum - 0.02) / 0.1, 0.0, 1.0) * smooth_annulus

    final_composite = hdr_img.copy()
    for ch in range(3):
        # Maximum intensity blend along the chromosphere
        final_composite[..., ch] = np.maximum(
            hdr_img[..., ch], short_img[..., ch] * prom_weight * 1.5
        )

    # Mask inner lunar disk
    moon_gate = np.clip((dist_map - (r_lunar - 3.0)) / 2.0, 0.0, 1.0)[:, :, None]
    final_composite = np.clip(final_composite * moon_gate, 0.0, 1.0)

    # Save Final Composite
    out_jpg = output_dir / f"{hdr_master_path.stem}_Prominences_Composite.jpg"
    Image.fromarray((final_composite * 255).astype(np.uint8)).save(out_jpg, quality=96)
    print(f"  [Exported Final Composite] -> {out_jpg.resolve()}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    import sys

    short_frame = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path("path_to_1_4000_aligned.tif")
    )
    hdr_frame = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else Path("Eclipse_HDR_Master_Artistic_HDR.tif")
    )
    out = Path("workspace/prominences_debug")
    debug_and_extract_prominences(short_frame, hdr_frame, out)
