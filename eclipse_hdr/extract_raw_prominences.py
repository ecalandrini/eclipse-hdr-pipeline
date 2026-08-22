"""Direct Raw Prominence Extractor with zero percentile scaling."""

from pathlib import Path
from astropy.io import fits
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tifffile


def load_raw_fits_rgb(fits_path: Path) -> np.ndarray:
    """Loads FITS preserving exact physical ADU ratios without stretching."""
    with fits.open(fits_path, memmap=False) as hdul:
        data = hdul[0].data.astype(np.float32)

    # Convert (C, H, W) -> (H, W, C)
    if data.ndim == 3 and data.shape[0] == 3:
        img = np.transpose(data, (1, 2, 0))
    elif data.ndim == 2:
        img = np.repeat(data[:, :, None], 3, axis=2)
    else:
        img = data

    # Scale strictly by physical bit depth
    max_adu = float(img.max())
    if max_adu > 255.0:
        img_norm = img / 65535.0
    else:
        img_norm = img / 255.0

    return np.clip(img_norm, 0.0, 1.0)


def extract_true_raw_prominences(
    short_fits: Path,
    hdr_master_file: Path,
    output_dir: Path,
    cx: float = 2492.4,
    cy: float = 1612.9,
    r_lunar: float = 222.7,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rgb = load_raw_fits_rgb(short_fits)

    # Load HDR master for base
    if hdr_master_file.suffix.lower() in (".fit", ".fits"):
        hdr_rgb = load_raw_fits_rgb(hdr_master_file)
    else:
        hdr_raw = tifffile.imread(str(hdr_master_file)).astype(np.float32)
        if hdr_raw.shape[2] > 3:
            hdr_raw = hdr_raw[:, :, :3]
        hdr_rgb = (
            hdr_raw / 65535.0
            if hdr_raw.dtype == np.uint16 or hdr_raw.max() > 255.0
            else hdr_raw / 255.0
        )

    h, w, _ = raw_rgb.shape

    # 1. Annular mask strictly around the chromosphere
    y_idx, x_idx = np.ogrid[:h, :w]
    dist_map = np.hypot(x_idx - cx, y_idx - cy)
    annulus = (dist_map >= r_lunar - 4.0) & (dist_map <= r_lunar + 25.0)

    # 2. Inspect true ADU values on the limb
    r_raw = raw_rgb[..., 0][annulus]
    g_raw = raw_rgb[..., 1][annulus]
    b_raw = raw_rgb[..., 2][annulus]

    print("\n" + "=" * 65)
    print("       TRUE RAW UNSTRETCHED PROMINENCE EXTRACTION                ")
    print("=" * 65)
    print(f"  * Raw Short Frame    : {short_fits.name}")
    print(
        f"  * Peak Limb Flux     : R={r_raw.max():.4f}, G={g_raw.max():.4f}, B={b_raw.max():.4f}"
    )
    print(
        f"  * Median Limb Flux   : R={np.median(r_raw):.4f}, G={np.median(g_raw):.4f}, B={np.median(b_raw):.4f}"
    )
    print("-" * 65)

    # Plot True Unclipped Histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(r_raw, bins=100, color="red", alpha=0.6, label="Red Channel")
    ax.hist(g_raw, bins=100, color="green", alpha=0.6, label="Green Channel")
    ax.hist(b_raw, bins=100, color="blue", alpha=0.6, label="Blue Channel")
    ax.set_title("True Raw Limb Annulus RGB Distribution (No Percentile Stretch)")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{short_fits.stem}_true_raw_histogram.png", dpi=150)
    plt.close(fig)

    # 3. Continuum Subtraction: Prominences are Red - Green in raw Bayer data
    continuum = (raw_rgb[..., 1] + raw_rgb[..., 2]) / 2.0
    h_alpha_raw = np.maximum(0.0, raw_rgb[..., 0] - continuum)

    smooth_gate = np.clip((dist_map - (r_lunar - 4.0)) / 2.0, 0.0, 1.0) * np.clip(
        ((r_lunar + 25.0) - dist_map) / 5.0, 0.0, 1.0
    )

    h_alpha_isolated = h_alpha_raw * smooth_gate

    # Scale prominence layer based on actual max signal
    peak_sig = float(h_alpha_isolated.max())
    print(f"  * True Peak H-alpha Signal : {peak_sig:.5f}")

    scale = 1.0 / max(peak_sig, 1e-4)
    prom_rgb = np.zeros_like(raw_rgb)
    prom_rgb[..., 0] = np.clip(h_alpha_isolated * scale, 0.0, 1.0)
    prom_rgb[..., 1] = np.clip(h_alpha_isolated * scale * 0.05, 0.0, 1.0)
    prom_rgb[..., 2] = np.clip(h_alpha_isolated * scale * 0.20, 0.0, 1.0)

    # Save isolated prominence layer
    out_prom = output_dir / f"{short_fits.stem}_prominences_clean.jpg"
    Image.fromarray((prom_rgb * 255.0).astype(np.uint8)).save(out_prom, quality=96)
    print(f"  -> Saved Clean Prominences : {out_prom.name}")

    # 4. Composite onto HDR Master
    composite = hdr_rgb.copy()
    for ch in range(3):
        # Soft-knee additive blend
        composite[..., ch] = np.clip(
            composite[..., ch] + prom_rgb[..., ch] * 0.8, 0.0, 1.0
        )

    # Moon Mask
    moon_gate = np.clip((dist_map - (r_lunar - 3.0)) / 2.0, 0.0, 1.0)[:, :, None]
    composite = np.clip(composite * moon_gate, 0.0, 1.0)

    out_comp = output_dir / f"{hdr_master_file.stem}_Prominences_Composite.jpg"
    Image.fromarray((composite * 255.0).astype(np.uint8)).save(out_comp, quality=96)
    print(f"  [Success] Exported Final Composite -> {out_comp.resolve()}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    import sys

    raw_f = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Aligned_Master_1_4000s.fit")
    )
    hdr_f = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else Path("Eclipse_HDR_Master_Artistic_HDR.tif")
    )
    out_d = Path("workspace/prominences_raw")
    extract_true_raw_prominences(raw_f, hdr_f, out_d)
