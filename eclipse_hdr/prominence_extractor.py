"""Chromospheric and prominence (H-alpha 656.3nm) extraction and HDR blending."""

from pathlib import Path
from astropy.io import fits
import cv2
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
    return np.clip(img / max(p99, 1e-5), 0.0, 1.0)


def extract_halpha_layer(
    short_exposure_rgb: np.ndarray,
    center: tuple[float, float],
    r_lunar: float,
    annulus_width_px: float = 35.0,
    chroma_boost: float = 2.5,
) -> np.ndarray:
    """Isolates pure H-alpha emission (R - Continuum) localized to the solar limb."""
    h, w, _ = short_exposure_rgb.shape
    cx, cy = center

    r_ch = short_exposure_rgb[..., 0]
    g_ch = short_exposure_rgb[..., 1]
    b_ch = short_exposure_rgb[..., 2]

    # 1. Estimate photospheric/coronal continuum (neutral white)
    continuum = (g_ch + b_ch) / 2.0

    # 2. Subtract continuum: Prominences have high R relative to G and B
    h_alpha_raw = np.maximum(0.0, r_ch - continuum)

    # 3. Create limb-confined annular mask
    y_idx, x_idx = np.ogrid[:h, :w]
    dist_map = np.hypot(x_idx - cx, y_idx - cy)

    # Sharp cut at inner moon limb, smooth decay outside prominence zone
    inner_gate = np.clip((dist_map - (r_lunar - 2.0)) / 3.0, 0.0, 1.0)
    outer_gate = np.clip(
        ((r_lunar + annulus_width_px) - dist_map) / (annulus_width_px * 0.5), 0.0, 1.0
    )
    limb_annulus = inner_gate * outer_gate

    # 4. Synthesize vivid H-alpha color (characteristic deep crimson / magenta)
    # H-alpha: 100% Red, ~5% Green, ~25% Blue (H-beta / Balmer mixing)
    h_alpha_signal = h_alpha_raw * limb_annulus * chroma_boost

    prom_layer = np.zeros_like(short_exposure_rgb)
    prom_layer[..., 0] = h_alpha_signal * 1.00  # Deep Red
    prom_layer[..., 1] = h_alpha_signal * 0.08  # Slight Green
    prom_layer[..., 2] = h_alpha_signal * 0.28  # Violet Balmer emission

    return prom_layer, limb_annulus


def blend_prominences_onto_hdr(
    hdr_master_rgb: np.ndarray,
    short_exposure_rgb: np.ndarray,
    center: tuple[float, float],
    r_lunar: float,
    blend_intensity: float = 1.4,
) -> np.ndarray:
    """Blends the high-resolution prominence layer directly onto the HDR composite."""
    prom_rgb, annulus_mask = extract_halpha_layer(
        short_exposure_rgb=short_exposure_rgb,
        center=center,
        r_lunar=r_lunar,
        annulus_width_px=40.0,
        chroma_boost=blend_intensity * 2.0,
    )

    # Screen / Lighten blend mode along the chromosphere
    # Composite: Base + Prominence details where H-alpha is active
    composite = hdr_master_rgb.copy()

    # Apply soft screen blending in the annular zone
    for ch in range(3):
        base_ch = composite[..., ch]
        p_ch = prom_rgb[..., ch]
        # Screen blending: 1 - (1 - A) * (1 - B)
        blended = 1.0 - (1.0 - base_ch) * (1.0 - p_ch)
        # Apply specifically to prominence mask
        composite[..., ch] = np.where(p_ch > 0.01, blended, base_ch)

    # Mask inner lunar disk clean
    h, w, _ = hdr_master_rgb.shape
    y_idx, x_idx = np.ogrid[:h, :w]
    dist_map = np.hypot(x_idx - center[0], y_idx - center[1])
    moon_mask = np.clip((dist_map - r_lunar) / 2.0, 0.0, 1.0)[:, :, None]

    final_rgb = np.clip(composite * moon_mask, 0.0, 1.0)
    return final_rgb


def run_prominence_pipeline(
    hdr_master_path: Path,
    short_exposure_path: Path,
    output_dir: Path,
    center: tuple[float, float] = (2492.4, 1612.9),
    r_lunar: float = 222.7,
    intensity: float = 1.5,
) -> None:
    """Full execution pipeline for prominence extraction."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 65)
    print("       H-ALPHA PROMINENCE EXTRACTION & HDR COMPOSITING           ")
    print("=" * 65)
    print(f"  * HDR Master File        : {hdr_master_path.name}")
    print(f"  * Short Exposure (Limb)  : {short_exposure_path.name}")
    print(f"  * Solar Center           : ({center[0]:.1f}, {center[1]:.1f})")
    print(f"  * Lunar Limb Radius      : {r_lunar:.1f} px")
    print("-" * 65)

    hdr_rgb = load_normalized_rgb(hdr_master_path)
    short_rgb = load_normalized_rgb(short_exposure_path)

    # Blend
    composite = blend_prominences_onto_hdr(
        hdr_master_rgb=hdr_rgb,
        short_exposure_rgb=short_rgb,
        center=center,
        r_lunar=r_lunar,
        blend_intensity=intensity,
    )

    # Export
    out_tif = output_dir / f"{hdr_master_path.stem}_Prominences_Blended.tif"
    tifffile.imwrite(
        str(out_tif), (composite * 65535.0).astype(np.uint16), photometric="rgb"
    )
    print(f"  [Exported TIFF] -> {out_tif.resolve()}")

    out_jpg = output_dir / f"{hdr_master_path.stem}_Prominences_Blended.jpg"
    Image.fromarray((composite * 255.0).astype(np.uint8)).save(out_jpg, quality=96)
    print(f"  [Exported Preview JPG] -> {out_jpg.resolve()}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    import sys

    # Default to Artistic HDR and short exposure file if present
    master = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("Eclipse_HDR_Master_Artistic_HDR.tif")
    )
    short_exp = Path(sys.argv[2]) if len(sys.argv) > 2 else master
    out = Path("workspace/prominences")

    run_prominence_pipeline(
        hdr_master_path=master,
        short_exposure_path=short_exp,
        output_dir=out,
        center=(2492.4, 1612.9),
        r_lunar=222.7,
        intensity=1.5,
    )
