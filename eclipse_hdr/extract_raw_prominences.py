"""Direct Raw Prominence Extractor and HDR Compositor with dynamic range matching."""

from pathlib import Path
from astropy.io import fits
import cv2
import numpy as np
from PIL import Image
import tifffile


def load_raw_fits_rgb(fits_path: Path) -> np.ndarray:
    """Loads FITS preserving exact physical linear ADU ratios."""
    with fits.open(fits_path, memmap=False) as hdul:
        data = hdul[0].data.astype(np.float32)

    if data.ndim == 3 and data.shape[0] == 3:
        img = np.transpose(data, (1, 2, 0))
    elif data.ndim == 2:
        img = np.repeat(data[:, :, None], 3, axis=2)
    else:
        img = data

    max_adu = float(img.max())
    if max_adu > 255.0:
        img_norm = img / 65535.0
    else:
        img_norm = img / 255.0

    return np.clip(img_norm, 0.0, 1.0)


def extract_and_composite_prominences(
    short_fits: Path,
    hdr_master_file: Path,
    output_dir: Path,
    cx: float = 2492.4,
    cy: float = 1612.9,
    r_lunar: float = 222.7,
    prominence_gain: float = 4.0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rgb = load_raw_fits_rgb(short_fits)
    h, w, _ = raw_rgb.shape

    # 1. Load HDR master
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

    # 2. Extract H-Alpha (Red - Continuum)
    continuum = (raw_rgb[..., 1] + raw_rgb[..., 2]) / 2.0
    h_alpha_raw = np.maximum(0.0, raw_rgb[..., 0] - continuum)

    # Annular spatial mask (covering chromosphere and prominences)
    y_idx, x_idx = np.ogrid[:h, :w]
    dist_map = np.hypot(x_idx - cx, y_idx - cy)
    smooth_gate = np.clip((dist_map - (r_lunar - 4.0)) / 2.0, 0.0, 1.0) * np.clip(
        ((r_lunar + 30.0) - dist_map) / 5.0, 0.0, 1.0
    )

    h_alpha_signal = h_alpha_raw * smooth_gate
    peak_sig = float(h_alpha_signal.max())

    # 3. Non-linear stretch for high dynamic range visibility
    # Compresses the peak and lifts the faint prominence loops
    norm_halpha = h_alpha_signal / max(peak_sig, 1e-6)
    stretched_halpha = np.arcsinh(norm_halpha * 15.0) / np.arcsinh(15.0)

    # Synthesize characteristic H-alpha emission (Ruby-red with Balmer Violet)
    prom_rgb = np.zeros_like(raw_rgb)
    prom_rgb[..., 0] = np.clip(stretched_halpha * 1.00, 0.0, 1.0)  # Red
    prom_rgb[..., 1] = np.clip(stretched_halpha * 0.08, 0.0, 1.0)  # Green
    prom_rgb[..., 2] = np.clip(stretched_halpha * 0.28, 0.0, 1.0)  # Blue (Balmer)

    # Save isolated high-contrast prominence layer
    out_prom = output_dir / f"{short_fits.stem}_prominences_vivid.jpg"
    Image.fromarray((prom_rgb * 255.0).astype(np.uint8)).save(out_prom, quality=96)
    print(f"  -> Saved High-Contrast Prominence Layer: {out_prom.name}")

    # 4. Composite over HDR Master using Screen / Additive blending
    composite = hdr_rgb.copy()
    for ch in range(3):
        # Blend: preserve corona while adding the saturated prominence light
        base = composite[..., ch]
        prom = prom_rgb[..., ch] * (prominence_gain * 0.35)
        # Soft-knee screen blend
        composite[..., ch] = 1.0 - (1.0 - base) * (1.0 - np.clip(prom, 0.0, 1.0))

    # Mask inner lunar disk
    moon_gate = np.clip((dist_map - (r_lunar - 2.5)) / 2.0, 0.0, 1.0)[:, :, None]
    composite = np.clip(composite * moon_gate, 0.0, 1.0)

    # 5. Export 16-Bit TIFF & High-Res JPG
    out_tif = output_dir / f"{hdr_master_file.stem}_Prominences_Final.tif"
    tifffile.imwrite(
        str(out_tif), (composite * 65535.0).astype(np.uint16), photometric="rgb"
    )
    print(f"  [Exported 16-Bit Master] -> {out_tif.resolve()}")

    out_jpg = output_dir / f"{hdr_master_file.stem}_Prominences_Final.jpg"
    Image.fromarray((composite * 255.0).astype(np.uint8)).save(out_jpg, quality=96)
    print(f"  [Exported Final Preview] -> {out_jpg.resolve()}")


if __name__ == "__main__":
    import sys

    raw_f = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("workspace/bucket_1_4000s/Aligned_Master_1_4000s.fit")
    )
    hdr_f = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else Path("Eclipse_HDR_Master_Artistic_HDR.tif")
    )
    out_d = Path("workspace/prominences_final")

    extract_and_composite_prominences(raw_f, hdr_f, out_d, prominence_gain=4.0)
