"""Artifact-free astronomical coronal detail enhancement via log-space bandpass filtering."""

from pathlib import Path
from astropy.io import fits
import cv2
import numpy as np
from PIL import Image
import tifffile


def load_any_master(input_path: Path) -> np.ndarray:
    """Robustly loads FITS, TIFF, or JPG masters into normalized [0.0, 1.0] float32 RGB."""
    ext = input_path.suffix.lower()

    if ext in (".fit", ".fits"):
        with fits.open(input_path, memmap=False) as hdul:
            data = hdul[0].data.astype(np.float32)
        if data.ndim == 3:
            if data.shape[0] == 3:
                img = np.transpose(data, (1, 2, 0))
            else:
                img = data
        elif data.ndim == 2:
            img = np.repeat(data[:, :, np.newaxis], 3, axis=2)
        else:
            raise ValueError(f"Unsupported FITS shape: {data.shape}")

    else:
        raw = tifffile.imread(str(input_path))
        if raw.ndim == 2:
            raw = np.repeat(raw[:, :, np.newaxis], 3, axis=2)
        elif raw.ndim == 3 and raw.shape[2] > 3:
            raw = raw[:, :, :3]

        if raw.dtype == np.uint8:
            img = raw.astype(np.float32) / 255.0
        elif raw.dtype == np.uint16:
            img = raw.astype(np.float32) / 65535.0
        else:
            img = raw.astype(np.float32)

    # Clean extreme outliers & normalize non-zero peak to 1.0
    p_high = float(np.percentile(img, 99.99)) or 1.0
    img = np.clip(img / p_high, 0.0, 1.0)
    return img


def enhance_coronal_structures(
    img_rgb: np.ndarray,
    asinh_stretch: float = 10.0,
    fine_sharpen: float = 1.2,
    streamer_boost: float = 1.5,
) -> np.ndarray:
    """Enhances fine magnetic loops and outer coronal streamers without geometric masks."""
    h, w, c = img_rgb.shape

    # 1. Estimate background level from corners
    border_px = np.concatenate([
        img_rgb[:20, :, :].reshape(-1, c),
        img_rgb[-20:, :, :].reshape(-1, c),
        img_rgb[:, :20, :].reshape(-1, c),
        img_rgb[:, -20:, :].reshape(-1, c),
    ], axis=0)
    bg_pedestal = np.median(border_px, axis=0)

    # 2. Subtract background & apply Asinh tone mapping for compression
    flux = np.maximum(0.0, img_rgb - bg_pedestal)
    p99 = np.percentile(flux, 99.9, axis=(0, 1)) + 1e-6
    norm_flux = np.clip(flux / p99, 0.0, 1.0)

    # Log/Asinh domain: maps faint outer streamers to equal footing with inner corona
    log_base = np.arcsinh(norm_flux * asinh_stretch) / np.arcsinh(asinh_stretch)

    # 3. Multi-Scale Frequency Decomposition (Spatial Bandpass)
    # Fine details: prominences, chromosphere spikes (sigma = 1.5 px)
    fine_blur = cv2.GaussianBlur(log_base, (0, 0), sigmaX=1.5)
    fine_detail = log_base - fine_blur

    # Medium details: coronal magnetic filaments (sigma = 6.0 vs sigma = 24.0 px)
    med_blur_small = cv2.GaussianBlur(log_base, (0, 0), sigmaX=6.0)
    med_blur_large = cv2.GaussianBlur(log_base, (0, 0), sigmaX=24.0)
    streamer_detail = med_blur_small - med_blur_large

    # 4. Synthesize Enhanced Master
    # Blend high frequencies back into compressed base
    enhanced = (
        log_base
        + (fine_sharpen * fine_detail)
        + (streamer_boost * streamer_detail)
    )
    enhanced = np.clip(enhanced, 0.0, 1.0)

    # 5. Black-level calibration: ensure sky background stays neutral dark
    dark_cut = float(np.percentile(enhanced, 1.0))
    calibrated = np.clip((enhanced - dark_cut) / (1.0 - dark_cut), 0.0, 1.0)

    return calibrated.astype(np.float32)


def process_coronal_features(
    input_master_path: Path,
    output_dir: Path,
    sharpen_amount: float = 1.2,
) -> None:
    """Full post-processing workflow for HDR eclipse composites."""
    print("\n" + "=" * 65, flush=True)
    print("       POST-PROCESSING: LOG-SPACE CORONAL FILAMENT EXTRACTION     ", flush=True)
    print("=" * 65, flush=True)
    print(f"  * Input File            : {input_master_path.resolve()}", flush=True)
    print(f"  * Sharpening Multiplier : {sharpen_amount:.2f}", flush=True)
    print("-" * 65, flush=True)

    img = load_any_master(input_master_path)

    enhanced = enhance_coronal_structures(
        img_rgb=img,
        asinh_stretch=12.0,
        fine_sharpen=sharpen_amount,
        streamer_boost=sharpen_amount * 1.2,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 16-Bit TIFF
    out_tiff = output_dir / f"{input_master_path.stem}_Enhanced.tif"
    tifffile.imwrite(
        str(out_tiff),
        (enhanced * 65535.0).astype(np.uint16),
        photometric="rgb",
    )
    print(f"  [Exported 16-bit Enhanced TIFF] -> {out_tiff.resolve()}", flush=True)

    # 2. Preview JPG
    out_jpg = output_dir / f"{input_master_path.stem}_Enhanced.jpg"
    preview_8u = (enhanced * 255.0).astype(np.uint8)
    Image.fromarray(preview_8u, mode="RGB").save(out_jpg, quality=95)
    print(f"  [Exported Enhanced JPG Preview] -> {out_jpg.resolve()}", flush=True)
    print("=" * 65 + "\n", flush=True)