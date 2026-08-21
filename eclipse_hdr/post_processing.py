"""Artifact-free astronomical coronal detail enhancement via local contrast bandpass."""

from pathlib import Path
from astropy.io import fits
import cv2
import numpy as np
from PIL import Image
import tifffile


def load_master_for_enhancement(input_path: Path) -> tuple[np.ndarray, bool]:
    """Loads master and detects if it is linear (FITS/linear TIFF) or already tone-mapped."""
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

        # Linear flux normalization
        p99 = float(np.percentile(img[img > 0], 99.95)) if np.any(img > 0) else 1.0
        img = np.clip(img / max(p99, 1e-6), 0.0, 1.0)

    else:
        raw = tifffile.imread(str(input_path))
        if raw.ndim == 2:
            raw = np.repeat(raw[:, :, np.newaxis], 3, axis=2)
        elif raw.ndim == 3 and raw.shape[2] > 3:
            raw = raw[:, :, :3]

        if raw.dtype == np.uint16:
            img = raw.astype(np.float32) / 65535.0
        elif raw.dtype == np.uint8:
            img = raw.astype(np.float32) / 255.0
        else:
            img = raw.astype(np.float32)
            if "linear" in input_path.stem.lower():
                is_linear = True

        p_max = float(np.percentile(img, 99.99)) or 1.0
        img = np.clip(img / p_max, 0.0, 1.0)

    return img, is_linear


def enhance_coronal_structures(
    img_rgb: np.ndarray,
    is_linear: bool = False,
    fine_sharpen: float = 0.8,
    streamer_boost: float = 1.0,
) -> np.ndarray:
    """Enhances fine magnetic loops and coronal streamers without blowing out the core."""
    h, w, c = img_rgb.shape

    # 1. Tone curve: Only apply asinh if input is strictly linear
    if is_linear:
        border_px = np.concatenate([
            img_rgb[:20, :, :].reshape(-1, c),
            img_rgb[-20:, :, :].reshape(-1, c),
            img_rgb[:, :20, :].reshape(-1, c),
            img_rgb[:, -20:, :].reshape(-1, c),
        ], axis=0)
        bg = np.median(border_px, axis=0)
        flux = np.maximum(0.0, img_rgb - bg)
        base = np.arcsinh(flux * 20.0) / np.arcsinh(20.0)
    else:
        # Artistic Mertens TIFF is ALREADY tone-mapped; do NOT re-stretch
        base = img_rgb.copy()

    # 2. Multi-Scale Frequency Decomposition (Spatial Bandpass)
    # Fine details: prominences, chromosphere (sigma = 1.5 px)
    fine_blur = cv2.GaussianBlur(base, (0, 0), sigmaX=1.5)
    fine_detail = base - fine_blur

    # Medium details: coronal streamers and magnetic arches (sigma = 6 vs sigma = 24 px)
    med_blur_small = cv2.GaussianBlur(base, (0, 0), sigmaX=6.0)
    med_blur_large = cv2.GaussianBlur(base, (0, 0), sigmaX=24.0)
    streamer_detail = med_blur_small - med_blur_large

    # 3. High-Pass Detail Recombination (Add without global pedestal shift)
    enhanced = base + (fine_sharpen * fine_detail) + (streamer_boost * streamer_detail)

    # 4. Safe Highlight Preservation (Prevents clipping saturated cores to 1.0)
    # Soft knee compression on upper 10% of dynamic range
    enhanced = np.where(
        enhanced > 0.90,
        0.90 + 0.10 * np.tanh((enhanced - 0.90) / 0.10),
        enhanced,
    )

    return np.clip(enhanced, 0.0, 1.0).astype(np.float32)


def process_coronal_features(
    input_master_path: Path,
    output_dir: Path,
    sharpen_amount: float = 0.8,
) -> None:
    """Post-processing pipeline for HDR eclipse composites."""
    print("\n" + "=" * 65, flush=True)
    print("       POST-PROCESSING: LOG-SPACE CORONAL FILAMENT EXTRACTION     ", flush=True)
    print("=" * 65, flush=True)
    print(f"  * Input File            : {input_master_path.resolve()}", flush=True)
    print(f"  * Sharpening Multiplier : {sharpen_amount:.2f}", flush=True)

    img, is_linear = load_master_for_enhancement(input_master_path)
    print(f"  * Detected Data Mode    : {'Strict Linear' if is_linear else 'Tone-Mapped (Mertens)'}", flush=True)
    print("-" * 65, flush=True)

    enhanced = enhance_coronal_structures(
        img_rgb=img,
        is_linear=is_linear,
        fine_sharpen=sharpen_amount,
        streamer_boost=sharpen_amount * 1.1,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Export 16-Bit Master TIFF
    out_tiff = output_dir / f"{input_master_path.stem}_Enhanced.tif"
    tifffile.imwrite(
        str(out_tiff),
        (enhanced * 65535.0).astype(np.uint16),
        photometric="rgb",
    )
    print(f"  [Exported 16-bit Enhanced TIFF] -> {out_tiff.resolve()}", flush=True)

    # Export Preview JPG
    out_jpg = output_dir / f"{input_master_path.stem}_Enhanced.jpg"
    preview_8u = (enhanced * 255.0).astype(np.uint8)
    Image.fromarray(preview_8u, mode="RGB").save(out_jpg, quality=95)
    print(f"  [Exported Enhanced JPG Preview] -> {out_jpg.resolve()}", flush=True)
    print("=" * 65 + "\n", flush=True)