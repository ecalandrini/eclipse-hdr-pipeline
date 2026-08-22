"""Perceptual CIELAB Bilateral Filtering & CLAHE Coronal Sculpting."""

from pathlib import Path
from astropy.io import fits
import cv2
import numpy as np
from PIL import Image
import tifffile


def load_master_rgb(file_path: Path) -> np.ndarray:
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


def apply_bilateral_clahe(
    img_rgb: np.ndarray,
    center: tuple[float, float] = (2492.4, 1612.9),
    r_lunar: float = 222.7,
    clahe_clip: float = 2.5,
    tile_grid_size: tuple[int, int] = (16, 16),
    bilateral_sigma_color: float = 0.15,
    bilateral_sigma_space: float = 7.0,
    detail_mix: float = 1.3,
) -> np.ndarray:
    """Edge-preserving local dynamic range compression via CIELAB Bilateral CLAHE."""
    h, w, _ = img_rgb.shape
    cx, cy = center

    # 1. Convert to CIELAB space to decouple Luminance (L*) from Chrominance (a*, b*)
    rgb_8u = (np.clip(img_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    lab = cv2.cvtColor(rgb_8u, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    lum = lab[..., 0]  # L* channel [0.0, 1.0]

    # 2. Bilateral Decomposition on L* (Separates Large-scale base from fine streamer texture)
    base_lum = cv2.bilateralFilter(
        lum.astype(np.float32),
        d=9,
        sigmaColor=bilateral_sigma_color,
        sigmaSpace=bilateral_sigma_space,
    )
    detail_lum = lum - base_lum  # Fine magnetic loop residuals

    # 3. CLAHE on the Base Luminance
    base_8u = (np.clip(base_lum, 0.0, 1.0) * 255.0).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=tile_grid_size)
    clahe_base = clahe.apply(base_8u).astype(np.float32) / 255.0

    # 4. Recombine Enhanced Base + Boosted High-frequency Detail
    enhanced_lum = clahe_base + (detail_mix * detail_lum)

    # 5. Moon Mask to keep lunar disk clear and dark
    y_idx, x_idx = np.ogrid[:h, :w]
    dist_map = np.hypot(x_idx - cx, y_idx - cy)
    moon_gate = np.clip((dist_map - r_lunar) / 3.0, 0.0, 1.0)

    enhanced_lum = np.clip(enhanced_lum * moon_gate, 0.0, 1.0)

    # 6. Reconstruct CIELAB -> RGB
    lab[..., 0] = enhanced_lum
    lab_8u = (np.clip(lab, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgb_out = cv2.cvtColor(lab_8u, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0

    return rgb_out


def run_bilateral_clahe_pipeline(
    input_file: Path,
    output_dir: Path,
    center: tuple[float, float] = (2492.4, 1612.9),
    r_lunar: float = 222.7,
    clahe_clip: float = 2.5,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 65)
    print("       CIELAB BILATERAL CLAHE CORONAL TONAL ENHANCEMENT          ")
    print("=" * 65)
    print(f"  * Input Master File      : {input_file.name}")
    print(f"  * Center Coordinates     : ({center[0]:.1f}, {center[1]:.1f})")
    print(f"  * Lunar Radius           : {r_lunar:.1f} px")
    print(f"  * CLAHE Clip Limit       : {clahe_clip:.2f}")
    print("-" * 65)

    img = load_master_rgb(input_file)
    enhanced = apply_bilateral_clahe(
        img_rgb=img,
        center=center,
        r_lunar=r_lunar,
        clahe_clip=clahe_clip,
    )

    out_tif = output_dir / f"{input_file.stem}_Bilateral_CLAHE.tif"
    tifffile.imwrite(
        str(out_tif), (enhanced * 65535.0).astype(np.uint16), photometric="rgb"
    )
    print(f"  [Exported 16-bit TIFF] -> {out_tif.resolve()}")

    out_jpg = output_dir / f"{input_file.stem}_Bilateral_CLAHE.jpg"
    Image.fromarray((enhanced * 255.0).astype(np.uint8)).save(out_jpg, quality=96)
    print(f"  [Exported Preview JPG] -> {out_jpg.resolve()}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    import sys

    target = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("Eclipse_HDR_Master_Artistic_HDR.tif")
    )
    out = Path("workspace/bilateral_clahe")
    run_bilateral_clahe_pipeline(target, out)
