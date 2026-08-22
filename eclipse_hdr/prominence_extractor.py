"""Chromospheric prominence (H-alpha 656.3nm) extraction and HDR blending."""

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
    return np.clip(img / max(p99, 1e-5), 0.0, 1.0).astype(np.float32)


def solve_center_and_limb_raycast(
    img_rgb: np.ndarray,
    n_rays: int = 360,
    r_min: int = 100,
    r_max: int = 380,
) -> tuple[float, float, float]:
    """Finds sub-pixel center and radius via radial gradient ray-casting."""
    lum = 0.299 * img_rgb[..., 0] + 0.587 * img_rgb[..., 1] + 0.114 * img_rgb[..., 2]
    h, w = lum.shape
    cx_init, cy_init = float(w / 2.0), float(h / 2.0)

    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    radii = np.arange(r_min, r_max, 1.0, dtype=np.float32)

    limb_xs, limb_ys = [], []

    for theta in angles:
        ray_x = cx_init + radii * np.cos(theta)
        ray_y = cy_init + radii * np.sin(theta)

        valid = (ray_x >= 0) & (ray_x < w - 1) & (ray_y >= 0) & (ray_y < h - 1)
        if not np.all(valid):
            continue

        profile = cv2.remap(
            lum.astype(np.float32),
            ray_x.astype(np.float32).reshape(1, -1),
            ray_y.astype(np.float32).reshape(1, -1),
            interpolation=cv2.INTER_LINEAR,
        ).ravel()

        d_profile = np.gradient(profile)
        peak_idx = int(np.argmax(d_profile))
        r_peak = radii[peak_idx]

        limb_xs.append(cx_init + r_peak * np.cos(theta))
        limb_ys.append(cy_init + r_peak * np.sin(theta))

    xs = np.array(limb_xs, dtype=np.float64)
    ys = np.array(limb_ys, dtype=np.float64)

    A = np.column_stack([2.0 * xs, 2.0 * ys, np.ones_like(xs)])
    b = xs**2 + ys**2
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    cx = float(sol[0])
    cy = float(sol[1])
    r_lunar = float(np.sqrt(sol[2] + cx**2 + cy**2))

    return cx, cy, r_lunar


def extract_isolated_prominences(
    short_exp_rgb: np.ndarray,
    center: tuple[float, float],
    r_lunar: float,
    boost: float = 2.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Extracts H-alpha emission features using color-ratio chromatic subtraction."""
    h, w, _ = short_exp_rgb.shape
    cx, cy = center

    r_ch = short_exp_rgb[..., 0]
    g_ch = short_exp_rgb[..., 1]
    b_ch = short_exp_rgb[..., 2]

    # Continuum level (average of green and blue)
    continuum = (g_ch + b_ch) / 2.0

    # Chromatic ratio: Prominences are strongly red compared to neutral continuum
    # (R - Continuum) normalized by continuum to isolate spectral purity
    chromatic_excess = np.maximum(0.0, r_ch - 1.15 * continuum)

    # Distance map to solar centroid
    y_idx, x_idx = np.ogrid[:h, :w]
    dist_map = np.hypot(x_idx - cx, y_idx - cy)

    # Annular spatial mask: allow prominences that sit slightly inside or outside the limb
    # Range: from (R_lunar - 6px) to (R_lunar + 45px)
    inner_edge = np.clip((dist_map - (r_lunar - 6.0)) / 3.0, 0.0, 1.0)
    outer_edge = np.clip(((r_lunar + 45.0) - dist_map) / 15.0, 0.0, 1.0)
    annulus = inner_edge * outer_edge

    # Isolated prominence intensity
    h_alpha_signal = chromatic_excess * annulus * boost

    # Synthesize characteristic H-alpha emission color (Ruby Red + Balmer Violet)
    prom_rgb = np.zeros_like(short_exp_rgb)
    prom_rgb[..., 0] = np.clip(h_alpha_signal * 1.6, 0.0, 1.0)  # Red
    prom_rgb[..., 1] = np.clip(h_alpha_signal * 0.15, 0.0, 1.0)  # Green (low)
    prom_rgb[..., 2] = np.clip(h_alpha_signal * 0.35, 0.0, 1.0)  # Blue (Balmer)

    # Alpha mask for blending
    alpha_mask = np.clip(h_alpha_signal * 2.0, 0.0, 1.0)

    return prom_rgb, alpha_mask


def blend_prominences_to_master(
    hdr_master_rgb: np.ndarray,
    short_exp_rgb: np.ndarray,
    output_dir: Path,
    stem: str,
    boost: float = 2.5,
) -> np.ndarray:
    """Detects alignment and blends prominences directly over HDR composite."""
    # 1. Solve geometry dynamically on the short exposure frame
    cx, cy, r_lunar = solve_center_and_limb_raycast(short_exp_rgb)
    print(
        f"  * Detected Short Exposure Limb : Center=({cx:.1f}, {cy:.1f}), R={r_lunar:.1f}px"
    )

    # 2. Extract prominence layer
    prom_rgb, alpha_mask = extract_isolated_prominences(
        short_exp_rgb, center=(cx, cy), r_lunar=r_lunar, boost=boost
    )

    # Save isolated prominence debug layer
    out_debug = output_dir / f"{stem}_debug_prominences_isolated.jpg"
    Image.fromarray((prom_rgb * 255.0).astype(np.uint8)).save(out_debug, quality=95)
    print(f"  -> Saved Debug Prominences  : {out_debug.name}")

    # 3. Composite onto HDR Master
    composite = hdr_master_rgb.copy()
    for ch in range(3):
        # Linear addition with soft highlight knee
        composite[..., ch] = composite[..., ch] + prom_rgb[..., ch]

    # Clean the lunar interior (strictly inside R_lunar - 6px)
    h, w, _ = composite.shape
    y_idx, x_idx = np.ogrid[:h, :w]
    dist_map = np.hypot(x_idx - cx, y_idx - cy)
    moon_gate = np.clip((dist_map - (r_lunar - 6.0)) / 2.0, 0.0, 1.0)[:, :, None]

    final_composite = np.clip(composite * moon_gate, 0.0, 1.0)
    return final_composite


def run_prominence_pipeline(
    hdr_master_path: Path,
    short_exposure_path: Path,
    output_dir: Path,
    boost: float = 2.5,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 65)
    print("       DYNAMIC H-ALPHA PROMINENCE EXTRACTION & COMPOSITING       ")
    print("=" * 65)
    print(f"  * HDR Master File        : {hdr_master_path.name}")
    print(f"  * Short Exposure Frame   : {short_exposure_path.name}")
    print(f"  * Prominence Gain Boost  : {boost:.2f}")
    print("-" * 65)

    hdr_rgb = load_normalized_rgb(hdr_master_path)
    short_rgb = load_normalized_rgb(short_exposure_path)

    composite = blend_prominences_to_master(
        hdr_master_rgb=hdr_rgb,
        short_exp_rgb=short_rgb,
        output_dir=output_dir,
        stem=hdr_master_path.stem,
        boost=boost,
    )

    out_tif = output_dir / f"{hdr_master_path.stem}_Prominences_Enhanced.tif"
    tifffile.imwrite(
        str(out_tif), (composite * 65535.0).astype(np.uint16), photometric="rgb"
    )
    print(f"  [Exported 16-bit Master] -> {out_tif.resolve()}")

    out_jpg = output_dir / f"{hdr_master_path.stem}_Prominences_Enhanced.jpg"
    Image.fromarray((composite * 255.0).astype(np.uint8)).save(out_jpg, quality=96)
    print(f"  [Exported Preview JPG]   -> {out_jpg.resolve()}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    import sys

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
        boost=3.0,
    )
