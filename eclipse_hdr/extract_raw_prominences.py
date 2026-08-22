"""Direct Raw Prominence Extractor with Sub-Pixel Shift Auto-Registration."""

from pathlib import Path
from astropy.io import fits
import cv2
import numpy as np
from PIL import Image
import tifffile


def load_raw_rgb(file_path: Path) -> np.ndarray:
    """Loads FITS/TIFF preserving exact physical linear ADU ratios."""
    ext = file_path.suffix.lower()
    if ext in (".fit", ".fits"):
        with fits.open(file_path, memmap=False) as hdul:
            data = hdul[0].data.astype(np.float32)

        if data.ndim == 3 and data.shape[0] == 3:
            img = np.transpose(data, (1, 2, 0))
        elif data.ndim == 2:
            img = np.repeat(data[:, :, None], 3, axis=2)
        else:
            img = data

        max_adu = float(img.max())
        img_norm = img / 65535.0 if max_adu > 255.0 else img / 255.0
    else:
        raw = tifffile.imread(str(file_path)).astype(np.float32)
        if raw.ndim == 2:
            raw = np.repeat(raw[:, :, None], 3, axis=2)
        elif raw.ndim == 3 and raw.shape[2] > 3:
            raw = raw[:, :, :3]
        img_norm = (
            raw / 65535.0
            if raw.dtype == np.uint16 or raw.max() > 255.0
            else raw / 255.0
        )

    return np.clip(img_norm, 0.0, 1.0)


def solve_center_and_limb_raycast(
    img_rgb: np.ndarray,
    n_rays: int = 360,
    r_min: int = 100,
    r_max: int = 380,
) -> tuple[float, float, float]:
    """Deterministically finds sub-pixel lunar center and radius."""
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


def extract_and_composite_prominences(
    short_fits: Path,
    hdr_master_file: Path,
    output_dir: Path,
    prominence_gain: float = 1.8,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rgb = load_raw_rgb(short_fits)
    hdr_rgb = load_raw_rgb(hdr_master_file)
    h, w, _ = raw_rgb.shape

    print("\n" + "=" * 70)
    print("   DYNAMIC SUB-PIXEL AUTO-REGISTERED PROMINENCE COMPOSITING      ")
    print("=" * 70)

    # 1. Independent Geometry Detection
    cx_s, cy_s, r_s = solve_center_and_limb_raycast(raw_rgb)
    cx_m, cy_m, r_m = solve_center_and_limb_raycast(hdr_rgb)

    shift_x = cx_m - cx_s
    shift_y = cy_m - cy_s

    print(f"  * Short Frame Centroid  : ({cx_s:.2f}, {cy_s:.2f}), R={r_s:.2f}px")
    print(f"  * Master HDR Centroid   : ({cx_m:.2f}, {cy_m:.2f}), R={r_m:.2f}px")
    print(f"  * Calculated Offset     : dx = {shift_x:+.2f} px, dy = {shift_y:+.2f} px")
    print("-" * 70)

    # 2. Extract H-Alpha in short frame native coordinate system
    continuum = (raw_rgb[..., 1] + raw_rgb[..., 2]) / 2.0
    h_alpha_raw = np.maximum(0.0, raw_rgb[..., 0] - continuum)

    y_idx, x_idx = np.ogrid[:h, :w]
    dist_short = np.hypot(x_idx - cx_s, y_idx - cy_s)
    smooth_gate_s = np.clip((dist_short - (r_s - 6.0)) / 2.0, 0.0, 1.0) * np.clip(
        ((r_s + 35.0) - dist_short) / 5.0, 0.0, 1.0
    )

    h_alpha_signal = h_alpha_raw * smooth_gate_s
    peak_sig = float(h_alpha_signal.max())

    norm_halpha = h_alpha_signal / max(peak_sig, 1e-6)
    stretched_halpha = np.arcsinh(norm_halpha * 20.0) / np.arcsinh(20.0)

    prom_layer_s = np.zeros_like(raw_rgb)
    prom_layer_s[..., 0] = np.clip(stretched_halpha * 1.00, 0.0, 1.0)
    prom_layer_s[..., 1] = np.clip(stretched_halpha * 0.06, 0.0, 1.0)
    prom_layer_s[..., 2] = np.clip(stretched_halpha * 0.28, 0.0, 1.0)

    alpha_prom_s = np.clip((stretched_halpha - 0.03) / 0.35, 0.0, 1.0) * smooth_gate_s

    # 3. Sub-Pixel Geometric Registration (Warping Short Layer -> Master Canvas)
    M_trans = np.float32([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]])
    prom_layer_aligned = cv2.warpAffine(
        prom_layer_s,
        M_trans,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    alpha_prom_aligned = cv2.warpAffine(
        alpha_prom_s,
        M_trans,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )[:, :, None]

    # Save aligned isolated prominence layer
    out_prom = output_dir / f"{short_fits.stem}_prominences_aligned.jpg"
    Image.fromarray((prom_layer_aligned * 255.0).astype(np.uint8)).save(
        out_prom, quality=96
    )
    print(f"  -> Saved Aligned Prominence Layer : {out_prom.name}")

    # 4. Chromatic Injection Blending on Master Frame
    composite = (1.0 - alpha_prom_aligned) * hdr_rgb + alpha_prom_aligned * (
        prom_layer_aligned * prominence_gain
    )

    # 5. Moon Mask on Master Centroid (preserving chromospheric base)
    dist_master = np.hypot(x_idx - cx_m, y_idx - cy_m)
    moon_gate = np.clip((dist_master - (r_m - 3.0)) / 2.0, 0.0, 1.0)[:, :, None]
    final_composite = np.clip(composite * moon_gate, 0.0, 1.0)

    # 6. Export
    out_tif = output_dir / f"{hdr_master_file.stem}_Prominences_Aligned_Final.tif"
    tifffile.imwrite(
        str(out_tif), (final_composite * 65535.0).astype(np.uint16), photometric="rgb"
    )
    print(f"  [Exported 16-Bit Master] -> {out_tif.resolve()}")

    out_jpg = output_dir / f"{hdr_master_file.stem}_Prominences_Aligned_Final.jpg"
    Image.fromarray((final_composite * 255.0).astype(np.uint8)).save(
        out_jpg, quality=96
    )
    print(f"  [Exported Final Preview] -> {out_jpg.resolve()}")
    print("=" * 70 + "\n")


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
    out_d = Path("workspace/prominences_aligned")

    extract_and_composite_prominences(raw_f, hdr_f, out_d, prominence_gain=1.8)
