"""Dual-mode HDR engine: Scientific Photometric Linear Radiance & Artistic Mertens Fusion."""

import json
from pathlib import Path
from astropy.io import fits
import cv2
import numpy as np
from PIL import Image
import tifffile


def parse_exposure_seconds(name: str) -> float:
    """Parses exposure time in seconds from bracket or file name (e.g. 'bucket_1_1000s' -> 0.001)."""
    clean = name.lower().replace("aligned_master_", "").replace("master_", "").replace("bucket_", "").replace("s", "").replace(".fit", "")
    if "_" in clean:
        parts = clean.split("_")
        try:
            return float(parts[0]) / float(parts[1])
        except (ValueError, ZeroDivisionError):
            pass
    try:
        return float(clean)
    except ValueError:
        return 1.0


def fuse_scientific_linear_radiance(
    aligned_masters: list[np.ndarray],
    master_names: list[str],
    output_path: Path,
    sat_threshold: float = 0.90,
) -> None:
    """Reconstructs a photometric 32-bit linear radiance map using linear photon SNR weighting."""
    num_masters = len(aligned_masters)
    h, w, c = aligned_masters[0].shape

    print("\n" + "=" * 65, flush=True)
    print("      SCIENTIFIC PHOTOMETRIC RADIANCE RECONSTRUCTION (LINEAR)     ", flush=True)
    print("=" * 65, flush=True)
    print(f"  * Brackets in Stack      : {num_masters} frames", flush=True)
    print(f"  * Saturation Ceiling     : {sat_threshold:.2f}", flush=True)
    print("-" * 65, flush=True)

    numerator = np.zeros((h, w, c), dtype=np.float64)
    denominator = np.zeros((h, w, c), dtype=np.float64)

    for img, name in zip(aligned_masters, master_names):
        t_exp = parse_exposure_seconds(name)

        # 1. Channel-wise background estimation from image borders
        border_px = np.concatenate([
            img[:20, :, :].reshape(-1, c),
            img[-20:, :, :].reshape(-1, c),
            img[:, :20, :].reshape(-1, c),
            img[:, -20:, :].reshape(-1, c),
        ], axis=0)

        bg = np.median(border_px, axis=0)  # Shape (3,)

        # Net linear flux above pedestal
        flux = np.maximum(0.0, img.astype(np.float64) - bg)

        # 2. Continuous Weight Function:
        # Rejects saturation, scales with SNR (proportional to flux * t_exp)
        # Linear ramp near saturation to avoid hard boundary steps
        sat_mask = np.clip((sat_threshold - img) / 0.05, 0.0, 1.0)
        weight = np.where(img < sat_threshold, flux * sat_mask, 0.0).astype(np.float64)

        # Radiance rate in counts/second (photons / sec equivalent)
        radiance_rate = flux / t_exp

        active_px_count = int(np.count_nonzero(weight > 1e-4))
        print(f"  * {name:22s} (t={t_exp:8.5f}s, bg={np.mean(bg):.5f}) -> Active Non-Zero Pixels: {active_px_count:,}", flush=True)

        numerator += weight * radiance_rate
        denominator += weight

    # 3. Handle regions:
    # - Active pixels: weighted average
    # - Saturated everywhere (inner core): fastest bracket radiance
    # - Pure sky/Moon shadow (zero weight): 0.0
    radiance_map = np.zeros((h, w, c), dtype=np.float32)
    valid_mask = denominator > 1e-9

    radiance_map[valid_mask] = (numerator[valid_mask] / denominator[valid_mask]).astype(np.float32)

    # 4. Save 32-bit FITS and TIFF
    fits_out = output_path.with_name(f"{output_path.stem}_Scientific_Linear.fits")
    fits.writeto(fits_out, np.transpose(radiance_map, (2, 0, 1)), overwrite=True)
    print(f"\n  [Exported 32-bit FITS Linear Radiance] -> {fits_out.resolve()}", flush=True)

    tiff_out = output_path.with_name(f"{output_path.stem}_Scientific_Linear.tif")
    tifffile.imwrite(str(tiff_out), radiance_map, photometric="rgb")
    print(f"  [Exported 32-bit TIFF Linear Radiance] -> {tiff_out.resolve()}", flush=True)

    # 5. Scientific Preview (Asinh Density Stretch)
    preview_jpg = output_path.with_name(f"{output_path.stem}_Scientific_Preview.jpg")
    p99_val = float(np.percentile(radiance_map[radiance_map > 0], 99.9)) if np.any(radiance_map > 0) else 1.0
    norm_rad = np.clip(radiance_map / max(p99_val, 1e-4), 0.0, 1.0)

    beta = 50.0
    stretched = np.arcsinh(norm_rad * beta) / np.arcsinh(beta)
    preview_8u = (np.clip(stretched, 0.0, 1.0) * 255.0).astype(np.uint8)

    Image.fromarray(preview_8u, mode="RGB").save(preview_jpg, quality=95)
    print(f"  [Exported Scientific Preview JPG]      -> {preview_jpg.resolve()}", flush=True)
    print("=" * 65 + "\n", flush=True)

def fuse_artistic_mertens(
    aligned_masters: list[np.ndarray],
    master_names: list[str],
    output_path: Path,
    contrast_w: float = 0.8,
    sat_w: float = 1.0,
    exp_w: float = 0.8,
    pre_stretch_factor: float = 12.0,
) -> None:
    """Fuses bracketed astronomical masters using tone-curved multi-scale Mertens fusion."""
    num_masters = len(aligned_masters)

    print("\n" + "=" * 65, flush=True)
    print("           ARTISTIC OPENCV MERTENS HDR PYRAMID FUSION         ", flush=True)
    print("=" * 65, flush=True)
    print(f"  * Input Brackets            : {num_masters} frames", flush=True)
    print(f"  * Contrast Weight (W_c)     : {contrast_w:.3f}", flush=True)
    print(f"  * Saturation Weight (W_s)   : {sat_w:.3f}", flush=True)
    print(f"  * Exposure Weight (W_e)     : {exp_w:.3f}", flush=True)
    print(f"  * Pre-Stretch Factor (Asinh): {pre_stretch_factor:.1f}", flush=True)
    print("-" * 65, flush=True)

    input_frames_32f = []
    for name, img in zip(master_names, aligned_masters):
        border_px = np.concatenate([
            img[:20, :, :].ravel(),
            img[-20:, :, :].ravel(),
            img[:, :20, :].ravel(),
            img[:, -20:, :].ravel(),
        ])
        bg = float(np.median(border_px))
        sub = np.maximum(0.0, img - bg)
        p99_9 = float(np.percentile(sub, 99.95)) or 1.0
        norm_linear = np.clip(sub / p99_9, 0.0, 1.0)

        # Tone curve pre-conditioning for Mertens
        norm_stretched = np.arcsinh(norm_linear * pre_stretch_factor) / np.arcsinh(pre_stretch_factor)
        input_frames_32f.append(norm_stretched.astype(np.float32))

    merge_mertens = cv2.createMergeMertens(
        contrast_weight=contrast_w,
        saturation_weight=sat_w,
        exposure_weight=exp_w,
    )
    fusion_rgb = merge_mertens.process(input_frames_32f)
    fusion_rgb = np.nan_to_num(fusion_rgb, nan=0.0, posinf=1.0, neginf=0.0)
    fusion_rgb = np.clip(fusion_rgb, 0.0, 1.0)

    f_max = float(np.percentile(fusion_rgb, 99.99)) or 1.0
    fusion_rgb = np.clip(fusion_rgb / f_max, 0.0, 1.0)

    # Save 16-Bit Master TIFF
    tiff_out = output_path.with_name(f"{output_path.stem}_Artistic_HDR.tif")
    fusion_16u = (fusion_rgb * 65535.0).astype(np.uint16)
    tifffile.imwrite(
        str(tiff_out),
        fusion_16u,
        photometric="rgb",
        compression=None,
    )
    print(f"\n  [Exported 16-bit Master TIFF] -> {tiff_out.resolve()}", flush=True)

    # Save Preview JPG
    preview_jpg = output_path.with_name(f"{output_path.stem}_Artistic_Preview.jpg")
    preview_8u = (fusion_rgb * 255.0).astype(np.uint8)
    Image.fromarray(preview_8u, mode="RGB").save(preview_jpg, quality=95)
    print(f"  [Exported Preview JPG]        -> {preview_jpg.resolve()}", flush=True)

    # Save Parameter Report JSON
    params_json_path = output_path.with_name(f"{output_path.stem}_Artistic_Report.json")
    with open(params_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": "Artistic OpenCV MergeMertens",
                "parameters": {
                    "contrast_weight": contrast_w,
                    "saturation_weight": sat_w,
                    "exposure_weight": exp_w,
                    "pre_stretch_factor": pre_stretch_factor,
                },
                "input_brackets": master_names,
            },
            f,
            indent=2,
        )
    print(f"  [Saved Parameter Report]      -> {params_json_path.resolve()}", flush=True)
    print("=" * 65 + "\n", flush=True)


def fuse_and_export_hdr(
    aligned_masters: list[np.ndarray],
    master_names: list[str],
    output_path: Path,
    mode: str = "both",
    contrast_w: float = 0.8,
    sat_w: float = 1.0,
    exp_w: float = 0.8,
    pre_stretch_factor: float = 12.0,
) -> None:
    """Router to execute scientific linear fusion, artistic Mertens fusion, or both."""
    if mode in ("scientific", "both"):
        fuse_scientific_linear_radiance(
            aligned_masters=aligned_masters,
            master_names=master_names,
            output_path=output_path,
        )

    if mode in ("artistic", "both"):
        fuse_artistic_mertens(
            aligned_masters=aligned_masters,
            master_names=master_names,
            output_path=output_path,
            contrast_w=contrast_w,
            sat_w=sat_w,
            exp_w=exp_w,
            pre_stretch_factor=pre_stretch_factor,
        )