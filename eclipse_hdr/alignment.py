"""Sub-pixel Fourier phase shift stacking and algebraic Taubin limb alignment."""

import gc
from pathlib import Path
from astropy.io import fits
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.registration import phase_cross_correlation


def load_fits_to_float32(fits_path: Path) -> np.ndarray:
    """Robustly loads a FITS file and maps it to a normalized (H, W, C) float32 array in [0.0, 1.0]."""
    with fits.open(fits_path, memmap=False) as hdul:
        data = hdul[0].data.astype(np.float32)

    # Reorder (C, H, W) -> (H, W, C)
    if data.ndim == 3:
        if data.shape[0] == 3:
            img = np.transpose(data, (1, 2, 0))
        else:
            img = data
    elif data.ndim == 2:
        img = np.repeat(data[:, :, np.newaxis], 3, axis=2)
    else:
        raise ValueError(f"Unsupported FITS shape: {data.shape} in {fits_path}")

    # Robust ADU auto-scaling
    max_val = float(np.max(img))
    if max_val > 255.0:
        img /= 65535.0
    elif max_val > 1.0:
        img /= 255.0

    return np.clip(img, 0.0, 1.0)


def save_preview_jpg(
    img_float32: np.ndarray, output_jpg_path: Path, mtf_mid: float = 0.05
) -> None:
    """Exports an 8-bit JPEG preview using an Astronomical Midtone Transfer Function (MTF).

    Reveals faint coronal streamers from linear float data without clipping highlights.
    """
    # 1. Estimate background level (median of sky perimeter)
    h, w, _ = img_float32.shape
    border_pixels = np.concatenate(
        [
            img_float32[:20, :, :].ravel(),
            img_float32[-20:, :, :].ravel(),
            img_float32[:, :20, :].ravel(),
            img_float32[:, -20:, :].ravel(),
        ]
    )
    bg = float(np.median(border_pixels))

    # 2. Subtract sky background pedestal
    img_sub = np.maximum(0.0, img_float32 - bg)
    img_max = float(np.percentile(img_sub, 99.99)) or 1.0
    img_norm = np.clip(img_sub / img_max, 0.0, 1.0)

    # 3. PixInsight-style MTF Stretch: f(x, m) = (m - 1)*x / ((2m - 1)*x - m)
    # where m = mtf_mid (controls stretch strength; lower = brighter corona)
    m = mtf_mid
    stretched = np.where(
        img_norm <= 0.0,
        0.0,
        ((m - 1.0) * img_norm) / (((2.0 * m - 1.0) * img_norm) - m),
    )

    img_8bit = (np.clip(stretched, 0.0, 1.0) * 255.0).astype(np.uint8)
    pil_img = Image.fromarray(img_8bit, mode="RGB")
    pil_img.save(output_jpg_path, quality=94)


def apply_fourier_shift_rgb(
    img_rgb: np.ndarray, shift_y: float, shift_x: float
) -> np.ndarray:
    """Applies sub-pixel 2D translation across all 3 color channels in Fourier space."""
    if abs(shift_y) < 0.05 and abs(shift_x) < 0.05:
        return img_rgb

    shifted_rgb = np.zeros_like(img_rgb)
    for c in range(3):
        shifted_c = ndimage.fourier_shift(
            np.fft.fftn(img_rgb[:, :, c]), (shift_y, shift_x)
        )
        shifted_rgb[:, :, c] = np.fft.ifftn(shifted_c).real

    return np.clip(shifted_rgb, 0.0, 1.0)


def align_and_stack_bucket_dft(
    fits_paths: list[Path],
    sigma_low: float = 3.0,
    sigma_high: float = 3.0,
    upsample_factor: int = 100,
    max_allowed_shift: float = 50.0,
) -> np.ndarray:
    """High-speed single-pass intra-bucket DFT alignment and sigma-clipped stacking."""
    num_frames = len(fits_paths)
    if num_frames == 0:
        raise ValueError("No FITS paths provided for stacking.")

    if num_frames == 1:
        print("  * Single frame bucket: skipping alignment.", flush=True)
        return load_fits_to_float32(fits_paths[0])

    print(f"  * Loading reference frame: {fits_paths[0].name}", flush=True)
    ref_img = load_fits_to_float32(fits_paths[0])
    h, w, _ = ref_img.shape

    # Fast 2x binned luminance for phase cross-correlation
    ref_lum_small = (
        0.299 * ref_img[::2, ::2, 0]
        + 0.587 * ref_img[::2, ::2, 1]
        + 0.114 * ref_img[::2, ::2, 2]
    )
    win_y = np.hanning(ref_lum_small.shape[0])
    win_x = np.hanning(ref_lum_small.shape[1])
    window_2d = np.outer(win_y, win_x).astype(np.float32)
    ref_lum_win = ref_lum_small * window_2d

    # List of valid in-memory aligned frames
    valid_frames: list[np.ndarray] = [ref_img]

    for i in range(1, num_frames):
        sub_img = load_fits_to_float32(fits_paths[i])
        sub_lum_small = (
            0.299 * sub_img[::2, ::2, 0]
            + 0.587 * sub_img[::2, ::2, 1]
            + 0.114 * sub_img[::2, ::2, 2]
        ) * window_2d

        shift, _, _ = phase_cross_correlation(
            ref_lum_win,
            sub_lum_small,
            upsample_factor=upsample_factor,
        )
        shift_y, shift_x = float(shift[0] * 2.0), float(shift[1] * 2.0)
        dist = float(np.hypot(shift_x, shift_y))

        if dist > max_allowed_shift:
            print(
                f"    [{i + 1}/{num_frames}] {fits_paths[i].name} -> [EXCLUDED] Shift {dist:.2f}px exceeds limit {max_allowed_shift:.1f}px (dy={shift_y:+.2f}px, dx={shift_x:+.2f}px)",
                flush=True,
            )
            del sub_img
            continue

        print(
            f"    [{i + 1}/{num_frames}] {fits_paths[i].name} -> [ACCEPTED] Shift: (dy={shift_y:+.2f}px, dx={shift_x:+.2f}px, dist={dist:.2f}px)",
            flush=True,
        )

        shifted_sub = apply_fourier_shift_rgb(sub_img, shift_y, shift_x)
        valid_frames.append(shifted_sub)
        del sub_img

    del ref_lum_small, ref_lum_win, window_2d
    gc.collect()

    n_valid = len(valid_frames)
    print(f"  * Stacking {n_valid}/{num_frames} accepted frames...", flush=True)

    if n_valid == 1:
        return valid_frames[0]

    # Convert list of 3D arrays to (N, H, W, C)
    stack_array = np.stack(valid_frames, axis=0)
    del valid_frames
    gc.collect()

    # Fast 1-pass SIMD Vectorized Sigma Clip
    mean = np.mean(stack_array, axis=0, dtype=np.float32)
    std = np.std(stack_array, axis=0, dtype=np.float32)
    std = np.where(std == 0, 1e-6, std)

    lower_bound = mean - (sigma_low * std)
    upper_bound = mean + (sigma_high * std)
    del std, mean

    # In-place clipping and mean calculation
    np.clip(stack_array, lower_bound, upper_bound, out=stack_array)
    master_stacked = np.mean(stack_array, axis=0, dtype=np.float32)
    del stack_array, lower_bound, upper_bound
    gc.collect()

    return np.clip(master_stacked, 0.0, 1.0)


def fit_circle_taubin(points: np.ndarray) -> tuple[float, float, float]:
    """Algebraic Taubin circle fitting (X^2 + Y^2 + aX + bY + c = 0)."""
    x = points[:, 0]
    y = points[:, 1]
    n = points.shape[0]

    mean_x = np.mean(x)
    mean_y = np.mean(y)
    u = x - mean_x
    v = y - mean_y

    z = u**2 + v**2
    mz = np.mean(z)
    cov_xz = np.mean(u * z)
    cov_yz = np.mean(v * z)
    cov_xx = np.mean(u**2)
    cov_yy = np.mean(v**2)
    cov_xy = np.mean(u * v)

    a_mat = np.array(
        [
            [cov_xx, cov_xy, mz * cov_xx],
            [cov_xy, cov_yy, mz * cov_xy],
            [cov_xx + cov_yy, cov_xy, mz * (cov_xx + cov_yy)],
        ]
    )

    m_xx = np.sum(u * u) / n
    m_yy = np.sum(v * v) / n
    m_xy = np.sum(u * v) / n
    m_xz = np.sum(u * z) / n
    m_yz = np.sum(v * z) / n
    m_zz = np.sum(z * z) / n

    c_mat = np.array(
        [
            [m_zz, m_xz, m_yz],
            [m_xz, m_xx, m_xy],
            [m_yz, m_xy, m_yy],
        ]
    )

    eigvals, eigvecs = np.linalg.eig(c_mat)
    idx = np.argmin(np.abs(eigvals))
    a, b, c = eigvecs[:, idx]

    if a == 0:
        return mean_x, mean_y, 0.0

    center_u = -b / (2 * a)
    center_v = -c / (2 * a)
    radius = np.sqrt(max(0.0, center_u**2 + center_v**2 - mz))

    return center_u + mean_x, center_v + mean_y, radius


def extract_limb_edges_parabolic(
    image: np.ndarray,
    center_est: tuple[float, float],
    r_est: float,
    num_rays: int = 360,
) -> np.ndarray:
    """Extracts sub-pixel radial edge points along the lunar/solar limb."""
    cx, cy = center_est
    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    edge_points = []

    h, w = image.shape
    radial_dist = np.linspace(r_est - 40, r_est + 40, 160)

    for theta in angles:
        xs = cx + radial_dist * np.cos(theta)
        ys = cy + radial_dist * np.sin(theta)

        valid = (xs >= 1) & (xs < w - 2) & (ys >= 1) & (ys < h - 2)
        if not np.all(valid):
            continue

        profile = ndimage.map_coordinates(image, [ys, xs], order=1)
        gradient = -np.gradient(profile)

        max_idx = np.argmax(gradient)
        if 1 < max_idx < len(gradient) - 2:
            y0, y1, y2 = gradient[max_idx - 1], gradient[max_idx], gradient[max_idx + 1]
            denom = 2 * (2 * y1 - y0 - y2)
            delta = (y0 - y2) / denom if denom != 0 else 0.0
            sub_r = radial_dist[max_idx] + delta * (radial_dist[1] - radial_dist[0])

            edge_points.append([cx + sub_r * np.cos(theta), cy + sub_r * np.sin(theta)])

    return np.array(edge_points) if edge_points else np.empty((0, 2))


def align_masters_taubin(
    masters: list[np.ndarray],
    master_names: list[str],
    ref_idx: int = 0,
) -> list[np.ndarray]:
    """Aligns all bracket master frames to the lunar limb centroid of the reference master."""
    print(
        f"\n--- [Stage 3] Inter-Master Sub-Pixel Limb Alignment (Ref: {master_names[ref_idx]}) ---",
        flush=True,
    )
    h, w, _ = masters[ref_idx].shape
    ref_lum = np.mean(masters[ref_idx], axis=2)

    center_est = (w / 2.0, h / 2.0)
    r_est = min(h, w) * 0.22

    ref_edges = extract_limb_edges_parabolic(ref_lum, center_est, r_est)
    if len(ref_edges) < 20:
        print(
            "  [Warning] Insufficient limb edge contrast in reference. Using image center as origin.",
            flush=True,
        )
        ref_cx, ref_cy = center_est
    else:
        ref_cx, ref_cy, _ = fit_circle_taubin(ref_edges)
        print(
            f"  * Reference Lunar Centroid: (cx={ref_cx:.2f}, cy={ref_cy:.2f})",
            flush=True,
        )

    aligned_masters = []
    for img, name in zip(masters, master_names):
        lum = np.mean(img, axis=2)
        edges = extract_limb_edges_parabolic(lum, (ref_cx, ref_cy), r_est)

        if len(edges) < 20:
            print(
                f"  * {name}: Limb too faint/saturated -> Phase correlation fallback.",
                flush=True,
            )
            shift, _, _ = phase_cross_correlation(
                ref_lum[::2, ::2], lum[::2, ::2], upsample_factor=100
            )
            dy, dx = shift[0] * 2.0, shift[1] * 2.0
        else:
            cx, cy, _ = fit_circle_taubin(edges)
            dx = ref_cx - cx
            dy = ref_cy - cy

        print(f"  * {name} -> Offset: (dy={dy:+.2f}px, dx={dx:+.2f}px)", flush=True)
        shifted = apply_fourier_shift_rgb(img, dy, dx)
        aligned_masters.append(shifted)

    return aligned_masters


def winsorized_sigma_clip_stack(
    stack_array: np.ndarray,
    sigma_low: float = 3.0,
    sigma_high: float = 3.0,
) -> np.ndarray:
    """Fast vectorized sigma-clipping stack for pre-allocated arrays in memory.

    Args:
        stack_array: (N, H, W, 3) float32 array
    Returns:
        (H, W, 3) float32 master stacked image
    """
    n_frames = stack_array.shape[0]
    if n_frames == 1:
        return stack_array[0]
    if n_frames == 2:
        return np.mean(stack_array, axis=0, dtype=np.float32)

    mean = np.mean(stack_array, axis=0, dtype=np.float32)
    std = np.std(stack_array, axis=0, dtype=np.float32)
    std = np.where(std == 0, 1e-6, std)

    lower_bound = mean - (sigma_low * std)
    upper_bound = mean + (sigma_high * std)

    winsorized = np.clip(stack_array, lower_bound, upper_bound)
    return np.mean(winsorized, axis=0, dtype=np.float32)
