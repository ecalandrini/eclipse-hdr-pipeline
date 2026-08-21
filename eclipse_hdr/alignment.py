"""Sub-pixel Fourier phase shift stacking and algebraic Taubin limb alignment."""

from pathlib import Path
from astropy.io import fits
import numpy as np
from scipy import ndimage
from skimage.registration import phase_cross_correlation


def load_fits_to_float32(fits_path: Path) -> np.ndarray:
    """Loads a FITS file and returns an (H, W, C) float32 RGB array normalized to [0.0, 1.0]."""
    with fits.open(fits_path) as hdul:
        data = hdul[0].data.astype(np.float32)

    # Convert from FITS (C, H, W) to standard image (H, W, C)
    if data.ndim == 3:
        if data.shape[0] == 3:
            img = np.transpose(data, (1, 2, 0))
        else:
            img = data
    elif data.ndim == 2:
        img = np.repeat(data[:, :, np.newaxis], 3, axis=2)
    else:
        raise ValueError(f"Unsupported FITS dimension: {data.ndim} in {fits_path}")

    max_val = np.max(img)
    if max_val > 1.0:
        # 16-bit ADU normalization
        if max_val > 255.0:
            img /= 65535.0
        else:
            img /= 255.0

    return np.clip(img, 0.0, 1.0)


def winsorized_sigma_clip_stack(
    stack_array: np.ndarray,
    sigma_low: float = 3.0,
    sigma_high: float = 3.0,
) -> np.ndarray:
    """Vectorized Median Absolute Deviation (MAD) Winsorized sigma-clipping stack.

    Args:
        stack_array: (N, H, W, 3) float32 array
    Returns:
        (H, W, 3) float32 master stacked image
    """
    median = np.median(stack_array, axis=0)
    mad = np.median(np.abs(stack_array - median), axis=0)
    sigma_est = 1.4826 * mad

    # Guard against zero-variance in dead pixels
    sigma_est = np.where(sigma_est == 0, 1e-6, sigma_est)

    lower_bound = median - (sigma_low * sigma_est)
    upper_bound = median + (sigma_high * sigma_est)

    # Winsorize: clamp outliers to threshold boundaries
    winsorized = np.clip(stack_array, lower_bound, upper_bound)
    return np.mean(winsorized, axis=0).astype(np.float32)


def apply_fourier_shift_rgb(
    img_rgb: np.ndarray, shift_y: float, shift_x: float
) -> np.ndarray:
    """Applies sub-pixel 2D translation across all 3 color channels in Fourier space."""
    if abs(shift_y) < 1e-4 and abs(shift_x) < 1e-4:
        return img_rgb

    shifted_rgb = np.zeros_like(img_rgb)
    for c in range(3):
        # fourier_shift operates in frequency domain without spatial kernel blur
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
) -> np.ndarray:
    """High-performance intra-bucket sub-pixel DFT alignment & MAD sigma-clip stacking."""
    num_frames = len(fits_paths)
    if num_frames == 0:
        raise ValueError("No FITS paths provided for stacking.")

    # Single frame shortcut
    if num_frames == 1:
        print("  * Single frame bucket: skipping alignment.")
        return load_fits_to_float32(fits_paths[0])

    # Load reference frame (first frame)
    print(f"  * Loading reference frame: {fits_paths[0].name}")
    ref_img = load_fits_to_float32(fits_paths[0])
    h, w, _ = ref_img.shape

    # Fast 2x downsampled luminance for phase cross-correlation
    ref_lum_small = (
        0.299 * ref_img[::2, ::2, 0]
        + 0.587 * ref_img[::2, ::2, 1]
        + 0.114 * ref_img[::2, ::2, 2]
    )

    aligned_stack = np.zeros((num_frames, h, w, 3), dtype=np.float32)
    aligned_stack[0] = ref_img

    for i in range(1, num_frames):
        sub_img = load_fits_to_float32(fits_paths[i])

        sub_lum_small = (
            0.299 * sub_img[::2, ::2, 0]
            + 0.587 * sub_img[::2, ::2, 1]
            + 0.114 * sub_img[::2, ::2, 2]
        )

        # Single-step DFT sub-pixel cross correlation on binned luminance
        shift, _, _ = phase_cross_correlation(
            ref_lum_small,
            sub_lum_small,
            upsample_factor=upsample_factor,
        )

        # Scale shift back to full-resolution space (factor of 2)
        shift_y, shift_x = shift[0] * 2.0, shift[1] * 2.0
        print(
            f"    [{i + 1}/{num_frames}] {fits_paths[i].name} -> Shift: (dy={shift_y:+.2f}px, dx={shift_x:+.2f}px)"
        )

        # Apply exact Fourier shift to full-res 3-channel array
        aligned_stack[i] = apply_fourier_shift_rgb(sub_img, shift_y, shift_x)

    print(
        f"  * Performing MAD Winsorized sigma clipping (low={sigma_low}, high={sigma_high})..."
    )
    master_stacked = winsorized_sigma_clip_stack(
        aligned_stack,
        sigma_low=sigma_low,
        sigma_high=sigma_high,
    )
    return master_stacked


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

    # Standard algebraic circle fit
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
            # 3-point sub-pixel parabolic vertex estimation
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
        f"\n--- [Stage 3] Inter-Master Sub-Pixel Limb Alignment (Ref: {master_names[ref_idx]}) ---"
    )
    h, w, _ = masters[ref_idx].shape
    ref_lum = np.mean(masters[ref_idx], axis=2)

    center_est = (w / 2.0, h / 2.0)
    r_est = min(h, w) * 0.22

    ref_edges = extract_limb_edges_parabolic(ref_lum, center_est, r_est)
    if len(ref_edges) < 20:
        print(
            "  [Warning] Insufficient limb edge contrast in reference. Using image center as origin."
        )
        ref_cx, ref_cy = center_est
    else:
        ref_cx, ref_cy, _ = fit_circle_taubin(ref_edges)
        print(f"  * Reference Lunar Centroid: (cx={ref_cx:.2f}, cy={ref_cy:.2f})")

    aligned_masters = []
    for img, name in zip(masters, master_names):
        lum = np.mean(img, axis=2)
        edges = extract_limb_edges_parabolic(lum, (ref_cx, ref_cy), r_est)

        if len(edges) < 20:
            print(
                f"  * {name}: Limb too faint/saturated -> Phase correlation fallback."
            )
            shift, _, _ = phase_cross_correlation(
                ref_lum[::2, ::2], lum[::2, ::2], upsample_factor=100
            )
            dy, dx = shift[0] * 2.0, shift[1] * 2.0
        else:
            cx, cy, _ = fit_circle_taubin(edges)
            dx = ref_cx - cx
            dy = ref_cy - cy

        print(f"  * {name} -> Offset: (dy={dy:+.2f}px, dx={dx:+.2f}px)")
        shifted = apply_fourier_shift_rgb(img, dy, dx)
        aligned_masters.append(shifted)

    return aligned_masters
