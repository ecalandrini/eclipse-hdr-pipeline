"""Sub-pixel registration and alignment algorithms for solar eclipse imagery.

Implements:
1. Algorithm 1: Intra-bucket sub-pixel DFT phase cross-correlation + MAD sigma-clipped stacking.
2. Algorithm 2: Inter-master parabolic radial limb peak extraction + Taubin algebraic circle fitting.
"""

from pathlib import Path
from astropy.io import fits
import cv2
import numpy as np
from scipy.ndimage import fourier_shift
from skimage.registration import phase_cross_correlation

# ==============================================================================
# I/O UTILITIES
# ==============================================================================


def load_fits_to_float32(fits_path: Path) -> np.ndarray:
    """Loads a 32-bit FITS file and formats it to normalized (H, W, C) float32 [0.0, 1.0]."""
    with fits.open(fits_path) as hdul:
        data = hdul[0].data.astype(np.float32)

    # Reorder channels: FITS (C, H, W) -> OpenCV/NumPy (H, W, C)
    if data.ndim == 3 and data.shape[0] in (3, 4):
        data = np.transpose(data, (1, 2, 0))
        if data.shape[2] == 4:
            data = data[:, :, :3]  # Strip alpha channel if present

    max_val = np.nanmax(data)
    if max_val > 1.0:
        data /= max_val

    return np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)


# ==============================================================================
# ALGORITHM 1: INTRA-BUCKET SUB-PIXEL DFT PHASE CROSS-CORRELATION & STACKING
# ==============================================================================


def align_and_stack_bucket_dft(
    fits_paths: list[Path],
    sigma_low: float = 3.0,
    sigma_high: float = 3.0,
    upsample_factor: int = 100,
) -> np.ndarray:
    """Aligns subframes of the identical exposure time using single-step DFT

    phase cross-correlation and performs Winsorized MAD sigma-clipped averaging.

    Args:
        fits_paths: List of 32-bit demosaiced FITS files for a single exposure tier.
        sigma_low: Low outlier clipping factor.
        sigma_high: High outlier clipping factor.
        upsample_factor: Sub-pixel resolution multiplier (100 = 1/100th pixel).

    Returns:
        32-bit floating-point master image tensor normalized to [0.0, 1.0].
    """
    if len(fits_paths) == 1:
        return load_fits_to_float32(fits_paths[0])

    loaded_frames = [load_fits_to_float32(p) for p in fits_paths]
    ref_frame = loaded_frames[0]
    ref_gray = (
        cv2.cvtColor(ref_frame, cv2.COLOR_RGB2GRAY)
        if ref_frame.ndim == 3
        else ref_frame
    )

    aligned_stack = [ref_frame]

    for idx in range(1, len(loaded_frames)):
        cur_frame = loaded_frames[idx]
        cur_gray = (
            cv2.cvtColor(cur_frame, cv2.COLOR_RGB2GRAY)
            if cur_frame.ndim == 3
            else cur_frame
        )

        # Compute sub-pixel shift (dy, dx) via Matrix Multiplication DFT
        shifts, error, _ = phase_cross_correlation(
            ref_gray, cur_gray, upsample_factor=upsample_factor
        )
        dy, dx = shifts

        # Shift in Fourier domain via linear phase ramp (zero spatial interpolation blur)
        if cur_frame.ndim == 3:
            shifted_chans = []
            for c in range(cur_frame.shape[2]):
                fft_c = np.fft.fftn(cur_frame[:, :, c])
                shifted_fft_c = fourier_shift(fft_c, shift=(dy, dx))
                shifted_chans.append(np.fft.ifftn(shifted_fft_c).real)
            aligned_frame = np.stack(shifted_chans, axis=2)
        else:
            fft_img = np.fft.fftn(cur_frame)
            shifted_fft = fourier_shift(fft_img, shift=(dy, dx))
            aligned_frame = np.fft.ifftn(shifted_fft).real

        aligned_stack.append(aligned_frame.astype(np.float32))

    stack_tensor = np.stack(aligned_stack, axis=0)

    # Winsorized Sigma-Clipped Average Integration
    if len(loaded_frames) >= 4:
        median = np.median(stack_tensor, axis=0)
        mad = np.median(np.abs(stack_tensor - median), axis=0)
        std_est = 1.4826 * (mad + 1e-7)

        lower_bound = median - sigma_low * std_est
        upper_bound = median + sigma_high * std_est

        valid_mask = (stack_tensor >= lower_bound) & (stack_tensor <= upper_bound)
        sum_valid = np.sum(np.where(valid_mask, stack_tensor, 0.0), axis=0)
        count_valid = np.maximum(np.sum(valid_mask.astype(np.float32), axis=0), 1.0)
        master = sum_valid / count_valid
    else:
        master = np.mean(stack_tensor, axis=0)

    return np.clip(master, 0.0, 1.0).astype(np.float32)


# ==============================================================================
# ALGORITHM 2: INTER-MASTER SUB-PIXEL TAUBIN LIMB CENTROID ALIGNMENT
# ==============================================================================


def taubin_circle_fit(points: np.ndarray) -> tuple[float, float, float]:
    """Fits a circle to a 2D point cloud (N, 2) using Taubin's algebraic method.

    Minimizes the algebraic distance (r_i^2 - R^2)^2 with singular matrix constraint,
    eliminating partial-arc and small-sample bias.
    """
    x, y = points[:, 0], points[:, 1]
    n = len(points)
    if n < 3:
        raise ValueError("Taubin fit requires >= 3 points.")

    x_mean, y_mean = np.mean(x), np.mean(y)
    u, v = x - x_mean, y - y_mean
    z = u**2 + v**2

    m_zz = np.sum(z**2) / n
    m_xz = np.sum(u * z) / n
    m_yz = np.sum(v * z) / n
    m_xx = np.sum(u**2) / n
    m_yy = np.sum(v**2) / n
    m_xy = np.sum(u * v) / n

    # Characteristic polynomial coefficients: det(M - lambda * N) = 0
    c3 = 8.0
    c2 = -4.0 * (m_xx + m_yy)
    c1 = 4.0 * (m_xx * m_yy - m_xy**2) - 2.0 * m_zz
    c0 = (
        m_zz * (m_xx + m_yy)
        - (m_xz**2 + m_yz**2)
        - 4.0 * (m_xx * m_yy - m_xy**2) * (m_xx + m_yy)
    )

    # Newton-Raphson solver for the root
    eta = 0.0
    for _ in range(25):
        f = c0 + eta * (c1 + eta * (c2 + eta * c3))
        f_prime = c1 + eta * (2.0 * c2 + eta * 3.0 * c3)
        if abs(f_prime) < 1e-12:
            break
        eta_next = eta - f / f_prime
        if abs(eta_next - eta) < 1e-10:
            eta = eta_next
            break
        eta = eta_next

    det = (m_xx - eta) * (m_yy - eta) - m_xy**2
    if abs(det) < 1e-12:
        raise RuntimeError("Degenerate matrix in Taubin fit.")

    u_c = (m_xz * (m_yy - eta) - m_yz * m_xy) / (2.0 * det)
    v_c = (m_yz * (m_xx - eta) - m_xz * m_xy) / (2.0 * det)

    xc = float(u_c + x_mean)
    yc = float(v_c + y_mean)
    radius = float(np.sqrt(u_c**2 + v_c**2 + (m_xx + m_yy)))
    return xc, yc, radius


def extract_subpixel_limb_points(
    image: np.ndarray,
    approx_center: tuple[float, float],
    approx_radius: float,
    num_rays: int = 720,
    search_range_px: float = 35.0,
) -> np.ndarray:
    """Samples radial profiles and interpolates peak gradient to sub-pixel precision."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image.copy()
    gray = gray.astype(np.float32)
    max_val = gray.max()
    if max_val > 0:
        gray /= max_val

    # Sobel spatial gradient magnitude
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.hypot(gx, gy)

    cx, cy = approx_center
    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    r_steps = np.linspace(
        -search_range_px, search_range_px, int(search_range_px * 4) + 1
    )

    subpixel_points = []
    for theta in angles:
        r_coords = approx_radius + r_steps
        x_samples = cx + r_coords * np.cos(theta)
        y_samples = cy + r_coords * np.sin(theta)

        # Bilinear sampling along radial ray
        profile = cv2.remap(
            grad_mag,
            x_samples.astype(np.float32).reshape(1, -1),
            y_samples.astype(np.float32).reshape(1, -1),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        ).flatten()

        peak_idx = np.argmax(profile)
        if 0 < peak_idx < len(profile) - 1:
            y0, y1, y2 = (
                profile[peak_idx - 1],
                profile[peak_idx],
                profile[peak_idx + 1],
            )
            denom = 2.0 * (y0 - 2.0 * y1 + y2)
            if denom < -1e-6:  # Strict local maximum
                delta = (y0 - y2) / denom  # Parabolic sub-pixel peak offset
                sub_r = r_coords[peak_idx] + delta * (r_steps[1] - r_steps[0])
                if profile[peak_idx] > 0.05 * np.max(grad_mag):
                    subpixel_points.append(
                        [cx + sub_r * np.cos(theta), cy + sub_r * np.sin(theta)]
                    )

    return np.array(subpixel_points, dtype=np.float64)


def align_masters_taubin(
    masters: list[np.ndarray], master_names: list[str], ref_idx: int = 4
) -> list[np.ndarray]:
    """Fits sub-pixel limb centroids on all disparate exposure masters

    and registers them to the reference master's solar center using Fourier phase shifts.
    """
    fitted_geometries = []

    print(
        f"\n--- [Algorithm 2] Fitting Sub-Pixel Limb Centroids on {len(masters)} Masters ---"
    )
    for idx, (img, name) in enumerate(zip(masters, master_names)):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
        _, thresh = cv2.threshold(
            (gray / gray.max() * 255).astype(np.uint8),
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        largest_c = max(contours, key=cv2.contourArea)
        (init_x, init_y), init_r = cv2.minEnclosingCircle(largest_c)

        pts = extract_subpixel_limb_points(img, (init_x, init_y), init_r)
        xc, yc, r = taubin_circle_fit(pts)
        fitted_geometries.append((xc, yc, r))
        print(f"  * [{name}] -> Centroid: ({xc:.3f}, {yc:.3f}) px | Radius: {r:.3f} px")

    ref_xc, ref_yc, _ = fitted_geometries[ref_idx]
    aligned_masters = []

    print(
        f"\n--- Aligning all masters to Reference Master [{master_names[ref_idx]}] ---"
    )
    for idx, (img, name) in enumerate(zip(masters, master_names)):
        cur_xc, cur_yc, _ = fitted_geometries[idx]
        dx = ref_xc - cur_xc
        dy = ref_yc - cur_yc
        print(f"  * [{name}] Applying Fourier Shift: dx={dx:+.3f}, dy={dy:+.3f}")

        if img.ndim == 3:
            shifted_chans = []
            for c in range(img.shape[2]):
                fft_c = np.fft.fftn(img[:, :, c])
                shifted_fft = fourier_shift(fft_c, shift=(dy, dx))
                shifted_chans.append(np.fft.ifftn(shifted_fft).real)
            aligned = np.stack(shifted_chans, axis=2)
        else:
            fft_img = np.fft.fftn(img)
            shifted_fft = fourier_shift(fft_img, shift=(dy, dx))
            aligned = np.fft.ifftn(shifted_fft).real

        aligned_masters.append(np.clip(aligned, 0.0, 1.0).astype(np.float32))

    return aligned_masters
