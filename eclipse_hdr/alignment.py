"""Solar-frame robust multi-bracket alignment with bandpass filtering."""

import gc
from pathlib import Path
from astropy.io import fits
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.sparse.csgraph import minimum_spanning_tree
from skimage.registration import phase_cross_correlation


def load_fits_to_float32(fits_path: Path) -> np.ndarray:
    """Loads a FITS file and returns an (H, W, 3) float32 RGB array in [0.0, 1.0]."""
    with fits.open(fits_path, memmap=False) as hdul:
        data = hdul[0].data.astype(np.float32)

    if data.ndim == 3:
        if data.shape[0] == 3:
            img = np.transpose(data, (1, 2, 0))
        else:
            img = data
    elif data.ndim == 2:
        img = np.repeat(data[:, :, np.newaxis], 3, axis=2)
    else:
        raise ValueError(f"Unsupported FITS dimension: {data.ndim} in {fits_path}")

    max_val = float(np.max(img))
    if max_val > 255.0:
        img /= 65535.0
    elif max_val > 1.0:
        img /= 255.0

    return np.clip(img, 0.0, 1.0)


def save_preview_jpg(
    img_float32: np.ndarray, output_jpg_path: Path, mtf_mid: float = 0.06
) -> None:
    """Exports an 8-bit JPEG preview using an Astronomical Midtone Transfer Function."""
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

    img_sub = np.maximum(0.0, img_float32 - bg)
    img_max = float(np.percentile(img_sub, 99.99)) or 1.0
    img_norm = np.clip(img_sub / img_max, 0.0, 1.0)

    m = mtf_mid
    stretched = np.where(
        img_norm <= 0.0,
        0.0,
        ((m - 1.0) * img_norm) / (((2.0 * m - 1.0) * img_norm) - m),
    )
    img_8bit = (np.clip(stretched, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(img_8bit, mode="RGB").save(output_jpg_path, quality=94)


def apply_spatial_shift_rgb(
    img_rgb: np.ndarray, shift_y: float, shift_x: float
) -> np.ndarray:
    """Applies sub-pixel 2D translation in spatial domain with constant zero padding."""
    if abs(shift_y) < 0.01 and abs(shift_x) < 0.01:
        return img_rgb

    shifted_rgb = np.zeros_like(img_rgb)
    for c in range(3):
        shifted_rgb[:, :, c] = ndimage.shift(
            img_rgb[:, :, c],
            shift=(shift_y, shift_x),
            order=3,
            mode="constant",
            cval=0.0,
        )
    return np.clip(shifted_rgb, 0.0, 1.0)


def extract_solar_features(img_rgb: np.ndarray) -> np.ndarray:
    """Extracts scale-invariant high-pass solar/chromosphere features via DoG bandpass."""
    lum = 0.299 * img_rgb[:, :, 0] + 0.587 * img_rgb[:, :, 1] + 0.114 * img_rgb[:, :, 2]

    # Difference of Gaussians (DoG) isolates fixed prominences & fine coronal filaments
    g1 = cv2.GaussianBlur(lum, (0, 0), sigmaX=1.5)
    g2 = cv2.GaussianBlur(lum, (0, 0), sigmaX=7.0)
    dog = np.maximum(0.0, g1 - g2)

    p99 = float(np.percentile(dog, 99.7)) or 1.0
    norm_features = np.clip(dog / p99, 0.0, 1.0)

    # 2D Hanning window to prevent Fourier border leakage
    win_y = np.hanning(norm_features.shape[0])
    win_x = np.hanning(norm_features.shape[1])
    return norm_features * np.outer(win_y, win_x).astype(np.float32)


def align_masters_graph(
    masters: list[np.ndarray],
    master_names: list[str],
    anchor_idx: int = 0,
    max_shift: float = 60.0,
    upsample_factor: int = 100,
) -> list[np.ndarray]:
    """Graph MST feature alignment anchored to the Solar Coordinate Frame."""
    n = len(masters)
    if n <= 1:
        return masters

    print(
        f"\n--- [Stage 3] Solar-Frame Graph Registration ({n} Masters) [Anchor: {master_names[anchor_idx]}] ---",
        flush=True,
    )

    features = [extract_solar_features(m)[::2, ::2] for m in masters]
    cost_matrix = np.full((n, n), np.inf)
    shift_matrix = np.zeros((n, n, 2), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            shift, error, _ = phase_cross_correlation(
                features[i],
                features[j],
                upsample_factor=upsample_factor,
            )
            dy, dx = float(shift[0] * 2.0), float(shift[1] * 2.0)
            dist = float(np.hypot(dy, dx))

            if dist <= max_shift:
                cost_matrix[i, j] = error
                cost_matrix[j, i] = error
                shift_matrix[i, j] = [dy, dx]
                shift_matrix[j, i] = [-dy, -dx]

    adj_matrix = np.where(np.isinf(cost_matrix), 1e6, cost_matrix)
    mst = minimum_spanning_tree(adj_matrix).toarray()

    graph: dict[int, list[tuple[int, tuple[float, float]]]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if 0 < mst[i, j] < 1e5:
                graph[i].append((j, tuple(shift_matrix[i, j])))
                graph[j].append((i, tuple(shift_matrix[j, i])))

    final_shifts = np.zeros((n, 2), dtype=np.float64)
    visited = {anchor_idx}
    queue = [anchor_idx]

    while queue:
        curr = queue.pop(0)
        for neighbor, (dy, dx) in graph[curr]:
            if neighbor not in visited:
                visited.add(neighbor)
                final_shifts[neighbor] = final_shifts[curr] + np.array([dy, dx])
                queue.append(neighbor)

    aligned_masters = []
    for i, (img, name) in enumerate(zip(masters, master_names)):
        dy, dx = float(final_shifts[i, 0]), float(final_shifts[i, 1])
        print(
            f"  * {name} -> Offset vs Solar Anchor: (dy={dy:+.2f}px, dx={dx:+.2f}px)",
            flush=True,
        )
        shifted = apply_spatial_shift_rgb(img, dy, dx)
        aligned_masters.append(shifted)

    return aligned_masters


def align_and_stack_bucket_dft(
    fits_paths: list[Path],
    sigma_low: float = 3.0,
    sigma_high: float = 3.0,
    upsample_factor: int = 100,
    max_allowed_shift: float = 50.0,
) -> np.ndarray:
    """Intra-bucket sub-pixel alignment & sigma-clipped stacking."""
    num_frames = len(fits_paths)
    if num_frames == 0:
        raise ValueError("No FITS paths provided.")
    if num_frames == 1:
        return load_fits_to_float32(fits_paths[0])

    ref_img = load_fits_to_float32(fits_paths[0])
    h, w, _ = ref_img.shape

    ref_lum = (
        0.299 * ref_img[::2, ::2, 0]
        + 0.587 * ref_img[::2, ::2, 1]
        + 0.114 * ref_img[::2, ::2, 2]
    )
    win = np.outer(np.hanning(ref_lum.shape[0]), np.hanning(ref_lum.shape[1])).astype(
        np.float32
    )
    ref_lum_win = ref_lum * win

    valid_frames = [ref_img]

    for i in range(1, num_frames):
        sub_img = load_fits_to_float32(fits_paths[i])
        sub_lum = (
            0.299 * sub_img[::2, ::2, 0]
            + 0.587 * sub_img[::2, ::2, 1]
            + 0.114 * sub_img[::2, ::2, 2]
        ) * win

        shift, _, _ = phase_cross_correlation(
            ref_lum_win, sub_lum, upsample_factor=upsample_factor
        )
        shift_y, shift_x = float(shift[0] * 2.0), float(shift[1] * 2.0)
        dist = float(np.hypot(shift_x, shift_y))

        if dist > max_allowed_shift:
            del sub_img
            continue

        shifted_sub = apply_spatial_shift_rgb(sub_img, shift_y, shift_x)
        valid_frames.append(shifted_sub)
        del sub_img

    del ref_lum_win, win
    gc.collect()

    if len(valid_frames) == 1:
        return valid_frames[0]

    stack_array = np.stack(valid_frames, axis=0)
    del valid_frames

    mean = np.mean(stack_array, axis=0, dtype=np.float32)
    std = np.std(stack_array, axis=0, dtype=np.float32)
    std = np.where(std == 0, 1e-6, std)

    np.clip(
        stack_array,
        mean - (sigma_low * std),
        mean + (sigma_high * std),
        out=stack_array,
    )
    master_stacked = np.mean(stack_array, axis=0, dtype=np.float32)
    del stack_array
    gc.collect()

    return np.clip(master_stacked, 0.0, 1.0)


def fit_circle_taubin(points: np.ndarray) -> tuple[float, float, float]:
    """Algebraic Taubin circle fitting."""
    x = points[:, 0]
    y = points[:, 1]
    n = points.shape[0]

    mean_x, mean_y = np.mean(x), np.mean(y)
    u, v = x - mean_x, y - mean_y
    z = u**2 + v**2
    mz = np.mean(z)

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
) -> tuple[np.ndarray, float]:
    """Extracts sub-pixel limb edges and calculates total angular coverage (in degrees)."""
    cx, cy = center_est
    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    edge_points = []
    detected_angles = []

    h, w = image.shape
    radial_dist = np.linspace(r_est - 45, r_est + 45, 180)

    for theta in angles:
        xs = cx + radial_dist * np.cos(theta)
        ys = cy + radial_dist * np.sin(theta)

        valid = (xs >= 1) & (xs < w - 2) & (ys >= 1) & (ys < h - 2)
        if not np.all(valid):
            continue

        profile = ndimage.map_coordinates(image, [ys, xs], order=1)
        gradient = -np.gradient(profile)

        max_idx = np.argmax(gradient)
        if 1 < max_idx < len(gradient) - 2 and gradient[max_idx] > 0.005:
            y0, y1, y2 = gradient[max_idx - 1], gradient[max_idx], gradient[max_idx + 1]
            denom = 2 * (2 * y1 - y0 - y2)
            delta = (y0 - y2) / denom if denom != 0 else 0.0
            sub_r = radial_dist[max_idx] + delta * (radial_dist[1] - radial_dist[0])

            edge_points.append([cx + sub_r * np.cos(theta), cy + sub_r * np.sin(theta)])
            detected_angles.append(theta)

    if len(edge_points) < 20:
        return np.empty((0, 2)), 0.0

    # Calculate angular span
    detected_angles = np.array(detected_angles)
    angular_span_deg = np.rad2deg(np.ptp(detected_angles))
    return np.array(edge_points), angular_span_deg


def align_masters_taubin(
    masters: list[np.ndarray],
    master_names: list[str],
    ref_idx: int = 0,
    max_inter_master_shift: float = 60.0,
) -> list[np.ndarray]:
    """Inter-master alignment with angular span validation to prevent small-arc circle divergence."""
    h, w, _ = masters[ref_idx].shape
    ref_lum = (
        0.299 * masters[ref_idx][:, :, 0]
        + 0.587 * masters[ref_idx][:, :, 1]
        + 0.114 * masters[ref_idx][:, :, 2]
    )

    center_est = (w / 2.0, h / 2.0)
    r_est = min(h, w) * 0.22

    # Detect reference centroid
    ref_edges, ref_span = extract_limb_edges_parabolic(ref_lum, center_est, r_est)
    if len(ref_edges) >= 40 and ref_span >= 180.0:
        ref_cx, ref_cy, _ = fit_circle_taubin(ref_edges)
        print(
            f"  * Reference Centroid (Taubin limb, span={ref_span:.1f}°): ({ref_cx:.2f}, {ref_cy:.2f})",
            flush=True,
        )
    else:
        ref_cx, ref_cy = center_est
        print(
            f"  * Reference Centroid (Image Center): ({ref_cx:.2f}, {ref_cy:.2f})",
            flush=True,
        )

    # Prepare windowed annular crop for phase correlation fallback
    y_idx, x_idx = np.ogrid[:h, :w]
    dist_from_center = np.hypot(x_idx - ref_cx, y_idx - ref_cy)
    annulus_mask = (
        (dist_from_center >= r_est - 80) & (dist_from_center <= r_est + 80)
    ).astype(np.float32)
    ref_masked = (ref_lum * annulus_mask)[::2, ::2]

    aligned_masters = []

    for img, name in zip(masters, master_names):
        lum = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        edges, span = extract_limb_edges_parabolic(lum, (ref_cx, ref_cy), r_est)

        # Only use Taubin fit if limb is visible across at least 150 degrees
        if len(edges) >= 40 and span >= 150.0:
            cx, cy, _ = fit_circle_taubin(edges)
            dx = float(ref_cx - cx)
            dy = float(ref_cy - cy)
            method = f"Taubin (span={span:.0f}°)"
        else:
            # Short exposure fallback: Phase correlation restricted to the limb annulus
            target_masked = (lum * annulus_mask)[::2, ::2]
            shift, _, _ = phase_cross_correlation(
                ref_masked, target_masked, upsample_factor=100
            )
            dy = float(shift[0] * 2.0)
            dx = float(shift[1] * 2.0)
            method = f"Annular PhaseCorr (span={span:.0f}° < 150°)"

        dist = float(np.hypot(dx, dy))
        if dist > max_inter_master_shift:
            print(
                f"  * {name} -> [CLAMPED] Shift {dist:.2f}px via {method} exceeded {max_inter_master_shift:.1f}px. Set to (0,0).",
                flush=True,
            )
            dy, dx = 0.0, 0.0
        else:
            print(
                f"  * {name} -> Shift: (dy={dy:+.2f}px, dx={dx:+.2f}px, dist={dist:.2f}px) [{method}]",
                flush=True,
            )

        shifted = apply_spatial_shift_rgb(img, dy, dx)
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
