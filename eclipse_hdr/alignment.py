"""Solar-frame robust multi-bracket alignment with Difference of Gaussians (DoG) bandpass graph solver."""

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
    """Loads a FITS file and returns an (H, W, 3) float32 RGB array normalized to [0.0, 1.0]."""
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
    g2 = cv2.GaussianBlur(lum, (0, 0), sigmaX=5.0)
    dog = np.maximum(0.0, g1 - g2)

    p99 = float(np.percentile(dog, 99.8)) or 1.0
    norm_features = np.clip(dog / p99, 0.0, 1.0)

    # 2D Hanning window to prevent Fourier border wrap-around leakage
    win_y = np.hanning(norm_features.shape[0])
    win_x = np.hanning(norm_features.shape[1])
    return (norm_features * np.outer(win_y, win_x)).astype(np.float32)


def align_masters_graph(
    masters: list[np.ndarray],
    master_names: list[str],
    anchor_idx: int = 0,
    max_shift: float = 80.0,
    upsample_factor: int = 100,
) -> list[np.ndarray]:
    """Pure 2D translation alignment using feature graph correlation and Minimum Spanning Tree."""
    n = len(masters)
    if n <= 1:
        return masters

    print(
        f"\n--- Solving Feature Correlation Graph ({n} Masters) [Anchor: {master_names[anchor_idx]}] ---",
        flush=True,
    )

    # Compute high-pass feature maps (binned 2x for correlation speed)
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
            f"  * {name:22s} -> Offset vs Solar Anchor: (dy={dy:+.2f}px, dx={dx:+.2f}px, dist={np.hypot(dy, dx):.2f}px)",
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
    """Fast vectorized sigma-clipping stack for pre-allocated arrays in memory."""
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

    stack_array = np.stack(valid_frames, axis=0)
    del valid_frames

    master_stacked = winsorized_sigma_clip_stack(
        stack_array, sigma_low=sigma_low, sigma_high=sigma_high
    )
    del stack_array
    gc.collect()

    return np.clip(master_stacked, 0.0, 1.0)
