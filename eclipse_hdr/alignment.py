"""Solar-frame robust multi-bracket alignment with DoG bandpass graph solver and hot-pixel rejection."""

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


def load_fits_luminance_proxy(fits_path: Path, downsample_factor: int = 4) -> np.ndarray:
    """Loads a lightweight binned 2D grayscale proxy of a FITS file (<2 MB RAM)."""
    with fits.open(fits_path, memmap=True) as hdul:
        data = hdul[0].data

    if data.ndim == 3:
        if data.shape[0] == 3:
            lum = data[1, ::downsample_factor, ::downsample_factor].astype(np.float32)
        else:
            lum = data[::downsample_factor, ::downsample_factor, 1].astype(np.float32)
    elif data.ndim == 2:
        lum = data[::downsample_factor, ::downsample_factor].astype(np.float32)
    else:
        raise ValueError(f"Unsupported FITS dimension: {data.ndim}")

    p_max = float(np.percentile(lum, 99.9)) or 1.0
    return np.clip(lum / p_max, 0.0, 1.0)


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


def extract_proxy_features(lum_proxy: np.ndarray) -> np.ndarray:
    """Computes DoG bandpass features on the proxy image, suppressing hot pixel spikes."""
    # Light median pre-pass to remove isolated single-pixel hot defects
    med_plane = cv2.medianBlur((lum_proxy * 65535.0).astype(np.uint16), 3) / 65535.0
    spike = lum_proxy - med_plane
    cleaned = np.where(spike > 0.05, med_plane, lum_proxy)

    g1 = cv2.GaussianBlur(cleaned, (0, 0), sigmaX=1.0)
    g2 = cv2.GaussianBlur(cleaned, (0, 0), sigmaX=3.5)
    dog = np.maximum(0.0, g1 - g2)

    p99 = float(np.percentile(dog, 99.8)) or 1.0
    norm_features = np.clip(dog / p99, 0.0, 1.0)

    win_y = np.hanning(norm_features.shape[0])
    win_x = np.hanning(norm_features.shape[1])
    return (norm_features * np.outer(win_y, win_x)).astype(np.float32)


def solve_alignment_shifts_graph(
    master_files: list[Path],
    downsample: int = 4,
    upsample_factor: int = 100,
    max_shift_fullres: float = 80.0,
) -> tuple[int, list[tuple[float, float]]]:
    """Solves the Minimum Spanning Tree alignment offsets using low-memory proxies (<50MB RAM)."""
    n = len(master_files)
    print(f"\n--- [Stage 3] Low-Memory Proxy Registration ({n} Masters) ---", flush=True)

    # 1. Load lightweight proxies
    proxies = [load_fits_luminance_proxy(p, downsample_factor=downsample) for p in master_files]
    features = [extract_proxy_features(p) for p in proxies]

    # Select anchor frame with highest mean coronal/prominence signal
    anchor_idx = int(np.argmax([float(np.mean(p)) for p in proxies]))
    del proxies
    gc.collect()

    print(f"  * Selected Solar Anchor: {master_files[anchor_idx].stem}", flush=True)

    # 2. Pairwise cross-correlation
    cost_matrix = np.full((n, n), np.inf)
    shift_matrix = np.zeros((n, n, 2), dtype=np.float64)
    max_proxy_shift = max_shift_fullres / float(downsample)

    for i in range(n):
        for j in range(i + 1, n):
            shift, error, _ = phase_cross_correlation(
                features[i],
                features[j],
                upsample_factor=upsample_factor,
            )
            dy_proxy, dx_proxy = float(shift[0]), float(shift[1])
            dist_proxy = float(np.hypot(dy_proxy, dx_proxy))

            if dist_proxy <= max_proxy_shift:
                dy_full = dy_proxy * float(downsample)
                dx_full = dx_proxy * float(downsample)

                cost_matrix[i, j] = error
                cost_matrix[j, i] = error
                shift_matrix[i, j] = [dy_full, dx_full]
                shift_matrix[j, i] = [-dy_full, -dx_full]

    del features
    gc.collect()

    # 3. Minimum Spanning Tree
    adj_matrix = np.where(np.isinf(cost_matrix), 1e6, cost_matrix)
    mst = minimum_spanning_tree(adj_matrix).toarray()

    graph: dict[int, list[tuple[int, tuple[float, float]]]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if 0 < mst[i, j] < 1e5:
                graph[i].append((j, tuple(shift_matrix[i, j])))
                graph[j].append((i, tuple(shift_matrix[j, i])))

    # 4. BFS graph traversal to calculate cumulative shifts
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

    shift_tuples = [(float(final_shifts[i, 0]), float(final_shifts[i, 1])) for i in range(n)]
    return anchor_idx, shift_tuples


def winsorized_sigma_clip_stack(
    stack_array: np.ndarray,
    sigma_low: float = 3.0,
    sigma_high: float = 2.2,
) -> np.ndarray:
    """Vectorized sigma-clipping stack with single-pixel defect suppression."""
    n_frames, h, w, c = stack_array.shape

    if n_frames >= 3:
        median = np.median(stack_array, axis=0)
        mad = np.median(np.abs(stack_array - median), axis=0)
        std_est = 1.4826 * np.where(mad == 0, 1e-5, mad)

        lower_bound = median - (sigma_low * std_est)
        upper_bound = median + (sigma_high * std_est)

        winsorized = np.clip(stack_array, lower_bound, upper_bound)
        master = np.mean(winsorized, axis=0, dtype=np.float32)
    else:
        master = np.mean(stack_array, axis=0, dtype=np.float32)

    # Spatial Hot-Pixel Cleaner to suppress surviving stationary defects
    cleaned_master = np.zeros_like(master)
    for ch in range(c):
        plane = master[:, :, ch]
        med_plane = cv2.medianBlur((plane * 65535.0).astype(np.uint16), 3) / 65535.0
        spike = plane - med_plane
        cleaned_master[:, :, ch] = np.where(spike > 0.05, med_plane, plane)

    return np.clip(cleaned_master, 0.0, 1.0)


def align_and_stack_bucket_dft(
    fits_paths: list[Path],
    sigma_low: float = 3.0,
    sigma_high: float = 2.2,
    upsample_factor: int = 100,
    max_allowed_shift: float = 50.0,
) -> np.ndarray:
    """Intra-bucket sub-pixel alignment & sigma-clipped stacking."""
    num_frames = len(fits_paths)
    if num_frames == 0:
        raise ValueError("No FITS paths provided.")
    if num_frames == 1:
        img = load_fits_to_float32(fits_paths[0])
        # Clean hot pixels even on single-frame buckets
        cleaned_single = np.zeros_like(img)
        for ch in range(3):
            plane = img[:, :, ch]
            med_plane = cv2.medianBlur((plane * 65535.0).astype(np.uint16), 3) / 65535.0
            spike = plane - med_plane
            cleaned_single[:, :, ch] = np.where(spike > 0.05, med_plane, plane)
        return cleaned_single

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