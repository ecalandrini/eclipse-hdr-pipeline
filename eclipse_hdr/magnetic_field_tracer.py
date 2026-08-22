"""Solar Coronal Magnetic Field Line Extraction and Streamline Vector Tracing."""

from pathlib import Path
from astropy.io import fits
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tifffile


def load_normalized_lum(file_path: Path) -> np.ndarray:
    """Loads master image and returns float32 single-channel luminance [0.0, 1.0]."""
    ext = file_path.suffix.lower()
    if ext in (".fit", ".fits"):
        with fits.open(file_path, memmap=False) as h:
            d = h[0].data.astype(np.float32)
        img = np.transpose(d, (1, 2, 0)) if d.ndim == 3 and d.shape[0] == 3 else d
        if img.ndim == 3:
            lum = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
        else:
            lum = img
    else:
        raw = tifffile.imread(str(file_path)).astype(np.float32)
        if raw.ndim == 3 and raw.shape[2] >= 3:
            lum = 0.299 * raw[..., 0] + 0.587 * raw[..., 1] + 0.114 * raw[..., 2]
        else:
            lum = raw
        lum = (
            lum / 65535.0
            if raw.dtype == np.uint16 or raw.max() > 255.0
            else lum / 255.0
        )

    p99 = float(np.percentile(lum[lum > 0], 99.95)) if np.any(lum > 0) else 1.0
    return np.clip(lum / max(p99, 1e-5), 0.0, 1.0)


def compute_magnetic_vector_field(
    lum: np.ndarray,
    center: tuple[float, float],
    r_lunar: float,
    smoothing_sigma: float = 3.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates normalized orthogonal vector field (B || Streamers, B perp Grad(I))."""
    h, w = lum.shape
    cx, cy = center

    # 1. Non-linear stretch to boost outer field gradients
    stretched = np.arcsinh(lum * 25.0) / np.arcsinh(25.0)

    # 2. Gaussian smoothing to calculate continuous magnetic flux derivatives
    smoothed = cv2.GaussianBlur(stretched, (0, 0), sigmaX=smoothing_sigma)

    # 3. Compute Cartesian image gradients
    gx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
    gmag = cv2.magnitude(gx, gy)

    # 4. Orthogonal vector field: Plasma filaments run perpendicular to brightness gradients
    # Vector B = (-gy, gx) oriented radially outward
    bx = -gy
    by = gx

    # Ensure vectors point outward away from solar center
    y_idx, x_idx = np.ogrid[:h, :w]
    rx = x_idx - cx
    ry = y_idx - cy
    dot_radial = (bx * rx) + (by * ry)

    # Flip vectors pointing inward
    flip_mask = dot_radial < 0
    bx[flip_mask] = -bx[flip_mask]
    by[flip_mask] = -by[flip_mask]

    # Normalize vectors
    v_norm = np.sqrt(bx**2 + by**2) + 1e-5
    bx_unit = bx / v_norm
    by_unit = by / v_norm

    return bx_unit, by_unit, gmag


def trace_magnetic_streamlines(
    input_file: Path,
    output_dir: Path,
    center: tuple[float, float] = (2492.4, 1612.9),
    r_lunar: float = 222.7,
    density: float = 1.6,
    max_radius_px: float = 1200.0,
) -> None:
    """Renders magnetic streamline integration overlaid on coronal master."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 65)
    print("       SOLAR CORONAL MAGNETIC FIELD LINE VECTOR TRACER           ")
    print("=" * 65)
    print(f"  * Master Image           : {input_file.name}")
    print(f"  * Solar Center           : ({center[0]:.1f}, {center[1]:.1f})")
    print(f"  * Lunar Radius           : {r_lunar:.1f} px")
    print(f"  * Streamline Density     : {density:.2f}")
    print("-" * 65)

    lum = load_normalized_lum(input_file)
    h, w = lum.shape
    cx, cy = center

    # Compute unit vector field
    bx, by, gmag = compute_magnetic_vector_field(
        lum, center, r_lunar, smoothing_sigma=3.0
    )

    # Downsample grid for Matplotlib streamline integration
    step = 4
    x_coords = np.arange(0, w, step)
    y_coords = np.arange(0, h, step)
    X, Y = np.meshgrid(x_coords, y_coords)

    U = bx[::step, ::step]
    V = by[::step, ::step]
    speed = gmag[::step, ::step]

    # Distance mask from center for streamline seeds
    dist_from_sun = np.hypot(X - cx, Y - cy)
    valid_region = (dist_from_sun >= r_lunar * 0.98) & (dist_from_sun <= max_radius_px)

    U_masked = np.where(valid_region, U, np.nan)
    V_masked = np.where(valid_region, V, np.nan)

    # Matplotlib High-Resolution Streamline Plot
    fig, ax = plt.subplots(figsize=(14, 10), facecolor="black")

    # Background: Stretched monochrome eclipse master
    bg_display = np.arcsinh(lum * 20.0) / np.arcsinh(20.0)
    ax.imshow(bg_display, cmap="gray", origin="upper")

    # Streamline layer colored by local magnetic gradient intensity
    stream = ax.streamplot(
        X,
        Y,
        U_masked,
        V_masked,
        color=speed,
        cmap="plasma",
        density=density,
        linewidth=1.0,
        arrowsize=0.8,
        integration_direction="forward",
    )

    # Black disk over lunar interior
    moon_patch = plt.Circle((cx, cy), r_lunar, color="black", zorder=10)
    ax.add_patch(moon_patch)
    limb_ring = plt.Circle(
        (cx, cy), r_lunar, color="#ff4444", fill=False, linewidth=1.2, zorder=11
    )
    ax.add_patch(limb_ring)

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    out_png = output_dir / f"{input_file.stem}_Magnetic_Field_Trace.png"
    fig.savefig(out_png, dpi=200, facecolor="black")
    plt.close(fig)

    print(f"  [Exported Field Map] -> {out_png.resolve()}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    import sys

    target = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("Eclipse_HDR_Master_Artistic_HDR.tif")
    )
    out = Path("workspace/magnetic_field")
    trace_magnetic_streamlines(target, out)
