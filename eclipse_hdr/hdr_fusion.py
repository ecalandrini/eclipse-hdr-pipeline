"""Multi-scale Mertens HDR exposure fusion and 16-bit TIFF export."""

from pathlib import Path
import cv2
import numpy as np
import tifffile


def fuse_and_export_hdr(
    aligned_masters: list[np.ndarray],
    output_path: Path,
    contrast_w: float = 1.0,
    sat_w: float = 1.0,
    exp_w: float = 0.2,
) -> None:
    """Performs multi-scale Laplacian pyramid Mertens exposure fusion

    across all aligned exposure tiers and exports a 16-bit master TIFF.

    Args:
        aligned_masters: List of (H, W, 3) float32 master images in [0.0, 1.0].
        output_path: Destination path for the master TIFF file.
        contrast_w: Weight for the Laplacian local contrast metric.
        sat_w: Weight for the color saturation metric.
        exp_w: Weight for the Gaussian well-exposedness metric.
    """
    print(
        f"\n--- [Step 5] Executing Mertens Multi-Scale HDR Fusion ({len(aligned_masters)} Masters) ---"
    )
    print(
        f"  * Blending Weights -> Contrast: {contrast_w:.2f}, Saturation: {sat_w:.2f}, Exposure: {exp_w:.2f}"
    )

    # OpenCV createMergeMertens expects 8-bit uint8 or 32-bit float32 inputs in [0.0, 1.0]
    merge_mertens = cv2.createMergeMertens(
        contrast_weight=contrast_w,
        saturation_weight=sat_w,
        exposure_weight=exp_w,
    )

    # Convert RGB to BGR for OpenCV C++ processing engine
    bgr_masters = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in aligned_masters]

    # Execute Laplacian pyramid fusion in 32-bit float
    fusion_bgr = merge_mertens.process(bgr_masters)

    # Convert BGR back to RGB
    fusion_rgb = cv2.cvtColor(fusion_bgr, cv2.COLOR_BGR2RGB)

    # Normalize and scale to 16-bit integer [0, 65535] without clipping artifacts
    fusion_clipped = np.clip(fusion_rgb, 0.0, 1.0)
    fusion_uint16 = (fusion_clipped * 65535.0).astype(np.uint16)

    # Write uncompressed 16-bit TIFF with full dynamic range preserved
    out_file = output_path.resolve()
    tifffile.imwrite(
        out_file,
        fusion_uint16,
        photometric="rgb",
        compression=None,
    )

    print(f"\n[Success] Master HDR TIFF exported to: {out_file}")
