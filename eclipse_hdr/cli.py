"""Command Line Interface and execution orchestrator for the Eclipse HDR Pipeline."""

import argparse
from pathlib import Path
from astropy.io import fits
import numpy as np

from .alignment import (
    align_and_stack_bucket_dft,
    align_masters_taubin,
    load_fits_to_float32,
)
from .exif_parser import sort_rafs_by_exposure
from .hdr_fusion import fuse_and_export_hdr
from .siril_bridge import (
    demosaic_bucket_with_siril,
    register_and_stack_with_siril,
)


def main() -> None:
    """Entry point for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Automated 32-bit Sub-Pixel Alignment & HDR Fusion for Total Solar Eclipse Data"
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=Path,
        required=True,
        help="Path containing input RAW (.RAF) files",
    )
    parser.add_argument(
        "--work-dir",
        "-w",
        type=Path,
        default=Path("./eclipse_workspace"),
        help="Intermediate processing workspace folder",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("./Eclipse_HDR_Master.tif"),
        help="Output path for the 16-bit master HDR TIFF",
    )
    parser.add_argument(
        "--stacking-engine",
        choices=["python", "siril"],
        default="python",
        help="Engine for intra-bucket alignment & stacking: 'python' (in-memory sub-pixel DFT + MAD) or 'siril' (native C engine on disk)",
    )
    parser.add_argument(
        "--sigma-low",
        type=float,
        default=3.0,
        help="Sigma clipping low threshold for intra-bucket stacking",
    )
    parser.add_argument(
        "--sigma-high",
        type=float,
        default=3.0,
        help="Sigma clipping high threshold for intra-bucket stacking",
    )
    parser.add_argument(
        "--contrast-weight",
        type=float,
        default=1.0,
        help="Mertens contrast weight",
    )
    parser.add_argument(
        "--sat-weight",
        type=float,
        default=1.0,
        help="Mertens saturation weight",
    )
    parser.add_argument(
        "--exp-weight",
        type=float,
        default=0.2,
        help="Mertens well-exposedness weight",
    )

    args = parser.parse_args()

    input_path = args.input_dir.resolve()
    work_path = args.work_dir.resolve()
    output_path = args.output.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")

    # Step 1: EXIF parsing and bucket sorting
    bucket_map = sort_rafs_by_exposure(input_path, work_path)

    # Step 2 & 3: Intra-Bucket Stacking
    stacked_masters: list[np.ndarray] = []
    master_names: list[str] = []

    print(
        f"\n--- Running Intra-Bucket Stacking [Engine: {args.stacking_engine.upper()}] ---"
    )
    for exp_str, b_path in sorted(bucket_map.items()):
        master_fit_name = f"Master_{exp_str}"

        if args.stacking_engine == "python":
            print(f"\n[Bucket {exp_str}] Demosaicing with PySiril...")
            conv_fits_paths = demosaic_bucket_with_siril(b_path)

            print(
                f"[Bucket {exp_str}] Python DFT Sub-Pixel Stacking ({len(conv_fits_paths)} frames)..."
            )
            master_img = align_and_stack_bucket_dft(
                fits_paths=conv_fits_paths,
                sigma_low=args.sigma_low,
                sigma_high=args.sigma_high,
                upsample_factor=100,
            )

            # Archival write to disk
            master_fits_path = b_path / f"{master_fit_name}.fit"
            fits.writeto(
                master_fits_path,
                np.transpose(master_img, (2, 0, 1)),
                overwrite=True,
            )
        else:
            print(
                f"\n[Bucket {exp_str}] Siril Native Registration & Stacking on Disk..."
            )
            master_fits_path = register_and_stack_with_siril(
                bucket_dir=b_path,
                output_master_name=master_fit_name,
                sigma_low=args.sigma_low,
                sigma_high=args.sigma_high,
            )
            master_img = load_fits_to_float32(master_fits_path)

        stacked_masters.append(master_img)
        master_names.append(master_fit_name)

    # Step 4: Inter-Master Taubin Sub-Pixel Alignment
    ref_idx = min(len(stacked_masters) - 1, 4)
    aligned_masters = align_masters_taubin(
        masters=stacked_masters,
        master_names=master_names,
        ref_idx=ref_idx,
    )

    # Step 5: 32-Bit Multi-Scale Mertens HDR Fusion
    fuse_and_export_hdr(
        aligned_masters=aligned_masters,
        output_path=output_path,
        contrast_w=args.contrast_weight,
        sat_w=args.sat_weight,
        exp_w=args.exp_weight,
    )


if __name__ == "__main__":
    main()
