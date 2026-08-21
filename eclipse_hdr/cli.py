"""Command Line Interface and modular step-by-step execution orchestrator."""

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


def run_sort(input_path: Path, work_path: Path) -> dict[str, Path]:
    """Stage 1: Scan EXIF and bucket sort raw frames."""
    return sort_rafs_by_exposure(input_path, work_path)


def run_stack(
    work_path: Path,
    stacking_engine: str = "python",
    sigma_low: float = 3.0,
    sigma_high: float = 3.0,
) -> list[Path]:
    """Stage 2: Demosaic and stack intra-bucket subframes to Master FITS."""
    bucket_dirs = sorted(work_path.glob("bucket_*"))
    if not bucket_dirs:
        raise FileNotFoundError(
            f"No bucket folders found in {work_path}. Run '--step sort' first."
        )

    master_paths: list[Path] = []
    print(
        f"\n--- [Stage 2] Intra-Bucket Stacking [Engine: {stacking_engine.upper()}] ---"
    )

    for b_path in bucket_dirs:
        exp_str = b_path.name.replace("bucket_", "")
        master_fit_name = f"Master_{exp_str}"
        master_fits_path = b_path / f"{master_fit_name}.fit"

        if stacking_engine == "python":
            print(f"\n[{b_path.name}] Demosaicing with PySiril...")
            conv_fits_paths = demosaic_bucket_with_siril(b_path)

            print(
                f"[{b_path.name}] Python DFT Sub-Pixel Stacking ({len(conv_fits_paths)} frames)..."
            )
            master_img = align_and_stack_bucket_dft(
                fits_paths=conv_fits_paths,
                sigma_low=sigma_low,
                sigma_high=sigma_high,
                upsample_factor=100,
            )

            fits.writeto(
                master_fits_path,
                np.transpose(master_img, (2, 0, 1)),
                overwrite=True,
            )
        else:
            print(f"\n[{b_path.name}] Siril Native Registration & Stacking on Disk...")
            master_fits_path = register_and_stack_with_siril(
                bucket_dir=b_path,
                output_master_name=master_fit_name,
                sigma_low=sigma_low,
                sigma_high=sigma_high,
            )

        master_paths.append(master_fits_path)

    return master_paths


def run_align(work_path: Path) -> list[Path]:
    """Stage 3: Inter-master Taubin circle fitting and Fourier registration."""
    master_files = sorted(work_path.glob("bucket_*/Master_*.fit"))
    if not master_files:
        raise FileNotFoundError(
            f"No Master_*.fit files found in {work_path}. Run '--step stack' first."
        )

    stacked_masters = [load_fits_to_float32(p) for p in master_files]
    master_names = [p.stem for p in master_files]

    ref_idx = min(len(stacked_masters) - 1, 4)
    aligned_masters = align_masters_taubin(
        masters=stacked_masters,
        master_names=master_names,
        ref_idx=ref_idx,
    )

    aligned_paths: list[Path] = []
    print("\n--- Saving Aligned Masters ---")
    for aligned_img, src_path in zip(aligned_masters, master_files):
        out_aligned_path = src_path.parent / f"Aligned_{src_path.name}"
        fits.writeto(
            out_aligned_path,
            np.transpose(aligned_img, (2, 0, 1)),
            overwrite=True,
        )
        aligned_paths.append(out_aligned_path)
        print(f"  * Saved: {out_aligned_path.name}")

    return aligned_paths


def run_fuse(
    work_path: Path,
    output_path: Path,
    contrast_w: float = 1.0,
    sat_w: float = 1.0,
    exp_w: float = 0.2,
) -> None:
    """Stage 4: 32-bit Mertens multi-scale Laplacian pyramid fusion."""
    aligned_files = sorted(work_path.glob("bucket_*/Aligned_Master_*.fit"))
    if not aligned_files:
        # Fallback to unaligned Master_*.fit if user skipped separate align step
        aligned_files = sorted(work_path.glob("bucket_*/Master_*.fit"))
        if not aligned_files:
            raise FileNotFoundError(
                f"No Master FITS files found in {work_path}. Run '--step stack' and '--step align' first."
            )

    aligned_masters = [load_fits_to_float32(p) for p in aligned_files]
    fuse_and_export_hdr(
        aligned_masters=aligned_masters,
        output_path=output_path,
        contrast_w=contrast_w,
        sat_w=sat_w,
        exp_w=exp_w,
    )


def main() -> None:
    """Entry point with modular step execution."""
    parser = argparse.ArgumentParser(
        description="Automated 32-bit Sub-Pixel Alignment & HDR Fusion for Total Solar Eclipse Data"
    )
    parser.add_argument(
        "--step",
        choices=["all", "sort", "stack", "align", "fuse"],
        default="all",
        help="Pipeline step to run: 'sort', 'stack', 'align', 'fuse', or 'all' (default: all)",
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=Path,
        default=None,
        help="Path containing input RAW (.RAF) files (required for 'sort' and 'all')",
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
        help="Intra-bucket stacking engine: 'python' or 'siril'",
    )
    parser.add_argument(
        "--sigma-low",
        type=float,
        default=3.0,
        help="Sigma clipping low threshold",
    )
    parser.add_argument(
        "--sigma-high",
        type=float,
        default=3.0,
        help="Sigma clipping high threshold",
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
    work_path = args.work_dir.resolve()
    output_path = args.output.resolve()

    if args.step in ("all", "sort"):
        if not args.input_dir:
            parser.error("--input-dir / -i is required for 'sort' or 'all' steps.")
        input_path = args.input_dir.resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_path}")

    # Execution dispatcher
    if args.step == "all":
        run_sort(input_path, work_path)
        run_stack(work_path, args.stacking_engine, args.sigma_low, args.sigma_high)
        run_align(work_path)
        run_fuse(
            work_path,
            output_path,
            args.contrast_weight,
            args.sat_weight,
            args.exp_weight,
        )

    elif args.step == "sort":
        run_sort(input_path, work_path)

    elif args.step == "stack":
        run_stack(work_path, args.stacking_engine, args.sigma_low, args.sigma_high)

    elif args.step == "align":
        run_align(work_path)

    elif args.step == "fuse":
        run_fuse(
            work_path,
            output_path,
            args.contrast_weight,
            args.sat_weight,
            args.exp_weight,
        )


if __name__ == "__main__":
    main()
