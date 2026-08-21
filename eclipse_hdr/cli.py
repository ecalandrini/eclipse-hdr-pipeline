"""Command Line Interface and modular step-by-step execution orchestrator."""

import argparse
from pathlib import Path
from astropy.io import fits
import numpy as np
import gc

from eclipse_hdr.alignment import (
    align_and_stack_bucket_dft,
    apply_spatial_shift_rgb,
    load_fits_to_float32,
    save_preview_jpg,
    solve_alignment_shifts_graph,
)
from .exif_parser import sort_rafs_by_exposure
from .hdr_fusion import fuse_and_export_hdr
from .siril_bridge import (
    demosaic_all_buckets_with_siril,
    register_and_stack_with_siril,
)
from eclipse_hdr.post_processing import process_coronal_features

def run_sort(input_path: Path, work_path: Path) -> dict[str, Path]:
    """Stage 1: Scan EXIF and bucket sort raw frames."""
    return sort_rafs_by_exposure(input_path, work_path)


def run_stack(
    work_path: Path,
    stacking_engine: str = "python",
    sigma_low: float = 3.0,
    sigma_high: float = 3.0,
    max_shift: float = 50.0,
) -> list[Path]:
    """Stage 2: Demosaic and stack intra-bucket subframes to Master FITS & JPG.

    Skips any bucket folder that already contains a Master_*.fit file.
    """
    all_bucket_dirs = sorted(work_path.glob("bucket_*"))
    if not all_bucket_dirs:
        raise FileNotFoundError(
            f"No bucket folders found in {work_path}. Run '--step sort' first."
        )

    # Filter only bucket directories that do NOT have a Master_*.fit file
    bucket_dirs: list[Path] = []
    for b in all_bucket_dirs:
        existing_masters = list(b.glob("Master_*.fit"))
        if existing_masters:
            print(
                f"[{b.name}] Master already exists ({existing_masters[0].name}). Skipping.",
                flush=True,
            )
        else:
            bucket_dirs.append(b)

    if not bucket_dirs:
        print(
            "\nAll buckets already have Master FITS files. Nothing to stack.",
            flush=True,
        )
        return sorted(work_path.glob("bucket_*/Master_*.fit"))

    print(
        f"\n--- [Stage 2] Intra-Bucket Stacking ({len(bucket_dirs)}/{len(all_bucket_dirs)} Buckets to Process) [Engine: {stacking_engine.upper()}] ---",
        flush=True,
    )

    master_paths: list[Path] = []

    if stacking_engine == "siril":
        for b_path in bucket_dirs:
            exp_str = b_path.name.replace("bucket_", "")
            master_fit_name = f"Master_{exp_str}"
            print(
                f"\n[{b_path.name}] Stacking via Siril native C engine...", flush=True
            )
            master_fits_path = register_and_stack_with_siril(
                bucket_dir=b_path,
                output_master_name=master_fit_name,
                sigma_low=sigma_low,
                sigma_high=sigma_high,
            )
            master_img = load_fits_to_float32(master_fits_path)
            jpg_out = b_path / f"{master_fit_name}.jpg"
            save_preview_jpg(master_img, jpg_out)
            print(f"  [Saved] -> {master_fits_path.name} & {jpg_out.name}", flush=True)
            master_paths.append(master_fits_path)
    else:
        # Demosaic only the buckets that need processing
        bucket_fits_map = demosaic_all_buckets_with_siril(bucket_dirs)

        for b_path in bucket_dirs:
            exp_str = b_path.name.replace("bucket_", "")
            master_fit_name = f"Master_{exp_str}.fit"
            master_fits_path = b_path / master_fit_name
            master_jpg_path = b_path / f"Master_{exp_str}.jpg"

            conv_fits_paths = bucket_fits_map.get(b_path, [])
            print(
                f"\n[{b_path.name}] Python DFT Stacking ({len(conv_fits_paths)} frames)...",
                flush=True,
            )

            master_img = align_and_stack_bucket_dft(
                fits_paths=conv_fits_paths,
                sigma_low=sigma_low,
                sigma_high=sigma_high,
                upsample_factor=100,
                max_allowed_shift=max_shift,
            )

            # Save 32-bit linear FITS
            fits.writeto(
                master_fits_path,
                np.transpose(master_img, (2, 0, 1)),
                overwrite=True,
            )
            # Save tonemapped preview JPG
            save_preview_jpg(master_img, master_jpg_path)
            print(
                f"  [Saved] -> {master_fits_path.name} & {master_jpg_path.name}",
                flush=True,
            )
            master_paths.append(master_fits_path)

    return sorted(work_path.glob("bucket_*/Master_*.fit"))


def run_align(work_path: Path) -> list[Path]:
    """Stage 3: Memory-safe streamed inter-master alignment."""
    master_files = sorted(work_path.glob("bucket_*/Master_*.fit"))
    if not master_files:
        raise FileNotFoundError(f"No Master_*.fit files found in {work_path}. Run '--step stack' first.")

    # 1. Solve shifts on lightweight proxies (<50MB RAM)
    anchor_idx, shifts = solve_alignment_shifts_graph(
        master_files=master_files,
        downsample=4,
        upsample_factor=100,
    )

    # 2. Stream and apply shifts one frame at a time
    aligned_paths: list[Path] = []
    print("\n--- Streaming Aligned Masters to Disk ---", flush=True)

    for src_path, (dy, dx) in zip(master_files, shifts):
        out_aligned_path = src_path.parent / f"Aligned_{src_path.name}"
        out_aligned_jpg = src_path.parent / f"Aligned_{src_path.stem}.jpg"

        print(
            f"  * {src_path.parent.name:20s} -> Offset: (dy={dy:+.2f}px, dx={dx:+.2f}px) -> Saving...",
            flush=True,
        )

        img = load_fits_to_float32(src_path)
        shifted = apply_spatial_shift_rgb(img, dy, dx)
        del img

        fits.writeto(
            out_aligned_path,
            np.transpose(shifted, (2, 0, 1)),
            overwrite=True,
        )
        save_preview_jpg(shifted, out_aligned_jpg)
        aligned_paths.append(out_aligned_path)

        del shifted
        gc.collect()

    return aligned_paths


def run_fuse(
    work_path: Path,
    output_path: Path,
    mode: str = "both",
    contrast_w: float = 0.8,
    sat_w: float = 1.0,
    exp_w: float = 0.8,
) -> Path:
    """Stage 4: Dual-mode HDR fusion (Scientific & Artistic)."""
    aligned_files = sorted(work_path.glob("bucket_*/Aligned_Master_*.fit"))
    if not aligned_files:
        raise FileNotFoundError(f"No Aligned_Master_*.fit files found in {work_path}. Run '--step align' first.")

    aligned_masters = [load_fits_to_float32(p) for p in aligned_files]
    master_names = [p.stem.replace("Aligned_Master_", "Bucket_") for p in aligned_files]

    fuse_and_export_hdr(
        aligned_masters=aligned_masters,
        master_names=master_names,
        output_path=output_path,
        mode=mode,
        contrast_w=contrast_w,
        sat_w=sat_w,
        exp_w=exp_w,
    )
    return output_path


def main() -> None:
    """Entry point with modular step execution."""
    parser = argparse.ArgumentParser(
        description="Automated 32-bit Sub-Pixel Alignment & HDR Fusion for Total Solar Eclipse Data"
    )
    parser.add_argument(
        "--step",
        type=str,
        choices=["all", "sort", "stack", "align", "fuse", "postprocess"],
        default="all",
        help="Execution step: 'sort', 'stack', 'align', 'fuse', 'postprocess', or 'all'",
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
        default=Path("./workspace"),
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
        help="Intra-bucket stacking engine: 'python' (in-memory DFT cross-correlation) or 'siril' (native C engine)",
    )
    parser.add_argument(
        "--max-shift",
        type=float,
        default=50.0,
        help="Maximum allowable shift in pixels before a subframe is excluded as an outlier (default: 50.0px)",
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
    parser.add_argument(
        "--mode",
        type=str,
        choices=["scientific", "artistic", "both"],
        default="both",
        help="HDR fusion mode: 'scientific' (linear radiance), 'artistic' (Mertens pyramid), or 'both' (default: both)",
    )
    parser.add_argument(
        "--sharpen",
        type=float,
        default=1.8,
        help="Unsharp mask sharpening strength for coronal loops (default: 1.8)",
    ) 
    parser.add_argument(
        "--input-file",
        "-f",
        type=Path,
        default=None,
        help="Explicit input file to process for '--step postprocess' (e.g. ./Eclipse_HDR_Master_Artistic_HDR.tif)",
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

    if args.step == "all":
        run_sort(input_path, work_path)
        run_stack(
            work_path,
            args.stacking_engine,
            args.sigma_low,
            args.sigma_high,
            args.max_shift,
        )
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
        run_stack(
            work_path,
            args.stacking_engine,
            args.sigma_low,
            args.sigma_high,
            args.max_shift,
        )

    elif args.step == "align":
        run_align(work_path)

    elif args.step == "fuse":
        run_fuse(
            work_path,
            output_path,
            args.mode,
            args.contrast_weight,
            args.sat_weight,
            args.exp_weight,
        )
    elif args.step == "postprocess":
        if args.input_file:
            target_input = args.input_file.resolve()
        elif args.input_dir and args.input_dir.is_file():
            target_input = args.input_dir.resolve()
        else:
            candidates = [
                output_path.with_name(f"{output_path.stem}_Artistic_HDR.tif"),
                output_path.with_name(f"{output_path.stem}_Scientific_Linear.fits"),
                output_path.with_name(f"{output_path.stem}_Scientific_Linear.tif"),
                output_path,
            ]
            target_input = next((p for p in candidates if p.exists()), None)

        if not target_input or not target_input.exists():
            parser.error("Could not find input file. Specify '--input-file <path>'.")

        process_coronal_features(
            input_master_path=target_input,
            output_dir=work_path,
            sharpen_amount=args.sharpen,
        )

if __name__ == "__main__":
    main()
