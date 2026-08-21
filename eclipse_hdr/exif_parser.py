"""EXIF metadata extraction and raw image file organization for exposure brackets."""

from fractions import Fraction
from pathlib import Path
import shutil
import exifread


def get_shutter_speed_str(raw_path: Path) -> str:
    """Extracts exposure time from raw image EXIF metadata and converts it

    into a clean, filesystem-safe string format (e.g., '1_4000s', '1_15s', '2_0s').

    Args:
        raw_path: Path to the raw camera file (.RAF, etc.).

    Returns:
        Standardized string identifier for the exposure speed.
    """
    with open(raw_path, "rb") as f:
        # Fast EXIF read stopping immediately at the ExposureTime tag
        tags = exifread.process_file(f, stop_tag="EXIF ExposureTime", details=False)

    exp_tag = tags.get("EXIF ExposureTime")
    if not exp_tag:
        raise ValueError(f"Could not locate 'EXIF ExposureTime' tag in {raw_path.name}")

    val_str = str(exp_tag).strip()

    # Sub-second fractional representations (e.g., '1/4000', '1/125')
    if "/" in val_str:
        num, den = map(int, val_str.split("/"))
        frac = Fraction(num, den)
        return f"{frac.numerator}_{frac.denominator}s"
    else:
        # Multi-second exposures (e.g., '2', '2.5')
        val_float = float(val_str)
        return f"{val_float:.1f}s".replace(".", "_")


def sort_rafs_by_exposure(source_dir: Path, output_base_dir: Path) -> dict[str, Path]:
    """Scans the source directory for raw files and organizes them into

    subdirectories grouped by their shutter speed.

    Args:
        source_dir: Directory containing unorganized raw (.RAF) files.
        output_base_dir: Root workspace directory where bucket folders will be created.

    Returns:
        Dictionary mapping exposure time strings to their destination folder Paths.
    """
    source_path = source_dir.resolve()
    base_dest = output_base_dir.resolve()
    base_dest.mkdir(parents=True, exist_ok=True)

    raw_files = list(source_path.glob("*.RAF")) + list(source_path.glob("*.raf"))
    if not raw_files:
        raise FileNotFoundError(f"No RAW (.RAF) files found in {source_path}")

    bucket_map: dict[str, Path] = {}

    print(f"\n--- Scanning and Sorting {len(raw_files)} RAW Files by Exposure ---")
    for raw in sorted(raw_files):
        exp_str = get_shutter_speed_str(raw)
        bucket_dir = base_dest / f"bucket_{exp_str}"
        bucket_dir.mkdir(exist_ok=True)
        bucket_map[exp_str] = bucket_dir

        dst_file = bucket_dir / raw.name
        if not dst_file.exists():
            shutil.copy2(raw, dst_file)

    for exp_str, b_path in sorted(bucket_map.items()):
        sub_count = len(list(b_path.glob("*.RAF")) + list(b_path.glob("*.raf")))
        print(f"  * Bucket '{exp_str}': {sub_count} frame(s)")

    return bucket_map
