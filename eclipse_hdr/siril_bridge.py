"""Bridge module interfacing with Siril via PySiril for demosaicing and stacking."""

from pathlib import Path
from pysiril.siril import Siril
from pysiril.wrapper import Wrapper


def demosaic_bucket_with_siril(bucket_dir: Path) -> list[Path]:
    """Uses PySiril to convert and demosaic Fujifilm X-Trans / Bayer RAW (.RAF)

    files to 32-bit linear FITS sequences for in-memory processing.
    """
    b_path = bucket_dir.resolve()
    conv_dir = b_path / "conv"

    # Check if demosaiced frames already exist from a previous run
    existing_fits = sorted(conv_dir.glob("*.fit")) + sorted(conv_dir.glob("*.fits"))
    if existing_fits:
        return existing_fits

    app = Siril()
    try:
        app.Open()
        cmd = Wrapper(app)

        # Set working directory inside Siril
        cmd.cd(str(b_path))

        # Demosaic RAW files
        try:
            cmd.convertraw("light", debayer=True, out="conv")
        except (AttributeError, Exception):
            app.Send("convertraw light -debayer -out=conv")

        # Search for any FITS produced inside conv/ or in the bucket root
        fits_files = (
            sorted(conv_dir.glob("*.fit"))
            + sorted(conv_dir.glob("*.fits"))
            + sorted(b_path.glob("light_*.fit"))
            + sorted(b_path.glob("conv_*.fit"))
        )

        if not fits_files:
            # Fallback if needed
            try:
                cmd.convert("light", debayer=True, out="conv")
            except (AttributeError, Exception):
                app.Send("convert light -debayer -out=conv")

            fits_files = sorted(conv_dir.glob("*.fit")) + sorted(
                conv_dir.glob("*.fits")
            )

        if not fits_files:
            raise FileNotFoundError(
                f"Siril debayer conversion failed: no FITS files found in {conv_dir}"
            )
        return fits_files
    finally:
        app.Close()
        del app


def register_and_stack_with_siril(
    bucket_dir: Path,
    output_master_name: str,
    sigma_low: float = 3.0,
    sigma_high: float = 3.0,
) -> Path:
    """Uses Siril's native C engine to demosaic, register with 2-pass shift,

    and stack with Winsorized sigma-clipping directly on disk.
    """
    b_path = bucket_dir.resolve()
    raw_files = list(b_path.glob("*.RAF")) + list(b_path.glob("*.raf"))

    app = Siril()
    try:
        app.Open()
        cmd = Wrapper(app)

        cmd.cd(str(b_path))

        # 1. Demosaic RAW sequence
        try:
            cmd.convertraw("light", debayer=True, out="conv")
        except (AttributeError, Exception):
            app.Send("convertraw light -debayer -out=conv")

        conv_dir = b_path / "conv"
        fits_files = sorted(conv_dir.glob("*.fit")) + sorted(conv_dir.glob("*.fits"))

        # If only 1 frame exists in the bucket, save directly as master
        if len(raw_files) == 1 and fits_files:
            master_path = b_path / f"{output_master_name}.fit"
            fits_files[0].rename(master_path)
            return master_path

        # 2. Change working dir into conv/ for registration and stacking
        cmd.cd(str(conv_dir))

        # Register sequence using 2-pass translation (shift-only)
        try:
            cmd.register("light", two_pass=True, transf="shift")
            cmd.seqapplyreg("light", framing="current")
        except (AttributeError, Exception):
            app.Send("register light -2pass -transf=shift")
            app.Send("seqapplyreg light -framing=current")

        # 3. Stack with Winsorized Sigma Clipping and additive normalization
        try:
            cmd.stack(
                "r_light",
                type="rej",
                low=sigma_low,
                high=sigma_high,
                norm="add",
                out=f"../{output_master_name}",
            )
        except (AttributeError, Exception):
            app.Send(
                f"stack r_light rej {sigma_low} {sigma_high} -norm=add -out=../{output_master_name}"
            )

        master_path = b_path / f"{output_master_name}.fit"
        if not master_path.exists():
            raise FileNotFoundError(
                f"Siril stacking failed: '{master_path.name}' was not created."
            )
        return master_path
    finally:
        app.Close()
        del app


def demosaic_all_buckets_with_siril(bucket_dirs: list[Path]) -> dict[Path, list[Path]]:
    """Opens a single Siril session to demosaic all bucket folders sequentially."""
    results: dict[Path, list[Path]] = {}

    # Check if all buckets already have converted FITS files
    buckets_to_process = []
    for b_path in bucket_dirs:
        conv_dir = b_path / "conv"
        fits_files = sorted(conv_dir.glob("*.fit")) + sorted(conv_dir.glob("*.fits"))
        if fits_files:
            results[b_path] = fits_files
        else:
            buckets_to_process.append(b_path)

    if not buckets_to_process:
        return results

    app = Siril()
    app.Open()
    cmd = Wrapper(app)

    try:
        for b_path in buckets_to_process:
            print(f"  * Demosaicing RAW frames in {b_path.name}...", flush=True)
            cmd.cd(str(b_path))

            try:
                cmd.convertraw("light", debayer=True, out="conv")
            except Exception:
                app.Send("convertraw light -debayer -out=conv")

            conv_dir = b_path / "conv"
            fits_files = sorted(conv_dir.glob("*.fit")) + sorted(
                conv_dir.glob("*.fits")
            )
            if not fits_files:
                # Fallback check in root
                fits_files = sorted(b_path.glob("light_*.fit")) + sorted(
                    b_path.glob("conv_*.fit")
                )

            if not fits_files:
                raise FileNotFoundError(
                    f"Failed to find demosaiced FITS files in {b_path.name}"
                )

            results[b_path] = fits_files
    finally:
        try:
            app.Close()
        except Exception:
            pass

    return results
