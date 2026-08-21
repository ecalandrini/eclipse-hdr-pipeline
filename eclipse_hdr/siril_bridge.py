"""Bridge module interfacing with Siril via PySiril for demosaicing and stacking."""

from pathlib import Path
from pysiril.siril import Siril
from pysiril.wrapper import Wrapper


def _send_siril_cmd(app: Siril, cmd_str: str) -> None:
    """Dispatches a command string directly to the Siril IPC pipe."""
    if hasattr(app, "send"):
        app.send(cmd_str)
    elif hasattr(app, "Send"):
        app.Send(cmd_str)
    elif hasattr(app, "Execute"):
        app.Execute(cmd_str)
    else:
        raise AttributeError(
            "Could not find a valid command dispatch method on Siril object."
        )


def demosaic_all_buckets_with_siril(bucket_dirs: list[Path]) -> dict[Path, list[Path]]:
    """Opens a single Siril session to demosaic all bucket folders sequentially."""
    results: dict[Path, list[Path]] = {}
    buckets_to_process: list[Path] = []

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

            # Send RAW conversion command
            _send_siril_cmd(app, "convertraw light -debayer -out=conv")

            conv_dir = b_path / "conv"
            fits_files = sorted(conv_dir.glob("*.fit")) + sorted(
                conv_dir.glob("*.fits")
            )
            if not fits_files:
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
    app.Open()
    cmd = Wrapper(app)

    try:
        cmd.cd(str(b_path))

        conv_dir = b_path / "conv"
        fits_files = sorted(conv_dir.glob("*.fit")) + sorted(conv_dir.glob("*.fits"))

        # 1. Demosaic if not already converted
        if not fits_files:
            _send_siril_cmd(app, "convertraw light -debayer -out=conv")
            fits_files = sorted(conv_dir.glob("*.fit")) + sorted(
                conv_dir.glob("*.fits")
            )

        # If only 1 frame exists in the bucket, save directly as master
        if len(raw_files) == 1 and fits_files:
            master_path = b_path / f"{output_master_name}.fit"
            fits_files[0].rename(master_path)
            return master_path

        # 2. Change working dir into conv/ for sequence operations
        cmd.cd(str(conv_dir))

        # 3. Register sequence using 2-pass translation (shift-only)
        _send_siril_cmd(app, "register light -2pass -transf=shift")
        _send_siril_cmd(app, "seqapplyreg light -framing=current")

        # 4. Stack with Winsorized Sigma Clipping and additive normalization
        _send_siril_cmd(
            app,
            f"stack r_light rej {sigma_low} {sigma_high} -norm=add -out=../{output_master_name}",
        )

        master_path = b_path / f"{output_master_name}.fit"
        if not master_path.exists():
            raise FileNotFoundError(
                f"Siril stacking failed: '{master_path.name}' was not created."
            )
        return master_path
    finally:
        try:
            app.Close()
        except Exception:
            pass


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
    app.Open()
    cmd = Wrapper(app)

    try:
        # Set working directory inside Siril
        cmd.cd(str(b_path))

        # Demosaic RAW files
        _send_siril_cmd(app, "convertraw light -debayer -out=conv")

        # Search for FITS produced inside conv/ or in the bucket root
        fits_files = (
            sorted(conv_dir.glob("*.fit"))
            + sorted(conv_dir.glob("*.fits"))
            + sorted(b_path.glob("light_*.fit"))
            + sorted(b_path.glob("conv_*.fit"))
        )

        if not fits_files:
            # Fallback if standard convert is required
            _send_siril_cmd(app, "convert light -debayer -out=conv")
            fits_files = sorted(conv_dir.glob("*.fit")) + sorted(
                conv_dir.glob("*.fits")
            )

        if not fits_files:
            raise FileNotFoundError(
                f"Siril debayer conversion failed: no FITS files found in {conv_dir}"
            )
        return fits_files
    finally:
        try:
            app.Close()
        except Exception:
            pass
