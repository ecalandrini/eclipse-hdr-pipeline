"""Bridge module interfacing with Siril via PySiril for demosaicing and stacking."""

from pathlib import Path
from pysiril.siril import Siril
from pysiril.wrapper import Wrapper


def demosaic_bucket_with_siril(bucket_dir: Path) -> list[Path]:
    """Uses PySiril to convert and demosaic Fujifilm X-Trans / Bayer RAW (.RAF)

    files to 32-bit linear FITS sequences for in-memory processing.
    """
    b_path = bucket_dir.resolve()
    app = Siril()
    try:
        app.Open()
        cmd = Wrapper(app)

        # Set working directory inside Siril
        cmd.cd(str(b_path))

        # Demosaic RAW files (convertraw or convert light -debayer -out=conv)
        try:
            cmd.convertraw("light", debayer=True, out="conv")
        except (AttributeError, Exception):
            app.Send("convertraw light -debayer -out=conv")

        fits_files = sorted(b_path.glob("conv_*.fit"))
        if not fits_files:
            try:
                cmd.convert("light", debayer=True, out="conv")
            except (AttributeError, Exception):
                app.Send("convert light -debayer -out=conv")
            fits_files = sorted(b_path.glob("conv_*.fit"))

        if not fits_files:
            raise FileNotFoundError(
                f"Siril debayer conversion failed: no FITS files found in {b_path.name}"
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

        # 1. Demosaic RAW sequence to 32-bit linear FITS
        try:
            cmd.convertraw("light", debayer=True, out="conv")
        except (AttributeError, Exception):
            app.Send("convertraw light -debayer -out=conv")

        fits_files = sorted(b_path.glob("conv_*.fit"))
        if not fits_files:
            try:
                cmd.convert("light", debayer=True, out="conv")
            except (AttributeError, Exception):
                app.Send("convert light -debayer -out=conv")

        # If only 1 frame exists in the bucket, save directly as master
        if len(raw_files) == 1:
            try:
                cmd.load("conv_00001.fit")
                cmd.save(output_master_name)
            except (AttributeError, Exception):
                app.Send("load conv_00001.fit")
                app.Send(f"save {output_master_name}")
            return b_path / f"{output_master_name}.fit"

        # 2. Register sequence using 2-pass translation (shift-only)
        try:
            cmd.register("conv", two_pass=True, transf="shift")
            cmd.seqapplyreg("conv", framing="current")
        except (AttributeError, Exception):
            app.Send("register conv -2pass -transf=shift")
            app.Send("seqapplyreg conv -framing=current")

        # 3. Stack with Winsorized Sigma Clipping and additive normalization
        try:
            cmd.stack(
                "r_conv",
                type="rej",
                low=sigma_low,
                high=sigma_high,
                norm="add",
                out=output_master_name,
            )
        except (AttributeError, Exception):
            app.Send(
                f"stack r_conv rej {sigma_low} {sigma_high} -norm=add -out={output_master_name}"
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
