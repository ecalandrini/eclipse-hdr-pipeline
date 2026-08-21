# Eclipse HDR Pipeline

An automated 32-bit astrophotography pipeline designed to eliminate posterization banding, align multi-exposure bracket sequences with sub-pixel precision, and blend coronal dynamic range using multi-scale Mertens HDR fusion.

Optimized for high-dynamic-range solar eclipse sequences (e.g., Fujifilm 14-bit X-Trans `.RAF` or Bayer RAW files) and managed with `uv`.

---

## Key Features

- **EXIF Automated Sorting**: Inspects shutter speed metadata directly from raw files and auto-sorts them into exposure bucket folders (`1/4000s` down to `1/15s`).
- **32-Bit Linear Demosaicing**: Interfaces with Siril via `pysiril` to execute CFA demosaicing directly to 32-bit floating-point linear FITS files.
- **Dual Intra-Bucket Stacking Engines**:
  - **`python` (Default)**: In-memory single-step Discrete Fourier Transform (DFT) phase cross-correlation ($1/100\text{th}$ px precision) paired with vectorized Median Absolute Deviation (MAD) Winsorized sigma-clipping.
  - **`siril`**: Native C engine disk-based 2-pass translation registration and Winsorized sigma-clipping via `pysiril` IPC commands.
- **Two-Tier Sub-Pixel Alignment**:
  - **Algorithm 1 (Intra-Bucket)**: Sub-pixel cross-correlation and outlier rejection within each exposure tier.
  - **Algorithm 2 (Inter-Master)**: Sub-pixel parabolic radial gradient edge tracing + algebraic Taubin circle fitting to align disparate exposure masters to the lunar/solar limb centroid.
- **Multi-Scale Mertens Exposure Fusion**: 32-bit Laplacian pyramid blending weighted by contrast, saturation, and well-exposedness, avoiding hard threshold clipping or color posterization rings.

---

## Prerequisites

1. **Siril**: [Siril](https://siril.org/) must be installed and accessible on your system `PATH`.
2. **Package Manager**: [uv](https://docs.astral.sh/uv/) for environment and dependency management.

---

## Installation & Setup

Clone the repository and sync the virtual environment using `uv`:

```bash
git clone [https://github.com/ecalandrini/eclipse-hdr-pipeline.git](https://github.com/ecalandrini/eclipse-hdr-pipeline.git)
cd eclipse-hdr-pipeline

# Creates .venv and locks dependencies from Git and PyPI
uv sync
```

## Usage

Run the pipeline using `uv run` with the default Python in-memory stacking engine:

```bash
uv run eclipse-hdr --input-dir /path/to/raw_raf_files --output ./Eclipse_HDR_Master.tif
```

Or switch to Siril's native C disk-based engine:

```bash
uv run eclipse-hdr -i /path/to/raw_raf_files --stacking-engine siril -o ./Eclipse_HDR_Master.tif
```

How to run step-by-step:

```bash
# 1. Sort files into buckets
uv run eclipse-hdr --step sort -i /path/to/raws -w ./workspace

# 2. Demosaic and stack brackets (inspect Master_*.fit files in Siril / DS9 if desired)
uv run eclipse-hdr --step stack -w ./workspace --stacking-engine python

# 3. Fit lunar limb circles and Fourier shift all masters to common center
uv run eclipse-hdr --step align -w ./workspace

# 4. Tweak fusion weights without re-stacking or re-aligning
uv run eclipse-hdr --step fuse -w ./workspace --contrast-weight 2.0 --exp-weight 0.0 -o ./output_punchy.tif
uv run eclipse-hdr --step fuse -w ./workspace --contrast-weight 0.8 --exp-weight 0.3 -o ./output_natural.tif
```

> **Note!** 
Currently, **Taubin Alignment** (Algorithm 2) and **Mertens Fusion** (Step 5) expect in-memory Python objects (`stacked_masters`, `aligned_masters`) passed directly from previous loop iterations. If you exit after stacking, those arrays are lost unless written to disk.
>To make stages independent:
>* `sort`: Reads raw `.RAF`, creates `bucket_<exp>`/ folders.
>* `stack`: Reads `bucket_<exp>/*.RAF`, demosaics, stacks, and writes `Master_<exp>.fit`.
>* `align`: Reads all `Master_<exp>.fit` files, fits Taubin centroids, shifts via Fourier, and saves `Aligned_Master_<exp>.fit`.
>* `fuse`: Reads all `Aligned_Master_*.fit` files, runs Mertens pyramid fusion, and exports the final 16-bit TIFF.

## CLI Options
| Flag | Shorthand | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input-dir` | `-i` | *Required* | Directory containing the uncompressed input `.RAF` files |
| `--work-dir` | `-w` | `./eclipse_workspace` | Working directory where bucket folders and intermediate 32-bit FITS files are stored |
| `--output` | `-o` | `./Eclipse_HDR_Master.tif` | Destination path for the final 16-bit uncompressed master TIFF |
| `--stacking-engine` | | `python` | Intra-bucket stacking engine: `python` (in-memory DFT + MAD) or `siril` (native C engine on disk) |
| `--sigma-low` | | `3.0` | Lower outlier rejection threshold (MAD sigma-clipping) for intra-bucket stacking |
| `--sigma-high` | | `3.0` | Upper outlier rejection threshold (MAD sigma-clipping) for intra-bucket stacking |
| `--contrast-weight` | | `1.0` | Mertens Laplacian contrast weight (rewards sharp coronal filaments) |
| `--sat-weight` | | `1.0` | Mertens color saturation weight (protects prominence colors) |
| `--exp-weight` | | `0.2` | Mertens well-exposedness Gaussian weight (governs mid-tone balance) |

## Parameter Tuning & Mathematical Meaning

### 1. Intra-Bucket Stacking Controls

* **`--stacking-engine` (`python` vs. `siril`)**
  * **Function**: Selects between in-memory single-step DFT phase cross-correlation + MAD sigma-clipping (`python`) and Siril's native disk-streaming C registration engine (`siril`).
  * **When to adjust**: Keep on `python` for mathematical precision ($1/100\text{th}$ px Fourier shift with zero spatial interpolation blur) and zero intermediate disk clutter. Use `siril` when processing massive bursts (e.g., 50+ frames per bracket) on low-RAM machines.
* **`--sigma-low` & `--sigma-high`**
  * **Mathematical meaning**: Sets the scale factor $k$ for outlier rejection based on the Median Absolute Deviation (MAD):
    $$[\text{Median} - k_{\text{low}} \cdot \hat{\sigma}, \;\; \text{Median} + k_{\text{high}} \cdot \hat{\sigma}], \quad \text{where } \hat{\sigma} = 1.4826 \cdot \text{MAD}$$
  * **When to adjust**: Lower to `2.0` – `2.5` to aggressively reject wind shake, cloud transients, or tracking drift. Increase to `3.5` – `4.5` on clean, high-SNR sequences to avoid clipping faint coronal signal.

---

### 2. Mertens Exposure Fusion Weights

The multi-scale Laplacian pyramid blends aligned masters according to normalized per-pixel weights:
$$W_{ij} = (C_{ij})^{w_c} \times (S_{ij})^{w_s} \times (E_{ij})^{w_e}$$

* **`--contrast-weight` ($w_c$, Default: `1.0`)**
  * **Mathematical meaning**: Evaluates the local high-pass response from a 2D discrete Laplacian filter.
  * **Eclipse Impact**: Primary control for **coronal streamer striations** and fine magnetic loop definition. Increasing this gives priority to structural sharpness across brackets.
* **`--sat-weight` ($w_s$, Default: `1.0`)**
  * **Mathematical meaning**: Computes the channel-wise standard deviation across RGB components for every pixel.
  * **Eclipse Impact**: Preserves saturated emissions from hydrogen-alpha ($H\alpha$) solar prominences and chromosphere beads without allowing them to blend into adjacent white coronal light.
* **`--exp-weight` ($w_e$, Default: `0.2`)**
  * **Mathematical meaning**: Evaluates closeness to midtone exposure using a Gaussian function centered at $0.5$ ($\sigma \approx 0.2$).
  * **Eclipse Impact**: While terrestrial photography uses $w_e = 1.0$, solar eclipse imagery features natural steep gradients from the inner to outer corona. A lower value (`0.1` – `0.2`) prevents artificial darkening and halo artifacts around the limb.

## Recommended Tuning Profiles

```bash
# Maximum Filament & Streamer Definition
uv run eclipse-hdr -i ./raws --contrast-weight 2.0 --sat-weight 1.2 --exp-weight 0.0

# Natural / Photometric Dynamic Range
uv run eclipse-hdr -i ./raws --contrast-weight 0.8 --sat-weight 1.0 --exp-weight 0.3

# Windy / Cloud-Transients Rejection
uv run eclipse-hdr -i ./raws --sigma-low 2.2 --sigma-high 2.2 --contrast-weight 1.2
```

## Processing Architecture

```
Raw .RAF Files
      │
      ▼
 [EXIF Parser] ──► Auto-sort into Exposure Buckets (e.g. 1/4000s ... 1/15s)
      │
      ▼
 [PySiril Bridge] ──► 32-bit Linear X-Trans Demosaicing (conv_*.fit)
      │
      ├─────────────────────────────────────────┐
      │ (Engine: python)                        │ (Engine: siril)
      ▼                                         ▼
 [Algorithm 1: In-Memory DFT]          [Siril 2-Pass Registration & Stacking]
      │                                         │
      └───────────────────┬─────────────────────┘
                          │
                          ▼
             11 Master FITS Files (RAM / Disk)
                          │
                          ▼
 [Algorithm 2: Taubin Sub-Pixel Limb Centroid Alignment]
                          │
                          ▼
 [Fourier Shift Registration to Reference Solar Center]
                          │
                          ▼
 [Mertens 32-Bit Multi-Scale Laplacian Pyramid Fusion]
                          │
                          ▼
             16-Bit Master HDR TIFF (De-banded, High-SNR Output)
 ```

## References & Further Reading

1. **Mertens Exposure Fusion**:
   - Mertens, T., Kautz, J., & Van Reeth, F. (2009). *Exposure Fusion: Blend Photographic Exposures to High Dynamic Range*. Computer Graphics Forum, 28(1), 161–171. [DOI: 10.1111/j.1467-8659.2008.01184.x](https://doi.org/10.1111/j.1467-8659.2008.01184.x)
2. **Sub-Pixel Registration via Matrix Multiplication DFT**:
   - Guizar-Sicairos, M., Thurman, S. T., & Fienup, J. R. (2008). *Efficient subpixel image registration by cross-correlation*. Optics Letters, 33(2), 156–158. [DOI: 10.1364/OL.33.000156](https://doi.org/10.1364/OL.33.000156)
3. **Taubin Algebraic Circle Fitting**:
   - Taubin, G. (1991). *Estimation of planar curves, surfaces, and nonplanar space curves defined by implicit equations with applications to edge and range image segmentation*. IEEE Transactions on Pattern Analysis and Machine Intelligence, 13(11), 1115–1138. [DOI: 10.1109/34.103273](https://doi.org/10.1109/34.103273)
4. **Pyramid-Based Image Blending**:
   - Burt, P. J., & Adelson, E. H. (1983). *A Multiresolution Spline with Application to Image Mosaics*. ACM Transactions on Graphics (TOG), 2(4), 217–236. [DOI: 10.1145/245.247](https://doi.org/10.1145/245.247)
5. **Astronomical Outlier Rejection & Stacking**:
   - Berry, R., & Burnell, J. (2005). *The Handbook of Astronomical Image Processing*. Willmann-Bell.