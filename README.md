# NanoObjectDetection 2.0

NanoObjectDetection analyzes TIFF image stacks of diffusing particles. It
localizes particles, links detections into trajectories, estimates diffusion
and hydrodynamic diameter, and exports analysis tables and plots. Version 2.0
keeps compatibility with the 1.0 parameter format and adds blockwise
processing and optional CUDA acceleration.

The original 1.0 package is still included in [`nanoobjectdetection_1_0/`](./nanoobjectdetection_1_0/).
For full parameter descriptions, scientific methods, compatibility details,
limitations, and output-file definitions, see the
[2.0 documentation](./NANOOBJECTDETECTION_2_0_DOCUMENTATION.md).

## Installation

Use Python with a virtual environment. From the repository folder, run these
commands in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-2.0.txt
```

On macOS or Linux, create and activate the environment with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-2.0.txt
```

### Optional GPU preprocessing

GPU mode requires a supported NVIDIA GPU, a compatible CUDA runtime, and the
matching CuPy package. For CUDA 12.x, the repository's optional requirements
file installs `cupy-cuda12x`:

```bash
python -m pip install -r requirements-gpu.txt
```

If your CUDA version differs, install the CuPy package matching your CUDA
runtime instead. The pipeline reports an error if GPU mode is requested but a
usable CUDA device is unavailable. Trackpy localization, linking, and MSD
fitting remain CPU operations.

## Run from an IDE (recommended)

The default settings in `main.py` point to the included example TIFF. You can
run that file directly from VS Code, PyCharm, or another Python IDE:

1. Open the repository folder in your IDE.
2. Select the Python interpreter from the `.venv` environment created above.
   In VS Code, use **Python: Select Interpreter**; in PyCharm, select the
   project interpreter in the Python interpreter settings.
3. Open `main.py` and review the settings near the top of the file. For your
   own experiment, update `INPUT_FILE`, `OUTPUT_DIR`, and the experiment and
   analysis settings in `main.py`. The input and output paths can be absolute
   paths or paths built from `ROOT`.
4. Run `main.py` with the IDE's **Run Python File** or **Run** action. The
   default example stack will run without command-line arguments.

To load a legacy-format JSON instead of the in-code `PARAMETERS`, set
`LEGACY_PARAMETER_FILE` in `main.py`, for example:

```python
LEGACY_PARAMETER_FILE = ROOT / "example" / "parameter.json"
```

Then update the input/output paths as needed and run `main.py`. To use
chunked processing or GPU preprocessing from an IDE, add the relevant
arguments to that IDE's run configuration; for example, use
`--processing-mode chunked` or `--gpu`. Select the same `.venv` interpreter
for the run configuration so the installed dependencies are available.

Before analyzing experimental data, review the camera calibration,
acquisition rate, temperature, viscosity, channel dimensions, ROI, and
Trackpy settings in `main.py` or the selected parameter file.

## Run from a terminal

The same script can be run with options from PowerShell:

```powershell
python .\main.py `
  --input-file "C:\data\my_stack.tif" `
  --output-dir "C:\data\my_analysis" `
  --processing-mode single_stack
```

To use a legacy-format parameter file:

```powershell
python .\main.py `
  --parameter-file ".\example\parameter.json" `
  --input-file ".\example\20_6µmKap_50Au_20_7C_100fps_5000fr_700us_V1j.tif" `
  --output-dir ".\example\my_analysis" `
  --processing-mode chunked
```

The input and output command-line paths override the corresponding paths in
the parameter JSON. If `--parameter-file` is omitted, the analysis settings
defined in `main.py` are used.

Available command-line options:

| Option | Purpose |
|---|---|
| `--processing-mode single_stack` | Read frames directly from the TIFF; this is the default. |
| `--processing-mode chunked` | Store/process the selected stack in blocks, useful for large inputs. |
| `--gpu` | Enable GPU preprocessing; requires compatible CuPy/CUDA installation. |
| `--input-file PATH` | Select the TIFF stack to analyze. |
| `--output-dir PATH` | Select where results are written. |
| `--parameter-file PATH` | Load analysis settings from a legacy-format JSON file. |

Chunking reduces localization working-block size, but does not make the exact
stack-wide static-background calculation bounded-memory. When that
preprocessing option is enabled, the selected ROI stack must fit in RAM even
in chunked mode. Reduce the ROI or frame range if memory is insufficient.

## Features

- TIFF stack input with spatial and frame-range ROI selection.
- Camera-offset, laser-fluctuation, static-background, and optional SNR
  preprocessing.
- Trackpy particle localization and trajectory linking, including optional
  stitching for repeatedly lost single-particle detections.
- Fixed-particle filtering, trajectory quality checks, and global drift
  correction.
- Mean-squared displacement analysis, diffusion estimation, and
  Stokes-Einstein hydrodynamic diameter calculations.
- Optional channel hindrance correction, including position-aware correction.
- Single-stack and chunked execution modes, plus optional CUDA preprocessing.
- CSV tables, compatibility output names, run configuration, and plots.

See the [2.0 documentation](./NANOOBJECTDETECTION_2_0_DOCUMENTATION.md) for
the complete list of supported settings, generated files, assumptions, and
known limitations.
