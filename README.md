# NanoObjectDetection 2.0

This is a new Version of Ronny Foersters Package. The old Package had lots of issues with type safety and RAM usage. This new implementation aims to improve RAM usage and enable us to easily develop new features.

NanoObjectDetection analyzes TIFF image stacks of diffusing particles. It
localizes particles, links detections into trajectories, estimates diffusion
and hydrodynamic diameter, and exports analysis tables and plots. Version 2.0
keeps compatibility with the 1.0 parameter format and adds blockwise
processing and optional CUDA acceleration.

The source checkout also contains the original 1.0 implementation in
[`nanoobjectdetection_1_0/`](./nanoobjectdetection_1_0/); the installable
distribution packages the 2.0 pipeline. For full parameter descriptions,
scientific methods, compatibility details, limitations, and output-file
definitions, see the
[2.0 documentation](./NANOOBJECTDETECTION_2_0_DOCUMENTATION.md).

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
- RAM saving options: Single-stack and chunked execution modes, plus optional CUDA preprocessing, Optional Conversion of pixels to 8bit depth, 16bit is probably overkill.
- CSV tables, compatibility output names, run configuration, and plots.

## New Features

- Try to implement them on your own! AI is powerful and easy too use. You can then try to submit a merge request here.

## Issues

- 1st Try to solve the issue yourself with AI
-> if its a bug caused by this repo, not your local installation open an Issue or write me a detailed  Bug report on Slack. Paste console logs or anything that helps me solve your issue.

## Installation

Use Python with a virtual environment. From the repository folder, install the
package and its dependencies in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install .
```

On macOS or Linux, create and activate the environment with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

The installation adds the `nanoobjectdetection` command. It can be run from
any working directory and does not require editing `main.py`. To install an
editable development copy instead, use `python -m pip install -e .` from the
repository folder.

### Python example

After installing the package, run the included `example.py` from any directory.
Use absolute paths, or paths relative to the directory where you run the
command:

```powershell
python "D:\Python\NanoObjectDetection\example.py" `
  --input-file "C:\data\my_stack.tif" `
  --output-dir "C:\data\my_analysis" `
  --background-stack-storage disk `
  --processing-bit-depth 8 `
  --dimension all
```

The script imports `run` from the installed `nanoobjectdetection_2_0`
package.
Its `PARAMETERS` dictionary and processing settings near the top of
`example.py` expose the analysis configuration for editing. Omit
`--parameter-file` to use those settings; optionally add
`--parameter-file "C:\data\parameter.json"` to use a legacy JSON instead.
Input/output paths can be set in the script or passed as options.
Run `python "D:\Python\NanoObjectDetection\example.py" --help` for options.

### Optional GPU preprocessing

GPU mode requires a supported NVIDIA GPU, a compatible CUDA runtime, and the
matching CuPy package. For CUDA 12.x, install the package's GPU extra:

```bash
python -m pip install ".[gpu]"
```

If your CUDA version differs, install the CuPy package matching your CUDA
runtime instead. The pipeline reports an error if GPU mode is requested but a
usable CUDA device is unavailable. Trackpy localization, linking, and MSD
fitting remain CPU operations.

## Run from an IDE

The source `main.py` defaults to the included example TIFF and can be run
directly from VS Code, PyCharm, or another Python IDE:

1. Open the repository folder in your IDE.
2. Select the Python interpreter from the `.venv` environment created above.
   In VS Code, use **Python: Select Interpreter**; in PyCharm, select the
   project interpreter in the Python interpreter settings.
3. Run `main.py` with the IDE's **Run Python File** or **Run** action to
   analyze the included example.

For your own data, use the installed command with a parameter JSON rather than
editing `main.py`. Review camera calibration, acquisition rate, temperature,
viscosity, channel dimensions, ROI, and Trackpy settings in that JSON.

## Run from a terminal

Run an installed analysis from any folder in PowerShell:

```powershell
nanoobjectdetection `
  --parameter-file "C:\data\parameter.json" `
  --input-file "C:\data\my_stack.tif" `
  --output-dir "C:\data\my_analysis" `
  --processing-mode single_stack `
  --background-stack-storage disk `
  --processing-bit-depth 8
```

To install the package from another folder or drive, use its repository path:

```powershell
python -m pip install "D:\Python\NanoObjectDetection"
```

The source script can also be run with options from the repository folder:

```powershell
python .\main.py `
  --input-file "C:\data\my_stack.tif" `
  --output-dir "C:\data\my_analysis" `
  --processing-mode single_stack `
  --dimension 2d `
  --background-stack-storage disk
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
| `--background-stack-storage memory` | Keep the background stack used for subtraction in RAM (default). |
| `--background-stack-storage disk` | Use a temporary disk-backed stack; needs free space approximately equal to the selected-depth ROI stack. |
| `--processing-bit-depth 16` | Preserve the input pixel values (default). |
| `--processing-bit-depth 8` | For uint16 input, keep the high 8 bits (`value >> 8`) for stored/raw processing data; halves raw/background stack storage but discards low-bit precision. |
| `--dimension all` | Analyze x, y, and 2d trajectories; this is the default. |
| `--dimension x`, `--dimension y`, `--dimension 2d` | Calculate only the selected trajectory dimension. |
| `--gpu` | Enable GPU preprocessing; requires compatible CuPy/CUDA installation. |
| `--input-file PATH` | Select the TIFF stack to analyze. |
| `--output-dir PATH` | Select where results are written. |
| `--parameter-file PATH` | Load analysis settings from a legacy-format JSON file. |

Disk-backed background storage preserves the same calculation as memory
storage at the selected bit depth and removes its temporary stack file after
localization. It can use more disk I/O and run more slowly than keeping the
stack in RAM.

8-bit processing is opt-in and quantizes uint16 values by shifting right eight
bits; it is not equivalent to the default 16-bit analysis and can change
background subtraction and particle detections. It reduces raw and background
stack storage by about half, but not total process memory because processing
also uses float32 working arrays. For uint8 TIFF input, this option leaves the
pixel values unchanged.

Chunking reduces localization working-block size, but does not make the exact
stack-wide static-background calculation bounded-memory. When that
preprocessing option is enabled, the selected ROI stack must fit in RAM even
in chunked mode. Reduce the ROI or frame range if memory is insufficient.



See the [2.0 documentation](./NANOOBJECTDETECTION_2_0_DOCUMENTATION.md) for
the complete list of supported settings, generated files, assumptions, and
known limitations.
