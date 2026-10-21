# NanoObjectDetection 2.0

## 1. Purpose

NanoObjectDetection 2.0 is the compatibility implementation of the original
NanoObjectDetection 1.0 analysis. It preserves the 1.0 parameter names,
physical conventions, analysis stages, trajectory logic, and legacy output
names while adding two execution modes:

1. blockwise TIFF storage and localization for large inputs;
2. optional CUDA preprocessing and uint16 conversion.

The original 1.0 package remains available in `nanoobjectdetection_1_0/`.
Version 2.0 is the active pipeline described in this guide.

The design principle is **same scientific settings, different execution
backend**. Changing from `single_stack` to `chunked` should not require
changing `parameter.json`. GPU processing is an acceleration of preprocessing,
not a different tracking algorithm.

## 2. Quick start

Install CPU dependencies:

```bash
/path/to/python -m pip install -r /path/to/NanoObjectDetection/requirements-2.0.txt
```

Run using the parameter values embedded in `main.py`:

```bash
/path/to/python /path/to/NanoObjectDetection/main.py
```

Every completed run writes the effective legacy and 2.0 settings to
`OUTPUT_DIR/parameters.json`.

Run with a legacy-format parameter JSON in chunked CPU mode:

```bash
/path/to/python /path/to/NanoObjectDetection/main.py \
  --parameter-file /path/to/example/parameter.json \
  --processing-mode chunked
```

Run in single-stack mode:

```bash
/path/to/python /path/to/NanoObjectDetection/main.py \
  --parameter-file /path/to/example/parameter.json \
  --processing-mode single_stack
```

Run with GPU preprocessing:

```bash
/path/to/python /path/to/NanoObjectDetection/main.py \
  --parameter-file /path/to/example/parameter.json \
  --processing-mode chunked --gpu
```

GPU mode requires a compatible CuPy package and CUDA runtime. The program
fails explicitly if `--gpu` is requested but no CUDA device is available.

### Configuration and command-line overrides

`main.py` contains the default experiment constants and the `PARAMETERS`
dictionary, which uses the legacy JSON section/key names. If
`--parameter-file` is omitted, these in-code values are translated into the
2.0 configuration. If a JSON file is supplied, its supported analysis values
are loaded instead. `INPUT_FILE` and `OUTPUT_DIR` in `main.py` (or the
`--input-file` and `--output-dir` flags) select the source and destination;
they override the JSON `File.data_file_name` and `Plot.SaveFolder`. The
`PROCESSING_MODE` constant (or `--processing-mode`) selects the execution
mode; it defaults to `single_stack`. The command-line script runs on CPU
unless `--gpu` is supplied. `CHUNK_FRAMES`, `CHUNK_OVERLAP_FRAMES`, and
`LOCATE_WORKERS` are set in `main.py`, not the legacy JSON file.

Only settings that have a 2.0 implementation affect the analysis. Several
legacy JSON keys are retained for compatibility but are explicitly unused;
they are identified in the comments in `main.py`. In particular, plotting
options from the old `Plot` section do not configure the 2.0 plots.

## 3. Processing stages

The 2.0 workflow follows the same conceptual order as 1.0:

1. Read the TIFF stack and apply the configured frame and spatial ROI.
2. Convert working data to float32.
3. Optionally remove camera offset.
4. Optionally normalize laser fluctuations frame by frame.
5. Estimate and subtract the temporal static background.
6. Optionally apply Gaussian SNR enhancement.
7. Scale to uint16 for Trackpy.
8. Localize particles with Trackpy.
9. Link detections into moving and stationary trajectories.
10. Remove detections in stationary-particle no-go regions.
11. Split or reject trajectories according to `Split`.
12. Estimate global drift and subtract accumulated drift.
13. Calculate MSD and fit diffusion.
14. Convert diffusion to hydrodynamic diameter.
15. Apply optional hindrance and diameter filters.
16. Write 2.0 tables and 1.0-compatible aliases.
17. Create legacy-style plots and export plot data.

Chunked mode writes the selected source frames and ROI to `image_store/`,
then localizes blocks with overlap if configured. Single-stack mode reads
source TIFF frames directly and does not create that chunk store. In both
modes, detections are combined before tracking and drift estimation; tracks
and drift are not estimated independently per block.

When static background removal is enabled, both modes calculate one exact
legacy background image: for each pixel, sort its values over the entire
selected stack and average the values from the 30th through the 70th
percentiles. The same image is subtracted from all blocks. It is saved as
`global_background.npy`. To preserve this whole-stack calculation without
temporary duplicate stacks, 2.0 preallocates the selected ROI as a uint16
array and reduces it in pixel strips. **This requires the complete selected
ROI stack to fit in RAM even in chunked mode.** Chunking limits localization
working blocks and keeps a disk-backed copy; it does not make this exact
global background calculation bounded-memory. On Linux, when `/proc/meminfo` is available, the code checks available memory
and raises an error when the estimated stack exceeds 80% of it. Reduce the
ROI or frame range if needed. With static background removal disabled, input
frames are streamed in processing blocks instead.

## 4. Version 1.0 parameter compatibility

### 4.1 `Exp`

| Parameter | Meaning |
|---|---|
| `NA` | Numerical aperture. Used by PSF-derived automatic settings and the `Estimate_X = "Theory"` MSD option. |
| `lambda` | Illumination wavelength in nm. Used by PSF-derived automatic settings and the `Estimate_X = "Theory"` MSD option. |
| `fps` | Acquisition frame rate. Used for time axes and MSD. |
| `ExposureTime` | Exposure time in seconds. Used by `Estimate_X = "Theory"`; there is no separate motion-blur correction in the ordinary MSD fit. |
| `Microns_per_pixel` | Spatial calibration used for diffusion and diameter. |
| `Temperature` | Temperature in K. |
| `Viscosity` | 1.0 viscosity convention, converted to Pa s internally. |

Wavelength and NA also determine the PSF estimate used by automatic Trackpy
diameter and Gaussian SNR-kernel sizing. Exposure time, NA, and wavelength are
used for the theory-based localization-error input to automatic MSD lag
selection; see [Automatic parameters](#automatic-parameters).

### 4.2 `Fiber`

| Parameter | Meaning |
|---|---|
| `TubeDiameter_nm` | Nominal channel diameter for hindrance correction. |
| `Center_y_px` | Channel centre in pixels, or `"unknown"` for automatic estimation. |
| `Mode`, `Waist` | Preserved experiment descriptors. The current position model assumes a circular channel and does not use Gaussian intensity fitting. |

### 4.3 `ROI`

`x_min`, `x_max`, `y_min`, `y_max` define the spatial slice. `frame_min` and
`frame_max` define the half-open frame interval. Large sentinel maxima are
interpreted as “until the end”.

### 4.4 `PreProcessing`

| Parameter | Effect |
|---|---|
| `Remove_CameraOffset` | Subtract the minimum intensity of the working data. |
| `Remove_Laserfluctuation` | Normalize each frame by its mean intensity relative to the median frame intensity. |
| `Remove_StaticBackground` | Estimate one exact stack-wide temporal background and subtract it consistently. |
| `BackgroundCores` | Retained for 1.0 compatibility; not used by 2.0. |
| `AllowNonPosValues` | Retained for compatibility; 2.0 clips processed values to non-negative and scales them to uint16. |
| `EnhanceSNR` | Apply the 1.0-compatible Gaussian spatial smoothing. |

The default static background is the trimmed temporal mean: values from the
30th through the 70th percentile are averaged independently for each pixel
over the full selected stack. This suppresses transient particles while
retaining stationary background structure. `CHUNK_FRAMES` controls processing
block size, but does not remove the full-stack RAM requirement for this exact
background. The saved `global_background.npy` is an intermediate diagnostic;
it is recalculated on each run.

### 4.5 `Find`

| Parameter | Effect |
|---|---|
| `tp_diameter` | Odd Trackpy feature diameter in pixels. |
| `tp_minmass` | Integrated feature-mass threshold. |
| `tp_separation` | Minimum separation between features. |
| `tp_percentile` | Percentile threshold used by Trackpy. |

These values retain the 1.0 intensity scaling convention. A threshold tuned
for another preprocessing method is not interchangeable.

`Help["Bead size"]` enables PSF-derived diameter and
`Help["Bead brightness"]` enables automatic minmass calibration; details are
in [Automatic parameters](#automatic-parameters). If both are disabled,
`tp_diameter` and `tp_minmass` are used directly.

### 4.6 `Link` and stationary objects

`Dark time` is passed to Trackpy as memory. `Max displacement` is the primary
search range. `Max displacement fix` and `Dwell time stationary objects`
control the stationary-particle pass. `Analyze fixed spots` enables that pass,
and `Min distance to stationary object` sets the exclusion radius around
detected fixed spots.

`Min tracking frames before drift` is the minimum track length used to estimate
drift. `Min_tracking_frames` is the minimum number of measured positions a
trajectory must have to be retained for sizing and analysis; it is applied
after drift correction. Thus it is the main setting for minimum valid
trajectory length. The linker retains enough shorter tracks to permit the
separate drift threshold to work.

When fixed spots are enabled, detections close to their trajectories are
removed before moving-particle linking. This preserves the 1.0 no-go-area
behavior but can remove genuine slow particles near a fixed bright object.

### 4.7 `Split`

`IntensityJump` enables intensity discontinuity splitting.
`Max rel median intensity step` controls the relative jump threshold.
`Max_traj_length` limits trajectory length; a very large 1.0 sentinel means
unlimited. `Max_traj_length_keep_tail` controls whether a short final fragment
is retained.

### 4.8 `Drift`

`Apply` enables or disables drift correction. In global mode, 2.0 uses the
same Trackpy `compute_drift` operation as 1.0, with the configured
`Avg frames`/rolling smoothing semantics. Missing frame values are linearly
interpolated before subtraction. The first frame is the zero-drift reference.
`Avg frames` is the smoothing window passed from `main.py`; saturated positions
are excluded from the estimate.

The exported drift table contains:

```text
frame, drift_x, drift_y
```

The legacy plot displays `drift_y` as transversal and `drift_x` as along fiber,
using accumulated drift in pixels.

`LaminarFlow` remains a compatibility setting. The global compatibility entry
point currently uses the global drift path; a dedicated transversal-flow mode
requires the 1.0 `Drift.TransversalEstimation` algorithm and should not be
silently substituted by a global correction.

### 4.9 `MSD`

`Amount lagtimes auto` selects automatic lag-time optimization when enabled;
otherwise `lagtimes_max` is the maximum lag used for the fit (at least two
lags are required). `lagtimes_min` and `Amount summands` are retained legacy
keys and are not used. `Estimate_X` selects experimental (`Exp`) or
theory-based (`Theory`) localization error for automatic lag optimization.
`CheckBrownianMotion` enables the quality filter, while `KSMethod` and
`KSMinSignificance` set its p-value method and rejection threshold.
`EstimateHindranceFactor` and `HindrancePositionAware` select the hindrance
mode described below.

For one dimension:

```text
MSD(tau) = intercept + 2 * D * tau
```

For two dimensions:

```text
MSD(tau) = intercept + 4 * D * tau
```

The diffusion coefficient is converted using the Stokes-Einstein relation:

```text
d = 2 * kB * T / (6 * pi * eta * D)
```

The fit uses physical time from `fps` and physical distance from
`Microns_per_pixel`.

### 4.10 `PostProcessing` and `Plot`

`MaxDiameter` removes diameters above the configured limit from statistics
and plots while retaining their filter reason in the particle table.
`ForceUltraUniform` (1 = on) applies the 1.0 ultra-uniform hypothesis: particles
are ordered by `valid_frames`, and those whose z-score (distance to the median
divided by a rolling spread, window 100) reaches 2.576 are removed iteratively,
first by diameter and then by diffusion. Removed particles keep a filter reason.
These legacy `Plot` settings are retained in `main.py` for parameter-file
compatibility but do not control the 2.0 plots. 2.0 writes both weighted and
unweighted diameter histograms when there are valid estimates. The histograms
include a reciprocal-space Gaussian fit and summary statistics. The
diameter-versus-trajectory-length plot uses explicit legacy-style tick labels
on logarithmic axes.

### 4.11 `SingleParticle`

Enable `SingleParticle["Enabled"]` only when analyzing a video expected to
contain one particle that is repeatedly lost and re-detected. The mode first
links detections normally, retains short fragments for stitching, then joins
plausible fragments using the configured gap, distance, and median-mass
limits. Fragments shorter than `Min fragment frames` are discarded.

`Keep` selects `longest` (most detections), `brightest` (greatest summed
Trackpy mass), or `all` stitched chains. Fragment matching is greedy, not a
globally optimized assignment; permissive distance/gap settings can join
different nearby spots. An ROI around the target is often safer. Global drift
correction is disabled in this mode because one particle's motion cannot be
separated from common drift using that particle alone. MSD calculations use
only observed frame pairs; missing frames in a stitched trajectory are not
interpolated.

## 5. Hindrance correction

Three modes are available:

### `none`

The unconfined Stokes-Einstein diameter is reported.

### `legacy`

One fixed `TubeDiameter_nm` is used for every trajectory, independent of
transverse position. This is the 1.0 global correction.

### `position`

The corrected transverse trajectory position is compared with `Center_y_px`.
For a circular channel, the available local diameter is approximated as twice
the wall distance. The mean local diameter over each trajectory is then used
for the hindrance inversion.

This is a geometric approximation. It is not a measurement of the optical
intensity profile and does not use `Fiber.Mode` or `Fiber.Waist`. A trajectory
outside the nominal channel is clipped to zero available wall distance; such
data should be inspected because the correction becomes physically unreliable.

## 6. Output files and plots

The primary 2.0 files retain descriptive column names, for example
`detections.csv`, `trajectories.csv`, `particle_sizes_x.csv`,
`particle_sizes_y.csv`, and `particle_sizes_2d.csv`. Additional per-dimension
files include corrected trajectories, drift tables, quality results, and
summaries. `analysis_config.json` records the complete legacy parameter
dictionary and the resolved 2.0 configuration, including runtime overrides
and automatically calibrated Trackpy values. It also records run metadata
and analysis results. `results_manifest.json` lists generated output paths;
automatic calibration also writes `auto_parameters.json`.

The effective configuration is also saved as `parameters.json` in the output
directory. It preserves the legacy
section/key layout, reflects the actual input/output paths and automatic
Trackpy calibration values, and adds a `NanoObjectDetection_2_0` section with
the resolved 2.0 configuration, including parameters that have no legacy
equivalent. The legacy JSON loader ignores this additional section, so the
file remains usable as a legacy-format input. If the selected input parameter
file is already named `parameters.json` in the output directory, the exported
copy is named `parameters_used.json` to avoid overwriting the input.

The following legacy-compatible aliases are also written when their source
tables exist:

```text
obj_all.csv
traj_fixed.csv
obj_moving_no_go_area.csv
traj_moving.csv
traj_no_drift.csv
traj_final.csv
sizes_df_lin_long.csv
Diameter_Histogramm-long-weighted.csv
Diameter_Histogramm-long-unweighted.csv
DiameterOverTrajLength_long.csv
```

The legacy plot names are:

```text
Global_Drift.png
Diameter_Histogramm-long-weighted.png
Diameter_Histogramm-long-unweighted.png
DiameterOverTrajLength_long.png
MSD_fit_long.png
Trajectories_long.png
```

The plots are generated once for each evaluated direction (`long`, `trans`,
and `2d`): trajectories, MSD fits, weighted and unweighted diameter
histograms, and diameter versus trajectory length; the global drift plot is
also written. Histogram PNGs show the reciprocal Gaussian fit. The histogram
CSV files are compatibility tables, not binned histogram counts: like
`sizes_df_lin_*` and `DiameterOverTrajLength_*`, each contains the
per-particle size table in the legacy 21-column layout. The weighted
histogram counts each particle once per valid frame; the unweighted histogram
counts each particle once.

## 7. Chunked and GPU limitations

Chunk overlap prevents many boundary-localization losses; overlapping
detections are de-duplicated. Results may still differ from a single
full-stack localization because Trackpy sees blocks independently and
numerical scaling can differ. Compare detection counts, trajectory lengths,
drift, and diameter distributions when changing block size or overlap.

Both modes localize in processing blocks. `single_stack` reads the TIFF
directly, while `chunked` also stores chunk files on disk. When exact
stack-wide background subtraction is enabled, both modes retain the complete
selected ROI in RAM as described above. Choose the mode based on storage and
localization needs, not on the assumption that chunking removes this memory
requirement.

GPU acceleration currently covers block preprocessing and uint16 conversion;
global background and scaling passes, Trackpy localization/linking, and MSD
fitting remain CPU operations. GPU output can differ slightly at threshold
boundaries because of floating-point operation order.

## 8. Reproducibility checklist

Record all of the following with each result:

1. the exact `parameter.json`, or the `main.py` version used for in-code settings;
2. processing mode;
3. block size (`CHUNK_FRAMES`) and overlap (`CHUNK_OVERLAP_FRAMES`);
4. GPU flag and CuPy/CUDA version;
5. input TIFF filename and frame range;
6. output `analysis_config.json`;
7. detection and trajectory counts;
8. the drift table and the diameter summary.

For a 1.0 comparison, run 2.0 with identical settings in `single_stack` CPU
mode first, then compare `chunked` CPU mode, and only then enable GPU mode.

## Automatic parameters (ported from 1.0)

- `Help["Bead size"] = 1`: `tp_diameter` is derived from the PSF width (sigma = 0.21·lambda/NA, 4 sigma in px, times sqrt(2) with `EnhanceSNR`, rounded up to an odd number).
- `Help["Bead brightness"] = 1`: `tp_minmass` is calibrated. `TryFrames` evenly spaced frames are searched by ZNCC (`ZNCC_min`); trackpy minmass is then tuned until it reproduces that particle set, and 0.9 × the optimum is used. The result is saved to `auto_parameters.json`.
- `PreProcessing["KernelSize"]`: `"auto"` uses the PSF sigma for the `EnhanceSNR` Gaussian filter, a number sets sigma in px.
- `MSD["Estimate_X"] = "Theory"`: the reduced localisation error in the lag-time optimisation comes from NA, lambda and ExposureTime (Michalet & Berglund; D = 1 µm²/s, 1 photon as in 1.0) instead of the MSD fit offset. Not ported: the 1.0 `FindChannel` mask and the interactive diameter selection.

If all candidate trajectories are rejected by analysis filters (for example,
the Brownian-motion quality test), 2.0 reports that condition and skips
sizing/plot generation for that dimension rather than failing with an
unhandled empty-result error.
