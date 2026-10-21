# NanoObjectDetection 1.0 Pipeline

## Purpose and scope

The 1.0 implementation is the original Python 3.9-oriented package in
`nanoobjectdetection_1_0/`. The complete example workflow is
`/path/to/example/auswertung.py`; its settings are stored in
`/path/to/example/parameter.json`.
The 1.0 code loads the complete image stack into memory and is therefore
substantially more memory-intensive than the 2.0 blockwise implementation.

The workflow estimates particle diffusion and hydrodynamic diameter from TIFF
video data. It is especially intended for particles diffusing in a cylindrical
fiber or channel. The supplied 100 fps example uses a nominal 20.6 micrometre
channel, 0.3467 micrometres/pixel, water at 293.85 K, and 50 nm gold particles.

## Running the 1.0 example

Edit `File.data_file_name`, `File.data_folder_name`, `File.json`, and the plot
save paths in `/path/to/example/parameter.json` so they point to the local
machine. `/path/to/example/auswertung.py` contains a
original acquisition computer and must also be edited.

The script performs:

```python
nd.CheckSystem.CheckAll(ParameterJsonFile)
settings = nd.handle_data.ReadJson(ParameterJsonFile)
rawframes_np = nd.handle_data.ReadData2Numpy(ParameterJsonFile)
rawframes_super = nd.handle_data.RoiAndSuperSampling(...)
rawframes_pre, rawframes_int, static_background = nd.PreProcessing.Main(...)
obj_all = nd.get_trajectorie.FindSpots(...)
traj_fixed = nd.get_trajectorie.Link(..., SearchFixedParticles=True)
obj_moving = nd.get_trajectorie.RemoveSpotsInNoGoAreas(...)
traj_moving = nd.get_trajectorie.Link(..., SearchFixedParticles=False)
traj_no_drift = nd.Drift.Main(...)
```

`CalcDiameter.Main2` is then called separately for `x`, `y`, and `2d`, and
`nd.visualize.PlotDiameters` writes the corresponding diameter plots and CSVs.
An optional animation can be produced at the end of the example script.

## Parameter file

The JSON is grouped into `Exp`, `Fiber`, `File`, `Logger`, `ROI`,
`PreProcessing`, `Find`, `Link`, `StationaryObjects`, `Split`, `Drift`, `MSD`,
`Time`, `PostProcessing`, `Plot`, and `Animation`.

### Experiment and physical calibration

| Key | Meaning |
|---|---|
| `Exp.NA` | Numerical aperture used by localization-precision theory. |
| `Exp.lambda` | Illumination wavelength in nm. |
| `Exp.fps` | Acquisition rate in frames/s. |
| `Exp.ExposureTime` | Exposure duration in seconds; relevant to motion blur and precision models. |
| `Exp.Microns_per_pixel` | Image-plane calibration. |
| `Exp.Temperature` | Absolute temperature in kelvin. |
| `Exp.Viscosity` | Dynamic viscosity in the 1.0 internal unit convention, approximately 9.8466e-16 N*s/um^2 for the example. |
| `Exp.Viscosity_auto` | Obtain viscosity from the solvent model when enabled. |
| `Exp.solvent` | Solvent name used by automatic viscosity calculation. |

The supplied viscosity corresponds to approximately 0.0009847 Pa s. Unit
consistency is essential; an error here scales every diameter.

### Fiber and geometry

| Key | Meaning |
|---|---|
| `Fiber.Mode` | Optical/channel mode such as `Gaussian` or `Speckle`. |
| `Fiber.Waist` | Gaussian waist parameter where applicable. |
| `Fiber.TubeDiameter_nm` | Nominal channel diameter used in hindrance correction. |
| `Fiber.Center_y_px` | Optional calibrated channel centre for position-aware correction. |

The default 1.0 example uses `TubeDiameter_nm = 20600`. The global 1.0
correction uses this value for every trajectory. Position-aware correction is
optional and depends on the relevant settings and calibration.

### Input, ROI, and preprocessing

| Group/key | Meaning |
|---|---|
| `File.data_type` | Usually `tif_stack` for the example. |
| `File.use_num_frame` | `"all"` or a restricted frame count. |
| `ROI.Apply` | Enable ROI processing. |
| `ROI.x_min/x_max`, `ROI.y_min/y_max` | Spatial bounds. |
| `ROI.frame_min/frame_max` | Frame range limits. |
| `PreProcessing.Remove_CameraOffset` | Subtract a global camera offset estimate. |
| `PreProcessing.Remove_Laserfluctuation` | Correct illumination fluctuations; disabled in the reference example. |
| `PreProcessing.Remove_StaticBackground` | Estimate and subtract static background. |
| `PreProcessing.BackgroundCores` | Number of workers for static-background estimation. Use `1` for large TIFF stacks to avoid duplicating multi-gigabyte arrays in worker processes. |
| `PreProcessing.AllowNonPosValues` | Whether negative preprocessed values are retained. |
| `PreProcessing.EnhanceSNR` | Enable PSF/Gaussian-based preprocessing. |
| `PreProcessing.KernelSize` | PSF kernel size or automatic selection. |
| `PreProcessing.ZNCC_min` | Correlation threshold for applicable localization paths. |

The camera offset is estimated globally. Static background removal uses
temporal intensity statistics designed to reduce the influence of moving bright
particles. These assumptions fail with strong illumination gradients, changing
background structure, saturation, or substantial bleaching.

For large stacks, `BackgroundCores = 1` is the safe default. The 1.0
implementation converts the complete stack to `float32`; parallel workers can
then duplicate large line slices and cause the operating system to terminate
the process for out-of-memory. Increasing this value can improve speed only
when sufficient RAM is available.

### Detection and linking

| Group/key | Meaning |
|---|---|
| `Find.tp_diameter` | Trackpy feature diameter in pixels; reference value 7. |
| `Find.tp_minmass` | Trackpy integrated-mass threshold; reference value 358. |
| `Find.tp_separation` | Minimum feature separation; reference value 20 pixels. |
| `Find.tp_percentile` | Detection percentile; reference value 20. |
| `Link.Dark time` | Maximum gap allowed during linking; reference value 1 frame. |
| `Link.Max displacement` | Maximum linking displacement; reference value 10 pixels. |
| `Link.Max displacement fix` | Smaller search range for stationary objects; reference value 0.3 pixels. |
| `Link.Min tracking frames before drift` | Minimum moving-track length before drift; reference value 10. |
| `Link.Min_tracking_frames` | Final minimum track length; reference value 5. |

The numeric meaning of `tp_minmass` is specific to the 1.0 preprocessing
and intensity representation. It should not be copied directly to the 2.0
pipeline, where the reference equivalent was approximately 1000 after global
normalization.

### Stationary objects, splitting, and drift

| Group/key | Meaning |
|---|---|
| `StationaryObjects.Analyze fixed spots` | Run the stationary-object pass. |
| `StationaryObjects.Min distance to stationary object` | Exclusion radius around fixed spots. |
| `Split.Max_traj_length` | Split tracks longer than this length. |
| `Split.Max_traj_length_keep_tail` | Keep the final short fragment. |
| `Split.IntensityJump` | Split at intensity discontinuities. |
| `Split.Max rel median intensity step` | Relative intensity-jump threshold. |
| `Drift.Drift rolling window size` | Smoothing window for drift estimation. |
| `Drift.Min particle per block` | Minimum particles required for a drift block. |
| `Drift.Apply` | Enable drift correction. |

The reference workflow first identifies fixed tracks, removes nearby moving
detections, links moving particles, and then subtracts global drift. Fixed
objects can also be contaminants; no-go filtering can remove valid particles
near them.

## MSD, diffusion, and diameter physics

For a trajectory coordinate x(t), the 1.0 code calculates:

```text
MSD(tau) = average( (x(t+tau) - x(t))^2 )
```

For one coordinate:

```text
MSD = 2*D*tau + b
```

and for two dimensions:

```text
MSD = 4*D*tau + b
```

The slope is fitted over the configured lag range. The reference settings use
`MSD.lagtimes_min = 1`, `MSD.lagtimes_max = 3`, while automatic lag selection
can be enabled through `Amount lagtimes auto`. `Amount summands` is a minimum
statistical-data requirement at each lag. `Estimate_X` selects the localization
precision model used when optimizing lag selection.

The unconfined Stokes--Einstein inversion is:

```text
d = kB*T/(3*pi*eta*D)
```

The 1.0 code can invert a hindrance factor using
`MSD.EstimateHindranceFactor`. The hindrance polynomial is a Dechadilok--Deen-
style approximation based on particle/channel diameter ratio. With position-
aware correction enabled, `HindrancePositionAware` and the calibrated
transverse centre are used to estimate a local available channel diameter.

### Brownian-motion and post-processing checks

| Key | Meaning |
|---|---|
| `MSD.CheckBrownianMotion` | Apply the Kolmogorov--Smirnov Brownian-motion screening. |
| `MSD.EvalOnlyLongestTraj` | Evaluate only the longest trajectory when enabled. |
| `PostProcessing.MinimalBrightness` | Exclude dim particles. |
| `PostProcessing.MaxDiameter` | Exclude diameters above a chosen limit; zero disables the limit. |
| `PostProcessing.ForceUltraUniform` | Apply an additional distribution-based uniformity filter. |

The KS test is a statistical screen, not a physical proof of Brownian motion.
The longest-track option changes the population being reported and should not
be confused with length weighting.

## Plot and export settings

Important `Plot` keys are:

| Key | Meaning |
|---|---|
| `Plot.MSD_fit` | Display/save MSD fits. |
| `Plot.Histogramm` | Produce diameter histograms. |
| `Plot.Histogramm_abs_or_rel` | Unweighted (`abs`), trajectory-length-weighted (`rel`), or both. |
| `Plot.Histogramm_Fit_1_Particle` | Fit a distribution under the uniform-particle assumption. |
| `Plot.Histogramm_Bin_size_nm` | Histogram bin width. |
| `Plot.Histogramm_Bins` | Number of histogram bins where applicable. |
| `Plot.Histogramm_min/max` | Explicit diameter plot limits. |
| `Plot.DiamOverTraj` | Plot diameter against trajectory length. |
| `Plot.SaveFolder` | Output directory. |
| `Plot.save_data2csv` | Export numeric results as CSV. |

The reference analysis writes files such as `sizes_df_lin_long.csv`,
`Diameter_Histogramm-long-unweighted.csv`, `Diameter_Histogramm-long-weighted.csv`,
and `DiameterOverTrajLength_long.csv`.

## Important limitations

The 1.0 workflow loads the full TIFF stack and can exceed available RAM,
especially during background calculation and multiprocessing. It also relies
on machine-specific paths and older Trackpy/NumPy/Pandas behavior.

The MSD fit assumes approximately Brownian motion and a linear MSD regime.
It does not fully model motion blur, localization uncertainty, anomalous
diffusion, flow, wall chemistry, non-circular geometry, or polydispersity.
Hindrance correction is an approximation and is only physically defensible when
the channel geometry, particle size, viscosity, and position calibration are
valid. Detection thresholds depend strongly on background subtraction and
intensity scaling. Inspect trajectory and MSD diagnostics rather than relying
on a single histogram median.

## Reference 100 fps settings

The supplied `/path/to/example/parameter.json` uses `eval_dim = "x"` in the longitudinal
reference result, 100 fps, 0.3467 micrometres/pixel, 293.85 K, a 20.6
micrometre channel, `tp_diameter = 7`, `tp_minmass = 358`,
`tp_separation = 20`, maximum displacement 10 pixels, a 0.3-pixel stationary
search range, 10 frames before drift, and 5 frames for final analysis.
