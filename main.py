"""NanoObjectDetection 2.0: 1.0-compatible chunked/GPU analysis. 
1.0 - Ronny Foerster and Jisoo 2019,
2.0 - Tim Stechel 10/2026
-- Added Chunked processing and GPU support 
-- New Hindrance model with Position Dependence
-- Stitched trajectory for single particle detection"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
import resource
from pathlib import Path
import threading
import time

from nanoobjectdetection_2_0.legacy_config import (
    load_parameter_file,
    load_parameter_settings,
)
from nanoobjectdetection_2_0.auto_parameters import trackpy_diameter
from nanoobjectdetection_2_0.legacy_outputs import export_legacy_aliases
from nanoobjectdetection_2_0.pipeline import AnalysisPipeline
from nanoobjectdetection_2_0.storage import tiff_info
ROOT = Path(__file__).resolve().parent

## Processing parameters
PROCESSING_MODE = "single_stack"  # "chunked" = image store processed in blocks (large stacks), "single_stack" = frame by frame from the TIFF
INPUT_FILE = ROOT / "example" / "20_6µmKap_50Au_20_7C_100fps_5000fr_700us_V1j.tif"  # image stack to analyse
OUTPUT_DIR = ROOT / "example" / "output_single_cpu"  # results folder
LEGACY_PARAMETER_FILE = None #ROOT / "example" / "parameters.json"  # path to legacy parameter file
GPU_ENABLED = False  # True = GPU preprocessing (needs CuPy/CUDA), False = CPU #currently no point in using gpu since trackpy is CPU-bound
CHUNK_FRAMES = int((5000+500)/3)  # frames per block in chunked mode; lower if memory runs out
LOCATE_WORKERS = 0  # CPU processes for trackpy localization (the real bottleneck); 1 = serial, 0 = all cores, sensible: 4-8
CHUNK_OVERLAP_FRAMES = 20  # frames shared by neighbouring blocks; 0 <= value < CHUNK_FRAMES

## Camera parameters
CAMERA_MICRONS_PER_PIXEL = 0.3467  # pixel size in the sample plane in um 
NA = 0.25  # numerical aperture of the objective (typ. 0.1-1.4); used for PSF width (auto diameter/kernel) and Estimate_X "Theory"
LAMBDA = 532.0  # laser wavelength in nm; used together with NA (see NA)
FPS=100.0

## Sample parameters
ESTIMATED_PARTICLE_SIZE=50.0 # estimated particle size in nm (should be roughly the nominal value - 10%)
TEMPERATURE_K = 293.85  # sample temperature in K (20 C = 293.15)
VISCOSITY = 9.846562548618592e-16  # solvent viscosity in legacy units (kg/(um*s)); water 20 C ~ 1e-15
CHANNEL_DIAMETER_NM = 20600.0  # channel/fibre inner diameter in nm; needed for hindrance correction (0 = off)

## Trackpy Parameters
AUTO_TP = False  # automatically determine trackpy parameters, if false must provide tp_diameter and tp_minmass, true requires NA and lambda to calculate the PSF width
ENHANCE_SNR = False
def estimate_tp_diameter(
    numerical_aperture: float,
    wavelength_nm: float,
    pixel_size_um: float,
    enhance_snr: bool = False,
) -> int:
    """Estimate Trackpy's odd feature diameter from the optical PSF in pixels."""
    if numerical_aperture <= 0 or wavelength_nm <= 0 or pixel_size_um <= 0:
        raise ValueError("NA, wavelength, and pixel size must be positive")
    return trackpy_diameter(
        numerical_aperture,
        wavelength_nm,
        pixel_size_um,
        enhance_snr,
    )
TP_DIAMETER = 7# estimate_tp_diameter(NA, LAMBDA, CAMERA_MICRONS_PER_PIXEL, ENHANCE_SNR)  # only used if AUTO_TP is False #7 is sensible value
TP_MINMASS = 358  # trackpy minimum mass (only used if AUTO_TP is False)

## Other Processing Parameters
CORRECT_DRIFT = True
SATURATED_PIXEL_VALUE = "No Saturation"
ESTIMATE_HINDRANCE = True  # whether to estimate hindrance correction (True/False) needs TubeDiameter_nm 
HINDERANCE_POSITION_AWARE = False  # whether to consider the particle position within the channel for hindrance correction (True/False)

## Trajectory filtering
MIN_TRAJECTORY_LENGTH = 5  # minimum number of frames for a trajectory to be considered valid
MAX_DIAMETER = 0  # maximum particle diameter in nm (trajectories with larger particles are discarded) 0=off
FORCE_ULTRA_UNIFORM = False  # whether to force ultra uniform particle distribution (Z-score) (True/False)

## Single particle stitching mode
SINGLE_PARTICLE_ENABLED = False  # whether to enable single particle stitching mode (True/False)
SINGLE_PARTICLE_KEEP = "longest"  # which stitched trajectory to analyse: "longest" = most detections, "brightest" = largest total mass, "all" = every stitched trajectory

PARAMETERS = {
    # Keys marked "[unused]" are kept for 1.0 JSON compatibility but ignored by the 2.0 pipeline.
    "Exp": {
        "NA": NA,  # numerical aperture of the objective (typ. 0.1-1.4); used for PSF width (auto diameter/kernel) and Estimate_X "Theory"
        "lambda": LAMBDA,  # laser wavelength in nm; used together with NA (see NA)
        "fps": FPS,  # camera frame rate in frames/s
        "ExposureTime": 0.0007,  # camera exposure time in s (must be <= 1/fps); used by Estimate_X "Theory"
        "Microns_per_pixel": CAMERA_MICRONS_PER_PIXEL,  # pixel size in the sample plane in um
        "Temperature": TEMPERATURE_K,  # sample temperature in K (20 C = 293.15)
        "Viscosity": VISCOSITY,  # solvent viscosity in legacy units (kg/(um*s))
        "Viscosity_auto": 0,  # [unused] 1 = compute viscosity from Temperature/solvent in 1.0
        "solvent": "water",  # [unused] solvent name for Viscosity_auto: "water" or "glycerol"
    },
    "Fiber": {
        "Mode": "Gaussian",  # [unused] illumination profile in 1.0: "Gaussian" or "Flat"
        "Waist": 10.3,  # [unused] laser waist in um
        "TubeDiameter_nm": CHANNEL_DIAMETER_NM,  # channel/fibre inner diameter in nm; needed for hindrance correction (0 = off)
        "Center_y_px": "unknown",  # channel centre row in px; "unknown" = estimate from image, or give a number
    },
    "File": {
        "data_file_name": f"{INPUT_FILE}",  # image stack path; overridden by INPUT_FILE / --input-file
        "data_folder_name": f"{ROOT / 'example'}",  # [unused] data folder
        "data_type": "tif_stack",  # [unused] input format; 2.0 reads TIFF stacks only
        "use_num_frame": "all",  # [unused] use ROI.frame_min/frame_max instead
        "DefaultParameterJsonFile": (  # [unused] 1.0 defaults file
            f"{ROOT / 'example' / 'NanoObjectDetection_1_0' / 'default_json' / 'default_json.json'}"
        ),
        "json": f"{ROOT / 'example' / 'parameter.json'}",  # [unused] path of the 1.0 parameter file
    },
    "Logger": {"level": "info"},  # [unused] 1.0 log level: "debug", "info", "warning", "error"
    "ROI": {
        "Apply": 1,  # [unused] ROI is always applied; use large max values to disable
        "Save": 0,  # [unused] 1.0 option to save the cropped stack
        "x_min": 0,  # first column in px (0 = left edge)
        "x_max": 100000,  # last column in px; >= 1e9 = full width, larger than image is clipped
        "y_min": 0,  # first row in px (0 = top edge)
        "y_max": 100000,  # last row in px; larger than image is clipped
        "frame_min": 0,  # first frame to analyse
        "frame_max": 6000,  # last frame (exclusive); >= 1e6 = all frames
    },
    "PreProcessing": {
        "Remove_CameraOffset": 1,  # 1 = subtract camera offset, 0 = off
        "Remove_Laserfluctuation": 0,  # 1 = normalise per-frame laser intensity, 0 = off
        "Remove_StaticBackground": 1,  # 1 = subtract static background (global over all chunks), 0 = off
        "BackgroundCores": 1,  # [unused] CPU cores for background estimation in 1.0
        "AllowNonPosValues": 0,  # [unused] 1.0 option for negative pixels
        "EnhanceSNR": ENHANCE_SNR,  # 1 = SNR enhancement filter (then Trackpy's own preprocessing is skipped), 0 = off
        "KernelSize": "auto",  # Gaussian sigma (px) of the SNR filter: "auto" = PSF width 0.21*lambda/NA/pixel size, or a number such as 1.0
        "ZNCC_min": 0.7,  # ZNCC correlation threshold (0-1) for reference particles in the minmass calibration; sensible 0.6-0.8
    },
    "Help": {  # automatic parameter helpers; only Bead brightness, Bead size and TryFrames are used in 2.0
        "ROI": 0,  # 1 = help choosing ROI
        "Bead brightness": AUTO_TP,  # 1 = calibrate tp_minmass automatically via ZNCC (result x0.9), 0 = use tp_minmass
        "Bead size": AUTO_TP,  # 1 = set tp_diameter from PSF theory (NA, lambda, pixel size), 0 = use tp_diameter
        "TryFrames": 30,  # number of evenly spaced frames sampled by the minmass calibration; more = more robust, slower
        "Separation": 1,  # 1 = auto-estimate tp_separation
        "GuessLowestDiameter_nm": ESTIMATED_PARTICLE_SIZE,  # smallest expected diameter in nm for the helpers
        "Drift": CORRECT_DRIFT,  # 1 = auto-estimate drift averaging
        "Intensity Jump": 1,  # 1 = auto-estimate intensity jump threshold
    },
    "Find": {
        "tp_diameter": TP_DIAMETER,  # Trackpy feature size in px; odd integer >= 3, slightly above the particle spot size
        "SaturatedPixelValue": SATURATED_PIXEL_VALUE,  # "No Saturation" or raw pixel value (e.g. 255, 4095, 65535); spots reaching it are excluded from the MSD fit
        "tp_separation": 20,  # min distance between spots in px
        "tp_percentile": 20,  # Trackpy brightness percentile filter, 0-100
    },
    "Link": {
        "Dark time": 1,  # frames a particle may vanish and still be linked (Trackpy memory); 0 = none
        "Dwell time stationary objects": 100,  # min frames for an object to count as stationary
        "Max displacement": 10,  # max px jump between frames when linking moving particles
        "Max displacement fix": 0.3,  # max px jump when searching stationary objects
        "Min tracking frames before drift": 10.0,  # min trajectory length used for drift estimation
        "Min_tracking_frames": MIN_TRAJECTORY_LENGTH,  # min trajectory length kept after drift correction
    },
    "StationaryObjects": {
        "Analyze fixed spots": 1,  # 1 = find stationary objects and exclude their neighbourhood, 0 = off
        "Min distance to stationary object": 20,  # exclusion radius in px around stationary objects
    },
    "Split": {
        "Max_traj_length": 1000000.0,  # split trajectories longer than this many frames; >= 1e8 = never
        "Max_traj_length_keep_tail": 0,  # 1 = keep the short remainder after splitting, 0 = discard it
        "IntensityJump": 0,  # 1 = split trajectories at brightness jumps (clusters/aggregates), 0 = off
        "Max rel median intensity step": 0.906,  # relative brightness step that triggers a split, 0-1
        "ParticleWithLongestTrajBeforeSplit": 976,  # [unused] 1.0 bookkeeping value
    },
    "Drift": {
        "Apply": CORRECT_DRIFT,  # 1 = correct drift, 0 = off
        "Avg frames": 13,  # frames averaged for the global drift estimate (trackpy smoothing)
        "LaminarFlow": 0,  # [unused] 1 = per-position drift for laminar flow in 1.0
        "Drift rolling window size": 5,  # rolling window (frames) used to smooth the drift
        "Min particle per block": 40.0,  # min particles per block for a drift estimate
    },
    "SingleParticle": {  # video contains only one particle: stitch track fragments together
        "Enabled": SINGLE_PARTICLE_ENABLED,  # 1 = link without minimum length, stitch fragments, switch drift correction off; 0 = normal multi-particle analysis
        "Keep": SINGLE_PARTICLE_KEEP,  # which stitched trajectory to analyse: "longest" = most detections, "brightest" = largest total mass, "all" = every stitched trajectory
        "Max gap frames": 300,  # longest dropout (frames) that may be bridged; sensible 50-1000
        "Max distance px": 15.0,  # allowed jump between the end of one fragment and the start of the next, px; sensible 5-30
        "Distance growth px per sqrt frame": 0.5,  # extra allowed jump per sqrt(gap frames) to account for diffusion during the dropout; 0 = fixed distance
        "Max mass ratio": 3.0,  # fragments are only joined if their median brightness (mass) agrees within this factor; >= 1, sensible 2-5
        "Min fragment frames": 3,  # fragments with fewer detections are treated as noise and ignored; sensible 3-10
    },
    "MSD": {
        "Amount summands": 10,  # [unused] 1.0 value for block-wise MSD averaging
        "Amount lagtimes auto": 1,  # 1 = optimise lag times per track (Michalet 2012; legacy behaviour), 0 = fixed lagtimes_max
        "Estimate_X": "Exp",  # localisation error used in lag-time optimisation: "Exp" = from MSD fit offset, "Theory" = from NA, lambda, ExposureTime (Michalet & Berglund)
        "lagtimes_min": 1,  # [unused] first lag time; always 1
        "lagtimes_max": 3,  # fixed number of lag times (>= 2) used when Amount lagtimes auto = 0
        "EstimateHindranceFactor": ESTIMATE_HINDRANCE,  # 1 = correct diameter for wall hindrance (needs TubeDiameter_nm), 0 = off
        "EvalOnlyLongestTraj": 0,  # [unused] 1 = evaluate only the longest trajectory in 1.0
        "CheckBrownianMotion": 1,  # 1 = reject tracks failing the Kolmogorov-Smirnov test, 0 = keep all
        "KSMethod": "legacy",  # KS p-value: "legacy" = 2*exp(-2*N*D^2) like 1.0, "exact" = full Kolmogorov series with small-sample correction
        "KSMinSignificance": 0.01,  # reject tracks with p below this; sensible 0.001-0.1, higher = stricter
        "HindrancePositionAware": HINDERANCE_POSITION_AWARE,  # 1 = position-dependent hindrance from y position (overrides EstimateHindranceFactor), 0 = off
    },
    "Time": {"DoRolling": 0, "Frames": 100, "skip_image_fac": 10},  # [unused] 1.0 time-resolved analysis
    "PostProcessing": {
        "MinimalBrightness": 0,  # [unused] min brightness filter in 1.0
        "MaxDiameter": MAX_DIAMETER,  # discard diameters above this in nm; 0 = off
        "ForceUltraUniform": FORCE_ULTRA_UNIFORM,  # 1 = drop particles whose diameter/diffusion z-score vs. the median is >= 2.576 (ultra-uniform sample), 0 = off
    },
    "Plot": {  # only SaveFolder is read by 2.0; 2.0 creates its own plots
        "Background": 0,  # [unused] 1 = plot background
        "Laserfluctuation": 0,  # [unused] 1 = plot laser fluctuation
        "MSD_fit": 0,  # [unused] 1 = plot MSD fits
        "Histogramm": 1,  # [unused] 1 = diameter histogram
        "Histogramm_abs_or_rel": "both",  # [unused] "abs", "rel" or "both"
        "Histogramm_Fit_1_Particle": 1,  # [unused] 1 = fit single-particle distribution
        "Histogramm_time": 0,  # [unused] 1 = histogram over time
        "Histogramm_Bin_size_nm": 0.5,  # [unused] bin width in nm
        "Histogramm_Bins": 2592,  # [unused] number of bins
        "Histogramm_min_max_auto": 0,  # [unused] 1 = automatic histogram range
        "Histogramm_min": 10,  # [unused] histogram minimum in nm
        "Histogramm_max": 300.0,  # [unused] histogram maximum in nm
        "DiameterPDF": 0,  # [unused] 1 = diameter PDF plot
        "DiameterPDF_min_max_auto": 1,  # [unused] 1 = automatic PDF range
        "PDF_min": 0.0,  # [unused] PDF minimum in nm
        "PDF_max": 1000.0,  # [unused] PDF maximum in nm
        "PDF_show_fit": 0,  # [unused] 1 = show PDF fit
        "DiamOverTraj": 1,  # [unused] 1 = diameter vs trajectory length plot
        "DiamOverTraj_UseRawMass": "mean",  # [unused] "mean", "median" or "max"
        "fontsize": 14,  # [unused] plot font size
        "SaveFolder": f"{OUTPUT_DIR}",  # output folder; overridden by OUTPUT_DIR / --output-dir
        "save_json": 1,  # [unused] 1 = save plot properties as JSON
        "save_data2csv": 1,  # [unused] 2.0 always writes CSVs
        "dpi": 300,  # [unused] figure resolution
        "SaveProperties": (  # [unused] plot property file    
        ),
    },
    "Animation": {  # [unused] 1.0 animation settings
        "x_min": "auto",  # "auto" or px
        "x_max": "auto",  # "auto" or px
        "y_min": "auto",  # "auto" or px
        "y_max": "auto",  # "auto" or px
        "gamma": 0.5,  # display gamma
        "frames_tot": 5,  # number of frames animated
        "accelerate": 1,  # playback speed factor
    },
}


def _memory_stats() -> tuple[float | None, float | None, float | None]:
    meminfo = Path("/proc/meminfo")
    status = Path("/proc/self/status")
    if not meminfo.exists() or not status.exists():
        return None, None, None

    values = {}
    for line in meminfo.read_text().splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.split()[0]) / 1024**2
    rss = next(
        (
            int(line.split()[1]) / 1024**2
            for line in status.read_text().splitlines()
            if line.startswith("VmRSS:")
        ),
        None,
    )
    return values.get("MemTotal"), values.get("MemAvailable"), rss


def _memory_report() -> str:
    total, available, rss = _memory_stats()
    if total is None or available is None:
        return "RAM stats unavailable on this platform."
    process_usage = f", process RSS {rss:.1f} GB" if rss is not None else ""
    return (
        f"RAM: {available:.1f} GB available of {total:.1f} GB"
        f"{process_usage}."
    )


def _log_stage(stage: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {stage} - {_memory_report()}", flush=True)


def _peak_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def _watch_memory(stop: threading.Event, min_available_gb: float = 1.0) -> None:
    warned = False
    while not stop.wait(2):
        _, available, _ = _memory_stats()
        if available is not None and available < min_available_gb and not warned:
            print(
                "[LOW MEMORY] "
                f"{_memory_report()} If the process disappears abruptly, it may "
                "have been terminated by the OS out-of-memory killer; inspect "
                "the system log (for example, `dmesg | grep -i oom`).",
                flush=True,
            )
            warned = True


def run(parameter_file: Path | None = None, *, input_file: Path = INPUT_FILE,
        output_dir: Path = OUTPUT_DIR, processing_mode: str = PROCESSING_MODE,
        gpu_enabled: bool = GPU_ENABLED) -> dict:
    tiff = tiff_info(input_file)
    shape = tiff["shape"]
    frame_count = shape[0] if len(shape) > 2 else tiff["pages"]
    height, width = shape[-2:]
    _log_stage(
        f"Starting analysis of {input_file} "
        f"({frame_count} frames, {height} x {width} pixels)"
    )
    stop = threading.Event()
    watcher = threading.Thread(target=_watch_memory, args=(stop,), daemon=True)
    watcher.start()
    try:
        return _run_analysis(
            parameter_file,
            input_file=input_file,
            output_dir=output_dir,
            processing_mode=processing_mode,
            gpu_enabled=gpu_enabled,
        )
    except MemoryError:
        print(f"[OUT OF MEMORY] {_memory_report()}", flush=True)
        raise
    finally:
        stop.set()
        watcher.join(timeout=3)
        print(
            f"[PEAK MEMORY] Peak process RSS: {_peak_rss_gib():.2f} GiB",
            flush=True,
        )


def _run_analysis(parameter_file: Path | None = None, *, input_file: Path = INPUT_FILE,
                  output_dir: Path = OUTPUT_DIR, processing_mode: str = PROCESSING_MODE,
                  gpu_enabled: bool = GPU_ENABLED) -> dict:
    _log_stage("Loading configuration")
    if parameter_file is None:
        config, settings = load_parameter_settings(PARAMETERS, ROOT / "main.py")
        parameter_source = "main.py"
    else:
        config, settings = load_parameter_file(parameter_file)
        parameter_source = str(parameter_file.resolve())
    config = replace(
        config,
        input_file=input_file,
        output_dir=output_dir,
        processing_mode=processing_mode,
        gpu_enabled=gpu_enabled,
        chunk_frames=CHUNK_FRAMES,
        chunk_overlap_frames=CHUNK_OVERLAP_FRAMES,
        locate_workers=LOCATE_WORKERS,
    )
    pipeline = AnalysisPipeline(config)
    _log_stage(f"Preparing image store ({processing_mode} mode)")
    pipeline.prepare_store()
    _log_stage("Localizing particles")
    detections, detections_path = pipeline.localize()
    search_range = float(settings["Link"].get("Max displacement", 10))
    memory = int(settings["Link"].get("Dark time", 1))
    _log_stage("Linking particle detections into trajectories")
    trajectories, trajectories_path = pipeline.track(
        detections, search_range=search_range, memory=memory
    )
    analyses = {}
    plot_paths = {}
    for dimension in ("x", "y", "2d"):
        _log_stage(f"Analyzing {dimension} trajectories and saving plots")
        estimates, sizes_path = pipeline.analyze_trajectories(
            trajectories,
            dimension=dimension,
            drift_window=int(settings["Drift"].get("Avg frames", 13)),
            min_particles_for_drift=int(settings["Drift"].get("Min particle per block", 40)),
            output_suffix="x" if dimension == "x" else dimension,
        )
        analyses[dimension] = {
            "particle_sizes_path": str(sizes_path),
            "particles_analyzed": len(estimates),
        }
        plot_paths[dimension] = dict(pipeline.plot_paths)
        analyses[dimension]["plot_paths"] = plot_paths[dimension]
    _log_stage("Writing result metadata")
    export_legacy_aliases(config.output_dir)
    effective_config = pipeline.config
    effective_legacy_parameters = deepcopy(settings)
    effective_legacy_parameters.setdefault("File", {})["data_file_name"] = str(
        effective_config.input_file.resolve()
    )
    effective_legacy_parameters.setdefault("Plot", {})["SaveFolder"] = str(
        effective_config.output_dir.resolve()
    )
    effective_legacy_parameters.setdefault("Find", {})["tp_diameter"] = (
        effective_config.particle_diameter
    )

    resolved_config = asdict(effective_config)
    resolved_config["input_file"] = str(effective_config.input_file.resolve())
    resolved_config["output_dir"] = str(effective_config.output_dir.resolve())
    resolved_config["resolved_channel_center_y_px"] = pipeline.channel_center_y_px
    effective_legacy_parameters["Find"]["tp_minmass"] = (
        effective_config.particle_minmass
    )
    exported_parameters = deepcopy(effective_legacy_parameters)
    exported_parameters["NanoObjectDetection_2_0"] = resolved_config

    parameter_export_path = config.output_dir / "parameters.json"
    if (
        parameter_file is not None
        and parameter_file.resolve() == parameter_export_path.resolve()
    ):
        parameter_export_path = config.output_dir / "parameters_used.json"
    parameter_export_path.write_text(
        json.dumps(exported_parameters, indent=2), encoding="utf-8"
    )

    metadata = {
        "pipeline_version": "2.0",
        "compatibility": "NanoObjectDetection 1.0",
        "parameter_file": parameter_source,
        "processing_mode": processing_mode,
        "gpu_enabled": gpu_enabled,
        "chunk_frames": CHUNK_FRAMES,
        "chunk_overlap_frames": CHUNK_OVERLAP_FRAMES,
        "global_background": (
            str((config.output_dir / "global_background.npy").resolve())
            if config.processing_mode == "chunked"
            and config.remove_static_background
            else None
        ),
        "input_file": str(config.input_file.resolve()),
        "output_dir": str(config.output_dir.resolve()),
        "parameters": {
            "legacy": effective_legacy_parameters,
            "resolved_2_0": resolved_config,
        },
        "legacy_parameters_export": str(parameter_export_path.resolve()),
        "analyses": analyses,
        "plot_paths": plot_paths,
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "analysis_config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (config.output_dir / "results_manifest.json").write_text(
        json.dumps(
            {
                "output_dir": str(config.output_dir.resolve()),
                "files": sorted(
                    str(path.relative_to(config.output_dir))
                    for path in config.output_dir.rglob("*")
                    if path.is_file()
                ),
                "analyses": analyses,
                "plot_paths": plot_paths,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "detections_path": str(detections_path),
        "trajectories_path": str(trajectories_path),
        "analyses": analyses,
        "output_dir": str(config.output_dir),
        "legacy_parameters_path": str(parameter_export_path),
    }


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    if LEGACY_PARAMETER_FILE is not None:
        parser.add_argument("--parameter-file", type=Path, default=LEGACY_PARAMETER_FILE)
    parser.add_argument("--input-file", type=Path, default=INPUT_FILE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--processing-mode", choices=("chunked", "single_stack"),
                        default=PROCESSING_MODE)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(
        args.parameter_file,
        input_file=args.input_file,
        output_dir=args.output_dir,
        processing_mode=args.processing_mode,
        gpu_enabled=args.gpu,
    ), indent=2))
    print(f"Execution time: {time.time() - start_time} seconds")
