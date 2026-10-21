"""1.0-compatible configuration for the 2.0 chunked/GPU pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from .config import AnalysisConfig, Roi


def _number(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _flag(value) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "auto", "manual"}


def _kernel_sigma(value):
    """1.0 KernelSize: "auto" (PSF width) or the Gaussian sigma in pixels."""
    if str(value).strip().lower() == "auto":
        return None
    return _number(value, None)


def _saturation(value):
    """Return the saturation threshold, or None for 'No Saturation'."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_parameter_file(parameter_file: str | Path) -> tuple[AnalysisConfig, dict]:
    """Translate a 1.0 ``parameter.json`` into an equivalent 2.0 config."""
    parameter_path = Path(parameter_file)
    settings = json.loads(parameter_path.read_text(encoding="utf-8"))
    return load_parameter_settings(settings, parameter_path)


def load_parameter_settings(
    settings: dict, parameter_path: str | Path = "."
) -> tuple[AnalysisConfig, dict]:
    """Translate in-memory 1.0 settings into an equivalent 2.0 config."""
    parameter_path = Path(parameter_path)
    exp = settings["Exp"]
    fiber = settings["Fiber"]
    files = settings["File"]
    roi_settings = settings["ROI"]
    preprocessing = settings["PreProcessing"]
    find = settings["Find"]
    link = settings["Link"]
    stationary = settings["StationaryObjects"]
    split = settings["Split"]
    drift = settings["Drift"]
    msd = settings["MSD"]
    help_settings = settings.get("Help", {})
    single = settings.get("SingleParticle", {})
    post = settings["PostProcessing"]

    frame_stop = None
    if roi_settings.get("frame_max", 0) and roi_settings["frame_max"] < 10**6:
        frame_stop = int(roi_settings["frame_max"])
    roi = Roi(
        x_min=int(roi_settings.get("x_min", 0)),
        x_max=None if roi_settings.get("x_max", 10**9) >= 10**9 else int(roi_settings["x_max"]),
        y_min=int(roi_settings.get("y_min", 0)),
        y_max=None if roi_settings.get("y_max", 10**9) >= 10**9 else int(roi_settings["y_max"]),
    )
    hindrance_aware = bool(msd.get("HindrancePositionAware", 0))
    hindrance_mode = "position" if hindrance_aware else (
        "legacy" if bool(msd.get("EstimateHindranceFactor", 0)) else "none"
    )
    max_diameter = _number(post.get("MaxDiameter", 0), 0.0) or None
    max_trajectory = _number(split.get("Max_traj_length", 0), 0.0) or None
    if max_trajectory and max_trajectory >= 10**8:
        max_trajectory = None

    input_file = Path(files["data_file_name"])
    if not input_file.is_file():
        candidate = parameter_path.parent / input_file.name
        if candidate.is_file():
            input_file = candidate
    output_dir = Path(settings["Plot"].get("SaveFolder", parameter_path.parent))
    if not output_dir.is_absolute() or not output_dir.exists():
        output_dir = parameter_path.parent / "analysis" / "pipeline_2_0"

    config = AnalysisConfig(
        input_file=input_file,
        output_dir=output_dir,
        frame_start=int(roi_settings.get("frame_min", 0)),
        frame_stop=frame_stop,
        roi=roi,
        fps=_number(exp["fps"], 100.0),
        pixel_size_um=_number(exp["Microns_per_pixel"], 1.0),
        wavelength_nm=_number(exp.get("lambda"), 532.0),
        numerical_aperture=_number(exp.get("NA"), 0.25),
        exposure_time_s=_number(exp.get("ExposureTime"), 0.0007),
        temperature_k=_number(exp.get("Temperature"), 293.15),
        viscosity_pa_s=_number(exp.get("Viscosity"), 0.001) * 1e12,
        channel_diameter_nm=_number(fiber.get("TubeDiameter_nm"), 0.0) or None,
        channel_center_y_px=(
            _number(fiber.get("Center_y_px"), 0.0)
            if str(fiber.get("Center_y_px", "")).lower() != "unknown"
            else None
        ),
        auto_channel_center=str(fiber.get("Center_y_px", "")).lower() == "unknown",
        hindrance_mode=hindrance_mode,
        remove_camera_offset=bool(preprocessing.get("Remove_CameraOffset", 1)),
        remove_laser_fluctuation=bool(
            preprocessing.get("Remove_Laserfluctuation", 0)
        ),
        remove_static_background=bool(preprocessing.get("Remove_StaticBackground", 1)),
        enhance_snr=bool(preprocessing.get("EnhanceSNR", 0)),
        snr_kernel_sigma_px=_kernel_sigma(preprocessing.get("KernelSize", "auto")),
        zncc_min=_number(preprocessing.get("ZNCC_min"), 0.7),
        auto_minmass=_flag(help_settings.get("Bead brightness", 0)),
        auto_diameter=_flag(help_settings.get("Bead size", 0)),
        minmass_try_frames=int(_number(help_settings.get("TryFrames"), 30)),
        red_x_mode=str(msd.get("Estimate_X", "Exp")).lower(),
        single_particle=bool(single.get("Enabled", 0)),
        single_particle_keep=str(single.get("Keep", "longest")).lower(),
        stitch_max_gap_frames=int(single.get("Max gap frames", 300)),
        stitch_max_distance_px=float(single.get("Max distance px", 15.0)),
        stitch_distance_growth_px=float(single.get("Distance growth px per sqrt frame", 0.5)),
        stitch_max_mass_ratio=float(single.get("Max mass ratio", 3.0)),
        stitch_min_fragment_frames=int(single.get("Min fragment frames", 3)),
        background_method="legacy_trimmed_mean",
        intensity_normalization="global",
        particle_diameter=int(find.get("tp_diameter", 7)),
        particle_minmass=_number(find.get("tp_minmass"), 358.0),
        particle_separation=int(find.get("tp_separation", 20)),
        particle_percentile=_number(find.get("tp_percentile"), 20.0),
        minimum_trajectory_frames=int(link.get("Min_tracking_frames", 5)),
        minimum_trajectory_frames_before_drift=int(link.get("Min tracking frames before drift", 10)),
        drift_enabled=bool(drift.get("Apply", 1)),
        msd_max_lag_frames=max(2, int(msd.get("lagtimes_max", 3))),
        msd_lag_auto=bool(msd.get("Amount lagtimes auto", 0)),
        analyze_stationary_particles=bool(stationary.get("Analyze fixed spots", 1)),
        stationary_search_range=_number(link.get("Max displacement fix"), 0.3),
        stationary_minimum_frames=int(link.get("Dwell time stationary objects", 100)),
        no_go_distance=_number(stationary.get("Min distance to stationary object"), 20.0),
        split_intensity_jumps=bool(split.get("IntensityJump", 0)),
        max_relative_intensity_step=_number(split.get("Max rel median intensity step"), 0.906),
        maximum_trajectory_frames=None if max_trajectory is None else int(max_trajectory),
        keep_short_trajectory_tail=bool(split.get("Max_traj_length_keep_tail", 0)),
        ks_method=str(msd.get("KSMethod", "legacy")).lower(),
        minimum_ks_significance=_number(msd.get("KSMinSignificance"), 0.01),
        saturated_pixel_value=_saturation(find.get("SaturatedPixelValue")),
        apply_quality_filter=bool(msd.get("CheckBrownianMotion", 1)),
        maximum_diameter_nm=max_diameter,
        force_ultra_uniform=bool(post.get("ForceUltraUniform", 0)),
    )
    return config, settings
