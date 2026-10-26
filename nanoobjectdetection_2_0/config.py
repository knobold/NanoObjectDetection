"""Configuration for the independent analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .auto_parameters import sigma_psf_px


@dataclass(frozen=True)
class Roi:
    x_min: int = 0
    x_max: int | None = None
    y_min: int = 0
    y_max: int | None = None

    @property
    def x_slice(self) -> slice:
        return slice(self.x_min, self.x_max)

    @property
    def y_slice(self) -> slice:
        return slice(self.y_min, self.y_max)


@dataclass(frozen=True)
class AnalysisConfig:
    input_file: Path
    output_dir: Path
    frame_start: int = 0
    frame_stop: int | None = None
    chunk_frames: int = 512
    chunk_overlap_frames: int = 0
    processing_mode: str = "chunked"
    gpu_enabled: bool = False
    gpu_memory_fraction: float = 0.7
    locate_workers: int = 1
    roi: Roi = Roi()
    fps: float = 300.0
    pixel_size_um: float = 1.0
    wavelength_nm: float = 532.0
    numerical_aperture: float = 0.25
    exposure_time_s: float = 0.0007
    temperature_k: float = 295.15
    viscosity_pa_s: float = 0.00095
    channel_diameter_nm: float | None = None
    channel_center_y_px: float | None = None
    auto_channel_center: bool = True
    hindrance_mode: str = "none"
    remove_camera_offset: bool = True
    remove_laser_fluctuation: bool = False
    remove_static_background: bool = True
    enhance_snr: bool = False
    snr_kernel_sigma_px: float | None = None
    auto_diameter: bool = False
    auto_minmass: bool = False
    zncc_min: float = 0.7
    minmass_try_frames: int = 30
    red_x_mode: str = "exp"
    single_particle: bool = False
    single_particle_keep: str = "longest"
    stitch_max_gap_frames: int = 300
    stitch_max_distance_px: float = 15.0
    stitch_distance_growth_px: float = 0.5
    stitch_max_mass_ratio: float = 3.0
    stitch_min_fragment_frames: int = 3
    background_stack_storage: str = "memory"
    background_method: str = "legacy_trimmed_mean"
    intensity_normalization: str = "global"
    particle_diameter: int = 7
    particle_minmass: float = 100000.0
    particle_separation: int = 15
    particle_percentile: float = 20.0
    minimum_trajectory_frames: int = 100
    minimum_trajectory_frames_before_drift: int = 10
    drift_enabled: bool = True
    msd_max_lag_frames: int = 20
    msd_lag_auto: bool = False
    analyze_stationary_particles: bool = True
    stationary_search_range: float = 0.3
    stationary_minimum_frames: int = 100
    no_go_distance: float = 20.0
    split_intensity_jumps: bool = False
    max_relative_intensity_step: float = 0.906
    maximum_trajectory_frames: int | None = None
    keep_short_trajectory_tail: bool = False
    apply_quality_filter: bool = True
    minimum_msd_pairs_at_max_lag: int = 5
    minimum_ks_significance: float = 0.01
    ks_method: str = "exact"
    saturated_pixel_value: float | None = None
    maximum_diameter_nm: float | None = None
    force_ultra_uniform: bool = False
    ultra_uniform_z_max: float = 2.576

    def validate(self) -> None:
        if not self.input_file.is_file():
            raise FileNotFoundError(f"Input TIFF does not exist: {self.input_file}")
        if self.frame_start < 0:
            raise ValueError("frame_start must be non-negative")
        if self.frame_stop is not None and self.frame_stop <= self.frame_start:
            raise ValueError("frame_stop must be greater than frame_start")
        if self.chunk_frames <= 0:
            raise ValueError("chunk_frames must be positive")
        if self.chunk_overlap_frames < 0 or self.chunk_overlap_frames >= self.chunk_frames:
            raise ValueError("chunk_overlap_frames must be in [0, chunk_frames)")
        if self.processing_mode not in {"chunked", "single_stack"}:
            raise ValueError("processing_mode must be 'chunked' or 'single_stack'")
        if self.background_stack_storage not in {"memory", "disk"}:
            raise ValueError(
                "background_stack_storage must be 'memory' or 'disk'"
            )
        if self.background_method not in {"median", "legacy_trimmed_mean"}:
            raise ValueError("background_method must be 'median' or 'legacy_trimmed_mean'")
        if self.intensity_normalization not in {"global", "block"}:
            raise ValueError("intensity_normalization must be 'global' or 'block'")
        if self.hindrance_mode not in {"none", "legacy", "position"}:
            raise ValueError(
                "hindrance_mode must be 'none', 'legacy', or 'position'"
            )
        if self.hindrance_mode in {"legacy", "position"}:
            if self.channel_diameter_nm is None or self.channel_diameter_nm <= 0:
                raise ValueError(
                    "channel_diameter_nm is required for hindrance correction"
                )
        if self.hindrance_mode == "position" and (
            self.channel_center_y_px is None and not self.auto_channel_center
        ):
            raise ValueError(
                "position hindrance correction requires channel_center_y_px "
                "or auto_channel_center=True"
            )
        if not 0 < self.gpu_memory_fraction < 1:
            raise ValueError("gpu_memory_fraction must be between 0 and 1")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.pixel_size_um <= 0:
            raise ValueError("pixel_size_um must be positive")
        if self.wavelength_nm <= 0:
            raise ValueError("wavelength_nm must be positive")
        if self.numerical_aperture <= 0:
            raise ValueError("numerical_aperture must be positive")
        if self.red_x_mode not in {"exp", "theory"}:
            raise ValueError("red_x_mode must be 'exp' or 'theory'")
        if self.single_particle_keep not in {"longest", "brightest", "all"}:
            raise ValueError("single_particle_keep must be 'longest', 'brightest' or 'all'")
        if (
            self.stitch_max_gap_frames < 1
            or self.stitch_max_distance_px <= 0
            or self.stitch_distance_growth_px < 0
            or self.stitch_max_mass_ratio < 1
            or self.stitch_min_fragment_frames < 1
        ):
            raise ValueError("Invalid stitching parameters")
        if not 0 < self.zncc_min < 1:
            raise ValueError("zncc_min must be between 0 and 1")
        if self.minmass_try_frames < 1:
            raise ValueError("minmass_try_frames must be at least 1")
        if self.snr_kernel_sigma_px is not None and self.snr_kernel_sigma_px <= 0:
            raise ValueError("snr_kernel_sigma_px must be positive")
        if self.exposure_time_s <= 0:
            raise ValueError("exposure_time_s must be positive")
        if self.minimum_trajectory_frames < 2 or self.minimum_trajectory_frames_before_drift < 2:
            raise ValueError("minimum trajectory lengths must be at least 2")
        if self.msd_max_lag_frames < 2:
            raise ValueError("msd_max_lag_frames must be at least 2")
        if self.particle_diameter < 3 or self.particle_diameter % 2 == 0:
            raise ValueError("particle_diameter must be an odd integer >= 3")
        if self.stationary_search_range <= 0 or self.stationary_minimum_frames < 2:
            raise ValueError("Stationary-particle parameters must be positive")
        if self.no_go_distance < 0:
            raise ValueError("no_go_distance must be non-negative")
        if self.max_relative_intensity_step <= 0:
            raise ValueError("max_relative_intensity_step must be positive")
        if self.minimum_msd_pairs_at_max_lag < 2:
            raise ValueError("minimum_msd_pairs_at_max_lag must be at least 2")
        if self.ks_method not in {"exact", "legacy"}:
            raise ValueError("ks_method must be 'exact' or 'legacy'")
        if not 0 < self.minimum_ks_significance < 1:
            raise ValueError("minimum_ks_significance must be between 0 and 1")
        if self.ultra_uniform_z_max <= 0:
            raise ValueError("ultra_uniform_z_max must be positive")
        if self.maximum_diameter_nm is not None and self.maximum_diameter_nm <= 0:
            raise ValueError("maximum_diameter_nm must be positive")
        if (
            self.maximum_trajectory_frames is not None
            and self.maximum_trajectory_frames < self.minimum_trajectory_frames_before_drift
        ):
            raise ValueError("maximum_trajectory_frames is too short")

    @property
    def effective_snr_sigma_px(self) -> float:
        """Gaussian sigma of the SNR filter; the PSF width when KernelSize is "auto"."""
        if self.snr_kernel_sigma_px is not None:
            return self.snr_kernel_sigma_px
        return sigma_psf_px(
            self.numerical_aperture, self.wavelength_nm, self.pixel_size_um
        )
