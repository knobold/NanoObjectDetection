"""Orchestration for the independent blockwise pipeline."""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import numpy as np

from .config import AnalysisConfig
from .localization import locate
from .preprocessing import preprocess, to_uint16
from .storage import create_store, iter_blocks, iter_tiff_frames, tiff_info
from .tracking import link, remove_near_stationary
from .trajectory_processing import (
    filter_minimum_frames,
    split_intensity_jumps,
    split_long_trajectories,
)
from .quality import filter_trajectories
from .statistics import diameter_summary
from .drift import correct_global
from .msd import estimate_all
from .ultra_uniform import ultra_uniform_mask
from .hindrance import corrected_diameter_nm, local_channel_diameter_nm
from .calibration import estimate_channel_center_from_tiff
from dataclasses import replace

from scipy.ndimage import gaussian_filter

from .auto_parameters import (
    optimize_minmass,
    red_x_from_theory,
    sigma_psf_px,
    trackpy_diameter,
    zncc_particles,
)
from .progress import update
from .stitching import keep_dominant_particle, stitch_fragments
from .legacy_outputs import (
    legacy_sizes_table,
    legacy_trajectories,
)
from .progress import progress
from .plotting import create_analysis_plots
from .gpu import choose_chunk_frames, gpu_available, preprocess_gpu, to_uint16_gpu


def _available_memory_bytes() -> int | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


class AnalysisPipeline:
    def __init__(self, config: AnalysisConfig):
        config.validate()
        self.config = config
        self.store = config.output_dir / "image_store"
        self.channel_center_y_px = config.channel_center_y_px
        self.plot_paths: dict[str, str] = {}
        self.global_background: np.ndarray | None = None
        self.global_offset: float | None = None
        self.effective_chunk_frames = config.chunk_frames
        self._stack: np.ndarray | None = None
        self._stack_first = 0
        if config.gpu_enabled:
            if not gpu_available():
                raise RuntimeError(
                    "GPU acceleration was requested, but CuPy/CUDA is unavailable. "
                    "Install a CUDA-compatible CuPy package or set gpu_enabled=False."
                )
            info = tiff_info(config.input_file)
            height, width = info["shape"][-2:]
            self.effective_chunk_frames = choose_chunk_frames(
                config.chunk_frames,
                height,
                width,
                config.gpu_memory_fraction,
            )
            update(
                f"GPU acceleration enabled; using {self.effective_chunk_frames} "
                "frames per chunk"
            )

    def resolve_channel_center(self) -> float | None:
        """Use the configured center or estimate it from the input TIFF."""
        if self.channel_center_y_px is not None:
            return self.channel_center_y_px
        if not self.config.auto_channel_center:
            return None
        self.channel_center_y_px = estimate_channel_center_from_tiff(
            self.config.input_file,
            self.config.frame_start,
            self.config.frame_start + 300
            if self.config.frame_stop is None
            else min(self.config.frame_stop, self.config.frame_start + 300),
        )
        return self.channel_center_y_px

    def prepare_store(self) -> dict:
        update("Step 1/4: Preparing TIFF working store")
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        if self.config.processing_mode == "single_stack":
            return {
                "format": "single-stack",
                "shape": list(tiff_info(self.config.input_file)["shape"]),
                "chunks": [],
            }
        return create_store(
            self.config.input_file,
            self.store,
            self.config.frame_start,
            self.config.frame_stop,
            self.effective_chunk_frames,
            self.config.chunk_overlap_frames,
            self.config.roi.y_slice,
            self.config.roi.x_slice,
        )

    def localize(self) -> tuple[pd.DataFrame, Path]:
        update("Step 2/4: Localizing particles block by block")
        detections: list[pd.DataFrame] = []
        scale_max = None
        if self.config.remove_static_background:
            self.global_background = self._global_background()
        if self.config.intensity_normalization == "global":
            scale_max = self._global_processed_max()
        self._calibrate_detection(scale_max)
        _, _, height, width = self._source_range_and_shape()
        resident = self._stack is not None
        total_blocks = (
            -(-len(self._stack) // self._block_frames(height, width)) if resident else None
        )
        workers = self.config.locate_workers
        workers = (os.cpu_count() or 1) if workers == 0 else workers
        pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
        for absolute_start, raw_block in progress(
            self._iter_blocks(),
            desc="Localizing particles",
            total=total_blocks,
            unit="block",
        ):
            if self.config.gpu_enabled:
                processed = preprocess_gpu(
                    raw_block,
                    self.config.remove_camera_offset,
                    self.config.remove_laser_fluctuation,
                    self.config.remove_static_background,
                    self.config.background_method,
                    self.config.enhance_snr,
                    background=self.global_background,
                    offset=self.global_offset,
                    snr_sigma_px=self.config.effective_snr_sigma_px,
                )
                image_block = to_uint16_gpu(processed, scale_max=scale_max)
            else:
                processed = preprocess(
                    raw_block,
                    self.config.remove_camera_offset,
                    self.config.remove_laser_fluctuation,
                    self.config.remove_static_background,
                    self.config.background_method,
                    self.config.enhance_snr,
                    background=self.global_background,
                    offset=self.global_offset,
                    snr_sigma_px=self.config.effective_snr_sigma_px,
                )
                image_block = to_uint16(processed, scale_max=scale_max)
            found = locate(
                image_block,
                self.config.particle_diameter,
                self.config.particle_minmass,
                self.config.particle_separation,
                self.config.particle_percentile,
                preprocess=not self.config.enhance_snr,
                workers=workers,
                executor=pool,
            )
            if not found.empty:
                found = found.copy()
                found["saturated"] = self._saturated_flags(raw_block, found)
                found["frame"] += absolute_start
                found["abstime"] = found["frame"] / self.config.fps
                detections.append(found)
            del raw_block, processed, image_block
        if pool is not None:
            pool.shutdown()
        self._stack = None
        if not detections:
            raise RuntimeError("No particles were localized in any image block")
        result = pd.concat(detections, ignore_index=True)
        if (
            self.config.processing_mode == "chunked"
            and self.config.chunk_overlap_frames
            and not resident
        ):
            result = result.drop_duplicates(subset=["frame", "x", "y"]).reset_index(
                drop=True
            )
        path = self.config.output_dir / "detections.csv"
        result.to_csv(path, index=False)
        return result, path

    def _saturated_flags(self, raw_block: np.ndarray, found: pd.DataFrame) -> np.ndarray:
        """Flag detections whose 3x3 neighbourhood reaches the saturation value."""
        limit = self.config.saturated_pixel_value
        if limit is None:
            return np.zeros(len(found), dtype=bool)
        flags = np.zeros(len(found), dtype=bool)
        height, width = raw_block.shape[1:]
        for index, (frame, x, y) in enumerate(
            zip(found["frame"].to_numpy(int), found["x"].to_numpy(), found["y"].to_numpy())
        ):
            px, py = int(x), int(y)
            window = raw_block[
                frame,
                max(py - 1, 0) : min(py + 2, height),
                max(px - 1, 0) : min(px + 2, width),
            ]
            flags[index] = window.size > 0 and window.max() >= limit
        return flags

    def _calibrate_detection(self, scale_max: float | None) -> None:
        """Set trackpy diameter and/or minmass automatically, as the 1.0 help functions do."""
        config = self.config
        if not (config.auto_diameter or config.auto_minmass):
            return
        updates: dict[str, float | int] = {}
        if config.auto_diameter:
            updates["particle_diameter"] = trackpy_diameter(
                config.numerical_aperture,
                config.wavelength_nm,
                config.pixel_size_um,
                config.enhance_snr,
            )
            update(f"Auto trackpy diameter: {updates['particle_diameter']} px")
        if config.auto_minmass:
            diameter = int(updates.get("particle_diameter", config.particle_diameter))
            updates["particle_minmass"] = self._auto_minmass(diameter, scale_max)
            update(f"Auto trackpy minmass: {updates['particle_minmass']:g}")
        self.config = replace(config, **updates)
        (config.output_dir / "auto_parameters.json").write_text(
            json.dumps(updates, indent=2), encoding="utf-8"
        )

    def _auto_minmass(self, diameter: int, scale_max: float | None) -> float:
        """Find the minmass at which trackpy best reproduces a ZNCC particle search."""
        config = self.config
        info = tiff_info(config.input_file)
        stop = info["shape"][0] if config.frame_stop is None else min(
            config.frame_stop, info["shape"][0]
        )
        indices = np.unique(
            np.round(np.linspace(config.frame_start, stop - 1, config.minmass_try_frames))
        ).astype(int)
        wanted = set(indices.tolist())
        if self._stack is not None:
            frames = [self._stack[number - self._stack_first] for number in sorted(wanted)]
        else:
            frames = [
                image[config.roi.y_slice, config.roi.x_slice]
                for number, image in iter_tiff_frames(
                    config.input_file, config.frame_start, stop
                )
                if number in wanted
            ]
        processed = preprocess(
            np.stack(frames),
            config.remove_camera_offset,
            config.remove_laser_fluctuation,
            config.remove_static_background,
            config.background_method,
            False,
            background=self.global_background,
            offset=self.global_offset,
        )
        sigma = sigma_psf_px(
            config.numerical_aperture, config.wavelength_nm, config.pixel_size_um
        )
        update("Auto minmass: locating reference particles by ZNCC")
        reference = zncc_particles(processed, sigma, config.enhance_snr, config.zncc_min)
        if len(reference) == 0:
            raise RuntimeError("ZNCC found no particles in the sampled frames; cannot set minmass")
        if len(reference) < 200:
            update(
                f"Warning: only {len(reference)} reference particles; "
                "consider raising minmass_try_frames"
            )
        trackpy_input = processed
        if config.enhance_snr:
            trackpy_input = gaussian_filter(
                processed, sigma=(0, config.effective_snr_sigma_px, config.effective_snr_sigma_px)
            )
        image_block = to_uint16(trackpy_input, scale_max=scale_max)
        update("Auto minmass: optimizing against trackpy")
        candidates = locate(
            image_block,
            diameter,
            1,
            config.particle_separation,
            config.particle_percentile,
            preprocess=not config.enhance_snr,
            workers=config.locate_workers,
        )
        optimum, _ = optimize_minmass(candidates, reference, diameter / 2.0)
        return float(int(optimum * 0.90))

    def _source_range_and_shape(self) -> tuple[int, int, int, int]:
        """Return first frame, stop frame, ROI height and ROI width of the analysed data."""
        if self.config.processing_mode == "single_stack":
            info = tiff_info(self.config.input_file)
            total, height, width = info["shape"][-3:]
            stop = total if self.config.frame_stop is None else min(self.config.frame_stop, total)
            height = len(range(*self.config.roi.y_slice.indices(height)))
            width = len(range(*self.config.roi.x_slice.indices(width)))
            return self.config.frame_start, stop, height, width
        manifest = json.loads((self.store / "manifest.json").read_text(encoding="utf-8"))
        _, height, width = manifest["shape"]
        return int(manifest["frame_start"]), int(manifest["frame_stop"]), height, width

    def _iter_source_blocks(self, batch_frames: int):
        """Yield (first frame, ROI-cropped block) by reading the source exactly once."""
        if self.config.processing_mode != "single_stack":
            yield from iter_blocks(self.store)
            return
        frames: list[np.ndarray] = []
        first_frame = self.config.frame_start
        roi = self.config.roi
        for frame_number, image in iter_tiff_frames(
            self.config.input_file, self.config.frame_start, self.config.frame_stop
        ):
            if not frames:
                first_frame = frame_number
            frames.append(image[roi.y_slice, roi.x_slice])
            if len(frames) == batch_frames:
                yield first_frame, np.stack(frames, axis=0)
                frames = []
        if frames:
            yield first_frame, np.stack(frames, axis=0)

    def _block_frames(self, height: int, width: int) -> int:
        """Frames per processing block: bounded in memory, large enough to feed all workers."""
        workers = self.config.locate_workers or (os.cpu_count() or 1)
        by_memory = max(1, int(128e6 // (height * width * 2)))
        frames = min(self.effective_chunk_frames, by_memory)
        return max(frames, workers * 4 if workers > 1 else 1)

    def _iter_blocks(self):
        """Yield (first frame, raw ROI block); served from RAM once the stack is resident."""
        _, _, height, width = self._source_range_and_shape()
        size = self._block_frames(height, width)
        if self._stack is not None:
            for start in range(0, len(self._stack), size):
                yield self._stack_first + start, self._stack[start : start + size]
            return
        yield from self._iter_source_blocks(size)

    def _global_background(self) -> np.ndarray:
        """Estimate the 1.0 background: per-pixel 30-70% mean over all frames.

        The ROI stack is read once into a preallocated array in its native integer
        dtype, kept for the later passes, and reduced in pixel strips, so the
        result is identical to the 1.0 calculation.
        """
        first, stop, height, width = self._source_range_and_shape()
        frames = stop - first
        needed = frames * height * width * 2
        available = _available_memory_bytes()
        if available is not None and needed > 0.8 * available:
            raise MemoryError(
                f"The exact 1.0 background needs the whole ROI stack in RAM "
                f"({needed / 1e9:.1f} GB) but only {available / 1e9:.1f} GB is available. "
                "Reduce the ROI or the frame range, or use "
                "background_method='median' on a subset."
            )
        stack = np.empty((frames, height, width), dtype=np.uint16)
        filled = 0
        for block_first, block in progress(
            self._iter_source_blocks(self._block_frames(height, width)),
            desc="Reading frames",
            unit="block",
        ):
            # Chunks may overlap; count every frame once
            skip = max(0, first + filled - block_first)
            block = block[skip:]
            block = block[: frames - filled]
            if len(block):
                stack[filled : filled + len(block)] = block
                filled += len(block)
        if filled != frames:
            raise RuntimeError(f"Expected {frames} frames but read {filled}")
        self._stack = stack
        self._stack_first = first
        offset = float(stack.min())
        flat = stack.reshape(frames, height * width)
        low = int(0.3 * frames)
        high = max(low + 1, int(0.7 * frames))
        background = np.empty(height * width, dtype=np.float64)
        strip = max(1, int(2**28 // (frames * flat.itemsize)))
        for start in range(0, height * width, strip):
            part = np.ascontiguousarray(flat[:, start : start + strip].T)
            if self.config.background_method == "median":
                background[start : start + strip] = np.median(part, axis=1)
            else:
                part.partition([low, high - 1], axis=1)
                background[start : start + strip] = part[:, low:high].mean(
                    axis=1, dtype=np.float64
                )
        background = background.reshape(height, width)
        if self.config.remove_camera_offset:
            self.global_offset = offset
            background = background - offset
        global_background = background.astype(np.float32)
        np.save(self.config.output_dir / "global_background.npy", global_background)
        update(f"Global background estimated from {frames} frames")
        return global_background

    def _global_processed_max(self) -> float:
        """Determine one intensity scale for all blocks, matching 1.0 scaling."""
        maximum = 0.0
        for _, raw_block in self._iter_blocks():
            processed = preprocess(
                raw_block,
                self.config.remove_camera_offset,
                self.config.remove_laser_fluctuation,
                self.config.remove_static_background,
                self.config.background_method,
                self.config.enhance_snr,
                background=self.global_background,
                offset=self.global_offset,
                snr_sigma_px=self.config.effective_snr_sigma_px,
            )
            maximum = max(maximum, float(np.max(np.maximum(processed, 0.0))))
        if maximum <= 0:
            raise RuntimeError("Processed image stack contains no positive intensity")
        return maximum

    @property
    def _kept_minimum_frames(self) -> int:
        """Shortest track kept: the valid-track minimum, never above the drift minimum."""
        return min(
            self.config.minimum_trajectory_frames,
            self.config.minimum_trajectory_frames_before_drift,
        )

    def track(
        self,
        detections: pd.DataFrame,
        search_range: float,
        memory: int = 1,
        minimum_frames: int | None = None,
    ) -> tuple[pd.DataFrame, Path]:
        """Link the compact detection table after blockwise localization."""
        update("Step 3/4: Linking trajectories")
        minimum_frames = (
            self.config.minimum_trajectory_frames
            if minimum_frames is None
            else minimum_frames
        )
        single = self.config.single_particle
        moving_detections = detections
        stationary = pd.DataFrame()
        if self.config.analyze_stationary_particles:
            stationary = link(
                detections,
                search_range=self.config.stationary_search_range,
                memory=1,
                minimum_frames=self.config.stationary_minimum_frames,
            )
            moving_detections = remove_near_stationary(
                detections, stationary, self.config.no_go_distance
            )
            stationary.to_csv(self.config.output_dir / "stationary_trajectories.csv", index=False)
        trajectories = link(
            moving_detections,
            search_range=search_range,
            memory=memory,
            minimum_frames=1 if single else self._kept_minimum_frames,
        )
        if single:
            config = self.config
            fragments = trajectories["particle"].nunique()
            trajectories = stitch_fragments(
                trajectories,
                config.stitch_max_gap_frames,
                config.stitch_max_distance_px,
                config.stitch_distance_growth_px,
                config.stitch_max_mass_ratio,
                config.stitch_min_fragment_frames,
            )
            stitched = trajectories["particle"].nunique()
            if config.single_particle_keep != "all":
                trajectories = keep_dominant_particle(trajectories, config.single_particle_keep)
            update(
                f"Single particle mode: {fragments} fragments stitched into {stitched} "
                f"trajectories, {trajectories['particle'].nunique()} kept"
            )
        trajectories = filter_minimum_frames(
            trajectories, self._kept_minimum_frames
        )
        if self.config.split_intensity_jumps:
            trajectories = split_intensity_jumps(
                trajectories,
                self.config.max_relative_intensity_step,
                self._kept_minimum_frames,
            )
        trajectories = split_long_trajectories(
            trajectories,
            self.config.maximum_trajectory_frames,
            self._kept_minimum_frames,
            self.config.keep_short_trajectory_tail,
        )
        path = self.config.output_dir / "trajectories.csv"
        trajectories.to_csv(path, index=False)
        return trajectories, path

    def analyze_trajectories(
        self,
        trajectories: pd.DataFrame,
        dimension: str = "2d",
        max_lag: int | None = None,
        drift_window: int = 20,
        min_particles_for_drift: int = 2,
        output_suffix: str = "",
    ) -> tuple[pd.DataFrame, Path]:
        """Correct drift, estimate diffusion, and calculate particle sizes."""
        update("Step 4/4: Correcting drift, estimating MSD, and sizing particles")
        # With one particle the "common drift" would just be its own motion
        if self.config.drift_enabled and not self.config.single_particle:
            corrected, drift = correct_global(
                trajectories,
                drift_window,
                min_particles_for_drift,
                self.config.minimum_trajectory_frames_before_drift,
            )
        else:
            corrected = trajectories.copy()
            corrected["x_corrected"] = corrected["x"]
            corrected["y_corrected"] = corrected["y"]
            drift = pd.DataFrame(
                {"frame": sorted(corrected["frame"].unique()), "drift_x": 0.0, "drift_y": 0.0}
            )
        if dimension == "x":
            legacy_trajectories(corrected).to_csv(
                self.config.output_dir / "traj_no_drift.csv", index=False
            )
        corrected = filter_minimum_frames(
            corrected, self.config.minimum_trajectory_frames
        )
        if "saturated" in corrected:
            # Saturated positions are poorly localized; the legacy fit ignores them
            corrected = filter_minimum_frames(
                corrected[~corrected["saturated"].astype(bool)],
                self.config.minimum_trajectory_frames,
            )
        quality = pd.DataFrame()
        if self.config.apply_quality_filter:
            corrected, quality = filter_trajectories(
                corrected,
                dimension=dimension,
                max_lag=(
                    self.config.msd_max_lag_frames
                    if max_lag is None
                    else max_lag
                ),
                pixel_size_um=self.config.pixel_size_um,
                fps=self.config.fps,
                minimum_pairs_at_max_lag=self.config.minimum_msd_pairs_at_max_lag,
                minimum_ks_significance=self.config.minimum_ks_significance,
                ks_method=self.config.ks_method,
            )
            quality.to_csv(
                self.config.output_dir / f"trajectory_quality_{dimension}.csv",
                index=False,
            )
            if dimension == "x":
                quality.to_csv(
                    self.config.output_dir / "trajectory_quality.csv",
                    index=False,
                )
        else:
            quality.to_csv(
                self.config.output_dir / f"trajectory_quality_{dimension}.csv",
                index=False,
            )
            if dimension == "x":
                quality.to_csv(
                    self.config.output_dir / "trajectory_quality.csv",
                    index=False,
                )
        estimates = estimate_all(
            corrected,
            dimension=dimension,
            max_lag=(
                self.config.msd_max_lag_frames
                if max_lag is None
                else max_lag
            ),
            pixel_size_um=self.config.pixel_size_um,
            fps=self.config.fps,
            auto_lag=self.config.msd_lag_auto and max_lag is None,
            fixed_red_x=(
                red_x_from_theory(
                    self.config.exposure_time_s,
                    1.0 / self.config.fps,
                    self.config.numerical_aperture,
                    self.config.wavelength_nm,
                )
                if self.config.red_x_mode == "theory"
                else None
            ),
        )
        if estimates.empty:
            rejected = (
                quality["reason"].value_counts().to_dict()
                if "reason" in quality
                else {}
            )
            update(
                f"No trajectory survived the filters for dimension '{dimension}' "
                f"(rejection reasons: {rejected or 'none recorded'}); skipping sizing and plots"
            )
            suffix = f"_{output_suffix}" if output_suffix else ""
            path = self.config.output_dir / f"particle_sizes{suffix}.csv"
            estimates.to_csv(path, index=False)
            self.plot_paths = {}
            return estimates, path
        local = {}
        if self.config.hindrance_mode == "position":
            center = self.resolve_channel_center()
            if center is None or self.config.channel_diameter_nm is None:
                raise RuntimeError(
                    "Position-dependent hindrance correction needs a channel center "
                    "and channel diameter"
                )
            for particle, group in corrected.groupby("particle"):
                local[particle] = local_channel_diameter_nm(
                    group["y_corrected"].to_numpy(),
                    center,
                    self.config.channel_diameter_nm,
                    self.config.pixel_size_um,
                )
            estimates["local_channel_diameter_nm"] = estimates["particle"].map(local)
        elif self.config.hindrance_mode == "legacy":
            estimates["local_channel_diameter_nm"] = self.config.channel_diameter_nm
        diameters = []
        for particle, diffusion in zip(
            estimates["particle"], estimates["diffusion_um2_s"]
        ):
            if not pd.notna(diffusion) or diffusion <= 0:
                diameters.append(float("nan"))
                continue
            channel_diameter = (
                local.get(particle)
                if self.config.hindrance_mode == "position"
                else self.config.channel_diameter_nm
                if self.config.hindrance_mode == "legacy"
                else None
            )
            diameters.append(
                corrected_diameter_nm(
                    diffusion,
                    self.config.temperature_k,
                    self.config.viscosity_pa_s,
                    channel_diameter,
                )
            )
        estimates["diameter_nm"] = diameters
        if self.config.maximum_diameter_nm is not None:
            too_large = estimates["diameter_nm"] > self.config.maximum_diameter_nm
            estimates["diameter_filter_reason"] = ""
            estimates.loc[too_large, "diameter_filter_reason"] = (
                f"above maximum_diameter_nm={self.config.maximum_diameter_nm:g}"
            )
            estimates.loc[too_large, "diameter_nm"] = float("nan")
        else:
            estimates["diameter_filter_reason"] = ""
        if self.config.force_ultra_uniform:
            outliers = ~ultra_uniform_mask(
                estimates, self.config.ultra_uniform_z_max
            )
            estimates.loc[outliers, "diameter_filter_reason"] = (
                f"ultra-uniform z-score >= {self.config.ultra_uniform_z_max:g}"
            )
            estimates.loc[outliers, "diameter_nm"] = float("nan")
        suffix = f"_{output_suffix}" if output_suffix else ""
        path = self.config.output_dir / f"particle_sizes{suffix}.csv"
        estimates.to_csv(path, index=False)
        summary = diameter_summary(estimates)
        summary.to_csv(
            self.config.output_dir / f"particle_size_summary{suffix}.csv",
            index=False,
        )
        if output_suffix == "x":
            estimates.to_csv(self.config.output_dir / "particle_sizes.csv", index=False)
            summary.to_csv(
                self.config.output_dir / "particle_size_summary.csv",
                index=False,
            )
        drift.to_csv(self.config.output_dir / f"drift_{dimension}.csv", index=False)
        if dimension == "x":
            drift.to_csv(self.config.output_dir / "drift.csv", index=False)
        corrected_path = self.config.output_dir / f"trajectories_corrected{suffix}.csv"
        corrected.to_csv(corrected_path, index=False)
        if output_suffix == "x":
            corrected.to_csv(
                self.config.output_dir / "trajectories_corrected.csv",
                index=False,
            )
        label = {"x": "long", "y": "trans"}.get(dimension, dimension)
        sizes_table = legacy_sizes_table(
            corrected, trajectories, estimates, quality, self.config.fps
        )
        for weighting in ("unweighted", "weighted"):
            sizes_table.to_csv(
                self.config.output_dir
                / f"Diameter_Histogramm-{label}-{weighting}.csv",
                index=False,
            )
        final = legacy_trajectories(corrected)
        final = final[final["particle"].isin(sizes_table["particle"])]
        sizes_table.to_csv(
            self.config.output_dir / f"sizes_df_lin_{label}.csv", index=False
        )
        sizes_table.to_csv(
            self.config.output_dir / f"DiameterOverTrajLength_{label}.csv", index=False
        )
        final.to_csv(self.config.output_dir / f"traj_final_{label}.csv", index=False)
        if dimension == "x":
            final.to_csv(self.config.output_dir / "traj_final.csv", index=False)
        plot_paths = create_analysis_plots(
            corrected,
            estimates,
            self.config.output_dir,
            pixel_size_um=self.config.pixel_size_um,
            fps=self.config.fps,
            dimension=dimension,
            max_lag=(
                self.config.msd_max_lag_frames
                if max_lag is None
                else max_lag
            ),
            dimension_label=output_suffix or dimension,
            drift=drift,
        )
        self.plot_paths = plot_paths
        return estimates, path

    def prepare(self) -> dict:
        manifest = self.prepare_store()
        center = self.resolve_channel_center()
        return {
            "tiff": tiff_info(self.config.input_file),
            "chunked_shape": tuple(manifest["shape"]),
            "chunk_count": len(manifest["chunks"]),
            "chunk_store": str(self.store.resolve()),
            "channel_center_y_px": center,
        }
