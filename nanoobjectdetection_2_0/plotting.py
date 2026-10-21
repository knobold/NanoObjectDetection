"""Diagnostic plots matching the 1.0 analysis views."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .msd import calculate_msd, estimate_diffusion
from .statistics import weighted_quantile


def _save(figure, path, **kwargs) -> None:
    # Overwriting files on synced drives (OneDrive via WSL) can fail with EINVAL; retry after removing the old file.
    for attempt in range(4):
        try:
            figure.savefig(path, **kwargs)
            return
        except OSError:
            if attempt == 3:
                raise
            Path(path).unlink(missing_ok=True)
            time.sleep(0.5)


def _statistics(values: np.ndarray) -> dict[str, float | tuple[float, float]]:
    inverse = 1000.0 / values
    inverse_mean = float(np.mean(inverse))
    inverse_std = float(np.std(inverse, ddof=1)) if len(inverse) > 1 else 0.0
    q16, q84 = np.quantile(inverse, [0.158655, 0.841345])
    q023, q977 = np.quantile(inverse, [0.02275, 0.97725])
    return {
        "median": float(1000.0 / np.median(inverse)),
        "mu": float(np.mean(values)),
        "cv_inv": inverse_std / inverse_mean if inverse_mean else float("nan"),
        "one_sigma": (float(1000.0 / q84), float(1000.0 / q16)),
        "two_sigma": (float(1000.0 / q977), float(1000.0 / q023)),
    }


def _dimension_title(dimension: str) -> str:
    return {"x": "long.", "y": "trans.", "2d": "2d"}.get(dimension, dimension)


def _add_statistics_box(axis, values: np.ndarray) -> None:
    stats = _statistics(values)
    one = stats["one_sigma"]
    two = stats["two_sigma"]
    text = "\n".join(
        [
            f"median = {stats['median']:.1f} nm",
            rf"$\mu$ = {stats['mu']:.1f} nm",
            rf"$CV_{{\text{{inv}}}}$ = {stats['cv_inv']:.3f}",
            rf"1$\sigma_q$ = [{one[0]:.1f}, {one[1]:.1f}] nm",
            rf"2$\sigma_q$ = [{two[0]:.1f}, {two[1]:.1f}] nm",
        ]
    )
    axis.text(
        0.55,
        0.95,
        text,
        transform=axis.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "honeydew", "alpha": 0.7},
    )


def _add_reciprocal_gaussian_fit(axis, diameters: np.ndarray, peak: float) -> None:
    """Plot the 1.0-style fit: Gaussian in inverse-size space, back-transformed and scaled to the histogram peak."""
    inverse = 1000.0 / diameters
    if len(inverse) < 2 or np.std(inverse) == 0:
        return
    grid = np.linspace(max(np.min(diameters) * 0.5, 1.0), min(np.max(diameters) * 1.5, 300.0), 1000)
    mean, std = np.mean(inverse), np.std(inverse, ddof=1)
    density = np.exp(-0.5 * ((1000.0 / grid - mean) / std) ** 2) / grid**2
    axis.plot(grid, density / density.max() * peak, color="C3")


def _legacy_log_ticks(axis, x: np.ndarray, y: np.ndarray) -> None:
    """Explicit tick labels instead of powers of ten, as in the 1.0 plots."""
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    x_ticks = np.array([1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000])
    y_ticks = np.array([1, 2, 3, 4, 6, 10, 15, 20, 30, 40, 60, 80, 100, 150, 200, 300, 400, 600, 800, 1000])
    x_low = x_ticks[x_ticks <= x.min()][-1] if (x_ticks <= x.min()).any() else x_ticks[0]
    x_high = x_ticks[x_ticks >= x.max()][0] if (x_ticks >= x.max()).any() else x_ticks[-1]
    y_low = y_ticks[y_ticks <= y.min()][-1] if (y_ticks <= y.min()).any() else y_ticks[0]
    y_high = y_ticks[y_ticks >= y.max()][0] if (y_ticks >= y.max()).any() else y_ticks[-1]
    axis.set_xticks(x_ticks)
    axis.set_yticks(y_ticks)
    axis.set_xlim(x_low, x_high)
    axis.set_ylim(y_low, y_high)
    for ticks in (axis.xaxis, axis.yaxis):
        ticks.set_major_formatter(ScalarFormatter())
        ticks.set_minor_formatter(NullFormatter())


def _valid_estimates(estimates: pd.DataFrame) -> pd.DataFrame:
    return estimates.loc[
        estimates["diameter_nm"].notna()
        & np.isfinite(estimates["diameter_nm"])
        & (estimates["diameter_nm"] > 0)
    ].copy()


def _histogram(
    values: pd.DataFrame,
    output_dir: Path,
    dimension: str,
    weighted: bool,
    plt,
) -> str:
    title_dimension = _dimension_title(dimension)
    diameters = values["diameter_nm"].to_numpy(float)
    if weighted:
        diameters = np.repeat(
            diameters, values["valid_frames"].to_numpy(dtype=int)
        )
    bin_size = 0.5
    minimum = np.floor(np.min(diameters) / bin_size) * bin_size
    maximum = np.ceil(np.max(diameters) / bin_size) * bin_size
    bins = np.arange(minimum, maximum + bin_size, bin_size)
    figure, axis = plt.subplots(figsize=(6, 6))
    counts, _, _ = axis.hist(diameters, bins=bins, color="C0")
    _add_reciprocal_gaussian_fit(axis, diameters, counts.max())
    _add_statistics_box(axis, diameters)
    weighting = "weighted by track length" if weighted else "each track counted once"
    axis.set(
        title=f"Amount of {title_dimension} trajectories analyzed = {len(values)}, {weighting}",
        xlabel="Diameter [nm]",
        ylabel="Absolute occurrence",
    )
    axis.set_xlim(10, 300)
    axis.grid(True)
    suffix = "weighted" if weighted else "unweighted"
    legacy_dimension = {"x": "long", "y": "trans", "2d": "2d"}.get(dimension, dimension)
    path = output_dir / f"Diameter_Histogramm-{legacy_dimension}-{suffix}.png"
    _save(figure, path, dpi=150)
    plt.close(figure)
    return str(path)


def create_analysis_plots(
    trajectories: pd.DataFrame,
    estimates: pd.DataFrame,
    output_dir: Path,
    *,
    pixel_size_um: float,
    fps: float,
    dimension: str,
    max_lag: int | None,
    dimension_label: str | None = None,
    drift: pd.DataFrame | None = None,
) -> dict[str, str]:
    """Save 1.0-compatible plots for one MSD dimension."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plot generation requires matplotlib.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    label = dimension_label or dimension
    paths: dict[str, str] = {}
    title_dimension = _dimension_title(dimension)

    figure, axis = plt.subplots(figsize=(10, 6))
    for particle, group in trajectories.groupby("particle"):
        axis.plot(group["x_corrected"], group["y_corrected"], linewidth=0.8, alpha=0.7)
    axis.set(title=f"{title_dimension} drift-corrected trajectories", xlabel="x [pixel]", ylabel="y [pixel]")
    legacy_label = {"x": "long", "y": "trans", "2d": "2d"}.get(dimension, dimension)
    paths["trajectories_plot"] = str(output_dir / f"Trajectories_{legacy_label}.png")
    _save(figure, paths["trajectories_plot"], dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 6))
    for particle, group in trajectories.groupby("particle"):
        msd = calculate_msd(group, dimension=dimension, max_lag=max_lag, pixel_size_um=pixel_size_um, fps=fps)
        if len(msd) < 2:
            continue
        fit = estimate_diffusion(msd, dimension)
        axis.plot(msd["time_s"], msd["msd_um2"], ".", alpha=0.45)
        axis.plot(msd["time_s"], fit["intercept_um2"] + fit["slope_um2_s"] * msd["time_s"], linewidth=1)
    axis.set(title=f"{title_dimension} MSD and linear fits", xlabel="lag time [s]", ylabel="MSD [um^2]")
    paths["msd_plot"] = str(output_dir / f"MSD_fit_{legacy_label}.png")
    _save(figure, paths["msd_plot"], dpi=150)
    plt.close(figure)

    valid = _valid_estimates(estimates)
    if not valid.empty:
        paths["diameter_histogram_unweighted"] = _histogram(valid, output_dir, label, False, plt)
        paths["diameter_histogram_weighted"] = _histogram(valid, output_dir, label, True, plt)

        figure, axis = plt.subplots(figsize=(10, 6))
        brightness = valid["mean_brightness"].to_numpy(float) if "mean_brightness" in valid else np.ones(len(valid))
        scatter = axis.scatter(
            valid["valid_frames"], valid["diameter_nm"],
            c=brightness, cmap="jet", alpha=0.9, edgecolors="black", linewidths=0.3,
        )
        colorbar = figure.colorbar(scatter, ax=axis)
        colorbar.set_label("Mean Brightness")
        axis.set(
            title=f"{title_dimension.capitalize()} Particle size over tracking time",
            xlabel="Trajectory length [frames]", ylabel="Diameter [nm]",
            xscale="log", yscale="log",
        )
        _legacy_log_ticks(axis, valid["valid_frames"].to_numpy(float), valid["diameter_nm"].to_numpy(float))
        axis.grid(alpha=0.25)
        paths["diameter_over_trajectory_plot"] = str(
            output_dir / f"DiameterOverTrajLength_{legacy_label}.png"
        )
        _save(figure, paths["diameter_over_trajectory_plot"], dpi=150)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 6))
    if drift is not None and {"frame", "drift_x", "drift_y"}.issubset(drift.columns):
        axis.plot(drift["frame"], drift["drift_y"], label="transversal")
        axis.plot(drift["frame"], drift["drift_x"], label="along fiber")
        axis.legend()
    else:
        axis.text(
            0.5,
            0.5,
            "No common drift estimate available",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.set(title="accumulated drift", xlabel="frames", ylabel="drift [px]")
    paths["global_drift_plot"] = str(output_dir / "Global_Drift.png")
    _save(figure, paths["global_drift_plot"], dpi=150)
    plt.close(figure)
    return paths
