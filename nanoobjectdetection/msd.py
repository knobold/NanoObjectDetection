"""Mean-squared-displacement and diffusion estimation."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .progress import progress


def calculate_msd(
    trajectory: pd.DataFrame,
    dimension: str = "2d",
    max_lag: int | None = None,
    pixel_size_um: float = 1.0,
    fps: float = 300.0,
) -> pd.DataFrame:
    """Calculate MSD in µm² for one trajectory."""
    if dimension not in {"x", "y", "2d"}:
        raise ValueError("dimension must be 'x', 'y', or '2d'")
    x_col = "x_corrected" if "x_corrected" in trajectory else "x"
    y_col = "y_corrected" if "y_corrected" in trajectory else "y"
    values = trajectory.sort_values("frame").drop_duplicates("frame")
    frames = values["frame"].to_numpy(int)
    x = values[x_col].to_numpy(float) * pixel_size_um
    y = values[y_col].to_numpy(float) * pixel_size_um
    n = len(values)
    # Short lags are less affected by channel confinement and residual drift.
    limit = min(
        int(frames[-1] - frames[0]) if n else 0,
        max_lag or min(max(1, n // 4), 20),
    )
    rows = []
    for lag in range(1, limit + 1):
        right = np.searchsorted(frames, frames + lag)
        valid = right < n
        valid &= frames[np.minimum(right, n - 1)] == frames + lag
        left = np.nonzero(valid)[0]
        if not len(left):
            continue
        dx = x[right[left]] - x[left]
        dy = y[right[left]] - y[left]
        squared = dx * dx if dimension == "x" else dy * dy if dimension == "y" else dx * dx + dy * dy
        rows.append({"lag": lag, "time_s": lag / fps, "msd_um2": float(np.mean(squared)), "pairs": len(squared)})
    return pd.DataFrame(rows)


def estimate_diffusion(msd: pd.DataFrame, dimension: str = "2d") -> dict[str, float]:
    """Fit MSD = intercept + 2*d*D*time and return D in µm²/s."""
    if len(msd) < 2:
        raise ValueError("At least two MSD points are required")
    time = msd["time_s"].to_numpy(dtype=float)
    values = msd["msd_um2"].to_numpy(dtype=float)
    centered_time = time - float(np.mean(time))
    centered_values = values - float(np.mean(values))
    denominator = float(np.dot(centered_time, centered_time))
    if denominator == 0:
        raise ValueError("MSD fit requires at least two distinct lag times")
    slope = float(np.dot(centered_time, centered_values) / denominator)
    intercept = float(np.mean(values) - slope * np.mean(time))
    factor = 4.0 if dimension == "2d" else 2.0
    return {
        "diffusion_um2_s": float(slope / factor),
        "slope_um2_s": float(slope),
        "intercept_um2": float(intercept),
        "fit_points": float(len(msd)),
    }


def _ols(time: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(time, values, 1)
    return float(slope), float(intercept)


def _block_msd(positions: list[np.ndarray], lag: int) -> tuple[float, int]:
    """Legacy MSD at one lag: mean of non-overlapping block means of the squared displacements."""
    total = len(positions[0])
    squared = np.zeros(total)
    for pos in positions:
        displacement = np.full(total, np.nan)
        displacement[lag:] = pos[lag:] - pos[:-lag]
        squared = squared + displacement**2
    pairs = int(np.count_nonzero(~np.isnan(squared)))
    blocks = (total - lag) // lag + 1
    if blocks < 1:
        return float("nan"), pairs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        block_means = np.nanmean(squared[: blocks * lag].reshape(blocks, lag), axis=1)
        return float(np.nanmean(block_means)), pairs


def estimate_diffusion_auto_lag(
    trajectory: pd.DataFrame,
    dimension: str,
    pixel_size_um: float,
    fps: float,
    max_iterations: int = 10,
    fixed_red_x: float | None = None,
) -> dict[str, float] | None:
    """Reproduce the 1.0 MSD fit with automatic lag selection (Michalet 2012, appendix B).

    Follows CalcDiameter.OptimizeMSD of 1.0 step by step, including its details:
    block-averaged MSD, the slope being fitted with the second entry of the lag pair
    returned by UpdateP_Min (the offset with the first), and convergence on that entry.
    Returns None if the track is too short, has too many holes or does not converge.
    """
    values = trajectory.sort_values("frame").drop_duplicates("frame")
    n_rows = len(values)
    frames = values["frame"].to_numpy(int)
    x_col = "x_corrected" if "x_corrected" in values else "x"
    y_col = "y_corrected" if "y_corrected" in values else "y"
    total = int(frames[-1] - frames[0]) + 1 if n_rows else 0
    positions = []
    for use, column in ((dimension in ("x", "2d"), x_col), (dimension in ("y", "2d"), y_col)):
        if use:
            pos = np.full(total, np.nan)
            pos[frames - frames[0]] = values[column].to_numpy(float) * pixel_size_um
            positions.append(pos)
    if not positions:
        raise ValueError("dimension must be 'x', 'y', or '2d'")
    start = min(max(int(np.ceil(n_rows / 10)), 2), 100)
    lags = [start, start]
    factor = 4.0 if dimension == "2d" else 2.0
    t_frame = 1.0 / fps
    for counter in range(max_iterations + 1):
        lag_max = max(lags)
        if total < lag_max:
            return None
        msd, pairs = zip(*(_block_msd(positions, lag) for lag in range(1, lag_max + 1)))
        if pairs[lag_max - 1] < lag_max + 1 or np.isnan(msd).any():
            return None
        pairs_lag1 = pairs[0]
        time = np.arange(1, lag_max + 1) / fps
        msd = np.asarray(msd)
        slope, _ = _ols(time[: lags[1]], msd[: lags[1]])
        _, intercept = _ols(time[: lags[0]], msd[: lags[0]])
        l_a = 3 + max(4.5 * pairs_lag1**0.4 - 8.5, 0.0) ** 1.2
        l_b = 0.8 + 0.564 * pairs_lag1
        if slope < 0:
            new = [int(l_a), int(l_b)]
        else:
            if fixed_red_x is not None:
                red_x = max(fixed_red_x, 0.0)
            else:
                red_x = max(intercept / (t_frame * slope), 0.0) if slope else 0.0
            f_a = 2 + 1.6 * red_x**0.51
            p_a = max(np.floor(f_a * l_a / (f_a**3 + l_a**3) ** (1 / 3)), 2)
            f_b = 2 + 1.35 * red_x**0.6
            p_b = max(np.floor(min(l_b, f_b * l_b / (f_b**3 + l_b**3) ** (1 / 3))), 2)
            new = [int(p_b), int(p_a)]
        if counter == max_iterations:
            return None
        if new[1] == lags[1]:
            return {
                "diffusion_um2_s": slope / factor,
                "slope_um2_s": slope,
                "intercept_um2": intercept,
                "fit_points": float(lags[1]),
                "msd_pairs": int(sum(pairs)),
                "valid_frames": int(pairs_lag1),
            }
        lags = new
    return None


def estimate_all(
    trajectories: pd.DataFrame,
    auto_lag: bool = False,
    fixed_red_x: float | None = None,
    **kwargs,
) -> pd.DataFrame:
    rows = []
    dimension = kwargs.get("dimension", "2d")
    groups = trajectories.groupby("particle")
    for particle, trajectory in progress(
        groups,
        desc="Estimating MSD and particle size",
        total=trajectories["particle"].nunique(),
        unit="particle",
    ):
        if auto_lag:
            row = estimate_diffusion_auto_lag(
                trajectory,
                dimension,
                kwargs.get("pixel_size_um", 1.0),
                kwargs.get("fps", 300.0),
                fixed_red_x=fixed_red_x,
            )
            if row is None:
                continue
        else:
            msd = calculate_msd(trajectory, **kwargs)
            if len(msd) < 2:
                continue
            row = estimate_diffusion(msd, dimension)
            row["msd_pairs"] = int(msd["pairs"].sum())
            row["valid_frames"] = int(msd["pairs"].iloc[0])
        row.update(
            {
                "particle": particle,
                "trajectory_frames": len(trajectory),
                "mean_brightness": float(
                    trajectory["mass"].mean()
                    if "mass" in trajectory
                    else trajectory["signal"].mean()
                    if "signal" in trajectory
                    else float("nan")
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
