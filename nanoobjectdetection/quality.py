"""Quality checks used before converting MSD fits into particle diameters."""

from __future__ import annotations

import numpy as np
import pandas as pd
from math import erf, exp, sqrt

from .msd import calculate_msd, estimate_diffusion


def trajectory_quality(
    trajectory: pd.DataFrame,
    *,
    dimension: str,
    max_lag: int,
    pixel_size_um: float,
    fps: float,
    minimum_pairs_at_max_lag: int = 5,
    minimum_ks_significance: float = 0.01,
    ks_method: str = "exact",
) -> dict[str, float | bool | str]:
    """Return explicit acceptance diagnostics for one linked trajectory."""
    msd = calculate_msd(
        trajectory,
        dimension=dimension,
        max_lag=max_lag,
        pixel_size_um=pixel_size_um,
        fps=fps,
    )
    if len(msd) < 2:
        return {"accepted": False, "reason": "too_few_msd_points"}
    if int(msd["pairs"].iloc[-1]) < minimum_pairs_at_max_lag:
        return {"accepted": False, "reason": "too_few_pairs"}

    fit = estimate_diffusion(msd, dimension)
    if not np.isfinite(fit["diffusion_um2_s"]) or fit["diffusion_um2_s"] <= 0:
        return {
            "accepted": False,
            "reason": "non_positive_diffusion",
            "diffusion_um2_s": fit["diffusion_um2_s"],
        }

    values = trajectory.sort_values("frame")
    coordinate = "x_corrected" if dimension in {"x", "2d"} else "y_corrected"
    displacement = values[coordinate].diff().dropna().to_numpy(float)
    if dimension == "2d":
        dx = values["x_corrected"].diff().dropna().to_numpy(float)
        dy = values["y_corrected"].diff().dropna().to_numpy(float)
        displacement = np.hypot(dx, dy)
    displacement = displacement[np.isfinite(displacement)]
    if len(displacement) < minimum_pairs_at_max_lag:
        return {"accepted": False, "reason": "too_few_displacements"}
    mean = float(np.mean(displacement))
    std = float(np.std(displacement))
    if not np.isfinite(std) or std == 0:
        return {"accepted": False, "reason": "zero_displacement_variance"}
    ordered = np.sort(displacement)
    cdf = np.array(
        [0.5 * (1.0 + erf((value - mean) / (std * sqrt(2.0)))) for value in ordered]
    )
    sample_size = len(ordered)
    ranks = np.arange(1, sample_size + 1, dtype=float)
    statistic = float(
        max(
            np.max(ranks / sample_size - cdf),
            np.max(cdf - (ranks - 1.0) / sample_size),
        )
    )
    if ks_method == "legacy":
        # Legacy formula: first term of the Kolmogorov series with N = trajectory length
        pvalue = float(2.0 * exp(-2.0 * len(values) * statistic**2))
        accepted = bool(pvalue >= minimum_ks_significance)
        return {
            "accepted": accepted,
            "reason": "accepted" if accepted else "non_brownian",
            "diffusion_um2_s": fit["diffusion_um2_s"],
            "fit_points": fit["fit_points"],
            "pairs_at_max_lag": float(msd["pairs"].iloc[-1]),
            "ks_significance": pvalue,
        }
    scaled_statistic = (sqrt(sample_size) + 0.12 + 0.11 / sqrt(sample_size)) * statistic
    pvalue = 0.0
    for term in range(1, 101):
        contribution = 2.0 * (-1.0) ** (term - 1) * exp(
            -2.0 * term**2 * scaled_statistic**2
        )
        pvalue += contribution
        if abs(contribution) < 1e-12:
            break
    pvalue = float(np.clip(pvalue, 0.0, 1.0))
    accepted = bool(pvalue >= minimum_ks_significance)
    return {
        "accepted": accepted,
        "reason": "accepted" if accepted else "non_brownian",
        "diffusion_um2_s": fit["diffusion_um2_s"],
        "fit_points": fit["fit_points"],
        "pairs_at_max_lag": float(msd["pairs"].iloc[-1]),
        "ks_significance": float(pvalue),
    }


def filter_trajectories(
    trajectories: pd.DataFrame,
    *,
    dimension: str,
    max_lag: int,
    pixel_size_um: float,
    fps: float,
    minimum_pairs_at_max_lag: int = 5,
    minimum_ks_significance: float = 0.01,
    ks_method: str = "exact",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter trajectories and return accepted rows plus per-track diagnostics."""
    diagnostics = []
    accepted_ids: list[object] = []
    for particle, group in trajectories.groupby("particle", sort=True):
        result = trajectory_quality(
            group,
            dimension=dimension,
            max_lag=max_lag,
            pixel_size_um=pixel_size_um,
            fps=fps,
            minimum_pairs_at_max_lag=minimum_pairs_at_max_lag,
            minimum_ks_significance=minimum_ks_significance,
            ks_method=ks_method,
        )
        result["particle"] = particle
        diagnostics.append(result)
        if result["accepted"]:
            accepted_ids.append(particle)
    filtered = trajectories[trajectories["particle"].isin(accepted_ids)].reset_index(drop=True)
    return filtered, pd.DataFrame(diagnostics)
