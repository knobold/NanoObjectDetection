"""Weighted summary statistics for particle-size estimates."""

from __future__ import annotations

import numpy as np
import pandas as pd


def weighted_quantile(
    values: pd.Series | np.ndarray,
    weights: pd.Series | np.ndarray,
    quantile: float,
) -> float:
    """Return a quantile using non-negative observation weights."""
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between 0 and 1")
    values_array = np.asarray(values, dtype=float)
    weights_array = np.asarray(weights, dtype=float)
    valid = np.isfinite(values_array) & np.isfinite(weights_array) & (weights_array > 0)
    if not np.any(valid):
        return float("nan")
    values_array = values_array[valid]
    weights_array = weights_array[valid]
    order = np.argsort(values_array)
    values_array = values_array[order]
    weights_array = weights_array[order]
    cumulative = np.cumsum(weights_array)
    return float(values_array[np.searchsorted(cumulative, quantile * cumulative[-1])])


def diameter_summary(estimates: pd.DataFrame) -> pd.DataFrame:
    """Create unweighted, frame-weighted, and MSD-pair-weighted summaries."""
    valid = estimates[
        estimates["diameter_nm"].notna()
        & np.isfinite(estimates["diameter_nm"])
        & (estimates["diameter_nm"] > 0)
    ].copy()
    if valid.empty:
        return pd.DataFrame(
            columns=["weighting", "particles", "weight_sum", "median_nm", "q25_nm", "q75_nm"]
        )
    rows = []
    weighting = {
        "unweighted": np.ones(len(valid)),
        "trajectory_frames": valid["trajectory_frames"].to_numpy(float),
        "msd_pairs": valid["msd_pairs"].to_numpy(float),
    }
    for name, weights in weighting.items():
        rows.append(
            {
                "weighting": name,
                "particles": len(valid),
                "weight_sum": float(weights.sum()),
                "median_nm": weighted_quantile(valid["diameter_nm"], weights, 0.5),
                "q25_nm": weighted_quantile(valid["diameter_nm"], weights, 0.25),
                "q75_nm": weighted_quantile(valid["diameter_nm"], weights, 0.75),
            }
        )
    return pd.DataFrame(rows)
