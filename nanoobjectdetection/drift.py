"""Drift estimation from linked trajectories."""

from __future__ import annotations

import numpy as np
import pandas as pd
import trackpy as tp


def correct_global(
    trajectories: pd.DataFrame,
    window: int = 20,
    min_particles: int = 2,
    minimum_frames: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate frame-to-frame common motion and subtract its cumulative sum."""
    required = {"frame", "x", "y"}
    missing = required.difference(trajectories.columns)
    if missing:
        raise ValueError(f"Trajectories are missing columns: {sorted(missing)}")
    result = trajectories.copy().sort_values(["frame", "particle"])
    # Saturated positions are poorly localized and, as in 1.0, excluded from the estimate
    estimation = result
    if "saturated" in result:
        estimation = result[~result["saturated"].astype(bool)]
    if minimum_frames > 1:
        lengths = estimation.groupby("particle")["frame"].transform("size")
        estimation = estimation[lengths >= minimum_frames]
    drift = tp.compute_drift(
        estimation[["particle", "frame", "x", "y"]],
        smoothing=window,
    )
    full_index = result["frame"].sort_values().unique()
    # trackpy starts at the second frame; the first frame is the zero-drift reference (as in 1.0)
    drift = drift.reindex(full_index).interpolate(method="linear").ffill().fillna(0.0)
    drift.index.name = "frame"
    drift = drift.rename(columns={"x": "drift_x", "y": "drift_y"}).reset_index()
    if drift.empty:
        result["x_corrected"] = result["x"]
        result["y_corrected"] = result["y"]
        return result, drift
    correction = drift.set_index("frame")[["drift_x", "drift_y"]]
    result = result.join(correction, on="frame")
    result[["drift_x", "drift_y"]] = result[["drift_x", "drift_y"]].fillna(0.0)
    result["x_corrected"] = result["x"] - result["drift_x"]
    result["y_corrected"] = result["y"] - result["drift_y"]
    return result.reset_index(drop=True), drift.reset_index(drop=True)
