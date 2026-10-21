"""Trajectory filtering and splitting compatible with the 1.0 workflow."""

from __future__ import annotations

import numpy as np
import pandas as pd


def filter_minimum_frames(
    trajectories: pd.DataFrame, minimum_frames: int
) -> pd.DataFrame:
    """Keep trajectories with at least ``minimum_frames`` measured positions."""
    if minimum_frames < 1:
        raise ValueError("minimum_frames must be positive")
    if trajectories.empty:
        return trajectories.copy()
    lengths = trajectories.groupby("particle")["frame"].transform("size")
    return trajectories.loc[lengths >= minimum_frames].reset_index(drop=True)


def _interpolated_mass(group: pd.DataFrame) -> pd.Series:
    indexed = group.sort_values("frame").set_index("frame")["mass"].astype(float)
    full = indexed.reindex(range(int(indexed.index.min()), int(indexed.index.max()) + 1))
    return full.interpolate(method="nearest").bfill().ffill()


def split_intensity_jumps(
    trajectories: pd.DataFrame,
    relative_step: float,
    minimum_frames: int,
) -> pd.DataFrame:
    """Split tracks at large relative mass jumps and discard short fragments."""
    if relative_step <= 0:
        raise ValueError("relative_step must be positive")
    required = {"particle", "frame", "mass"}
    missing = required.difference(trajectories.columns)
    if missing:
        raise ValueError(f"Trajectories are missing columns: {sorted(missing)}")

    result = trajectories.copy().sort_values(["particle", "frame"])
    next_particle = int(result["particle"].max()) + 1 if not result.empty else 0
    for particle, group in list(result.groupby("particle", sort=False)):
        mass = _interpolated_mass(group)
        smooth = mass.rolling(3, center=True, min_periods=1).median()
        previous = smooth.shift(1)
        relative = (smooth - previous).abs() / ((smooth + previous.abs()) / 2.0)
        split_frames = relative.index[relative > relative_step].tolist()
        for frame in split_frames:
            mask = (result["particle"] == particle) & (result["frame"] >= frame)
            result.loc[mask, "particle"] = next_particle
            next_particle += 1
            particle = next_particle - 1
    return filter_minimum_frames(result, minimum_frames)


def split_long_trajectories(
    trajectories: pd.DataFrame,
    maximum_frames: int | None,
    minimum_frames: int,
    keep_tail: bool = False,
) -> pd.DataFrame:
    """Split long tracks into bounded contiguous frame segments."""
    if maximum_frames is None:
        return trajectories.copy()
    if maximum_frames < minimum_frames:
        raise ValueError("maximum_frames must be at least minimum_frames")
    result_parts: list[pd.DataFrame] = []
    next_particle = int(trajectories["particle"].max()) + 1 if not trajectories.empty else 0
    for _, group in trajectories.sort_values(["particle", "frame"]).groupby("particle"):
        group = group.copy()
        for start in range(0, len(group), maximum_frames):
            part = group.iloc[start : start + maximum_frames].copy()
            if len(part) < minimum_frames and not keep_tail:
                continue
            if start:
                part["particle"] = next_particle
                next_particle += 1
            result_parts.append(part)
    if not result_parts:
        return trajectories.iloc[0:0].copy()
    return pd.concat(result_parts, ignore_index=True)
