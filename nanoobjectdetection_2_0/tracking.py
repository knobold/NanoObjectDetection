"""Trajectory linking for the independent pipeline."""

from __future__ import annotations

import pandas as pd
import sys
import numpy as np


def link(
    detections: pd.DataFrame,
    search_range: float,
    memory: int = 1,
    minimum_frames: int = 100,
    adaptive_stop: float | None = None,
    adaptive_step: float = 0.9,
) -> pd.DataFrame:
    """Link detections without importing any historical project module."""
    try:
        import trackpy as tp
        from trackpy.linking.utils import SubnetOversizeException
    except ImportError as exc:
        raise RuntimeError(
            "Trajectory linking requires 'trackpy', but it is not "
            f"installed in the active interpreter ({sys.executable}). "
            "Install it with: python -m pip install trackpy"
        ) from exc
    tp.quiet()
    required = {"frame", "x", "y"}
    missing = required.difference(detections.columns)
    if missing:
        raise ValueError(f"Detections are missing columns: {sorted(missing)}")
    if search_range <= 0:
        raise ValueError("search_range must be positive")
    if not 0 < adaptive_step < 1:
        raise ValueError("adaptive_step must be between 0 and 1")
    if adaptive_stop is None:
        adaptive_stop = min(0.5, search_range * 0.25)
    if not 0 < adaptive_stop < search_range:
        raise ValueError("adaptive_stop must be positive and smaller than search_range")
    try:
        linked = tp.link_df(
            detections.sort_values("frame"),
            search_range=search_range,
            memory=memory,
            adaptive_stop=adaptive_stop,
            adaptive_step=adaptive_step,
        )
    except SubnetOversizeException as exc:
        raise RuntimeError(
            "Trackpy could not resolve a dense linking subnetwork even after "
            f"adaptive search-range reduction to {adaptive_stop:g} pixels. "
            "Reduce search_range, increase particle_minmass, or inspect "
            "overlapping detections."
        ) from exc
    lengths = linked.groupby("particle")["frame"].transform("size")
    return linked[lengths >= minimum_frames].reset_index(drop=True)


def remove_near_stationary(
    detections: pd.DataFrame,
    stationary: pd.DataFrame,
    distance: float,
) -> pd.DataFrame:
    """Remove detections inside the 1.0 stationary-particle no-go areas."""
    if distance < 0:
        raise ValueError("distance must be non-negative")
    if stationary.empty or detections.empty:
        return detections.copy()
    centers = stationary.groupby("particle")[["x", "y"]].mean().to_numpy()
    points = detections[["x", "y"]].to_numpy()
    keep = np.ones(len(points), dtype=bool)
    for center in centers:
        keep &= np.hypot(points[:, 0] - center[0], points[:, 1] - center[1]) > distance
    return detections.loc[keep].reset_index(drop=True)
