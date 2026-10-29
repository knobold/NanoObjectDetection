"""Automatic geometric calibration from the image stack."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .storage import read_frames


def estimate_channel_center_y(
    frames: np.ndarray,
    intensity_percentile: float = 99.5,
) -> float:
    """Estimate the transverse channel center from a robust intensity profile.

    The high spatial percentile preserves the illuminated channel while the
    median over time suppresses moving particles. The center is the weighted
    centroid of the contiguous signal region above a conservative threshold.
    """
    data = np.asarray(frames, dtype=np.float32)
    if data.ndim != 3 or data.shape[1] < 3:
        raise ValueError("frames must have shape (frames, y, x)")
    if not 90.0 <= intensity_percentile < 100.0:
        raise ValueError("intensity_percentile must be in [90, 100)")

    profile = np.median(
        np.percentile(data, intensity_percentile, axis=2), axis=0
    )
    baseline = float(np.percentile(profile, 20))
    peak = float(np.max(profile))
    excess = profile - baseline
    if peak <= baseline or not np.isfinite(peak):
        raise ValueError("No channel signal found in transverse profile")

    threshold = baseline + 0.30 * (peak - baseline)
    active = profile >= threshold
    peak_index = int(np.argmax(profile))
    left = peak_index
    right = peak_index
    while left > 0 and active[left - 1]:
        left -= 1
    while right + 1 < len(active) and active[right + 1]:
        right += 1
    weights = np.maximum(excess[left : right + 1], 0)
    positions = np.arange(left, right + 1, dtype=float)
    # The midpoint is less sensitive to a few bright particles than a
    # brightness-weighted centroid when the illumination is asymmetric.
    return float((left + right) / 2.0)


def estimate_channel_center_from_tiff(
    path: str | Path,
    frame_start: int = 0,
    frame_stop: int | None = 300,
    intensity_percentile: float = 99.5,
) -> float:
    """Estimate the channel center from a bounded TIFF sample."""
    stop = frame_start + 300 if frame_stop is None else frame_stop
    return estimate_channel_center_y(
        read_frames(Path(path), frame_start, stop),
        intensity_percentile=intensity_percentile,
    )
