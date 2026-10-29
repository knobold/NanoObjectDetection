"""Array preprocessing that does not depend on the 1.0 package."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def preprocess(
    frames: np.ndarray,
    remove_camera_offset: bool = True,
    remove_laser_fluctuation: bool = False,
    remove_static_background: bool = True,
    background_method: str = "legacy_trimmed_mean",
    enhance_snr: bool = False,
    background: np.ndarray | None = None,
    offset: float | None = None,
    snr_sigma_px: float = 1.0,
) -> np.ndarray:
    """Return float32 frames after deterministic basic corrections."""
    result = np.asarray(frames, dtype=np.float32).copy()
    if remove_camera_offset:
        result -= np.min(result) if offset is None else np.float32(offset)
    if remove_laser_fluctuation:
        frame_intensity = np.mean(result, axis=(1, 2))
        reference = float(np.median(frame_intensity))
        if reference > 0:
            result /= np.maximum(frame_intensity[:, None, None] / reference, 1e-12)
    if remove_static_background:
        if background is not None:
            background = np.asarray(background, dtype=np.float32)
        elif background_method == "median":
            background = np.median(result, axis=0, keepdims=True)
        elif background_method == "legacy_trimmed_mean":
            ordered = np.sort(result, axis=0)
            first = int(0.3 * len(ordered))
            last = max(first + 1, int(0.7 * len(ordered)))
            background = np.mean(ordered[first:last], axis=0, keepdims=True)
        else:
            raise ValueError(
                "background_method must be 'median' or 'legacy_trimmed_mean'"
            )
        result -= background
    if enhance_snr:
        result = gaussian_filter(result, sigma=(0, snr_sigma_px, snr_sigma_px))
    return result


def to_uint16(frames: np.ndarray, scale_max: float | None = None) -> np.ndarray:
    """Convert processed frames to positive uint16 without overflow."""
    result = np.asarray(frames, dtype=np.float32)
    result = np.maximum(result, 0)
    maximum = float(np.max(result) if scale_max is None else scale_max)
    if maximum == 0:
        return np.zeros(result.shape, dtype=np.uint16)
    return np.rint(result * (65535.0 / maximum)).astype(np.uint16)
