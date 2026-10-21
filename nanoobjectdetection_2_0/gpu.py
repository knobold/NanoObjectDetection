"""Optional CuPy acceleration and GPU-memory-aware chunk sizing."""

from __future__ import annotations

import numpy as np


def get_cupy():
    """Return CuPy or ``None`` when the optional GPU backend is unavailable."""
    try:
        import cupy
    except ImportError:
        return None
    return cupy


def gpu_available() -> bool:
    """Return whether a CUDA device is available to CuPy."""
    cupy = get_cupy()
    if cupy is None:
        return False
    try:
        return int(cupy.cuda.runtime.getDeviceCount()) > 0
    except (RuntimeError, OSError):
        return False


def device_memory_bytes() -> int:
    """Return free memory on the active CUDA device."""
    cupy = get_cupy()
    if cupy is None:
        raise RuntimeError("CuPy is not installed")
    try:
        free, _ = cupy.cuda.Device().mem_info
    except (RuntimeError, OSError) as exc:
        raise RuntimeError("Could not query CUDA device memory") from exc
    return int(free)


def choose_chunk_frames(
    requested_frames: int,
    height: int,
    width: int,
    memory_fraction: float,
    bytes_per_pixel: int = 4,
) -> int:
    """Limit a chunk to a safe fraction of currently free GPU memory."""
    if not 0 < memory_fraction < 1:
        raise ValueError("gpu_memory_fraction must be between 0 and 1")
    free = device_memory_bytes()
    # Input, float32 working copy, background median, and output need headroom.
    bytes_per_frame = height * width * bytes_per_pixel
    safe_frames = int(free * memory_fraction / max(bytes_per_frame * 4, 1))
    return max(1, min(requested_frames, safe_frames))


def preprocess_gpu(
    frames: np.ndarray,
    remove_camera_offset: bool,
    remove_laser_fluctuation: bool = False,
    remove_static_background: bool = True,
    background_method: str = "legacy_trimmed_mean",
    enhance_snr: bool = False,
    background: np.ndarray | None = None,
    offset: float | None = None,
    snr_sigma_px: float = 1.0,
) -> np.ndarray:
    """Run preprocessing on CUDA and return a NumPy array for Trackpy."""
    cupy = get_cupy()
    if cupy is None or not gpu_available():
        raise RuntimeError("GPU preprocessing requested but no CUDA/CuPy device is available")
    device_frames = cupy.asarray(frames, dtype=cupy.float32)
    if remove_camera_offset:
        device_frames -= cupy.min(device_frames) if offset is None else offset
    if remove_laser_fluctuation:
        frame_intensity = cupy.mean(device_frames, axis=(1, 2))
        reference = cupy.median(frame_intensity)
        device_frames /= cupy.maximum(
            frame_intensity[:, None, None] / cupy.maximum(reference, 1e-12),
            1e-12,
        )
    if remove_static_background:
        if background is not None:
            device_frames -= cupy.asarray(background, dtype=cupy.float32)
        elif background_method == "median":
            background = cupy.median(device_frames, axis=0, keepdims=True)
        elif background_method == "legacy_trimmed_mean":
            ordered = cupy.sort(device_frames, axis=0)
            first = int(0.3 * len(ordered))
            last = max(first + 1, int(0.7 * len(ordered)))
            background = cupy.mean(ordered[first:last], axis=0, keepdims=True)
        else:
            raise ValueError(
                "background_method must be 'median' or 'legacy_trimmed_mean'"
            )
        device_frames -= background
    result = cupy.asnumpy(device_frames)
    if enhance_snr:
        from scipy.ndimage import gaussian_filter

        result = gaussian_filter(result, sigma=(0, snr_sigma_px, snr_sigma_px))
    del device_frames
    cupy.get_default_memory_pool().free_all_blocks()
    return result


def to_uint16_gpu(frames: np.ndarray, scale_max: float | None = None) -> np.ndarray:
    """Convert processed frames to uint16 on CUDA."""
    cupy = get_cupy()
    if cupy is None or not gpu_available():
        raise RuntimeError("GPU conversion requested but no CUDA/CuPy device is available")
    device_frames = cupy.maximum(cupy.asarray(frames, dtype=cupy.float32), 0)
    maximum = (
        float(cupy.asnumpy(cupy.max(device_frames)))
        if scale_max is None
        else scale_max
    )
    if maximum == 0:
        result = np.zeros(frames.shape, dtype=np.uint16)
    else:
        result = cupy.asnumpy(cupy.rint(device_frames * (65535.0 / maximum))).astype(
            np.uint16
        )
    del device_frames
    cupy.get_default_memory_pool().free_all_blocks()
    return result
