"""Automatic detection parameters ported from 1.0 (PSF width, trackpy diameter, minmass)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree


def sigma_psf_um(numerical_aperture: float, wavelength_nm: float) -> float:
    """Gaussian approximation of the PSF width in µm (Zhang et al. 2007, as in 1.0)."""
    return 0.21 * wavelength_nm / numerical_aperture / 1000.0


def sigma_psf_px(
    numerical_aperture: float, wavelength_nm: float, pixel_size_um: float
) -> float:
    return sigma_psf_um(numerical_aperture, wavelength_nm) / pixel_size_um


def trackpy_diameter(
    numerical_aperture: float,
    wavelength_nm: float,
    pixel_size_um: float,
    enhance_snr: bool,
) -> int:
    """Odd trackpy feature diameter covering the 2-sigma interval of the PSF."""
    diameter_px = 4.0 * sigma_psf_px(numerical_aperture, wavelength_nm, pixel_size_um)
    if enhance_snr:
        # Convolving the PSF with a Gaussian of equal width widens it by sqrt(2)
        diameter_px *= np.sqrt(2.0)
    diameter = int(np.ceil(diameter_px))
    return diameter + 1 if diameter % 2 == 0 else diameter


def red_x_from_theory(
    exposure_time_s: float,
    lagtime_s: float,
    numerical_aperture: float,
    wavelength_nm: float,
    diffusion_um2_s: float = 1.0,
    photons: float = 1.0,
) -> float:
    """Reduced squared localization error from theory (Michalet & Berglund 2012).

    Mirrors 1.0, which evaluates it with D = 1 µm²/s and one photon.
    """
    blur = exposure_time_s / lagtime_s / 6.0
    s0 = sigma_psf_um(numerical_aperture, wavelength_nm)
    sigma0 = s0 / np.sqrt(photons)
    sigma = sigma0 * np.sqrt(1.0 + diffusion_um2_s * exposure_time_s / s0**2)
    return float(sigma**2 / (diffusion_um2_s * lagtime_s) - 2.0 * blur)


def gaussian_kernel(sigma: float, factor: float = 6.0) -> np.ndarray:
    size = int(np.ceil(factor * 2 * sigma))
    size += 1 - size % 2
    radius = (size - 1) // 2
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    kernel = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def zncc_image(frame: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Zero-normalised cross-correlation of a frame with a kernel for every pixel.

    Uses the identity zncc = sum(p * (k - mean k)) / (N * std(p) * std(k)), so the
    whole image is handled by one FFT correlation and two box filters. Pixels whose
    window leaves the image are 0, as in 1.0.
    """
    frame = np.asarray(frame, dtype=np.float64)
    size = kernel.shape[0]
    count = float(size * size)
    centered = kernel - kernel.mean()
    kernel_std = np.sqrt(np.mean(centered**2))
    numerator = fftconvolve(frame, centered[::-1, ::-1], mode="same") / count
    local_mean = ndimage.uniform_filter(frame, size, mode="constant")
    local_square = ndimage.uniform_filter(frame**2, size, mode="constant")
    local_std = np.sqrt(np.maximum(local_square - local_mean**2, 0.0))
    denominator = local_std * kernel_std
    result = np.divide(
        numerator, denominator, out=np.zeros_like(frame), where=denominator > 1e-9
    )
    margin = (size - 1) // 2
    result[:margin] = 0
    result[-margin:] = 0
    result[:, :margin] = 0
    result[:, -margin:] = 0
    return result


def zncc_particles(
    frames: np.ndarray,
    sigma_px: float,
    enhance_snr: bool,
    minimum_correlation: float,
) -> pd.DataFrame:
    """Locate particles by ZNCC with the PSF; returns y, x, frame (local index)."""
    if enhance_snr:
        kernel = gaussian_kernel(sigma_px * np.sqrt(2.0), factor=10.0)
    else:
        kernel = gaussian_kernel(sigma_px)
    structure = ndimage.generate_binary_structure(2, 2)
    rows = []
    for index, frame in enumerate(frames):
        frame = np.maximum(np.asarray(frame, dtype=np.float64), 0.0)
        if enhance_snr:
            frame = ndimage.gaussian_filter(frame, sigma_px)
        labels, count = ndimage.label(
            zncc_image(frame, kernel) > minimum_correlation, structure=structure
        )
        if count:
            centers = ndimage.center_of_mass(np.ones_like(labels), labels, range(1, count + 1))
            rows.extend((int(y), int(x), index) for y, x in centers)
    return pd.DataFrame(rows, columns=["y", "x", "frame"])


def optimize_minmass(
    candidates: pd.DataFrame,
    reference: pd.DataFrame,
    max_distance: float,
    max_iterations: int = 100000,
) -> tuple[float, pd.DataFrame]:
    """Raise minmass until trackpy detections match the ZNCC reference best (1.0 algorithm).

    ``candidates`` are trackpy detections found with minmass=1; raising the threshold
    is then a simple mass filter. Returns the optimum (before the 0.9 safety factor)
    and the history of the search.
    """
    reference_points = {
        frame: group[["y", "x"]].to_numpy(float) for frame, group in reference.groupby("frame")
    }
    by_frame = {frame: group for frame, group in candidates.groupby("frame")}
    n_reference = len(reference)
    minmass = 1
    history = []
    first = True
    for _ in range(max_iterations):
        trees = {}
        n_found = 0
        for frame, group in by_frame.items():
            kept = group[group["mass"] > minmass]
            n_found += len(kept)
            if len(kept):
                trees[frame] = cKDTree(kept[["y", "x"]].to_numpy(float))
        if first and n_found < n_reference:
            raise RuntimeError(
                "Trackpy finds fewer particles than the ZNCC reference even at minmass=1. "
                "The sample may be too dense, or tp_percentile too aggressive."
            )
        first = False
        if n_found > 5 * n_reference:
            minmass = int(minmass * 1.2) + 10
            continue
        if n_found > 2 * n_reference:
            minmass = int(minmass * 1.1) + 10
            continue
        right = 0
        wrong = max(n_found - n_reference, 0)
        for frame, points in reference_points.items():
            tree = trees.get(frame)
            if tree is None:
                wrong += len(points)
                continue
            distance, _ = tree.query(points)
            hits = int(np.sum(distance <= max_distance))
            right += hits
            wrong += len(points) - hits
        ratio = wrong / right if right else np.inf
        history.append((minmass, right, wrong, ratio))
        if wrong == 0 or n_found < 0.8 * n_reference:
            break
        minmass = int(minmass * 1.02) + 1
    table = pd.DataFrame(history, columns=["minmass", "right_found", "wrong_found", "wrong_to_right"])
    return float(table.loc[table["wrong_to_right"].idxmin(), "minmass"]), table
