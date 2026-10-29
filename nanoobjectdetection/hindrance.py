"""Hydrodynamic diameter and position-aware channel correction."""

from __future__ import annotations

import numpy as np
from scipy.constants import Boltzmann


def local_channel_diameter_nm(
    y_px: np.ndarray,
    center_y_px: float,
    channel_diameter_nm: float,
    pixel_size_um: float,
) -> float:
    """Estimate the mean available diameter in a circular channel."""
    if channel_diameter_nm <= 0 or pixel_size_um <= 0:
        raise ValueError("Channel diameter and pixel size must be positive")
    radius_px = channel_diameter_nm / 1000.0 / pixel_size_um / 2.0
    available = np.maximum(radius_px - np.abs(np.asarray(y_px) - center_y_px), 0.0)
    return float(np.mean(2.0 * available * pixel_size_um * 1000.0))


def hindrance_factor(channel_diameter_nm: float, particle_diameter_nm: float) -> float:
    """Return the Dechadilok-Deen diffusive hindrance factor Kd."""
    ratio = particle_diameter_nm / channel_diameter_nm
    if not 0 < ratio < 1:
        raise ValueError("Particle must be smaller than the local channel")
    h = (
        1 + 9 / 8 * ratio * np.log(ratio) - 1.56034 * ratio
        + 0.528155 * ratio**2 + 1.915210 * ratio**3
        - 2.819030 * ratio**4 + 0.270788 * ratio**5
        + 1.101150 * ratio**6 - 0.435933 * ratio**7
    )
    return float(h / (1 - ratio) ** 2)


def corrected_diameter_nm(
    diffusion_um2_s: float,
    temperature_k: float,
    viscosity_pa_s: float,
    channel_diameter_nm: float | None = None,
) -> float:
    """Convert diffusion to diameter and optionally invert channel hindrance."""
    if diffusion_um2_s <= 0 or temperature_k <= 0 or viscosity_pa_s <= 0:
        raise ValueError("Diffusion, temperature, and viscosity must be positive")
    diameter = 2 * Boltzmann * temperature_k / (6 * np.pi * viscosity_pa_s) * 1e9 / (diffusion_um2_s * 1e-12)
    if channel_diameter_nm is None or diameter >= channel_diameter_nm:
        return float(diameter)
    # Same damped fixed-point iteration as 1.0 (geometric mean of new and old estimate)
    corrected, previous = diameter, 0.0
    for _ in range(100):
        if abs(corrected - previous) <= 0.1:
            return float(corrected)
        previous = corrected
        corrected = np.sqrt(diameter * hindrance_factor(channel_diameter_nm, previous) * previous)
    return float(previous)
