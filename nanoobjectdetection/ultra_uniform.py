"""Ultra-uniform particle hypothesis filter (z-score based, as in 1.0 PostProcessing)."""

from __future__ import annotations

import numpy as np
import pandas as pd

# 99 % two-sided confidence level
DEFAULT_Z_MAX = 2.576


def _spread(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean((values - np.median(values)) ** 2)))


def _remove_outliers(data: pd.DataFrame, column: str, roll_area: int, z_max: float) -> pd.DataFrame:
    """Iteratively drop rows whose z-score (median-centred, rolling spread) reaches z_max."""
    while len(data):
        spread = data[column].rolling(roll_area, min_periods=1, center=True).apply(_spread, raw=True)
        z = (data[column] - data[column].median()).abs() / spread
        if z.max() < z_max:
            break
        data = data[z < z_max]
    return data


def ultra_uniform_mask(
    estimates: pd.DataFrame,
    z_max: float = DEFAULT_Z_MAX,
    diameter_column: str = "diameter_nm",
    diffusion_column: str = "diffusion_um2_s",
    order_column: str = "valid_frames",
) -> pd.Series:
    """Return a boolean mask of particles that fit a single uniform particle species.

    Particles are ordered by trajectory precision, then diameter and diffusion are
    filtered one after the other (diffusion on the particles that survived the
    diameter step), exactly like ForceUltraUniformParticle in 1.0.
    """
    valid = estimates[estimates[diameter_column].notna() & estimates[diffusion_column].notna()]
    num_particles = len(valid)
    if num_particles < 10:
        return pd.Series(True, index=estimates.index)
    roll_area = 100 if num_particles >= 100 else num_particles - 2
    data = valid[[diameter_column, diffusion_column, order_column]].sort_values(order_column)
    data = _remove_outliers(data, diameter_column, roll_area, z_max)
    data = _remove_outliers(data, diffusion_column, roll_area, z_max)
    keep = pd.Series(False, index=estimates.index)
    keep.loc[data.index] = True
    # Rows without a valid estimate were already rejected earlier and are left untouched
    keep.loc[~estimates.index.isin(valid.index)] = True
    return keep
