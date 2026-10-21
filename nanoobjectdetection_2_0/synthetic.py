"""Deterministic synthetic trajectories for numerical pipeline tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def brownian_trajectories(
    particles: int = 3,
    frames: int = 200,
    diffusion_um2_s: float = 1.0,
    fps: float = 300.0,
    seed: int = 42,
    drift_per_frame: tuple[float, float] = (0.0, 0.0),
) -> pd.DataFrame:
    """Generate linked trajectories in pixel coordinates."""
    rng = np.random.default_rng(seed)
    rows = []
    step = np.sqrt(2 * diffusion_um2_s / fps)
    for particle in range(particles):
        positions = np.zeros((frames, 2))
        positions[0] = rng.normal(0, 1, 2)
        positions[1:] = positions[0] + np.cumsum(rng.normal(0, step, (frames - 1, 2)), axis=0)
        positions += np.arange(frames)[:, None] * drift_per_frame
        for frame, (x, y) in enumerate(positions):
            rows.append({"particle": particle, "frame": frame, "x": x, "y": y})
    return pd.DataFrame(rows)
