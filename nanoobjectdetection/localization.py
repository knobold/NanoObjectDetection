"""Particle localization using a direct, optional Trackpy dependency."""

from __future__ import annotations

import pandas as pd
import numpy as np
import os
import sys
from concurrent.futures import ProcessPoolExecutor

def _locate_frame(args):
    import trackpy as tp

    frame, diameter, minmass, separation, percentile, preprocess = args
    return tp.locate(
        frame,
        diameter,
        minmass=minmass,
        separation=separation,
        percentile=percentile,
        preprocess=preprocess,
        engine="python",
    )


def locate(
    frames: np.ndarray,
    diameter: int,
    minmass: float,
    separation: int,
    percentile: float,
    preprocess: bool = True,
    workers: int = 1,
    executor: ProcessPoolExecutor | None = None,
) -> pd.DataFrame:
    """Locate particles in one block and preserve local frame numbers.

    ``workers`` > 1 spreads frames over processes; 0 uses all cores.
    """
    try:
        import trackpy as tp  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Particle localization requires 'trackpy', but it is not "
            f"installed in the active interpreter ({sys.executable}). "
            "Install it with: python -m pip install trackpy"
        ) from exc

    frame_list = np.asarray(frames)
    jobs = [
        (frame, diameter, minmass, separation, percentile, preprocess)
        for frame in frame_list
    ]
    workers = (os.cpu_count() or 1) if workers == 0 else workers
    if workers > 1 and len(jobs) > 1:
        # A few large batches per worker keep process start-up and pickling overhead low
        batch = max(1, -(-len(jobs) // (workers * 4)))
        if executor is not None:
            results = list(executor.map(_locate_frame, jobs, chunksize=batch))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_locate_frame, jobs, chunksize=batch))
    else:
        results = [_locate_frame(job) for job in jobs]
    detections = []
    for frame_number, found in enumerate(results):
        if not found.empty:
            found = found.copy()
            found["frame"] = frame_number
            detections.append(found)
    if not detections:
        return pd.DataFrame(columns=["x", "y", "frame"])
    return pd.concat(detections, ignore_index=True)
