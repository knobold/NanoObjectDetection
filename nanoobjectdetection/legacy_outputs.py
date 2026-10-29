"""Write CSV files with the file names, column order and columns used by 1.0."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OBJECT_COLUMNS = [
    "y", "x", "mass", "size", "ecc", "signal", "raw_mass", "ep", "frame",
    "abstime", "saturated",
]
TRAJECTORY_COLUMNS = [
    "frame", "y", "x", "mass", "size", "ecc", "signal", "raw_mass", "ep",
    "abstime", "saturated", "particle",
]
SIZE_COLUMNS = [
    "particle", "diameter", "diffusion", "diffusion std", "ep", "red_x",
    "signal", "mass", "rawmass_mean", "rawmass_median", "rawmass_max",
    "max step", "first frame", "traj length", "valid frames", "size", "ecc",
    "stat_sign", "true_particle", "x in first frame", "y in first frame",
]


def _ordered(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame.reindex(columns=columns)


def write_legacy_table(frame: pd.DataFrame, path: Path, columns: list[str]) -> None:
    _ordered(frame, columns).to_csv(path, index=False)


def legacy_trajectories(corrected: pd.DataFrame) -> pd.DataFrame:
    """Return a trajectory table whose x/y are the drift-corrected positions."""
    result = corrected.copy()
    if "x_corrected" in result:
        result["x"] = result["x_corrected"]
        result["y"] = result["y_corrected"]
    return _ordered(result, TRAJECTORY_COLUMNS)


def legacy_sizes_table(
    corrected: pd.DataFrame,
    moving: pd.DataFrame,
    estimates: pd.DataFrame,
    quality: pd.DataFrame,
    fps: float,
) -> pd.DataFrame:
    """Build the per-particle table written as ``sizes_df_lin_*.csv`` by 1.0."""
    valid = estimates[estimates["diameter_nm"].notna()].copy()
    if valid.empty:
        return pd.DataFrame(columns=SIZE_COLUMNS)
    stat = {}
    if len(quality) and "ks_significance" in quality:
        stat = dict(zip(quality["particle"], quality["ks_significance"]))
    start = (
        moving.sort_values("frame").groupby("particle").first()
        if len(moving)
        else pd.DataFrame()
    )
    tracks = {particle: group for particle, group in corrected.groupby("particle")}
    lagtime = 1.0 / fps
    rows = []
    for _, row in valid.iterrows():
        particle = row["particle"]
        track = tracks.get(particle)
        if track is None:
            continue
        track = track.sort_values("frame")
        red_x = row["intercept_um2"] / (lagtime * row["slope_um2_s"])
        red_x = max(red_x, -0.5)
        valid_frames = row["valid_frames"]
        error = np.sqrt(
            2.0 / (valid_frames - 1) * (1.0 + 2.0 * np.sqrt(1.0 + 2.0 * red_x))
        )
        first = start.loc[particle] if particle in start.index else track.iloc[0]
        rows.append(
            {
                "particle": int(particle),
                "diameter": row["diameter_nm"],
                "diffusion": row["diffusion_um2_s"],
                "diffusion std": row["diffusion_um2_s"] * error,
                "ep": track["ep"].mean(),
                "red_x": red_x,
                "signal": track["signal"].mean(),
                "mass": track["mass"].mean(),
                "rawmass_mean": track["raw_mass"].mean(),
                "rawmass_median": track["raw_mass"].median(),
                "rawmass_max": track["raw_mass"].max(),
                "max step": 0.0,
                "first frame": float(track["frame"].iloc[0]),
                "traj length": float(track["frame"].iloc[-1] - track["frame"].iloc[0] + 1),
                "valid frames": float(valid_frames),
                "size": track["size"].mean(),
                "ecc": track["ecc"].mean(),
                "stat_sign": stat.get(particle, np.nan),
                "true_particle": float(particle),
                "x in first frame": first["x"],
                "y in first frame": first["y"],
            }
        )
    return pd.DataFrame(rows, columns=SIZE_COLUMNS)


def export_legacy_aliases(output_dir: Path) -> None:
    """Write 1.0-compatible detection and trajectory CSVs without removing 2.0 files."""
    detections = output_dir / "detections.csv"
    if detections.is_file():
        table = pd.read_csv(detections)
        write_legacy_table(table, output_dir / "obj_all.csv", OBJECT_COLUMNS)
        write_legacy_table(
            table, output_dir / "obj_moving_no_go_area.csv", OBJECT_COLUMNS
        )
    for source, target in (
        ("stationary_trajectories.csv", "traj_fixed.csv"),
        ("trajectories.csv", "traj_moving.csv"),
    ):
        path = output_dir / source
        if path.is_file():
            write_legacy_table(pd.read_csv(path), output_dir / target, TRAJECTORY_COLUMNS)
