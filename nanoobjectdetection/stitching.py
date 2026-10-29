"""Stitching of trajectory fragments for videos that contain a single particle."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _fragment_table(trajectories: pd.DataFrame, edge_frames: int) -> pd.DataFrame:
    rows = []
    for particle, group in trajectories.sort_values("frame").groupby("particle"):
        head, tail = group.iloc[:edge_frames], group.iloc[-edge_frames:]
        rows.append(
            {
                "particle": particle,
                "start": int(group["frame"].iloc[0]),
                "end": int(group["frame"].iloc[-1]),
                "start_x": head["x"].mean(),
                "start_y": head["y"].mean(),
                "end_x": tail["x"].mean(),
                "end_y": tail["y"].mean(),
                "mass": float(group["mass"].median()) if "mass" in group else 1.0,
            }
        )
    return pd.DataFrame(rows)


def stitch_fragments(
    trajectories: pd.DataFrame,
    max_gap_frames: int,
    max_distance_px: float,
    distance_growth_px: float,
    max_mass_ratio: float,
    min_fragment_frames: int,
    edge_frames: int = 5,
) -> pd.DataFrame:
    """Give fragments that plausibly belong to one particle the same ``particle`` id.

    A fragment may follow another if it starts after the other ended, within
    ``max_gap_frames``, and its first positions lie within
    ``max_distance_px + distance_growth_px * sqrt(gap)`` of the other's last
    positions (diffusion widens the allowed jump with the gap). Median masses must
    agree within ``max_mass_ratio``. Pairs are accepted greedily, best first, and
    every fragment gets at most one predecessor and one successor.
    """
    if trajectories.empty:
        return trajectories.copy()
    sizes = trajectories.groupby("particle")["frame"].transform("size")
    kept = trajectories.loc[sizes >= min_fragment_frames].copy()
    if kept.empty:
        return kept.reset_index(drop=True)
    fragments = _fragment_table(kept, edge_frames).sort_values("start").reset_index(drop=True)

    candidates = []
    for a in fragments.itertuples():
        for b in fragments.itertuples():
            gap = b.start - a.end
            if a.particle == b.particle or gap < 1 or gap > max_gap_frames:
                continue
            limit = max_distance_px + distance_growth_px * np.sqrt(gap)
            distance = float(np.hypot(b.start_x - a.end_x, b.start_y - a.end_y))
            ratio = max(a.mass, b.mass) / max(min(a.mass, b.mass), 1e-12)
            if distance <= limit and ratio <= max_mass_ratio:
                candidates.append((distance / limit + 0.1 * np.log(ratio), a.particle, b.particle))
    candidates.sort()

    successor: dict = {}
    has_predecessor: set = set()
    for _, first, second in candidates:
        if first in successor or second in has_predecessor:
            continue
        successor[first] = second
        has_predecessor.add(second)

    new_id = {}
    for particle in fragments["particle"]:
        if particle in has_predecessor:
            continue
        current = particle
        while True:
            new_id[current] = particle
            if current not in successor:
                break
            current = successor[current]
    kept["particle"] = kept["particle"].map(new_id)
    return kept.reset_index(drop=True)


def keep_dominant_particle(trajectories: pd.DataFrame, criterion: str = "longest") -> pd.DataFrame:
    """Keep one stitched trajectory: most detections ("longest") or most total mass ("brightest")."""
    if criterion not in {"longest", "brightest"}:
        raise ValueError("criterion must be 'longest' or 'brightest'")
    if trajectories.empty:
        return trajectories.copy()
    grouped = trajectories.groupby("particle")
    score = grouped["mass"].sum() if criterion == "brightest" else grouped["frame"].size()
    return trajectories[trajectories["particle"] == score.idxmax()].reset_index(drop=True)
