"""Chunked TIFF storage for the independent pipeline."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from .progress import progress


def _tifffile():
    try:
        import tifffile
    except ImportError as exc:
        raise RuntimeError("Install 'tifffile' to use TIFF storage.") from exc
    return tifffile


def tiff_info(path: Path) -> dict[str, Any]:
    tifffile = _tifffile()
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        return {
            "path": str(path.resolve()),
            "shape": tuple(series.shape),
            "axes": series.axes,
            "dtype": str(series.dtype),
            "pages": len(tif.pages),
        }


def to_processing_bit_depth(
    frames: np.ndarray, processing_bit_depth: int
) -> np.ndarray:
    """Convert unsigned TIFF frames to the selected processing bit depth."""
    if processing_bit_depth == 16:
        return frames
    if processing_bit_depth != 8:
        raise ValueError("processing_bit_depth must be 8 or 16")

    data = np.asarray(frames)
    if data.dtype.kind != "u":
        raise ValueError(
            "8-bit processing requires an unsigned 8-bit or 16-bit TIFF stack"
        )
    if data.dtype.itemsize == 1:
        return data
    if data.dtype.itemsize == 2:
        return np.right_shift(data, 8).astype(np.uint8)
    raise ValueError(
        "8-bit processing requires an unsigned 8-bit or 16-bit TIFF stack"
    )


def read_frames(path: Path, first: int, stop: int) -> np.ndarray:
    tifffile = _tifffile()
    try:
        mapped = tifffile.memmap(path, mode="r")
        result = np.array(mapped[first:stop], copy=True)
        del mapped
        return result
    except (OSError, ValueError, RuntimeError):
        return np.asarray(tifffile.imread(path, key=slice(first, stop)))


def iter_tiff_frames(
    path: Path, frame_start: int, frame_stop: int | None
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield source TIFF frames sequentially without creating a chunk store."""
    tifffile = _tifffile()
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        total = int(series.shape[0]) if len(series.shape) > 2 else len(tif.pages)
        stop = total if frame_stop is None else min(frame_stop, total)
        if frame_start < 0 or frame_start >= stop:
            raise ValueError("Configured frame range is outside the source TIFF")
        if len(tif.pages) >= total:
            for frame_number in range(frame_start, stop):
                yield frame_number, tif.pages[frame_number].asarray()
            return
    # ImageJ stacks store all frames behind a single page header; map them directly
    mapped = tifffile.memmap(path, mode="r")
    try:
        for frame_number in range(frame_start, stop):
            yield frame_number, np.array(mapped[frame_number])
    finally:
        del mapped


def create_store(
    source: Path,
    target: Path,
    frame_start: int,
    frame_stop: int | None,
    chunk_frames: int,
    chunk_overlap_frames: int,
    y_slice: slice,
    x_slice: slice,
    processing_bit_depth: int = 16,
) -> dict[str, Any]:
    """Create a reproducible chunk store without changing the source."""
    info = tiff_info(source)
    total = int(info["shape"][0])
    stop = total if frame_stop is None else min(frame_stop, total)
    if frame_start >= stop:
        raise ValueError("Configured frame range is outside the source TIFF")

    chunks_dir = target / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    tifffile = _tifffile()
    manifest: dict[str, Any] = {
        "format": "nanoobjectdetection-2.0-chunks-v1",
        "source": str(source.resolve()),
        "axes": info["axes"],
        "source_dtype": info["dtype"],
        "dtype": (
            "uint8"
            if processing_bit_depth == 8 and np.dtype(info["dtype"]).itemsize == 2
            else info["dtype"]
        ),
        "frame_start": frame_start,
        "frame_stop": stop,
        "chunk_frames": chunk_frames,
        "chunk_overlap_frames": chunk_overlap_frames,
        "shape": None,
        "chunks": [],
    }
    step = chunk_frames - chunk_overlap_frames
    chunk_starts = range(frame_start, stop, step)
    total_chunks = (stop - frame_start + step - 1) // step
    for chunk_number, first in enumerate(
        progress(
            chunk_starts,
            desc="Writing TIFF chunks",
            total=total_chunks,
            unit="chunk",
        )
    ):
        last = min(first + chunk_frames, stop)
        data = read_frames(source, first, last)
        data = data[:, y_slice, x_slice]
        data = to_processing_bit_depth(data, processing_bit_depth)
        if manifest["shape"] is None:
            manifest["shape"] = [stop - frame_start, data.shape[1], data.shape[2]]
        name = f"chunk_{chunk_number:06d}.tif"
        tifffile.imwrite(
            chunks_dir / name,
            data,
            compression="deflate",
            predictor=True,
            metadata={"axes": "ZYX"},
        )
        manifest["chunks"].append(
            {
                "file": f"chunks/{name}",
                "frame_start": first,
                "frame_stop": last,
                "shape": list(data.shape),
            }
        )
        del data

    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def iter_blocks(store: Path) -> Iterator[tuple[int, np.ndarray]]:
    manifest_path = store / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "nanoobjectdetection-2.0-chunks-v1":
        raise ValueError(f"Unsupported image store: {store}")
    for chunk in progress(
        manifest["chunks"],
        desc="Reading image chunks",
        total=len(manifest["chunks"]),
        unit="chunk",
    ):
        yield (
            int(chunk["frame_start"]),
            read_frames(store / chunk["file"], 0, int(chunk["shape"][0])),
        )
