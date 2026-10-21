"""Modern, bounded-memory input helpers for image stacks."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np


def _require_tifffile():
    try:
        import tifffile
    except ImportError as exc:
        raise RuntimeError(
            "The modern TIFF reader requires the 'tifffile' package."
        ) from exc
    return tifffile


def tiff_info(path: str | Path) -> dict[str, Any]:
    """Return stack metadata without reading all pixel data."""
    tifffile = _require_tifffile()
    path = Path(path)
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        return {
            "path": str(path.resolve()),
            "shape": tuple(series.shape),
            "axes": series.axes,
            "dtype": str(series.dtype),
            "pages": len(tif.pages),
        }


def read_tiff_frames(
    path: str | Path,
    frame_start: int = 0,
    frame_stop: int | None = None,
    y_slice: slice | None = None,
    x_slice: slice | None = None,
) -> np.ndarray:
    """Read only the requested frames and optional spatial ROI.

    A memory map is used for the uncompressed acquisition format. Compressed
    TIFFs use tifffile's sliced reader instead, so callers never need to load
    the complete acquisition accidentally.
    """
    tifffile = _require_tifffile()
    path = Path(path)
    if frame_start < 0:
        raise ValueError("frame_start must be non-negative")
    if frame_stop is not None and frame_stop <= frame_start:
        raise ValueError("frame_stop must be greater than frame_start")

    try:
        mapped = tifffile.memmap(path, mode="r")
        frame_count = mapped.shape[0]
        stop = frame_count if frame_stop is None else min(frame_stop, frame_count)
        result = (
            mapped[frame_start:stop]
            if frame_stop is None
            else np.array(mapped[frame_start:stop], copy=True)
        )
        del mapped
    except (OSError, ValueError, RuntimeError):
        result = tifffile.imread(path, key=slice(frame_start, frame_stop))

    if y_slice is not None or x_slice is not None:
        result = result[
            :,
            y_slice if y_slice is not None else slice(None),
            x_slice if x_slice is not None else slice(None),
        ]
    return np.asarray(result)


def iter_tiff_frames(
    path: str | Path,
    frame_start: int = 0,
    frame_stop: int | None = None,
) -> Iterator[np.ndarray]:
    """Yield frames one at a time for algorithms that can stream input."""
    tifffile = _require_tifffile()
    path = Path(path)
    try:
        mapped = tifffile.memmap(path, mode="r")
        stop = mapped.shape[0] if frame_stop is None else min(frame_stop, mapped.shape[0])
        for index in range(frame_start, stop):
            yield np.asarray(mapped[index])
        del mapped
    except (OSError, ValueError, RuntimeError):
        with tifffile.TiffFile(path) as tif:
            stop = len(tif.pages) if frame_stop is None else min(frame_stop, len(tif.pages))
            for index in range(frame_start, stop):
                yield tif.pages[index].asarray()


def convert_tiff_to_chunks(
    source: str | Path,
    output_dir: str | Path,
    frame_start: int = 0,
    frame_stop: int | None = None,
    chunk_frames: int = 512,
    y_slice: slice | None = None,
    x_slice: slice | None = None,
) -> dict[str, Any]:
    """Convert a TIFF acquisition into independently readable TIFF chunks.

    The original file is never modified. Each chunk is a valid TIFF stack and
    ``manifest.json`` makes the resulting directory self-describing.
    """
    tifffile = _require_tifffile()
    source = Path(source)
    output_dir = Path(output_dir)
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")

    info = tiff_info(source)
    total_frames = int(info["shape"][0])
    stop = total_frames if frame_stop is None else min(frame_stop, total_frames)
    if frame_start < 0 or frame_start >= stop:
        raise ValueError("Invalid frame range")

    manifest = {
        "format": "nanoobjectdetection-chunked-tiff-v1",
        "source": str(source.resolve()),
        "shape": None,
        "axes": info["axes"],
        "dtype": info["dtype"],
        "frame_start": frame_start,
        "frame_stop": stop,
        "chunk_frames": chunk_frames,
        "chunks": [],
    }
    chunk_index = 0
    for first in range(frame_start, stop, chunk_frames):
        last = min(first + chunk_frames, stop)
        data = read_tiff_frames(source, first, last, y_slice, x_slice)
        if manifest["shape"] is None:
            manifest["shape"] = [stop - frame_start, data.shape[1], data.shape[2]]
        name = f"chunk_{chunk_index:06d}.tif"
        target = chunks_dir / name
        tifffile.imwrite(
            target,
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
        chunk_index += 1

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        __import__("json").dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def read_chunk_manifest(path: str | Path) -> dict[str, Any]:
    """Read and validate a chunked TIFF manifest."""
    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path /= "manifest.json"
    manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "nanoobjectdetection-chunked-tiff-v1":
        raise ValueError(f"Unsupported chunk store: {manifest_path}")
    return manifest


def iter_chunk_blocks(path: str | Path) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(absolute_frame_start, block)`` from a chunked TIFF store."""
    store = Path(path)
    manifest = read_chunk_manifest(store)
    for chunk in manifest["chunks"]:
        block = read_tiff_frames(store / chunk["file"])
        yield int(chunk["frame_start"]), block
