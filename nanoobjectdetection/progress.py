"""Progress reporting with a graceful fallback when tqdm is unavailable."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")


def progress(
    iterable: Iterable[T],
    *,
    desc: str,
    total: int | None = None,
    unit: str = "it",
) -> Iterator[T]:
    try:
        from tqdm import tqdm
    except ImportError:
        yield from iterable
        return
    yield from tqdm(iterable, desc=desc, total=total, unit=unit)


def update(message: str) -> None:
    try:
        from tqdm import tqdm
    except ImportError:
        print(message, flush=True)
        return
    tqdm.write(message)
