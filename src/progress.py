from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

T = TypeVar("T")


def progress(items: Iterable[T], **kwargs) -> Iterable[T]:
    """Use tqdm when installed, otherwise return the original iterable."""

    try:
        from tqdm import tqdm
    except ImportError:
        return items
    return tqdm(items, **kwargs)
