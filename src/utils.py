from __future__ import annotations

import logging
import random
from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar

import numpy as np

T = TypeVar("T")


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_parent(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def parse_indices(value: str | None) -> list[int] | None:
    """Parse ``0,2,5-8`` style integer lists."""

    if value is None or not value.strip():
        return None
    result: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", maxsplit=1)
            start, stop = int(left), int(right)
            if stop < start:
                raise ValueError(f"invalid descending range: {part}")
            result.update(range(start, stop + 1))
        else:
            result.add(int(part))
    return sorted(result)


def limit_items(items: Iterable[T], limit: int | None) -> list[T]:
    values = list(items)
    return values if limit is None else values[:limit]
