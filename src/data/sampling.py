from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable

from ..schema import Sample


def stratified_sample(
    samples: Iterable[Sample],
    per_group: int,
    group_fields: tuple[str, ...] = ("dataset", "label", "category"),
    seed: int = 42,
) -> list[Sample]:
    if per_group <= 0:
        raise ValueError("per_group must be positive")
    groups: dict[tuple[object, ...], list[Sample]] = defaultdict(list)
    for sample in samples:
        key = tuple(getattr(sample, field) for field in group_fields)
        groups[key].append(sample)
    rng = random.Random(seed)
    selected: list[Sample] = []
    for key in sorted(groups, key=str):
        group = list(groups[key])
        rng.shuffle(group)
        selected.extend(group[:per_group])
    return sorted(selected, key=lambda sample: sample.id)


def train_validation_split(
    samples: Iterable[Sample], validation_fraction: float, seed: int = 42
) -> tuple[list[Sample], list[Sample]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    groups: dict[tuple[str, int], list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[(sample.dataset, sample.label)].append(sample)
    rng = random.Random(seed)
    train: list[Sample] = []
    validation: list[Sample] = []
    for group in groups.values():
        rng.shuffle(group)
        count = max(1, round(len(group) * validation_fraction))
        validation.extend(group[:count])
        train.extend(group[count:])
    return train, validation

