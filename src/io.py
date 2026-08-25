from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from .schema import Prediction, Sample


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc


def load_manifest(path: str | Path, validate_images: bool = False) -> list[Sample]:
    samples = [Sample.from_dict(row) for row in read_jsonl(path)]
    ids = [sample.id for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError(f"manifest contains duplicate sample ids: {path}")
    if validate_images:
        for sample in samples:
            sample.validate_image()
    return samples


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_manifest(path: str | Path, samples: Iterable[Sample]) -> None:
    write_jsonl(path, (sample.to_dict() for sample in samples))


def write_predictions(path: str | Path, predictions: Iterable[Prediction]) -> None:
    write_jsonl(path, (prediction.to_dict() for prediction in predictions))

