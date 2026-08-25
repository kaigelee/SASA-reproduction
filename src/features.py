from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np

from .model.llava_adapter import LlavaSASAAdapter
from .progress import progress
from .schema import Sample

LOGGER = logging.getLogger(__name__)


def extract_features(
    adapter: LlavaSASAAdapter,
    samples: Iterable[Sample],
    output_path: str | Path,
    projected: bool = True,
    feature_kind: str = "logits",
    storage_dtype: str = "float32",
    continue_on_error: bool = False,
) -> Path:
    sample_list = list(samples)
    dtype = np.dtype(storage_dtype)
    if dtype not in (np.dtype("float16"), np.dtype("float32")):
        raise ValueError("storage_dtype must be float16 or float32")
    features: list[np.ndarray] = []
    labels: list[int] = []
    ids: list[str] = []
    datasets: list[str] = []
    categories: list[str] = []
    stats: list[str] = []
    failures: list[dict[str, str]] = []

    for sample in progress(sample_list, desc="extracting features"):
        try:
            feature, projection_stats = adapter.extract_feature(
                sample,
                projected=projected,
                kind=feature_kind,
            )
        except Exception as exc:
            if not continue_on_error:
                raise
            LOGGER.exception("failed sample %s", sample.id)
            failures.append({"id": sample.id, "error": repr(exc)})
            continue
        features.append(feature.astype(dtype, copy=False))
        labels.append(sample.label)
        ids.append(sample.id)
        datasets.append(sample.dataset)
        categories.append(sample.category)
        stats.append(json.dumps(asdict(projection_stats) if projection_stats else {}))

    if not features:
        raise RuntimeError("no features were extracted")
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        X=np.stack(features),
        y=np.asarray(labels, dtype=np.int64),
        ids=np.asarray(ids, dtype=str),
        datasets=np.asarray(datasets, dtype=str),
        categories=np.asarray(categories, dtype=str),
        projection_stats=np.asarray(stats, dtype=str),
        projected=np.asarray([int(projected)], dtype=np.int8),
        feature_kind=np.asarray([feature_kind], dtype=str),
        storage_dtype=np.asarray([storage_dtype], dtype=str),
    )
    if failures:
        failure_path = output.with_suffix(".failures.json")
        failure_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
    LOGGER.info("saved %d feature vectors to %s", len(features), output)
    return output


def load_feature_archive(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path).expanduser(), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}
