from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ..model.llava_adapter import LlavaSASAAdapter
from ..progress import progress
from ..schema import Sample


def top_singular_vector(matrix: np.ndarray, side: str = "left") -> np.ndarray:
    """Return the leading singular vector.

    The SASA paper explicitly says "left singular vector". `right` is retained
    as a sensitivity analysis because a feature-space right vector is also a
    plausible interpretation of the method.
    """

    if matrix.ndim != 2:
        raise ValueError("activation matrix must be two-dimensional")
    u, _singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    if side == "left":
        return u[:, 0]
    if side == "right":
        return vh[0]
    raise ValueError("side must be left or right")


def principal_angle(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if denominator == 0:
        raise ValueError("principal angle is undefined for a zero vector")
    cosine = abs(float(np.dot(vector_a, vector_b) / denominator))
    return float(math.acos(np.clip(cosine, -1.0, 1.0)))


def activation_matrix(
    adapter: LlavaSASAAdapter,
    samples: Sequence[Sample],
    ablate: tuple[int, int] | None = None,
) -> np.ndarray:
    return np.stack([adapter.extract_final_hidden(sample, ablate=ablate) for sample in samples])


def score_heads(
    adapter: LlavaSASAAdapter,
    samples: Sequence[Sample],
    output_csv: str | Path,
    layers: Iterable[int] | None = None,
    heads: Iterable[int] | None = None,
    singular_side: str = "left",
) -> Path:
    output = Path(output_csv).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    text_config = getattr(adapter.model.config, "text_config", adapter.model.config)
    layer_ids = list(layers if layers is not None else range(len(adapter.layers)))
    head_ids = list(heads if heads is not None else range(int(text_config.num_attention_heads)))

    completed: set[tuple[int, int]] = set()
    if output.is_file():
        with output.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                completed.add((int(row["layer"]), int(row["head"])))

    baseline = activation_matrix(adapter, samples)
    baseline_vector = top_singular_vector(baseline, side=singular_side)
    write_header = not output.exists() or output.stat().st_size == 0
    with output.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["layer", "head", "importance", "singular_side", "sample_count"],
        )
        if write_header:
            writer.writeheader()
        pairs = [(layer, head) for layer in layer_ids for head in head_ids]
        for layer, head in progress(pairs, desc="scoring attention heads"):
            if (layer, head) in completed:
                continue
            ablated = activation_matrix(adapter, samples, ablate=(layer, head))
            ablated_vector = top_singular_vector(ablated, side=singular_side)
            score = principal_angle(baseline_vector, ablated_vector)
            writer.writerow(
                {
                    "layer": layer,
                    "head": head,
                    "importance": score,
                    "singular_side": singular_side,
                    "sample_count": len(samples),
                }
            )
            handle.flush()
    return output


def safety_specific_heads(
    safety_csv: str | Path,
    utility_csv: str | Path,
    top_k: int = 10,
) -> list[dict[str, float | int]]:
    def read_scores(path: str | Path) -> list[dict[str, float | int]]:
        rows: list[dict[str, float | int]] = []
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    {
                        "layer": int(row["layer"]),
                        "head": int(row["head"]),
                        "importance": float(row["importance"]),
                    }
                )
        return sorted(rows, key=lambda row: float(row["importance"]), reverse=True)

    safety = read_scores(safety_csv)[:top_k]
    utility_keys = {
        (int(row["layer"]), int(row["head"])) for row in read_scores(utility_csv)[:top_k]
    }
    return [
        row
        for row in safety
        if (int(row["layer"]), int(row["head"])) not in utility_keys
    ]
