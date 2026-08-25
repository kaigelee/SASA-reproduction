from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.manifold import TSNE

from ..model.llava_adapter import LlavaSASAAdapter
from ..progress import progress
from ..schema import Sample


def is_readable_token(token: str) -> bool:
    cleaned = token.replace("▁", " ").replace("Ġ", " ").strip()
    if not cleaned:
        return False
    if cleaned.startswith("<") and cleaned.endswith(">"):
        return False
    return bool(re.search(r"[A-Za-z0-9]", cleaned))


def readability_rates(
    adapter: LlavaSASAAdapter,
    samples: Sequence[Sample],
    top_k: int = 5,
) -> tuple[np.ndarray, list[list[list[str]]]]:
    counts: np.ndarray | None = None
    all_tokens: list[list[list[str]]] = []
    for sample in progress(samples, desc="logit-lens readability"):
        layer_tokens = adapter.layerwise_top_tokens(sample, top_k=top_k)
        all_tokens.append(layer_tokens)
        readable = np.asarray(
            [[int(is_readable_token(token)) for token in tokens] for tokens in layer_tokens],
            dtype=np.float64,
        )
        if counts is None:
            counts = np.zeros_like(readable)
        counts += readable
    if counts is None:
        raise ValueError("readability analysis requires at least one sample")
    rates = counts.mean(axis=1) / len(samples)
    return rates, all_tokens


def collect_layer_activations(
    adapter: LlavaSASAAdapter,
    samples: Sequence[Sample],
) -> np.ndarray:
    per_sample = [adapter.layerwise_last_hidden(sample) for sample in progress(samples)]
    return np.asarray(per_sample, dtype=np.float32).transpose(1, 0, 2)


def tsne_by_layer(
    activations: np.ndarray,
    labels: np.ndarray,
    layers: Sequence[int],
    seed: int = 42,
) -> dict[int, np.ndarray]:
    if activations.ndim != 3:
        raise ValueError("activations must have shape [layers, samples, hidden]")
    if activations.shape[1] != len(labels):
        raise ValueError("number of labels does not match activations")
    result: dict[int, np.ndarray] = {}
    sample_count = activations.shape[1]
    if sample_count < 3:
        raise ValueError("t-SNE requires at least three samples")
    perplexity = max(1, min(30, sample_count // 3, sample_count - 1))
    for layer in layers:
        if not 0 <= layer < activations.shape[0]:
            raise IndexError(f"layer {layer} outside [0, {activations.shape[0]})")
        result[int(layer)] = TSNE(
            n_components=2,
            random_state=seed,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
        ).fit_transform(activations[layer])
    return result


def save_layer_analysis(
    output_dir: str | Path,
    activations: np.ndarray,
    labels: np.ndarray,
    ids: Sequence[str],
    readability: np.ndarray | None = None,
    tokens: list[list[list[str]]] | None = None,
) -> Path:
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "layer_activations.npz",
        activations=activations,
        labels=np.asarray(labels, dtype=np.int64),
        ids=np.asarray(ids, dtype=str),
        readability=np.asarray(readability if readability is not None else [], dtype=np.float64),
    )
    if tokens is not None:
        (output / "top_tokens.json").write_text(
            json.dumps(tokens, ensure_ascii=False),
            encoding="utf-8",
        )
    return output


def plot_tsne(
    embeddings: dict[int, np.ndarray],
    labels: np.ndarray,
    output_path: str | Path,
) -> Path:
    import matplotlib.pyplot as plt

    layer_ids = list(embeddings)
    if not layer_ids:
        raise ValueError("at least one t-SNE embedding is required")
    columns = min(3, len(layer_ids))
    rows = int(np.ceil(len(layer_ids) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows), squeeze=False)
    for axis, layer in zip(axes.ravel(), layer_ids):
        points = embeddings[layer]
        axis.scatter(points[labels == 0, 0], points[labels == 0, 1], s=13, c="#2E8B57", label="Benign")
        axis.scatter(points[labels == 1, 0], points[labels == 1, 1], s=13, c="#D94841", label="Harmful")
        axis.set_title(f"Layer {layer}")
        axis.grid(alpha=0.2)
    for axis in axes.ravel()[len(layer_ids) :]:
        axis.set_visible(False)
    axes.ravel()[0].legend(frameon=False)
    figure.tight_layout()
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_readability(rates: np.ndarray, output_path: str | Path, fused_layer: int) -> Path:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(np.arange(len(rates)), rates, marker="o", markersize=3, linewidth=1.6)
    axis.axvline(fused_layer, color="#D94841", linestyle="--", label=f"Fused layer {fused_layer}")
    axis.set_xlabel("Layer")
    axis.set_ylabel("Readable top-token rate")
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path
