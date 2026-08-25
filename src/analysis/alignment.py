from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np

from ..model.llava_adapter import LlavaSASAAdapter
from ..progress import progress
from ..schema import Sample


def cosine_similarity(left: np.ndarray, right: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """Row-wise cosine similarity for arrays shaped ``[layers, hidden]``."""

    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("alignment inputs must have equal [layers, hidden] shapes")
    numerator = np.sum(left * right, axis=-1)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    return numerator / np.maximum(denominator, epsilon)


def modality_alignment(
    adapter: LlavaSASAAdapter,
    samples: Sequence[Sample],
    neutral_prompt: str = "Describe the image.",
) -> dict[str, np.ndarray]:
    """Measure layer-wise alignment of combined inputs with modality controls.

    This is a reproducible proxy for the paper's modality-alignment analysis:
    image-only behavior is operationalized with a fixed neutral instruction,
    since an autoregressive LLaVA prompt still needs a textual query.
    """

    image_scores: list[np.ndarray] = []
    text_scores: list[np.ndarray] = []
    for sample in progress(samples, desc="modality alignment"):
        combined = np.asarray(adapter.layerwise_last_hidden(sample))
        text_only = np.asarray(
            adapter.layerwise_last_hidden_from_inputs(adapter.prepare_text_only(sample.prompt))
        )
        image_only = np.asarray(
            adapter.layerwise_last_hidden_from_inputs(
                adapter.prepare_image_only(sample, neutral_prompt=neutral_prompt)
            )
        )
        image_scores.append(cosine_similarity(combined, image_only))
        text_scores.append(cosine_similarity(combined, text_only))

    if not image_scores:
        raise ValueError("alignment analysis requires at least one sample")
    return {
        "combined_image_mean": np.mean(image_scores, axis=0),
        "combined_image_std": np.std(image_scores, axis=0),
        "combined_text_mean": np.mean(text_scores, axis=0),
        "combined_text_std": np.std(text_scores, axis=0),
    }


def save_alignment_csv(result: dict[str, np.ndarray], output_path: str | Path) -> Path:
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    layer_count = len(result["combined_image_mean"])
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "layer",
                "combined_image_mean",
                "combined_image_std",
                "combined_text_mean",
                "combined_text_std",
            ]
        )
        for layer in range(layer_count):
            writer.writerow(
                [
                    layer,
                    result["combined_image_mean"][layer],
                    result["combined_image_std"][layer],
                    result["combined_text_mean"][layer],
                    result["combined_text_std"][layer],
                ]
            )
    return output


def plot_alignment(
    result: dict[str, np.ndarray],
    output_path: str | Path,
    fused_layer: int,
) -> Path:
    import matplotlib.pyplot as plt

    layers = np.arange(len(result["combined_image_mean"]))
    figure, axis = plt.subplots(figsize=(7, 4))
    for prefix, label, color in (
        ("combined_image", "Combined vs image control", "#4C78A8"),
        ("combined_text", "Combined vs text control", "#E45756"),
    ):
        mean = result[f"{prefix}_mean"]
        std = result[f"{prefix}_std"]
        axis.plot(layers, mean, label=label, color=color)
        axis.fill_between(layers, mean - std, mean + std, color=color, alpha=0.15)
    axis.axvline(fused_layer, color="#333333", linestyle="--", label=f"Fused layer {fused_layer}")
    axis.set(xlabel="Layer", ylabel="Cosine similarity", ylim=(-1.02, 1.02))
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output
