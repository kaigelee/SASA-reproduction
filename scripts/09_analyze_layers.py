#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sasa_repro.analysis.layerwise import (
    collect_layer_activations,
    plot_readability,
    plot_tsne,
    readability_rates,
    save_layer_analysis,
    tsne_by_layer,
)
from sasa_repro.config import load_config
from sasa_repro.io import load_manifest
from sasa_repro.model.llava_adapter import LlavaSASAAdapter
from sasa_repro.utils import configure_logging, limit_items, parse_indices, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SASA layer-wise mechanistic analyses")
    parser.add_argument("--config", default="configs/llava15_7b.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default="0,5,10,13,15,20,25,31")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-readability", action="store_true")
    parser.add_argument("--skip-tsne", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    config = load_config(args.config)
    set_seed(config.seed)
    samples = limit_items(load_manifest(args.manifest, validate_images=True), args.limit)
    adapter = LlavaSASAAdapter(config)
    output = Path(args.output_dir).expanduser()

    readability = None
    tokens = None
    if not args.skip_readability:
        readability, tokens = readability_rates(adapter, samples, top_k=args.top_k)
        plot_readability(readability, output / "readability.png", config.sasa.fused_layer)

    activations = collect_layer_activations(adapter, samples)
    labels = np.asarray([sample.label for sample in samples], dtype=np.int64)
    save_layer_analysis(
        output,
        activations,
        labels,
        [sample.id for sample in samples],
        readability=readability,
        tokens=tokens,
    )
    if not args.skip_tsne:
        layers = parse_indices(args.layers) or []
        embeddings = tsne_by_layer(activations, labels, layers=layers, seed=config.seed)
        plot_tsne(embeddings, labels, output / "tsne.png")
        np.savez_compressed(
            output / "tsne_embeddings.npz",
            **{f"layer_{layer}": points for layer, points in embeddings.items()},
        )
    print(f"Layer analysis written to {output}")


if __name__ == "__main__":
    main()
