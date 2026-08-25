#!/usr/bin/env python3
from __future__ import annotations

import argparse

from sasa_repro.config import load_config
from sasa_repro.evaluation.keywords import is_refusal
from sasa_repro.io import load_manifest, write_predictions
from sasa_repro.model.llava_adapter import LlavaSASAAdapter
from sasa_repro.progress import progress
from sasa_repro.schema import Prediction
from sasa_repro.utils import configure_logging, limit_items, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate unmodified LLaVA baseline responses")
    parser.add_argument("--config", default="configs/llava15_7b.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    config = load_config(args.config)
    set_seed(config.seed)
    samples = limit_items(load_manifest(args.manifest, validate_images=True), args.limit)
    adapter = LlavaSASAAdapter(config)
    predictions: list[Prediction] = []
    for sample in progress(samples, desc="baseline generation"):
        response = adapter.generate_original(sample)
        predictions.append(
            Prediction(
                id=sample.id,
                dataset=sample.dataset,
                label=sample.label,
                unsafe_probability=None,
                predicted_label=None,
                response=response,
                refused=is_refusal(response),
                metadata={"method": "unmodified_llava"},
            )
        )
    write_predictions(args.output, predictions)
    print(f"Wrote {len(predictions)} responses to {args.output}")


if __name__ == "__main__":
    main()
