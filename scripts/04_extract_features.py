#!/usr/bin/env python3
from __future__ import annotations

import argparse

from sasa_repro.config import load_config
from sasa_repro.features import extract_features
from sasa_repro.io import load_manifest
from sasa_repro.model.llava_adapter import LlavaSASAAdapter
from sasa_repro.utils import configure_logging, limit_items, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract SASA first-token probe features")
    parser.add_argument("--config", default="configs/llava15_7b.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--kind", choices=("logits", "last_hidden"))
    parser.add_argument("--storage-dtype", choices=("float16", "float32"))
    parser.add_argument("--no-projection", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    config = load_config(args.config)
    set_seed(config.seed)
    samples = limit_items(load_manifest(args.manifest, validate_images=True), args.limit)
    adapter = LlavaSASAAdapter(config)
    extract_features(
        adapter,
        samples,
        args.output,
        projected=not args.no_projection,
        feature_kind=args.kind or config.features.kind,
        storage_dtype=args.storage_dtype or config.features.storage_dtype,
        continue_on_error=args.continue_on_error,
    )
    print(f"Feature archive written to {args.output}")


if __name__ == "__main__":
    main()
