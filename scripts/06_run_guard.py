#!/usr/bin/env python3
from __future__ import annotations

import argparse

from sasa_repro.config import load_config
from sasa_repro.guard import SASAGuard
from sasa_repro.io import load_manifest, write_predictions
from sasa_repro.model.llava_adapter import LlavaSASAAdapter
from sasa_repro.probe.model import SafetyProbe
from sasa_repro.progress import progress
from sasa_repro.utils import configure_logging, limit_items, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full SASA safety gate")
    parser.add_argument("--config", default="configs/llava15_7b.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    config = load_config(args.config)
    set_seed(config.seed)
    samples = limit_items(load_manifest(args.manifest, validate_images=True), args.limit)
    guard = SASAGuard(LlavaSASAAdapter(config), SafetyProbe.load(args.probe))
    predictions = [guard.predict(sample) for sample in progress(samples, desc="SASA guard")]
    write_predictions(args.output, predictions)
    print(f"Wrote {len(predictions)} guarded responses to {args.output}")


if __name__ == "__main__":
    main()
