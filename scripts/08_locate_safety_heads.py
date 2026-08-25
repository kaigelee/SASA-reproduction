#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sasa_repro.analysis.head_importance import safety_specific_heads, score_heads
from sasa_repro.config import load_config
from sasa_repro.io import load_manifest
from sasa_repro.model.llava_adapter import LlavaSASAAdapter
from sasa_repro.utils import configure_logging, limit_items, parse_indices, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Locate safety-specific attention heads")
    parser.add_argument("--config", default="configs/llava15_7b.yaml")
    parser.add_argument("--manifest", help="manifest to score; omit when only comparing CSV files")
    parser.add_argument("--output-csv")
    parser.add_argument("--layers", help="e.g. 0-31 or 10,13,15")
    parser.add_argument("--heads", help="e.g. 0-31 or 3,7,12")
    parser.add_argument("--singular-side", choices=("left", "right"), default="left")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--safety-csv")
    parser.add_argument("--utility-csv")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--comparison-output")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)

    if args.manifest:
        if not args.output_csv:
            parser.error("--output-csv is required with --manifest")
        config = load_config(args.config)
        set_seed(config.seed)
        samples = limit_items(load_manifest(args.manifest, validate_images=True), args.limit)
        adapter = LlavaSASAAdapter(config)
        score_heads(
            adapter,
            samples,
            args.output_csv,
            layers=parse_indices(args.layers),
            heads=parse_indices(args.heads),
            singular_side=args.singular_side,
        )
        print(f"Head scores written to {args.output_csv}")

    if args.safety_csv or args.utility_csv:
        if not (args.safety_csv and args.utility_csv):
            parser.error("--safety-csv and --utility-csv must be given together")
        rows = safety_specific_heads(args.safety_csv, args.utility_csv, top_k=args.top_k)
        text = json.dumps(rows, indent=2)
        print(text)
        if args.comparison_output:
            output = Path(args.comparison_output).expanduser()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text, encoding="utf-8")

    if not args.manifest and not args.safety_csv:
        parser.error("provide --manifest for scoring or both score CSV files for comparison")


if __name__ == "__main__":
    main()
