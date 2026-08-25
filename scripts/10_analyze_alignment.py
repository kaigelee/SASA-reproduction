#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from sasa_repro.analysis.alignment import modality_alignment, plot_alignment, save_alignment_csv
from sasa_repro.config import load_config
from sasa_repro.io import load_manifest
from sasa_repro.model.llava_adapter import LlavaSASAAdapter
from sasa_repro.utils import configure_logging, limit_items, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze layer-wise image/text alignment")
    parser.add_argument("--config", default="configs/llava15_7b.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--neutral-prompt", default="Describe the image.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    config = load_config(args.config)
    set_seed(config.seed)
    samples = limit_items(load_manifest(args.manifest, validate_images=True), args.limit)
    adapter = LlavaSASAAdapter(config)
    result = modality_alignment(adapter, samples, neutral_prompt=args.neutral_prompt)
    output = Path(args.output_dir).expanduser()
    save_alignment_csv(result, output / "alignment.csv")
    plot_alignment(result, output / "alignment.png", config.sasa.fused_layer)
    print(f"Alignment analysis written to {output}")


if __name__ == "__main__":
    main()
