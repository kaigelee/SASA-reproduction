#!/usr/bin/env python3
from __future__ import annotations

import argparse

from sasa_repro.data.sampling import stratified_sample
from sasa_repro.io import load_manifest, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a deterministic stratified subset")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-group", type=int, required=True)
    parser.add_argument("--group-fields", default="dataset,label,category")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate-images", action="store_true")
    args = parser.parse_args()
    fields = tuple(value.strip() for value in args.group_fields.split(",") if value.strip())
    samples = load_manifest(args.input, validate_images=args.validate_images)
    selected = stratified_sample(samples, args.per_group, group_fields=fields, seed=args.seed)
    write_manifest(args.output, selected)
    print(f"Selected {len(selected)} of {len(samples)} samples -> {args.output}")


if __name__ == "__main__":
    main()
