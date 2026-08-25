#!/usr/bin/env python3
from __future__ import annotations

import argparse

from sasa_repro.io import load_manifest, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge canonical manifests with duplicate checks")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--validate-images", action="store_true")
    args = parser.parse_args()
    samples = []
    seen: set[str] = set()
    for path in args.inputs:
        for sample in load_manifest(path, validate_images=args.validate_images):
            if sample.id in seen:
                parser.error(f"duplicate sample id across manifests: {sample.id}")
            seen.add(sample.id)
            samples.append(sample)
    write_manifest(args.output, samples)
    print(f"Merged {len(samples)} samples from {len(args.inputs)} manifests -> {args.output}")


if __name__ == "__main__":
    main()
