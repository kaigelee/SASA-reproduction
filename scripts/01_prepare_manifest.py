#!/usr/bin/env python3
from __future__ import annotations

import argparse

from sasa_repro.data.preparation import (
    prepare_figstep,
    prepare_mm_safetybench,
    prepare_vlguard,
)
from sasa_repro.io import write_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert safety datasets to SASA JSONL")
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    mm = subparsers.add_parser("mm-safetybench")
    mm.add_argument("--root", required=True)
    mm.add_argument("--variant", choices=("SD", "SD_TYPO", "TYPO"), default="SD_TYPO")
    mm.add_argument("--split", default="test")
    mm.add_argument("--output", required=True)
    mm.add_argument("--allow-missing-images", action="store_true")

    vl = subparsers.add_parser("vlguard")
    vl.add_argument("--metadata", required=True)
    vl.add_argument("--image-root", required=True)
    vl.add_argument("--subsets", default="unsafes,safe_unsafes,safe_safes")
    vl.add_argument("--split", default="test")
    vl.add_argument("--output", required=True)
    vl.add_argument("--allow-missing-images", action="store_true")

    fig = subparsers.add_parser("figstep")
    fig.add_argument("--csv", required=True)
    fig.add_argument("--image-root", required=True)
    fig.add_argument("--split", default="test")
    fig.add_argument("--output", required=True)
    fig.add_argument("--prompt", default=None, help="override the standard FigStep prompt")
    fig.add_argument("--allow-missing-images", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.dataset == "mm-safetybench":
        samples = prepare_mm_safetybench(
            args.root,
            split=args.split,
            image_variant=args.variant,
            strict_images=not args.allow_missing_images,
        )
    elif args.dataset == "vlguard":
        subsets = tuple(value.strip() for value in args.subsets.split(",") if value.strip())
        samples = prepare_vlguard(
            args.metadata,
            args.image_root,
            split=args.split,
            include_subsets=subsets,
            strict_images=not args.allow_missing_images,
        )
    else:
        kwargs = {}
        if args.prompt is not None:
            kwargs["fixed_prompt"] = args.prompt
        samples = prepare_figstep(
            args.csv,
            args.image_root,
            split=args.split,
            strict_images=not args.allow_missing_images,
            **kwargs,
        )
    write_manifest(args.output, samples)
    print(f"Wrote {len(samples)} samples to {args.output}")


if __name__ == "__main__":
    main()
