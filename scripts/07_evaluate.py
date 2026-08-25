#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sasa_repro.evaluation.metrics import classification_metrics, generation_metrics
from sasa_repro.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate probe decisions and refusal behavior")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    rows = list(read_jsonl(args.predictions))
    report: dict[str, object] = {"generation": generation_metrics(rows)}
    scored = [row for row in rows if row.get("unsafe_probability") is not None]
    if scored:
        report["classification"] = classification_metrics(
            [int(row["label"]) for row in scored],
            [float(row["unsafe_probability"]) for row in scored],
            threshold=args.threshold,
        )
    datasets = sorted({str(row["dataset"]) for row in rows})
    report["by_dataset"] = {
        dataset: generation_metrics(row for row in rows if str(row["dataset"]) == dataset)
        for dataset in datasets
    }
    text = json.dumps(report, indent=2, allow_nan=True)
    print(text)
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
