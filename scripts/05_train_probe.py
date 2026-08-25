#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from sasa_repro.config import load_config
from sasa_repro.evaluation.metrics import classification_metrics
from sasa_repro.features import load_feature_archive
from sasa_repro.probe.model import SafetyProbe
from sasa_repro.utils import configure_logging, set_seed


def _feature_kind(archive: dict[str, np.ndarray]) -> str:
    value = archive.get("feature_kind")
    return str(value[0]) if value is not None else "logits"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SASA logistic-regression probe")
    parser.add_argument("--config", default="configs/llava15_7b.yaml")
    parser.add_argument("--train-features", required=True)
    parser.add_argument("--validation-features")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    config = load_config(args.config)
    set_seed(config.seed)
    train_archive = load_feature_archive(args.train_features)
    X, y = train_archive["X"], train_archive["y"]

    if args.validation_features:
        validation_archive = load_feature_archive(args.validation_features)
        X_train, y_train = X, y
        X_validation, y_validation = validation_archive["X"], validation_archive["y"]
    else:
        X_train, X_validation, y_train, y_validation = train_test_split(
            X,
            y,
            test_size=config.probe.validation_fraction,
            random_state=config.seed,
            stratify=y,
        )

    feature_kind = _feature_kind(train_archive)
    probe = SafetyProbe(config.probe, feature_kind=feature_kind).fit(X_train, y_train)
    probe.save(args.output)
    train_probabilities = probe.predict_probability(X_train)
    validation_probabilities = probe.predict_probability(X_validation)
    report = {
        "feature_kind": feature_kind,
        "train_count": int(len(y_train)),
        "validation_count": int(len(y_validation)),
        "train": classification_metrics(y_train, train_probabilities, config.probe.threshold),
        "validation": classification_metrics(
            y_validation, validation_probabilities, config.probe.threshold
        ),
    }
    report_path = Path(args.report or f"{args.output}.metrics.json").expanduser()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=True))
    print(f"Saved probe to {args.output}; metrics to {report_path}")


if __name__ == "__main__":
    main()
