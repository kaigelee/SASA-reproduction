from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler

from ..config import ProbeConfig


class SafetyProbe:
    def __init__(self, config: ProbeConfig, feature_kind: str = "logits") -> None:
        self.config = config
        self.feature_kind = feature_kind
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self) -> Pipeline:
        steps: list[tuple[str, Any]] = []
        if self.config.normalize == "l2":
            steps.append(("normalize", Normalizer(norm="l2")))
        elif self.config.normalize == "standard":
            steps.append(("normalize", StandardScaler()))
        steps.append(
            (
                "classifier",
                LogisticRegression(
                    C=self.config.regularization_c,
                    class_weight=self.config.class_weight,
                    solver=self.config.solver,
                    max_iter=self.config.max_iter,
                    random_state=42,
                ),
            )
        )
        return Pipeline(steps)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "SafetyProbe":
        features = _as_2d(features)
        labels = np.asarray(labels, dtype=np.int64)
        if set(np.unique(labels).tolist()) != {0, 1}:
            raise ValueError("probe training requires both labels 0 and 1")
        self.pipeline.fit(features, labels)
        return self

    def predict_probability(self, features: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(_as_2d(features))[:, 1]

    def predict(self, features: np.ndarray, threshold: float | None = None) -> np.ndarray:
        threshold = self.config.threshold if threshold is None else threshold
        return (self.predict_probability(features) >= threshold).astype(np.int64)

    def save(self, path: str | Path) -> Path:
        output = Path(path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "config": asdict(self.config),
                "feature_kind": self.feature_kind,
                "format_version": 1,
            },
            output,
        )
        return output

    @classmethod
    def load(cls, path: str | Path) -> "SafetyProbe":
        bundle = joblib.load(Path(path).expanduser())
        probe = cls(ProbeConfig(**bundle["config"]), feature_kind=bundle["feature_kind"])
        probe.pipeline = bundle["pipeline"]
        return probe


def _as_2d(features: np.ndarray) -> np.ndarray:
    value = np.asarray(features, dtype=np.float32)
    if value.ndim == 1:
        value = value[None, :]
    if value.ndim != 2:
        raise ValueError(f"expected a 1D or 2D feature array, got {value.shape}")
    return value

