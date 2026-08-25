from __future__ import annotations

from dataclasses import asdict

from .evaluation.keywords import is_refusal
from .model.llava_adapter import LlavaSASAAdapter
from .probe.model import SafetyProbe
from .schema import Prediction, Sample


class SASAGuard:
    def __init__(self, adapter: LlavaSASAAdapter, probe: SafetyProbe) -> None:
        self.adapter = adapter
        self.probe = probe

    def predict(self, sample: Sample) -> Prediction:
        feature, projection_stats = self.adapter.extract_feature(
            sample,
            projected=True,
            kind=self.probe.feature_kind,
        )
        probability = float(self.probe.predict_probability(feature)[0])
        predicted = int(probability >= self.probe.config.threshold)
        if predicted == 1:
            response = self.adapter.config.generation.refusal_text
        elif self.adapter.config.generation.safe_generation_mode == "projected_greedy":
            response = self.adapter.generate_projected_greedy(sample)
        else:
            response = self.adapter.generate_original(sample)
        return Prediction(
            id=sample.id,
            dataset=sample.dataset,
            label=sample.label,
            unsafe_probability=probability,
            predicted_label=predicted,
            response=response,
            refused=is_refusal(response),
            metadata={
                "feature_kind": self.probe.feature_kind,
                "safe_generation_mode": self.adapter.config.generation.safe_generation_mode,
                "projection": asdict(projection_stats) if projection_stats else {},
            },
        )

