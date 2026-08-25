from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    name_or_path: str = "llava-hf/llava-1.5-7b-hf"
    dtype: str = "float16"
    device_map: str | None = "auto"
    attention_implementation: str | None = "eager"
    trust_remote_code: bool = False
    low_cpu_mem_usage: bool = True
    prompt_template: str = "USER: <image>\n{prompt}\nASSISTANT:"


@dataclass(frozen=True)
class SASAConfig:
    fused_layer: int = 15
    safety_layer: int = 13
    epsilon: float = 1e-6
    projection_mode: str = "replace"
    mix_coefficient: float = 1.0

    def __post_init__(self) -> None:
        if self.safety_layer >= self.fused_layer:
            raise ValueError("safety_layer must precede fused_layer")
        if self.projection_mode not in {"replace", "residual", "interpolate"}:
            raise ValueError("projection_mode must be replace, residual, or interpolate")
        if not 0.0 <= self.mix_coefficient <= 1.0:
            raise ValueError("mix_coefficient must be in [0, 1]")


@dataclass(frozen=True)
class FeatureConfig:
    kind: str = "logits"
    storage_dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.kind not in {"logits", "last_hidden"}:
            raise ValueError("features.kind must be logits or last_hidden")
        if self.storage_dtype not in {"float16", "float32"}:
            raise ValueError("features.storage_dtype must be float16 or float32")


@dataclass(frozen=True)
class ProbeConfig:
    threshold: float = 0.5
    regularization_c: float = 1.0
    class_weight: str | None = "balanced"
    solver: str = "liblinear"
    max_iter: int = 2000
    normalize: str = "none"
    validation_fraction: float = 0.2

    def __post_init__(self) -> None:
        if self.normalize not in {"none", "l2", "standard"}:
            raise ValueError("probe.normalize must be none, l2, or standard")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("probe.threshold must be in (0, 1)")


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    num_beams: int = 1
    refusal_text: str = "I'm sorry, but I cannot help with that request."
    safe_generation_mode: str = "original"

    def __post_init__(self) -> None:
        if self.safe_generation_mode not in {"original", "projected_greedy"}:
            raise ValueError("safe_generation_mode must be original or projected_greedy")


@dataclass(frozen=True)
class AppConfig:
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    sasa: SASAConfig = field(default_factory=SASAConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)


def _section(cls: type, raw: dict[str, Any], name: str):
    return cls(**dict(raw.get(name) or {}))


def load_config(path: str | Path) -> AppConfig:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return AppConfig(
        seed=int(raw.get("seed", 42)),
        model=_section(ModelConfig, raw, "model"),
        sasa=_section(SASAConfig, raw, "sasa"),
        features=_section(FeatureConfig, raw, "features"),
        probe=_section(ProbeConfig, raw, "probe"),
        generation=_section(GenerationConfig, raw, "generation"),
    )
