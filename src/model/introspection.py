from __future__ import annotations

from typing import Any, Iterable


def resolve_attr(root: Any, candidates: Iterable[str]) -> Any:
    errors: list[str] = []
    for path in candidates:
        value = root
        try:
            for part in path.split("."):
                value = getattr(value, part)
            return value
        except AttributeError:
            errors.append(path)
    raise AttributeError(
        f"none of the expected attribute paths exist on {type(root).__name__}: {errors}"
    )


def decoder_layers(model: Any) -> Any:
    return resolve_attr(
        model,
        (
            "language_model.model.layers",
            "model.language_model.layers",
            "model.language_model.model.layers",
            "language_model.base_model.model.layers",
        ),
    )


def final_norm(model: Any) -> Any:
    return resolve_attr(
        model,
        (
            "language_model.model.norm",
            "model.language_model.norm",
            "model.language_model.model.norm",
            "language_model.base_model.model.norm",
        ),
    )


def language_head(model: Any) -> Any:
    return resolve_attr(model, ("language_model.lm_head", "lm_head", "model.lm_head"))


def attention_output_projection(layer: Any) -> Any:
    return resolve_attr(layer, ("self_attn.o_proj", "self_attn.out_proj"))

