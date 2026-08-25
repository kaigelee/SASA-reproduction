from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ..config import AppConfig
from ..schema import Sample
from .introspection import (
    attention_output_projection,
    decoder_layers,
    final_norm,
    language_head,
)
from .projection import ProjectionStats, first_tensor, project_hidden_states, replace_first_tensor

LOGGER = logging.getLogger(__name__)


class LlavaSASAAdapter:
    """Hugging Face LLaVA-v1.5 adapter with a reference two-pass SASA implementation."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        try:
            import torch
            from transformers import AutoProcessor, LlavaForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "LLaVA dependencies are missing. Run `python -m pip install -e .`."
            ) from exc

        self.torch = torch
        dtype = self._resolve_dtype(config.model.dtype)
        model_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": config.model.device_map,
            "trust_remote_code": config.model.trust_remote_code,
            "low_cpu_mem_usage": config.model.low_cpu_mem_usage,
        }
        if config.model.attention_implementation:
            model_kwargs["attn_implementation"] = config.model.attention_implementation

        LOGGER.info("loading processor: %s", config.model.name_or_path)
        self.processor = AutoProcessor.from_pretrained(
            config.model.name_or_path,
            trust_remote_code=config.model.trust_remote_code,
        )
        LOGGER.info("loading model: %s", config.model.name_or_path)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            config.model.name_or_path,
            **model_kwargs,
        )
        self.model.eval()
        self.layers = decoder_layers(self.model)
        self._validate_layer_indices()

    def _resolve_dtype(self, name: str):
        mapping = {
            "auto": "auto",
            "float16": self.torch.float16,
            "fp16": self.torch.float16,
            "bfloat16": self.torch.bfloat16,
            "bf16": self.torch.bfloat16,
            "float32": self.torch.float32,
            "fp32": self.torch.float32,
        }
        try:
            return mapping[name.lower()]
        except KeyError as exc:
            raise ValueError(f"unsupported model dtype: {name}") from exc

    def _validate_layer_indices(self) -> None:
        layer_count = len(self.layers)
        for name, index in (
            ("safety_layer", self.config.sasa.safety_layer),
            ("fused_layer", self.config.sasa.fused_layer),
        ):
            if not 0 <= index < layer_count:
                raise IndexError(f"{name}={index} outside model with {layer_count} layers")
        LOGGER.info(
            "resolved %d decoder layers; SASA mapping: layer %d -> layer %d",
            layer_count,
            self.config.sasa.fused_layer,
            self.config.sasa.safety_layer,
        )

    @property
    def input_device(self):
        try:
            return self.model.device
        except AttributeError:
            return next(self.model.parameters()).device

    def format_prompt(self, prompt: str) -> str:
        return self.config.model.prompt_template.format(prompt=prompt)

    def prepare(self, sample: Sample) -> dict[str, Any]:
        from PIL import Image

        image = Image.open(Path(sample.image).expanduser()).convert("RGB")
        batch = self.processor(images=image, text=self.format_prompt(sample.prompt), return_tensors="pt")
        return self._move_batch(batch)

    def prepare_text_only(self, prompt: str) -> dict[str, Any]:
        """Prepare a text-only control without an image token."""

        text = self.format_prompt(prompt).replace("<image>\n", "")
        batch = self.processor(text=text, return_tensors="pt")
        return self._move_batch(batch)

    def prepare_image_only(
        self,
        sample: Sample,
        neutral_prompt: str = "Describe the image.",
    ) -> dict[str, Any]:
        """Prepare an image-dominant control using a fixed neutral instruction."""

        from PIL import Image

        image = Image.open(Path(sample.image).expanduser()).convert("RGB")
        batch = self.processor(
            images=image,
            text=self.format_prompt(neutral_prompt),
            return_tensors="pt",
        )
        return self._move_batch(batch)

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        device = self.input_device
        converted: dict[str, Any] = {}
        for key, value in batch.items():
            if not hasattr(value, "to"):
                converted[key] = value
                continue
            value = value.to(device)
            if key == "pixel_values" and value.is_floating_point():
                value = value.to(dtype=self.model.dtype)
            converted[key] = value
        return converted

    def raw_forward(self, inputs: dict[str, Any], output_hidden_states: bool = True):
        return self.model(
            **inputs,
            use_cache=False,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

    def projected_forward(
        self,
        inputs: dict[str, Any],
        output_hidden_states: bool = False,
    ) -> tuple[Any, ProjectionStats]:
        """Reference implementation using two deterministic prompt forwards.

        Pass 1 obtains H_f. Pass 2 installs a forward hook at layer s and
        replaces H_s with Equation-8 projection before later layers run.
        """

        with self.torch.inference_mode():
            first_pass = self.raw_forward(inputs, output_hidden_states=True)
            if first_pass.hidden_states is None:
                raise RuntimeError("model did not return hidden_states")
            # hidden_states[0] is the embedding output; block f is therefore f+1.
            fused_hidden = first_pass.hidden_states[self.config.sasa.fused_layer + 1].detach()
            captured: dict[str, ProjectionStats] = {}

            def hook(_module, _args, output):
                safety_hidden = first_tensor(output)
                replacement, stats = project_hidden_states(
                    safety_hidden,
                    fused_hidden,
                    epsilon=self.config.sasa.epsilon,
                    mode=self.config.sasa.projection_mode,
                    mix_coefficient=self.config.sasa.mix_coefficient,
                )
                captured["stats"] = stats
                return replace_first_tensor(output, replacement)

            handle = self.layers[self.config.sasa.safety_layer].register_forward_hook(hook)
            try:
                second_pass = self.raw_forward(
                    inputs,
                    output_hidden_states=output_hidden_states,
                )
            finally:
                handle.remove()
            if "stats" not in captured:
                raise RuntimeError("SASA safety-layer hook did not execute")
            return second_pass, captured["stats"]

    def extract_feature(
        self,
        sample: Sample,
        projected: bool = True,
        kind: str | None = None,
    ) -> tuple[np.ndarray, ProjectionStats | None]:
        kind = kind or self.config.features.kind
        inputs = self.prepare(sample)
        with self.torch.inference_mode():
            if projected:
                outputs, stats = self.projected_forward(
                    inputs,
                    output_hidden_states=(kind == "last_hidden"),
                )
            else:
                outputs = self.raw_forward(inputs, output_hidden_states=(kind == "last_hidden"))
                stats = None
            if kind == "logits":
                feature = outputs.logits[:, -1, :]
            elif kind == "last_hidden":
                if outputs.hidden_states is None:
                    raise RuntimeError("last_hidden feature requested but hidden states are absent")
                feature = outputs.hidden_states[-1][:, -1, :]
            else:
                raise ValueError(f"unknown feature kind: {kind}")
        return feature.detach().float().cpu().numpy()[0], stats

    def generate_original(self, sample: Sample) -> str:
        inputs = self.prepare(sample)
        config = self.config.generation
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.do_sample,
            "num_beams": config.num_beams,
        }
        if config.do_sample:
            generate_kwargs.update(temperature=config.temperature, top_p=config.top_p)
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, **generate_kwargs)
        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = generated[:, prompt_length:]
        return self.processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def generate_projected_greedy(self, sample: Sample) -> str:
        """Diagnostic projected generation.

        This deliberately disables KV cache and performs two full forwards for
        every token. It is slow but directly answers whether projected features
        remain generative. The paper does not specify a projected decoding loop.
        """

        inputs = self.prepare(sample)
        generated: list[int] = []
        eos_ids = self._eos_token_ids()
        with self.torch.inference_mode():
            for _ in range(self.config.generation.max_new_tokens):
                outputs, _ = self.projected_forward(inputs, output_hidden_states=False)
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                token_id = int(next_token.item())
                generated.append(token_id)
                inputs["input_ids"] = self.torch.cat([inputs["input_ids"], next_token], dim=1)
                inputs["attention_mask"] = self.torch.cat(
                    [
                        inputs["attention_mask"],
                        self.torch.ones_like(next_token, dtype=inputs["attention_mask"].dtype),
                    ],
                    dim=1,
                )
                if token_id in eos_ids:
                    break
        return self.processor.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

    def _eos_token_ids(self) -> set[int]:
        value = self.model.generation_config.eos_token_id
        if value is None:
            return set()
        if isinstance(value, int):
            return {value}
        return {int(item) for item in value}

    @contextlib.contextmanager
    def ablate_head(self, layer_index: int, head_index: int) -> Iterator[None]:
        """Zero one attention head immediately before the attention output projection."""

        layer = self.layers[layer_index]
        output_projection = attention_output_projection(layer)
        text_config = getattr(self.model.config, "text_config", self.model.config)
        num_heads = int(text_config.num_attention_heads)
        hidden_size = int(text_config.hidden_size)
        head_dim = hidden_size // num_heads
        if not 0 <= head_index < num_heads:
            raise IndexError(f"head {head_index} outside [0, {num_heads})")
        start = head_index * head_dim
        stop = start + head_dim

        def pre_hook(_module, args):
            hidden = args[0].clone()
            hidden[..., start:stop] = 0
            return (hidden,) + tuple(args[1:])

        handle = output_projection.register_forward_pre_hook(pre_hook)
        try:
            yield
        finally:
            handle.remove()

    def extract_final_hidden(
        self,
        sample: Sample,
        ablate: tuple[int, int] | None = None,
    ) -> np.ndarray:
        inputs = self.prepare(sample)
        context = self.ablate_head(*ablate) if ablate is not None else contextlib.nullcontext()
        with context, self.torch.inference_mode():
            outputs = self.raw_forward(inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][:, -1, :]
        return hidden.detach().float().cpu().numpy()[0]

    def layerwise_last_hidden(self, sample: Sample) -> list[np.ndarray]:
        return self.layerwise_last_hidden_from_inputs(self.prepare(sample))

    def layerwise_last_hidden_from_inputs(self, inputs: dict[str, Any]) -> list[np.ndarray]:
        with self.torch.inference_mode():
            outputs = self.raw_forward(inputs, output_hidden_states=True)
        # Exclude embedding output so output index equals decoder block index.
        return [
            state[:, -1, :].detach().float().cpu().numpy()[0]
            for state in outputs.hidden_states[1:]
        ]

    def layerwise_top_tokens(self, sample: Sample, top_k: int = 5) -> list[list[str]]:
        inputs = self.prepare(sample)
        norm = final_norm(self.model)
        lm_head = language_head(self.model)
        result: list[list[str]] = []
        with self.torch.inference_mode():
            outputs = self.raw_forward(inputs, output_hidden_states=True)
            for state in outputs.hidden_states[1:]:
                last = state[:, -1, :]
                logits = lm_head(norm(last))
                ids = logits.topk(top_k, dim=-1).indices[0].tolist()
                result.append([self.processor.tokenizer.decode([token_id]) for token_id in ids])
        return result
