from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProjectionStats:
    mean_alpha: float
    mean_source_norm: float
    mean_projected_norm: float


def project_hidden_states(
    safety_hidden: Any,
    fused_hidden: Any,
    epsilon: float = 1e-6,
    mode: str = "replace",
    mix_coefficient: float = 1.0,
) -> tuple[Any, ProjectionStats]:
    """Apply Equation 8 from SASA token by token.

    Equation 8 projects H_s onto the direction defined by H_f:
        alpha = <H_s, H_f> / (<H_f, H_f> + epsilon)
        projected = alpha * H_f

    `replace` is the paper-faithful setting. The other modes are diagnostics
    because the paper does not report them.
    """

    if safety_hidden.shape != fused_hidden.shape:
        raise ValueError(
            f"safety/fused hidden-state shape mismatch: "
            f"{tuple(safety_hidden.shape)} vs {tuple(fused_hidden.shape)}"
        )
    if mode not in {"replace", "residual", "interpolate"}:
        raise ValueError(f"unsupported projection mode: {mode}")

    fused_hidden = fused_hidden.to(device=safety_hidden.device, dtype=safety_hidden.dtype)
    numerator = (safety_hidden * fused_hidden).sum(dim=-1, keepdim=True)
    denominator = fused_hidden.square().sum(dim=-1, keepdim=True).clamp_min(epsilon)
    alpha = numerator / denominator
    projected = alpha * fused_hidden

    if mode == "replace":
        output = projected
    elif mode == "residual":
        output = safety_hidden + mix_coefficient * projected
    else:
        output = (1.0 - mix_coefficient) * safety_hidden + mix_coefficient * projected

    stats = ProjectionStats(
        mean_alpha=float(alpha.detach().float().mean().cpu()),
        mean_source_norm=float(safety_hidden.detach().float().norm(dim=-1).mean().cpu()),
        mean_projected_norm=float(projected.detach().float().norm(dim=-1).mean().cpu()),
    )
    return output, stats


def replace_first_tensor(output: Any, replacement: Any) -> Any:
    """Preserve a Transformer block's output container while replacing hidden states."""

    if isinstance(output, tuple):
        return (replacement,) + output[1:]
    if isinstance(output, list):
        return [replacement, *output[1:]]
    return replacement


def first_tensor(output: Any) -> Any:
    if isinstance(output, (tuple, list)):
        return output[0]
    return output

