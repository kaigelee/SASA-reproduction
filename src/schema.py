from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Sample:
    """Canonical image-text sample used by every pipeline stage."""

    id: str
    image: str
    prompt: str
    label: int
    dataset: str
    split: str = "unspecified"
    category: str = "unspecified"
    answer: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("sample id must not be empty")
        if not self.prompt:
            raise ValueError(f"sample {self.id!r} has an empty prompt")
        if self.label not in (0, 1):
            raise ValueError(f"sample {self.id!r} label must be 0 or 1")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Sample":
        return cls(
            id=str(value["id"]),
            image=str(value["image"]),
            prompt=str(value["prompt"]),
            label=int(value["label"]),
            dataset=str(value.get("dataset", "unknown")),
            split=str(value.get("split", "unspecified")),
            category=str(value.get("category", "unspecified")),
            answer=str(value.get("answer", "")),
            metadata=dict(value.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate_image(self) -> None:
        path = Path(self.image).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"image for sample {self.id!r} not found: {path}")


@dataclass(frozen=True)
class Prediction:
    id: str
    dataset: str
    label: int
    unsafe_probability: float | None
    predicted_label: int | None
    response: str
    refused: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

