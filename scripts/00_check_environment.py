#!/usr/bin/env python3
from __future__ import annotations

import importlib
import platform
import sys


PACKAGES = (
    "torch",
    "transformers",
    "accelerate",
    "PIL",
    "numpy",
    "sklearn",
    "yaml",
    "joblib",
    "tqdm",
    "matplotlib",
)


def main() -> int:
    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")
    missing: list[str] = []
    for name in PACKAGES:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "unknown")
            print(f"{name:14s} {version}")
        except ImportError:
            missing.append(name)
            print(f"{name:14s} MISSING")
    try:
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA devices:   {torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            memory_gib = properties.total_memory / 1024**3
            print(f"  [{index}] {properties.name} ({memory_gib:.1f} GiB)")
    except ImportError:
        pass
    if missing:
        print("\nInstall missing packages with: python -m pip install -e .")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
