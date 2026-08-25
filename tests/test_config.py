from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sasa_repro.config import SASAConfig, load_config


class ConfigTests(unittest.TestCase):
    def test_defaults_and_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                "seed: 7\nsasa:\n  fused_layer: 10\n  safety_layer: 8\n",
                encoding="utf-8",
            )
            config = load_config(path)
        self.assertEqual(config.seed, 7)
        self.assertEqual(config.sasa.fused_layer, 10)
        self.assertEqual(config.sasa.safety_layer, 8)
        self.assertEqual(config.model.name_or_path, "llava-hf/llava-1.5-7b-hf")

    def test_invalid_layer_order(self) -> None:
        with self.assertRaises(ValueError):
            SASAConfig(fused_layer=13, safety_layer=13)


if __name__ == "__main__":
    unittest.main()
