from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sasa_repro.io import load_manifest, write_manifest
from sasa_repro.schema import Sample


class ManifestTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        sample = Sample(
            id="one",
            image="/tmp/not-required.jpg",
            prompt="What is shown?",
            label=0,
            dataset="custom",
            metadata={"source": "unit-test"},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            write_manifest(path, [sample])
            loaded = load_manifest(path)
        self.assertEqual(loaded, [sample])

    def test_duplicate_ids_are_rejected(self) -> None:
        sample = Sample("same", "/tmp/a.jpg", "prompt", 1, "test")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            write_manifest(path, [sample, sample])
            with self.assertRaises(ValueError):
                load_manifest(path)


if __name__ == "__main__":
    unittest.main()
