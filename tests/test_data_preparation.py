from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sasa_repro.data.preparation import prepare_figstep, prepare_mm_safetybench, prepare_vlguard


class DataPreparationTests(unittest.TestCase):
    def test_mm_safetybench_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions = root / "data" / "processed_questions"
            questions.mkdir(parents=True)
            (questions / "Illegal Activity.json").write_text(
                json.dumps(
                    {
                        "0": {
                            "Question": "source",
                            "Rephrased Question": "controlled prompt",
                        }
                    }
                ),
                encoding="utf-8",
            )
            samples = prepare_mm_safetybench(root, strict_images=False)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].label, 1)
        self.assertEqual(samples[0].category, "Illegal Activity")

    def test_vlguard_expands_safe_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "test.json"
            metadata.write_text(
                json.dumps(
                    [
                        {
                            "id": "x",
                            "image": "x.jpg",
                            "safe": True,
                            "instr-resp": [
                                {
                                    "safe_instruction": "describe",
                                    "unsafe_instruction": "controlled unsafe prompt",
                                }
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            samples = prepare_vlguard(metadata, root, strict_images=False)
        self.assertEqual(sorted(sample.label for sample in samples), [0, 1])

    def test_figstep_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "SafeBench.csv"
            csv_path.write_text("id,category,question\n7,test,source question\n", encoding="utf-8")
            samples = prepare_figstep(csv_path, root, strict_images=False)
        self.assertEqual(samples[0].id, "figstep:7")
        self.assertEqual(samples[0].metadata["source_question"], "source question")


if __name__ == "__main__":
    unittest.main()
