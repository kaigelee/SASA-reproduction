from __future__ import annotations

import unittest

from sasa_repro.evaluation.keywords import is_refusal
from sasa_repro.evaluation.metrics import classification_metrics, generation_metrics


class MetricTests(unittest.TestCase):
    def test_classification_metrics(self) -> None:
        result = classification_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["f1"], 1.0)
        self.assertEqual(result["auc"], 1.0)

    def test_generation_metrics(self) -> None:
        rows = [
            {"label": 1, "refused": True},
            {"label": 1, "refused": False},
            {"label": 0, "refused": False},
        ]
        result = generation_metrics(rows)
        self.assertEqual(result["attack_success_rate"], 0.5)
        self.assertEqual(result["benign_refusal_rate"], 0.0)

    def test_refusal_is_case_insensitive(self) -> None:
        self.assertTrue(is_refusal("I CANNOT assist with that."))
        self.assertFalse(is_refusal("Here is a harmless description."))


if __name__ == "__main__":
    unittest.main()
