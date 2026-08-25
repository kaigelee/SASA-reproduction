from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from sasa_repro.config import ProbeConfig
from sasa_repro.probe.model import SafetyProbe


class ProbeTests(unittest.TestCase):
    def test_fit_predict_and_persist(self) -> None:
        features = np.asarray([[-2.0, 0.0], [-1.0, 0.1], [1.0, -0.1], [2.0, 0.0]])
        labels = np.asarray([0, 0, 1, 1])
        probe = SafetyProbe(ProbeConfig(class_weight=None)).fit(features, labels)
        prediction = probe.predict(features)
        np.testing.assert_array_equal(prediction, labels)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.joblib"
            probe.save(path)
            restored = SafetyProbe.load(path)
            np.testing.assert_allclose(
                restored.predict_probability(features), probe.predict_probability(features)
            )

    def test_fit_requires_both_labels(self) -> None:
        with self.assertRaises(ValueError):
            SafetyProbe(ProbeConfig()).fit(np.zeros((3, 2)), np.zeros(3))


if __name__ == "__main__":
    unittest.main()
