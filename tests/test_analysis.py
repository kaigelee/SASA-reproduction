from __future__ import annotations

import unittest

import numpy as np

from sasa_repro.analysis.alignment import cosine_similarity
from sasa_repro.analysis.layerwise import is_readable_token, tsne_by_layer


class AnalysisTests(unittest.TestCase):
    def test_cosine_similarity_by_layer(self) -> None:
        left = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        right = np.asarray([[1.0, 0.0], [1.0, 0.0]])
        np.testing.assert_allclose(cosine_similarity(left, right), [1.0, 0.0])

    def test_readable_token(self) -> None:
        self.assertTrue(is_readable_token("Ġhello"))
        self.assertFalse(is_readable_token("<s>"))

    def test_tsne_rejects_too_few_samples(self) -> None:
        with self.assertRaises(ValueError):
            tsne_by_layer(np.zeros((2, 2, 4)), np.asarray([0, 1]), layers=[0])


if __name__ == "__main__":
    unittest.main()
