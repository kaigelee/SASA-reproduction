from __future__ import annotations

import math
import unittest

import numpy as np

from sasa_repro.analysis.head_importance import principal_angle, top_singular_vector


class HeadMathTests(unittest.TestCase):
    def test_principal_angle_ignores_singular_vector_sign(self) -> None:
        vector = np.asarray([1.0, 2.0, 3.0])
        self.assertAlmostEqual(principal_angle(vector, -vector), 0.0)

    def test_orthogonal_angle(self) -> None:
        angle = principal_angle(np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0]))
        self.assertAlmostEqual(angle, math.pi / 2)

    def test_singular_vector_sides(self) -> None:
        matrix = np.asarray([[3.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        self.assertEqual(top_singular_vector(matrix, "left").shape, (3,))
        self.assertEqual(top_singular_vector(matrix, "right").shape, (2,))


if __name__ == "__main__":
    unittest.main()
