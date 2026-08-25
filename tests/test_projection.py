from __future__ import annotations

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class ProjectionTests(unittest.TestCase):
    def test_projection_matches_closed_form(self) -> None:
        import torch

        from sasa_repro.model.projection import project_hidden_states

        safety = torch.tensor([[[2.0, 2.0]]])
        fused = torch.tensor([[[1.0, 0.0]]])
        projected, stats = project_hidden_states(safety, fused, epsilon=1e-12)
        torch.testing.assert_close(projected, torch.tensor([[[2.0, 0.0]]]))
        self.assertAlmostEqual(stats.mean_alpha, 2.0)

    def test_zero_direction_stays_finite(self) -> None:
        import torch

        from sasa_repro.model.projection import project_hidden_states

        projected, _ = project_hidden_states(torch.ones(1, 2, 3), torch.zeros(1, 2, 3))
        self.assertTrue(torch.isfinite(projected).all())
        self.assertEqual(float(projected.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
