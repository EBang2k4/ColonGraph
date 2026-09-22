"""Tests for the sampling equation and coarse-gradient observation semantics."""
import unittest
import torch
from utils.node_sampling import (CoarseGradientImportance, farthest_point_sampling,
                                 weighted_farthest_point_sampling,
                                 validate_index_aligned_reference)


class WeightedFPSTests(unittest.TestCase):
    def test_matches_literal_euclidean_score(self):
        gen = torch.Generator().manual_seed(31)
        xyz = torch.randn(81, 3, generator=gen)
        importance = torch.rand(81, generator=gen)
        for alpha in (.25, .5, 1.):
            selected = [0]
            for _ in range(23):
                distance = torch.linalg.vector_norm(xyz[:, None] - xyz[selected], dim=-1).amin(1)
                score = distance * (1 + alpha * importance)
                score[selected] = -1
                selected.append(score.argmax().item())
            actual = weighted_farthest_point_sampling(xyz, 24, importance, alpha)
            self.assertEqual(actual.tolist(), selected)

    def test_zero_alpha_matches_standard_and_preserves_rng(self):
        xyz = torch.randn(67, 3, generator=torch.Generator().manual_seed(2))
        w = torch.linspace(0, 1, len(xyz))
        rng = torch.get_rng_state().clone()
        for alpha, importance in ((0., w), (1., torch.zeros_like(w))):
            actual = weighted_farthest_point_sampling(xyz, 25, importance, alpha)
            self.assertTrue(torch.equal(actual, farthest_point_sampling(xyz, 25)))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_distance_is_not_squared_before_weight(self):
        xyz = torch.tensor([[0., 0, 0], [3., 0, 0], [2., 0, 0]])
        w = torch.tensor([0., 0., 1.])
        self.assertEqual(weighted_farthest_point_sampling(xyz, 2, w).tolist(), [0, 2])

    def test_duplicates_remain_unique_deterministic(self):
        xyz = torch.zeros(30, 3)
        w = torch.linspace(0, 1, 30)
        a = weighted_farthest_point_sampling(xyz, 30, w)
        b = weighted_farthest_point_sampling(xyz, 30, w)
        self.assertEqual(a[0].item(), 0)
        self.assertEqual(len(a.unique()), 30)
        self.assertTrue(torch.equal(a, b))

    def test_moments_confidence_percentiles_and_no_input_mutation(self):
        stats = CoarseGradientImportance(torch.zeros(3, 3))
        g1 = torch.tensor([[1., 0, 0], [7., 0, 0], [4., 0, 0]])
        g2 = torch.tensor([[-1., 0, 0], [7., 0, 0], [-4., 0, 0]])
        before = g1.clone()
        stats.update(g1, torch.tensor([True, True, False]))
        stats.update(g2, torch.tensor([True, False, False]))
        out = stats.finalize()
        self.assertTrue(torch.equal(g1, before))
        self.assertEqual(out['grad_count'].tolist(), [2, 1, 0])
        torch.testing.assert_close(out['deviation'], torch.tensor([1., 0, 0], dtype=torch.float64))
        torch.testing.assert_close(out['confidence'], torch.tensor([.4, .2, 0], dtype=torch.float64))
        self.assertEqual(out['importance'].tolist(), [1., 0., 0.])
        self.assertEqual(out['coarse_steps'], 2)

    def test_zero_variance_fallback_and_invalid_inputs(self):
        stats = CoarseGradientImportance(torch.zeros(4, 3))
        for _ in range(7):
            stats.update(torch.ones(4, 3), torch.ones(4, dtype=torch.bool))
        self.assertEqual(stats.finalize()['importance'].tolist(), [0.] * 4)
        for w in (torch.full((4,), float('nan')), torch.full((4,), -1.), torch.full((4,), 2.)):
            with self.assertRaises(ValueError):
                weighted_farthest_point_sampling(torch.zeros(4, 3), 2, w)
        with self.assertRaises(FloatingPointError):
            stats.update(torch.full((4, 3), float('inf')), torch.ones(4, dtype=torch.bool))

    def test_paired_reference_alignment_accepts_sparse_small_drift(self):
        reference = torch.arange(3000, dtype=torch.float32).reshape(1000, 3)
        observed = reference.clone()
        observed[17, 1] += .05
        result = validate_index_aligned_reference(observed, reference)
        self.assertTrue(result['passed'])
        self.assertEqual(result['mismatched_gaussians'], 1)
        self.assertAlmostEqual(result['mismatched_fraction'], .001)

    def test_paired_reference_alignment_rejects_broad_or_large_drift(self):
        reference = torch.arange(3000, dtype=torch.float32).reshape(1000, 3)
        broad = reference.clone()
        broad[:21, 0] += .01
        with self.assertRaisesRegex(ValueError, 'drift is too large'):
            validate_index_aligned_reference(broad, reference)
        large = reference.clone()
        large[0, 0] += 10
        with self.assertRaisesRegex(ValueError, 'drift is too large'):
            validate_index_aligned_reference(large, reference)


if __name__ == '__main__':
    unittest.main()
