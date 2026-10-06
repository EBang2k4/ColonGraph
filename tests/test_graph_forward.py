import unittest

import torch

from utils.graph_forward import (
    GraphMotionResidual,
    k4_topology_tensors,
    repair_gaussian_k4_minimal,
    preserve_base_motion,
)


class GraphForwardTests(unittest.TestCase):
    def test_g4_limits_cancellation_and_preserves_perpendicular(self):
        base = torch.tensor([[2., 0., 0.], [0., 3., 0.], [1., 0., 0.]], dtype=torch.float64)
        raw = torch.tensor([[-1., 4., 5.], [4., 1., 2.], [-.05, 3., 0.]], dtype=torch.float64)
        safe = preserve_base_motion(base, raw)
        torch.testing.assert_close(safe[0], torch.tensor([-.2, 4., 5.], dtype=torch.float64))
        self.assertTrue(torch.equal(safe[1:], raw[1:]))
        unit = base / base.norm(dim=-1, keepdim=True)
        self.assertTrue(bool(((safe * unit).sum(-1) >= -.1 * base.norm(dim=-1) - 1e-12).all()))
        raw_perp = raw - (raw * unit).sum(-1, keepdim=True) * unit
        safe_perp = safe - (safe * unit).sum(-1, keepdim=True) * unit
        torch.testing.assert_close(raw_perp, safe_perp, rtol=0, atol=1e-12)

    def test_g4_zero_and_tiny_motion_finite_gradients(self):
        base = torch.tensor([[0., 0., 0.], [1e-10, 0., 0.], [1., 0., 0.]], requires_grad=True)
        raw = torch.tensor([[-1., 2., 3.], [-1., 2., 3.], [-1., 2., 3.]], requires_grad=True)
        safe = preserve_base_motion(base, raw)
        self.assertTrue(torch.equal(safe[:2], raw[:2]))
        (base + safe).square().mean().backward()
        self.assertTrue(bool(torch.isfinite(base.grad).all()))
        self.assertTrue(bool(torch.isfinite(raw.grad).all()))
        self.assertGreater(float(raw.grad[:, 1:].abs().sum()), 0)

    def test_g4_same_initialization_no_new_parameters_and_roundtrip(self):
        xyz = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        edges = torch.tensor([[0, 0, 1], [1, 2, 2]])
        weights = torch.tensor([1., 3., 2.])
        rng = torch.get_rng_state().clone()
        g3 = GraphMotionResidual(xyz, edges, weights, seed=4242, mode="g3")
        g4 = GraphMotionResidual(xyz, edges, weights, seed=4242, mode="g4")
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in g3.parameters()), sum(p.numel() for p in g4.parameters()))
        for key in g3.state_dict():
            self.assertTrue(torch.equal(g3.state_dict()[key], g4.state_dict()[key]))
        base = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 0.]], requires_grad=True)
        self.assertTrue(torch.equal(g3(base)[0], g4(base)[0]))
        g4(base)[0].square().mean().backward()
        self.assertGreater(float(g4.decoder[-1].weight.grad.abs().sum()), 0)
        with torch.no_grad():
            g4.decoder[-1].bias.copy_(torch.tensor([-1., -.5, .25]))
        restored = GraphMotionResidual.from_state_dict_payload(g4.state_dict_payload(), device="cpu")
        self.assertEqual(restored.mode, "g4")
        self.assertEqual(restored.cancellation_alpha, .1)
        for left, right in zip(g4(base), restored(base)):
            self.assertTrue(torch.equal(left, right))
        old_payload = g3.state_dict_payload()
        old_payload.pop("cancellation_alpha")
        legacy = GraphMotionResidual.from_state_dict_payload(old_payload, device="cpu")
        self.assertTrue(torch.equal(g3(base)[0], legacy(base)[0]))

    def test_minimal_k4_repair_preserves_compatible_rows_bit_exact(self):
        nodes = torch.tensor([
            [0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
            [4., 0., 0.], [5., 0., 0.],
        ])
        edges = torch.tensor([[0, 0, 0, 4], [1, 2, 3, 5]])
        gaussian = torch.tensor([[.1, .1, .1], [.2, .2, .2]])
        indices = torch.tensor([[0, 1, 2, 3], [0, 1, 4, 5]])
        weights = torch.tensor([[.4, .3, .2, .1], [.6, .2, .1, .1]])
        repaired_i, repaired_w, changed, fallback, audit = repair_gaussian_k4_minimal(
            gaussian, nodes, indices, weights, edges)
        self.assertTrue(torch.equal(repaired_i[0], indices[0]))
        self.assertTrue(torch.equal(repaired_w[0], weights[0]))
        self.assertFalse(bool(changed[0]))
        self.assertTrue(bool(changed[1]))
        self.assertFalse(bool(fallback.any()))
        topology = k4_topology_tensors(repaired_i, edges, len(nodes))
        self.assertTrue(bool(topology["connected"].all()))
        self.assertEqual(audit["changed_gaussians"], 1)
        self.assertEqual(audit["fallback_gaussians"], 0)
        torch.testing.assert_close(repaired_w.sum(1), torch.ones(2))

    def test_zero_init_residual_identity_boundaries_and_gradient(self):
        xyz = torch.tensor([[0., 0., 0.], [1., 0., 0.],
                            [1., 1., 0.], [0., 1., 0.]])
        edges = torch.tensor([[0, 0, 1, 2], [1, 3, 2, 3]])
        weights = torch.tensor([1., 2., 3., 4.])
        module = GraphMotionResidual(xyz, edges, weights, hidden_dim=8, seed=9)
        base = torch.randn(4, 3, requires_grad=True)
        final, residual = module(base)
        torch.testing.assert_close(residual, torch.zeros_like(residual), rtol=0, atol=0)
        torch.testing.assert_close(final, base, rtol=0, atol=0)
        final.square().mean().backward()
        self.assertGreater(module.decoder[-1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(base.grad.abs().sum().item(), 0)

    def test_weighted_aggregation_and_checkpoint_roundtrip(self):
        xyz = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.]])
        edges = torch.tensor([[0, 0, 1], [1, 2, 2]])
        weights = torch.tensor([1., 3., 2.])
        module = GraphMotionResidual(xyz, edges, weights, hidden_dim=7,
                                     seed=4, mode="g3")
        base = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        aggregate = module.aggregate(base)
        expected_motion0 = (1*torch.tensor([1., 0., 0.]) +
                            3*torch.tensor([0., 1., 0.]))/4
        torch.testing.assert_close(aggregate[0, :3], expected_motion0)
        with torch.no_grad():
            module.decoder[-1].bias.fill_(.25)
        restored = GraphMotionResidual.from_state_dict_payload(
            module.state_dict_payload(), device="cpu")
        self.assertEqual(restored.mode, "g3")
        for left, right in zip(module(base), restored(base)):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertTrue(torch.equal(module.edge_weights, restored.edge_weights))


if __name__ == "__main__":
    unittest.main()
