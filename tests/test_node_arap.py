import math
import unittest

import torch

from utils.node_arap import CanonicalNodeARAPGraph
from utils.loss_utils import canonical_xyz_laplacian


def _axis_angle_rotation(axis, angle, dtype=torch.float64):
    axis = torch.as_tensor(axis, dtype=dtype)
    axis = axis / axis.norm()
    x, y, z = axis
    skew = torch.tensor([[0., -z, y], [z, 0., -x], [-y, x, 0.]], dtype=dtype)
    eye = torch.eye(3, dtype=dtype)
    return eye + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)


class CanonicalNodeARAPTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.xyz = torch.randn(18, 3, dtype=torch.float64)
        self.graph = CanonicalNodeARAPGraph.build(self.xyz, k=6)

    def test_rigid_transform_has_zero_loss(self):
        rotation = _axis_angle_rotation([0.2, -0.7, 0.4], 0.73)
        translation = torch.tensor([1.2, -0.5, 2.1], dtype=self.xyz.dtype)
        transformed = self.xyz @ rotation.T + translation
        loss, diagnostics = self.graph(transformed, self.xyz, return_diagnostics=True)
        self.assertLess(loss.item(), 1e-20)
        self.assertTrue(torch.allclose(
            torch.det(diagnostics.rotations), torch.ones(len(self.xyz), dtype=self.xyz.dtype),
            atol=1e-10, rtol=1e-10))

    def test_uniform_scaling_is_positive_and_finite(self):
        loss = self.graph(self.xyz, 1.25 * self.xyz)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 1e-3)

    def test_reflection_is_corrected_to_so3(self):
        reflected = self.xyz.clone()
        reflected[:, 0] *= -1
        loss, diagnostics = self.graph(self.xyz, reflected, return_diagnostics=True)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(diagnostics.reflection_ratio, 0)
        determinant = torch.det(diagnostics.rotations)
        self.assertTrue((determinant > 0.999999).all())

    def test_detached_rotation_gradient_matches_refit_finite_difference(self):
        displacement = (0.025 * torch.randn_like(self.xyz)).requires_grad_(True)
        loss = self.graph(self.xyz, self.xyz + displacement)
        gradient, = torch.autograd.grad(loss, displacement)
        node, axis = 7, 1
        epsilon = 2e-6
        with torch.no_grad():
            plus = displacement.detach().clone()
            minus = displacement.detach().clone()
            plus[node, axis] += epsilon
            minus[node, axis] -= epsilon
            finite_difference = (
                self.graph(self.xyz, self.xyz + plus) -
                self.graph(self.xyz, self.xyz + minus)
            ) / (2 * epsilon)
        self.assertAlmostEqual(
            gradient[node, axis].item(), finite_difference.item(), delta=2e-6)

    def test_collinear_and_nearly_planar_neighborhoods_are_finite(self):
        line = torch.stack((torch.linspace(-1, 1, 12),
                            torch.zeros(12), torch.zeros(12)), dim=1).double()
        plane = torch.stack((torch.linspace(-1, 1, 12),
                             torch.linspace(1, -1, 12),
                             1e-9 * torch.arange(12)), dim=1).double()
        for canonical in (line, plane):
            graph = CanonicalNodeARAPGraph.build(canonical, k=6)
            target = canonical + 1e-3 * torch.randn_like(canonical)
            loss, diagnostics = graph(canonical, target, return_diagnostics=True)
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(diagnostics.rotations).all())
            self.assertTrue(torch.isfinite(diagnostics.singular_values).all())

    def test_topology_and_weights_are_symmetric_and_fixed(self):
        idx = self.graph.neighbor_indices
        mask = self.graph.neighbor_mask
        weights = self.graph.edge_weights
        edges = {(i, int(idx[i, j])): weights[i, j].item()
                 for i in range(len(idx)) for j in range(idx.shape[1]) if mask[i, j]}
        for (i, j), weight in edges.items():
            self.assertIn((j, i), edges)
            self.assertAlmostEqual(weight, edges[(j, i)], places=12)
        before = {key: value.clone() for key, value in self.graph.state_dict().items()}
        _ = self.graph(self.xyz, self.xyz + 0.01 * torch.randn_like(self.xyz))
        for key, value in before.items():
            self.assertTrue(torch.equal(value, self.graph.state_dict()[key]))

        restored = CanonicalNodeARAPGraph.from_state_dict_payload(
            self.graph.state_dict_payload())
        self.assertEqual(restored.k, self.graph.k)
        for key, value in self.graph.state_dict().items():
            self.assertTrue(torch.equal(value, restored.state_dict()[key]))

    def test_canonical_laplacian_is_exact_knn_component_and_xyz_only(self):
        xyz = self.xyz.detach().clone().requires_grad_(True)
        idx = torch.cdist(xyz.detach(), xyz.detach()).topk(
            7, largest=False).indices[:, 1:]
        expected = (xyz - xyz[idx].mean(1)).square().sum(1).mean()
        actual = canonical_xyz_laplacian(xyz, idx)
        self.assertTrue(torch.equal(actual, expected))
        gradient, = torch.autograd.grad(actual, xyz)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.norm()), 0)


if __name__ == "__main__":
    unittest.main()
