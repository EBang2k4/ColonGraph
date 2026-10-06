import unittest

import torch

from utils.gaussian_isometry import LocalGaussianIsometryGraph


class LocalGaussianIsometryTest(unittest.TestCase):
    def setUp(self):
        self.xyz = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 1.0],
        ])
        source = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 3])
        target = torch.tensor([1, 2, 3, 4, 2, 3, 4, 3, 4, 4])
        edges = torch.stack((source, target))
        lengths = (self.xyz[source] - self.xyz[target]).norm(dim=1)
        degree = torch.bincount(torch.cat((source, target)), minlength=len(self.xyz))
        self.graph = LocalGaussianIsometryGraph(
            edges, torch.ones_like(lengths), lengths,
            torch.ones_like(lengths, dtype=torch.long), degree,
            fallback_directed_edges=0, candidate_k=4, minimum_degree=4,
            sigma=float(lengths.median()))

    def test_rigid_translation_and_rotation_are_zero(self):
        theta = torch.tensor(0.7)
        rotation = torch.tensor([
            [torch.cos(theta), -torch.sin(theta), 0.0],
            [torch.sin(theta), torch.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ])
        rigid = self.xyz @ rotation.T + torch.tensor([2.0, -3.0, 0.5])
        self.assertLess(self.graph(self.xyz, rigid).item(), 1e-12)

    def test_scaling_and_local_stretch_are_detected(self):
        scaling = self.graph(self.xyz, 1.2 * self.xyz).item()
        stretched = self.xyz.clone()
        stretched[4, 2] += 0.5
        local = self.graph(self.xyz, stretched).item()
        self.assertGreater(scaling, 0.0)
        self.assertGreater(local, 0.0)

    def test_temporal_identity_is_zero(self):
        self.assertEqual(self.graph(self.xyz, self.xyz).item(), 0.0)

    def test_checkpoint_payload_roundtrip(self):
        restored = LocalGaussianIsometryGraph.from_state_dict_payload(
            self.graph.state_dict_payload(), device="cpu")
        self.assertTrue(torch.equal(restored.edge_indices, self.graph.edge_indices))
        self.assertTrue(torch.equal(restored.edge_weights, self.graph.edge_weights))
        self.assertTrue(torch.equal(restored.canonical_lengths,
                                    self.graph.canonical_lengths))
        self.assertEqual(restored.graph_statistics(), self.graph.graph_statistics())

    def test_builder_filters_by_shared_k4_and_guarantees_degree(self):
        distances = torch.cdist(self.xyz, self.xyz)
        candidates = distances.argsort(dim=1)[:, 1:5]
        k4 = torch.tensor([
            [0, 1, 2, 3],
            [0, 1, 4, 5],
            [6, 7, 8, 9],
            [6, 7, 10, 11],
            [12, 13, 14, 15],
        ])
        graph = LocalGaussianIsometryGraph.build(
            self.xyz, candidates, k4, candidate_k=4, minimum_degree=4,
            chunk_size=2)
        self.assertTrue((graph.degree >= 4).all())
        self.assertGreater(graph.fallback_directed_edges, 0)


if __name__ == "__main__":
    unittest.main()
