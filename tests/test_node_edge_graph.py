import unittest

import torch

from utils.node_edge_graph import NodeEdgeConsistencyGraph


class NodeEdgeGraphTests(unittest.TestCase):
    def setUp(self):
        xyz = torch.tensor([[0., 0., 0.], [1., 0., 0.],
                            [1., 1., 0.], [0., 1., 0.]])
        edges = torch.tensor([[0, 0, 1, 2], [1, 3, 2, 3]])
        lengths = (xyz[edges[0]] - xyz[edges[1]]).norm(dim=1)
        self.xyz = xyz
        self.graph = NodeEdgeConsistencyGraph(
            xyz, edges, torch.tensor([1., 2., 3., 4.]), lengths,
            "test", {"answer": 42})

    def test_rigid_motion_and_identity_are_zero(self):
        angle = torch.tensor(0.5)
        rotation = torch.tensor([[torch.cos(angle), -torch.sin(angle), 0.],
                                 [torch.sin(angle), torch.cos(angle), 0.],
                                 [0., 0., 1.]])
        moved = self.xyz @ rotation.T + torch.tensor([2., -4., 1.])
        self.assertLess(self.graph(self.xyz, moved).item(), 1e-12)
        self.assertEqual(self.graph(self.xyz, self.xyz).item(), 0.)

    def test_scaling_and_stretch_are_positive(self):
        self.assertGreater(self.graph(self.xyz, 1.2*self.xyz).item(), 0.)
        stretched = self.xyz.clone()
        stretched[0, 0] -= 0.5
        self.assertGreater(self.graph(self.xyz, stretched).item(), 0.)

    def test_normalized_weighted_mean_and_roundtrip(self):
        moved = self.xyz.clone()
        moved[0, 0] -= 0.5
        source, target = self.graph.edge_indices
        relative = (((self.xyz[source]-self.xyz[target]).norm(dim=1) -
                     (moved[source]-moved[target]).norm(dim=1)) /
                    self.graph.canonical_lengths)
        expected = (self.graph.edge_weights*relative.square()).sum()/self.graph.edge_weights.sum()
        self.assertTrue(torch.allclose(self.graph(self.xyz, moved), expected))
        restored = NodeEdgeConsistencyGraph.from_state_dict_payload(
            self.graph.state_dict_payload(), device="cpu")
        self.assertEqual(restored.topology, "test")
        self.assertEqual(restored.metadata, {"answer": 42})
        for key, value in self.graph.state_dict().items():
            self.assertTrue(torch.equal(value, restored.state_dict()[key]))


if __name__ == "__main__":
    unittest.main()
