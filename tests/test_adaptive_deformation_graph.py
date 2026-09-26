from types import SimpleNamespace
import tempfile
import unittest

import torch
from torch import nn

from scene.deformation_graph import SparseDeformationGraph
from utils.adaptive_deformation_graph import AdaptiveNodeConfig, AdaptiveNodeManager


class _Appearance(nn.Module):
    def __init__(self, rows, width=8):
        super().__init__()
        self.node_appearance = nn.Parameter(
            torch.arange(rows * width, dtype=torch.float32).reshape(rows, width) / 1000)

    def node_context(self, indices, weights):
        return (self.node_appearance[indices] * weights.unsqueeze(-1)).sum(1)


def _model():
    node_xyz = torch.stack((torch.linspace(-1, 1, 2000),
                            torch.zeros(2000), torch.zeros(2000)), dim=1)
    xyz = torch.tensor([
        [-0.95, 0.0, 0.0], [-0.90, 0.02, 0.0], [-0.85, -0.02, 0.0],
        [0.80, 0.0, 0.0], [0.85, 0.02, 0.0], [0.90, -0.02, 0.0],
    ])
    indices = torch.tensor([
        [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3],
        [1900, 1901, 1902, 1903], [1900, 1901, 1902, 1903], [1900, 1901, 1902, 1903],
    ])
    weights = torch.tensor([[0.4, 0.3, 0.2, 0.1]]).repeat(6, 1)
    appearance = _Appearance(2000)
    optimizer = torch.optim.Adam([appearance.node_appearance], lr=1e-3)
    model = SimpleNamespace(
        _xyz=nn.Parameter(xyz.clone()),
        deformation_graph=SparseDeformationGraph(node_xyz, indices, weights),
        appearance_model=appearance,
        optimizer=optimizer,
    )
    # Materialize Adam moments so resizing behavior is exercised.
    appearance.node_appearance.square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return model


def _config(**kwargs):
    values = dict(mode="split_once", min_coverage=1, min_age=0,
                  split_z=-100.0, max_splits=1)
    values.update(kwargs)
    return AdaptiveNodeConfig(**values)


class AdaptiveDeformationGraphTests(unittest.TestCase):
    def test_statistics_use_weighted_k4_support(self):
        with tempfile.TemporaryDirectory() as path:
            model = _model()
            manager = AdaptiveNodeManager(model, _config(mode="stats"), path)
            gradient = torch.tensor([[1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0],
                                     [4.0, 0, 0], [5.0, 0, 0], [6.0, 0, 0]])
            manager.update_statistics(gradient, torch.ones(6, dtype=torch.bool))
            stats = manager.node_statistics(iteration=5000)
            self.assertTrue(torch.isclose(stats["influence"][0], torch.tensor(1.2)))
            self.assertTrue(torch.isclose(stats["geometry"][0], torch.tensor(2.0)))
            self.assertEqual(stats["coverage"][0].item(), 3)
            self.assertTrue(torch.isfinite(stats["normalized"]).all())

    def test_split_is_local_and_new_adam_rows_are_zero(self):
        with tempfile.TemporaryDirectory() as path:
            model = _model()
            manager = AdaptiveNodeManager(model, _config(), path)
            old_indices = model.deformation_graph.gaussian_node_indices.clone()
            old_weights = model.deformation_graph.gaussian_node_weights.clone()
            old_feature = model.appearance_model.node_appearance[0].detach().clone()
            child_position = torch.tensor([-0.90, 0.08, 0.0])
            affected, old_ids = manager._append_children([0], [child_position])
            manager._resize_appearance([0])
            manager.birth_iterations = torch.cat((manager.birth_iterations, torch.tensor([5000])))
            manager.prune_counters = torch.cat((manager.prune_counters, torch.tensor([0])))

            outside = ~affected
            self.assertTrue(outside.any())
            self.assertTrue(torch.equal(
                manager.node_ids[model.deformation_graph.gaussian_node_indices[outside]],
                old_ids[outside]))
            self.assertTrue(torch.equal(
                model.deformation_graph.gaussian_node_indices[outside], old_indices[outside]))
            self.assertTrue(torch.equal(
                model.deformation_graph.gaussian_node_weights[outside], old_weights[outside]))
            self.assertTrue(torch.equal(model.appearance_model.node_appearance[-1], old_feature))
            adam_state = model.optimizer.state[model.appearance_model.node_appearance]
            self.assertEqual(torch.count_nonzero(adam_state["exp_avg"][-1]), 0)
            self.assertEqual(torch.count_nonzero(adam_state["exp_avg_sq"][-1]), 0)

    def test_adaptive_checkpoint_roundtrip_reproduces_context(self):
        with tempfile.TemporaryDirectory() as path:
            model = _model()
            config = _config()
            manager = AdaptiveNodeManager(model, config, path + "/one")
            manager._append_children([0], [torch.tensor([-0.90, 0.08, 0.0])])
            manager._resize_appearance([0])
            manager.birth_iterations = torch.cat((manager.birth_iterations, torch.tensor([5000])))
            manager.prune_counters = torch.cat((manager.prune_counters, torch.tensor([0])))
            manager.gaussian_score.copy_(torch.arange(6, dtype=torch.float32))
            manager.gaussian_observations.fill_(7)
            context = model.appearance_model.node_context(
                model.deformation_graph.gaussian_node_indices,
                model.deformation_graph.gaussian_node_weights).detach().clone()
            checkpoint = path + "/adaptive.pth"
            manager.save(checkpoint, 5000)

            restored_model = _model()
            restored = AdaptiveNodeManager(restored_model, config, path + "/two")
            restored.load(checkpoint)
            restored_context = restored_model.appearance_model.node_context(
                restored_model.deformation_graph.gaussian_node_indices,
                restored_model.deformation_graph.gaussian_node_weights).detach()
            self.assertTrue(torch.equal(context, restored_context))
            self.assertTrue(torch.equal(manager.node_ids, restored.node_ids))
            self.assertTrue(torch.equal(manager.gaussian_score, restored.gaussian_score))
            self.assertTrue(torch.equal(
                manager.gaussian_observations, restored.gaussian_observations))
            self.assertEqual(restored.state_dict(5000)["optimizer"]["state"].keys(),
                             manager.state_dict(5000)["optimizer"]["state"].keys())

    def test_prune_repairs_only_local_rows_and_preserves_adam(self):
        with tempfile.TemporaryDirectory() as path:
            model = _model()
            manager = AdaptiveNodeManager(
                model, _config(mode="prune_once", prune_windows=2), path)
            old_ids = manager.node_ids[
                model.deformation_graph.gaussian_node_indices].clone()
            old_weights = model.deformation_graph.gaussian_node_weights.clone()
            old_feature = model.appearance_model.node_appearance[1].detach().clone()
            old_moment = model.optimizer.state[
                model.appearance_model.node_appearance]["exp_avg"][1].clone()
            affected, _, _ = manager._prune_nodes([0])

            self.assertEqual(len(model.deformation_graph.node_xyz), 1999)
            self.assertEqual(len(model.appearance_model.node_appearance), 1999)
            outside = ~affected
            new_ids = manager.node_ids[model.deformation_graph.gaussian_node_indices]
            self.assertTrue(torch.equal(new_ids[outside], old_ids[outside]))
            self.assertTrue(torch.equal(
                model.deformation_graph.gaussian_node_weights[outside],
                old_weights[outside]))
            self.assertTrue(torch.equal(
                model.appearance_model.node_appearance[0], old_feature))
            moment = model.optimizer.state[
                model.appearance_model.node_appearance]["exp_avg"]
            self.assertTrue(torch.equal(moment[0], old_moment))

    def test_prune_requires_two_low_usage_windows_and_respects_age(self):
        with tempfile.TemporaryDirectory() as path:
            model = _model()
            manager = AdaptiveNodeManager(
                model, _config(mode="prune_once", prune_windows=2,
                               prune_quantile=0.2, min_age=5000), path)
            manager.gaussian_observations.fill_(10)
            manager.gaussian_score.fill_(1)
            stats = manager.node_statistics(5000)
            first = manager._prune_candidates(5000, stats)
            second = manager._prune_candidates(7500, stats)
            self.assertEqual(len(first), 0)
            self.assertGreater(len(second), 0)
            manager.birth_iterations[second[0]] = 7000
            third = manager._prune_candidates(7500, stats)
            self.assertNotIn(int(second[0]), third.tolist())


if __name__ == "__main__":
    unittest.main()
