"""Numerical, dispatch and renderer-gradient checks for spec.md."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import tempfile
import unittest
import numpy as np
import torch
from torch import nn
from arguments import ModelHiddenParams, OptimizationParams, PipelineParams
from scene.deformation import NodeDeformationMLP
from scene.deformation_graph import SparseDeformationGraph, node_rotation_matrices
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.node_sampling import farthest_point_sampling, build_gaussian_node_knn
from types import SimpleNamespace


class SparseNodesTests(unittest.TestCase):
    def test_separate_raw_and_pe_normalization(self):
        xyz = torch.randn(40,3)*10-300
        raw = NodeDeformationMLP()
        raw.configure_position_normalization(xyz)
        u = (xyz-raw.position_center)/raw.position_scale
        ex = raw.encode(xyz,raw.position_frequencies)
        eu = raw.encode(u,raw.position_frequencies)
        for mode, expected in (("raw_only",torch.cat((u,ex[:,3:]),-1)),
                               ("pe_only",torch.cat((xyz,eu[:,3:]),-1))):
            model=NodeDeformationMLP(position_normalization=mode)
            model.load_state_dict(raw.state_dict())
            torch.testing.assert_close(model.position_features(xyz),expected,rtol=0,atol=0)
            features=torch.cat((expected,raw.encode(torch.full((40,1),.4),raw.time_frequencies)),-1)
            torch.testing.assert_close(torch.cat(model(xyz,.4),-1),raw.net(features),rtol=0,atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA rasterizer")
    def test_raw_only_renderer_roundtrip(self):
        self._check_renderer_mode("direct", "raw_only")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA rasterizer")
    def test_pe_only_renderer_roundtrip(self):
        self._check_renderer_mode("direct", "pe_only")

    def test_normalization_and_legacy_mlp(self):
        torch.manual_seed(17)
        xyz = torch.randn(100, 3) * 10 - 300
        raw = NodeDeformationMLP()
        legacy = {k: v for k, v in raw.state_dict().items()
                  if k not in ("position_center", "position_scale")}
        restored = NodeDeformationMLP()
        restored.load_state_dict(legacy, strict=True)
        for a, b in zip(raw(xyz, .5), restored(xyz, .5)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        normalized = NodeDeformationMLP(position_normalization="bbox")
        normalized.load_state_dict(raw.state_dict())
        normalized.configure_position_normalization(xyz)
        x = (xyz - normalized.position_center) / normalized.position_scale
        self.assertLessEqual(x.abs().max(), 1.000001)
        torch.testing.assert_close(normalized.position_features(xyz),
                                   raw.position_features(x), rtol=0, atol=0)
        for a, b in zip(normalized(xyz, .5), raw(x, .5)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in raw.parameters()), 43658)
        self.assertEqual(sum(p.numel() for p in normalized.parameters()), 43658)
        self.assertFalse(normalized.position_center.requires_grad)
        with self.assertRaises(RuntimeError):
            normalized.load_state_dict(legacy, strict=True)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA rasterizer")
    def test_normalized_renderer_roundtrip(self):
        self._check_renderer_mode("direct", "bbox")

    def test_fps_exact_and_unique(self):
        torch.manual_seed(3)
        xyz = torch.randn(53, 3)
        got = farthest_point_sampling(xyz, 17)
        selected = [0]
        for _ in range(16):
            d = torch.cdist(xyz, xyz[selected]).square().min(1).values
            d[selected] = -1
            selected.append(d.argmax().item())
        self.assertEqual(got.tolist(), selected)
        self.assertEqual(len(farthest_point_sampling(torch.zeros(20, 3), 20).unique()), 20)

    def test_knn_inverse_euclidean_and_fixed_buffers(self):
        torch.manual_seed(7)
        xyz, nodes = torch.randn(97, 3), torch.randn(12, 3)
        xyz[0] = nodes[0]
        idx, w = build_gaussian_node_knn(xyz.requires_grad_(), nodes)
        distances, expected = torch.linalg.vector_norm(xyz[:, None] - nodes, dim=-1).topk(4, largest=False)
        self.assertTrue(torch.equal(idx, expected))
        expected_w = 1 / (distances + 1e-8)
        torch.testing.assert_close(w, expected_w / expected_w.sum(-1, keepdim=True))
        torch.testing.assert_close(w.sum(-1), torch.ones(len(xyz)))
        self.assertFalse(w.requires_grad)
        graph = SparseDeformationGraph(nodes, idx, w)
        self.assertEqual(len(list(graph.parameters())), 0)
        xyz.detach().add_(10)
        self.assertTrue(torch.equal(idx, graph.gaussian_node_indices))

    def test_blending_gradient_and_constant_motion(self):
        xyz = torch.randn(100, 3)
        graph = SparseDeformationGraph.from_gaussians(xyz, 20)
        delta = [torch.ones(20, n, requires_grad=True) for n in (3, 4, 3)]
        result = graph.blend(*delta)
        for x in result:
            torch.testing.assert_close(x, torch.ones_like(x))
        sum(x.square().mean() for x in result).backward()
        for x in delta:
            self.assertGreater(x.grad.abs().sum().item(), 0)
        mlp = NodeDeformationMLP()
        motion = mlp(graph.node_xyz, torch.tensor(0.4))
        self.assertEqual([tuple(x.shape) for x in motion], [(20, 3), (20, 4), (20, 3)])
        self.assertLess(max(x.abs().max().item() for x in motion), 0.01)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA rasterizer")
    def test_renderer_dispatch_constraints_optimizer_and_roundtrip(self):
        self._check_renderer_mode("direct")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA rasterizer")
    def test_local_renderer_dispatch_constraints_optimizer_and_roundtrip(self):
        self._check_renderer_mode("local_frame")

    def _check_renderer_mode(self, position_mode, normalization="none"):
        torch.manual_seed(11)
        parser = argparse.ArgumentParser()
        hp, op, pp = ModelHiddenParams(parser), OptimizationParams(parser), PipelineParams(parser)
        args = parser.parse_args([])
        args.deformation_mode = "node"
        args.node_position_mode = position_mode
        args.node_position_normalization = normalization
        args.num_nodes = 32
        model = GaussianModel(3, hp.extract(args))
        n = 256
        model._xyz = nn.Parameter(torch.randn(n, 3, device="cuda") * 0.15 + torch.tensor([0, 0, 2.], device="cuda"))
        model._scaling = nn.Parameter(torch.full((n, 3), -3.5, device="cuda"))
        model._rotation = nn.Parameter(torch.tensor([1., 0, 0, 0], device="cuda").repeat(n, 1))
        model._opacity = nn.Parameter(torch.zeros(n, 1, device="cuda"))
        model.features = nn.Parameter(torch.rand(n, 3, device="cuda"))
        model._deformation_table = nn.Parameter(torch.randn(n, 16, device="cuda") * .01)
        model._deformation.cuda()
        model.spatial_lr_scale = 1.
        model.initialize_deformation_nodes()
        model.training_setup(op.extract(args))
        optimizer_ids = {id(p) for group in model.optimizer.param_groups for p in group['params']}
        self.assertTrue(all(id(p) in optimizer_ids for p in model.node_mlp.parameters()))
        self.assertIn(id(model._xyz), optimizer_ids)
        old = model._deformation.deformation_net
        hooks = []
        def forbidden(*_):
            raise AssertionError("Original geometry head was called in node mode")
        for head in (old.pos_deform, old.scales_deform, old.rotations_deform):
            hooks.append(head.register_forward_pre_hook(forbidden))
        counts = []
        hooks.append(model.node_mlp.register_forward_pre_hook(lambda _, inputs: counts.append(len(inputs[0]))))
        t = torch.full((n, 1), .4, device="cuda")
        geometry = model.deform_geometry_from_nodes(t)
        self.assertTrue(((geometry[1] - model._scaling).abs() <= .1 * model._scaling.abs() + 1e-7).all())
        self.assertTrue(((geometry[2] - model._rotation).abs() <= .1 * model._rotation.abs() + 1e-7).all())
        model.deformation_mode = "original"
        for hook in hooks[:3]:
            hook.remove()
        inputs = (model._xyz, model._scaling, model._rotation, model._opacity, model.features, t, model._deformation_table)
        original = model._deformation(*inputs)
        dispatched = model.deform(*inputs)
        for a, b in zip(original[:4] + original[4], dispatched[:4] + dispatched[4]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        model.deformation_mode = "node"
        for head in (old.pos_deform, old.scales_deform, old.rotations_deform):
            hooks.append(head.register_forward_pre_hook(forbidden))
        appearance = model.deform(*inputs)[4]
        for a, b in zip(original[4], appearance):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        from utils.graphics_utils import getProjectionMatrix
        view = SimpleNamespace(FoVx=1.0, FoVy=1.0, image_height=64, image_width=64,
            world_view_transform=torch.eye(4, device="cuda"),
            full_proj_transform=getProjectionMatrix(.01, 100., 1., 1.).T.cuda(),
            camera_center=torch.zeros(3, device="cuda"), time=.4)
        image = render(view, model, pp.extract(args), torch.zeros(3, device="cuda"))["render"]
        self.assertTrue(torch.isfinite(image).all())
        image.square().mean().backward()
        self.assertGreater(model.node_mlp.net[-1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.node_mlp.net[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model._xyz.grad.abs().sum().item(), 0)
        self.assertGreater(old.grid.grids[0][0].grad.abs().sum().item(), 0)
        self.assertTrue(all(count == 32 for count in counts))
        # Same MLP output gives identical scale/Gaussian-rotation in both modes.
        args_mode = model.args.node_position_mode
        model.args.node_position_mode = "direct"
        direct_geometry = model.deform_geometry_from_nodes(t)
        model.args.node_position_mode = "local_frame"
        calls_before = len(counts)
        local_geometry = model.deform_geometry_from_nodes(t)
        self.assertEqual(len(counts) - calls_before, 1)
        for direct, local in zip(direct_geometry[1:], local_geometry[1:]):
            torch.testing.assert_close(direct, local, rtol=0, atol=0)
        model.args.node_position_mode = args_mode
        if position_mode == "local_frame":
            model.optimizer.zero_grad(set_to_none=True)
            model.args.no_dr = True
            model.args.no_ds = True
            # Rotation-output gradient can now reach RGB ONLY through position.
            position_only_image = render(view, model, pp.extract(args), torch.zeros(3, device="cuda"))["render"]
            position_only_image.square().mean().backward()
            self.assertGreater(model.node_mlp.net[-1].weight.grad[4:7].abs().sum().item(), 0)
            model.args.no_dr = False
            model.args.no_ds = False
        model.optimizer.step()
        with tempfile.TemporaryDirectory() as path:
            model.save_ply(str(Path(path) / "point_cloud.ply"))
            model.save_deformation(path)
            args.deformation_mode = "original"  # render defaults must auto-detect saved nodes
            args.node_position_mode = "direct"
            restored = GaussianModel(3, hp.extract(args))
            restored.load_ply(str(Path(path) / "point_cloud.ply"))
            restored.load_model(path)
            restored.spatial_lr_scale = 1.
            self.assertEqual(restored.deformation_mode, "node")
            self.assertEqual(restored.args.node_position_mode, position_mode)
            self.assertEqual(restored.args.node_position_normalization, normalization)
            a = render(view, model, pp.extract(args), torch.zeros(3, device="cuda"))["render"]
            b = render(view, restored, pp.extract(args), torch.zeros(3, device="cuda"))["render"]
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(model.deformation_graph.gaussian_node_weights,
                                       restored.deformation_graph.gaussian_node_weights, rtol=0, atol=0)
        for hook in hooks:
            hook.remove()

    def test_local_identity_equals_direct_and_pivot_rotation(self):
        torch.manual_seed(5)
        xyz = torch.randn(50, 3, dtype=torch.double)
        graph = SparseDeformationGraph.from_gaussians(xyz.float(), 8).double()
        delta = torch.randn(8, 3, dtype=torch.double)
        zeros = torch.zeros(8, 4, dtype=torch.double)
        direct = xyz + graph.blend(delta, zeros, delta)[0]
        local = graph.deform_positions_local(xyz, delta, zeros)
        # Cached float32 weights have ~1e-7 normalization rounding.
        torch.testing.assert_close(local, direct, rtol=1e-6, atol=1e-6)
        c = torch.tensor([[2., 3., 4.]], dtype=torch.double)
        pivot_graph = SparseDeformationGraph(c, torch.zeros(1, 1, dtype=torch.long), torch.ones(1, 1, dtype=torch.double))
        xyz = c + torch.tensor([[1., 0., 0.]], dtype=torch.double)
        root_half = 0.5 ** 0.5
        rotation = torch.tensor([[root_half - 1, 0., 0., root_half]], dtype=torch.double)
        translation = torch.tensor([[.1, .2, .3]], dtype=torch.double)
        rotated = pivot_graph.deform_positions_local(xyz, translation, rotation)
        torch.testing.assert_close(rotated, c + torch.tensor([[0., 1., 0.]], dtype=torch.double) + translation)
        # All nodes realizing the same rigid transform must blend to that transform.
        q = rotation.expand(8, -1)
        matrix = node_rotation_matrices(q)[0]
        shift = translation[0]
        node_dt = graph.node_xyz @ matrix.T + shift - graph.node_xyz
        xyz = torch.randn(50, 3, dtype=torch.double)
        local = graph.deform_positions_local(xyz, node_dt, q)
        torch.testing.assert_close(local, xyz @ matrix.T + shift, rtol=1e-6, atol=1e-6)

    def test_local_position_gradcheck(self):
        torch.manual_seed(9)
        graph = SparseDeformationGraph(torch.randn(4, 3, dtype=torch.double),
            torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]]),
            torch.tensor([[.1, .2, .3, .4], [.4, .3, .2, .1]], dtype=torch.double))
        xyz = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
        translation = (torch.randn(4, 3, dtype=torch.double) * .1).requires_grad_()
        rotation = (torch.randn(4, 4, dtype=torch.double) * .1).requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(graph.deform_positions_local, (xyz, translation, rotation)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
