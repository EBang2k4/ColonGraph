"""Phase-1 tests only: no dataset training or experiment sweep."""
import argparse
import copy
import io
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arguments import ModelHiddenParams
from scene.deformation import NodeDeformationMLP, deform_network
from refit_appearance_model import (Appearance, ResidualAppearance, NodeConditionedResidualAppearance, build_appearance,
    appearance_config, checkpoint_config, make_shared_initialization,
    optimizer_groups, seed_everything, sampling_sequence)

LRS = dict(deformation=.0001, grid=.0001, f_dc=.0025, deformation_table=.0025)


def config(**kwargs):
    parser = argparse.ArgumentParser()
    ModelHiddenParams(parser)
    cfg = parser.parse_args([])
    cfg.color_emb_dim = 16
    cfg.kplanes_config = dict(grid_dimensions=2, input_coordinate_dim=4,
                             output_coordinate_dim=4, resolution=[4, 4, 4, 4])
    cfg.multires = [1]
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    return cfg


class LegacyAppearance(nn.Module):
    """Verbatim pre-change constructor/forward, independent regression oracle."""
    def __init__(self, cfg):
        super().__init__()
        original = deform_network(cfg).deformation_net
        self.grid = original.grid
        self.feature_out = original.feature_out
        self.color_deform = original.color_deform
        self.grid.aabb.requires_grad_(False)

    def forward(self, points, times, colors, embedding):
        hidden = self.feature_out(self.grid(points, times)).float()
        delta = self.color_deform(torch.cat((hidden, embedding), -1))
        return colors * (1 + delta), delta


def same_tree(test, a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        test.assertEqual(a.keys(), b.keys())
        for key in a:
            same_tree(test, a[key], b[key])
    elif isinstance(a, (list, tuple)):
        test.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            same_tree(test, x, y)
    else:
        test.assertEqual(a, b)


class ResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def inputs(self, n=40):
        generator = torch.Generator().manual_seed(314)
        return (torch.randn(n, 3, generator=generator),
                torch.linspace(0, 1, n)[:, None],
                torch.rand(n, 3, generator=generator),
                torch.randn(n, 16, generator=generator))

    def test_zero_init_composition_and_time_encoding(self):
        app = build_appearance(config(color_model='residual_mlp'))
        inputs = self.inputs()
        color, delta = app(*inputs)
        self.assertEqual(torch.count_nonzero(delta).item(), 0)
        torch.testing.assert_close(color, inputs[2], rtol=0, atol=0)
        for layer in (app.residual_mlp[0], app.residual_mlp[2]):
            self.assertGreater(layer.weight.abs().sum().item(), 0)
            self.assertGreater(layer.bias.abs().sum().item(), 0)
        captured = []
        hook = app.residual_mlp[0].register_forward_pre_hook(lambda _, args: captured.append(args[0]))
        app(*inputs)
        hook.remove()
        gamma = NodeDeformationMLP.encode(inputs[1], 2. ** torch.arange(6))
        self.assertEqual(gamma.shape[1], 13)
        torch.testing.assert_close(captured[0], torch.cat((inputs[3], gamma), -1), rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in app.parameters()), 6275)

    def test_bound_and_unbounded_flag_and_no_xyz_dependence(self):
        for eta in (0., .05, .1, .3):
            cfg = config(color_model='residual_mlp', color_residual_scale=eta)
            app = build_appearance(cfg)
            with torch.no_grad():
                app.residual_mlp[-1].weight.fill_(1000.)
                app.residual_mlp[-1].bias.copy_(torch.tensor([1e6, -1e6, 1e6]))
            p, t, base, emb = self.inputs()
            color, delta = app(p, t, base, emb)
            self.assertTrue((delta.abs() <= eta).all())
            torch.testing.assert_close(color, base * (1 + delta), rtol=0, atol=0)
            torch.testing.assert_close(app(p * 100, t, base, emb)[0], color, rtol=0, atol=0)
            app.use_tanh = False
            if eta:
                self.assertGreater(app(p, t, base, emb)[1].abs().max().item(), eta)

    def test_hexplane_zero_control_is_before_feature_out(self):
        cfg = config(zero_hexplane_feature=True)
        app = build_appearance(cfg)
        p, t, base, emb = self.inputs()
        captured = []
        hook = app.feature_out.register_forward_pre_hook(lambda _, args: captured.append(args[0]))
        app(p, t, base, emb)
        hook.remove()
        self.assertEqual(torch.count_nonzero(captured[0]).item(), 0)
        self.assertEqual(tuple(captured[0].shape), (len(p), app.grid.feat_dim))

    def test_node_conditioning_blend_bound_gradients_and_roundtrip(self):
        cfg = config(color_model='node_residual_mlp', color_residual_scale=.3,
                     node_appearance_dim=8, appearance_seed=19)
        app = build_appearance(cfg, num_nodes=7)
        self.assertIsInstance(app, NodeConditionedResidualAppearance)
        p, t, base, emb = self.inputs()
        generator = torch.Generator().manual_seed(20)
        indices = torch.randint(0, 7, (len(p), 4), generator=generator)
        weights = torch.rand(len(p), 4, generator=generator)
        weights /= weights.sum(1, keepdim=True)
        expected_context = (app.node_appearance[indices] * weights[..., None]).sum(1)
        torch.testing.assert_close(app.node_context(indices, weights), expected_context, rtol=0, atol=0)
        captured = []
        hook = app.residual_mlp[0].register_forward_pre_hook(lambda _, args: captured.append(args[0]))
        color, delta = app(p, t, base, emb, indices, weights)
        hook.remove()
        self.assertEqual(tuple(captured[0].shape), (len(p), 37))
        self.assertEqual(torch.count_nonzero(delta).item(), 0)
        torch.testing.assert_close(color, base, rtol=0, atol=0)
        opt = torch.optim.Adam(app.parameters(), lr=1e-3)
        for step in range(3):
            opt.zero_grad(set_to_none=True)
            color, delta = app(p, t, base, emb, indices, weights)
            (color - .8).square().mean().backward()
            if step:
                self.assertIsNotNone(app.node_appearance.grad)
                self.assertGreater(app.node_appearance.grad.abs().sum().item(), 0)
            opt.step()
        self.assertLessEqual(delta.abs().max().item(), .3)
        state = app.state_dict()
        restored = build_appearance(cfg, num_nodes=7)
        restored.load_state_dict(state, strict=True)
        same_tree(self, app(p,t,base,emb,indices,weights), restored(p,t,base,emb,indices,weights))
        self.assertEqual(app.node_appearance.numel(), 56)
        self.assertEqual(sum(p.numel() for p in app.parameters()), 6843)

    def test_rng_isolation_seed_and_sampling(self):
        seed_everything(918)
        cpu = torch.get_rng_state().clone()
        cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        py, np_state = random.getstate(), np.random.get_state()
        a = build_appearance(config(color_model='residual_mlp', appearance_seed=5))
        torch.testing.assert_close(torch.get_rng_state(), cpu, rtol=0, atol=0)
        for old, new in zip(cuda, torch.cuda.get_rng_state_all() if cuda else []):
            torch.testing.assert_close(old, new, rtol=0, atol=0)
        self.assertEqual(random.getstate(), py)
        np.testing.assert_equal(np.random.get_state(), np_state)
        b = build_appearance(config(color_model='residual_mlp', appearance_seed=5))
        c = build_appearance(config(color_model='residual_mlp', appearance_seed=6))
        same_tree(self, a.state_dict(), b.state_dict())
        self.assertFalse(torch.equal(a.residual_mlp[0].weight, c.residual_mlp[0].weight))
        self.assertEqual(sampling_sequence(list(range(252)), 100, 5), sampling_sequence(list(range(252)), 100, 5))
        self.assertNotEqual(sampling_sequence(list(range(252)), 100, 5), sampling_sequence(list(range(252)), 100, 6))
        draws = []
        for _ in range(2):
            seed_everything(5)
            draws.append((random.random(), float(np.random.rand()), torch.rand(3)))
        same_tree(self, *draws)

    def test_shared_initialization_matches_legacy_and_cross_branch(self):
        model = SimpleNamespace(features=torch.empty(40, 3), _deformation_table=torch.empty(40, 16))
        box = torch.tensor([[2., 2., 2.], [-2., -2., -2.]])
        cfg = config()
        with torch.random.fork_rng():
            torch.manual_seed(0)
            legacy = LegacyAppearance(cfg)
            legacy.grid.aabb.copy_(box)
            base = torch.full((40, 3), .5)
            embedding = .01 * torch.randn(40, 16)
        before = torch.get_rng_state().clone()
        h = make_shared_initialization(model, cfg, box)
        torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
        same_tree(self, legacy.state_dict(), h['appearance'])
        same_tree(self, base, h['base_color'])
        same_tree(self, embedding, h['color_embedding'])
        cfg.color_model = 'residual_mlp'
        r = make_shared_initialization(model, cfg, box)
        same_tree(self, h['base_color'], r['base_color'])
        same_tree(self, h['color_embedding'], r['color_embedding'])
        cfg.appearance_seed = 1
        r1 = make_shared_initialization(model, cfg, box)
        self.assertFalse(torch.equal(r['color_embedding'], r1['color_embedding']))

    def test_hexplane_100_step_legacy_regression(self):
        cfg = config()  # No new flags: legacy default must apply.
        checkpoints = []
        curves = []
        for factory in (LegacyAppearance, build_appearance):
            seed_everything(44)
            app = factory(cfg)
            points, times, base, embedding = self.inputs()
            model = SimpleNamespace(features=nn.Parameter(base), _deformation_table=nn.Parameter(embedding))
            if factory is LegacyAppearance:
                groups = [dict(name='deformation', params=list(app.feature_out.parameters()) + list(app.color_deform.parameters()), lr=.0001),
                          dict(name='grid', params=[p for p in app.grid.parameters() if p.requires_grad], lr=.0001),
                          dict(name='f_dc', params=[model.features], lr=.0025),
                          dict(name='deformation_table', params=[model._deformation_table], lr=.0025)]
            else:
                groups = optimizer_groups(model, app, LRS)
            opt = torch.optim.Adam(groups, lr=0., eps=1e-15)
            curve = []
            for step in range(100):
                opt.zero_grad(set_to_none=True)
                color, delta = app(points, times, model.features, model._deformation_table)
                loss = (color - .7).abs().mean() + .005 * delta.square().mean() + .0001 * (color - color.mean(0)).square().mean()
                loss.backward()
                for group in groups:
                    torch.nn.utils.clip_grad_value_(group['params'], .5)
                opt.step()
                curve.append(loss.detach().clone())
            checkpoints.append(dict(appearance=app.state_dict(), base_color=model.features,
                                    color_embedding=model._deformation_table, optimizer=opt.state_dict()))
            curves.append(curve)
        same_tree(self, *curves)
        same_tree(self, *checkpoints)

    def test_roundtrip_config_and_gradient_after_zero_init(self):
        cfg = config(color_model='residual_mlp', color_residual_scale=.17, lambda_res=.013, appearance_seed=7)
        app = build_appearance(cfg)
        p, t, base, embedding = self.inputs()
        model = SimpleNamespace(features=nn.Parameter(base), _deformation_table=nn.Parameter(embedding))
        groups = optimizer_groups(model, app, LRS)
        self.assertEqual([(g['name'], g['lr']) for g in groups], [('deformation', .0001), ('f_dc', .0025), ('deformation_table', .0025)])
        opt = torch.optim.Adam(groups, lr=0., eps=1e-15)
        for step in range(3):
            opt.zero_grad(set_to_none=True)
            color, delta = app(p, t, model.features, model._deformation_table)
            ((color - .9).square().mean() + cfg.lambda_res * delta.square().mean()).backward()
            if step:
                self.assertGreater(model._deformation_table.grad.abs().sum().item(), 0)
                self.assertGreater(app.residual_mlp[0].weight.grad.abs().sum().item(), 0)
            opt.step()
        expected = app(p, t, model.features, model._deformation_table)
        stream = io.BytesIO()
        torch.save(dict(appearance=app.state_dict(), appearance_config=appearance_config(cfg),
                        base_color=model.features, color_embedding=model._deformation_table,
                        optimizer=opt.state_dict()), stream)
        stream.seek(0)
        saved = torch.load(stream, weights_only=False)
        new_cfg = config()
        checkpoint_config(new_cfg, saved)
        restored = build_appearance(new_cfg)
        restored.load_state_dict(saved['appearance'], strict=True)
        same_tree(self, appearance_config(cfg), appearance_config(new_cfg))
        same_tree(self, expected, restored(p, t, saved['base_color'], saved['color_embedding']))
        old_cfg = config(color_model='residual_mlp')
        checkpoint_config(old_cfg, {})
        self.assertEqual(old_cfg.color_model, 'hexplane')
        self.assertIsInstance(build_appearance(old_cfg), Appearance)

    @unittest.skipUnless(torch.cuda.is_available(), 'Training camera loss uses CUDA')
    def test_regularization_reduction_other_losses_unchanged(self):
        from refine_appearance import losses
        cfg = config(depth_weight=.25, dcol_weight=.005, col_smooth_weight=.0001, tv_weight=.001,
                     appearance_dssim_weight=.2, appearance_lpips_weight=0.)
        im = torch.rand(3, 32, 32, device='cuda', requires_grad=True)
        delta = torch.rand(40, 3, device='cuda', requires_grad=True)
        view = SimpleNamespace(original_image=im.detach() * .9,
                               original_depth=torch.rand(32, 32, device='cuda'), mask=None)
        pkg = dict(render=im, depth=torch.rand(1, 32, 32, device='cuda'),
                   dcol=delta, transformed_color=torch.rand(40, 3, device='cuda'))
        old, original = losses(pkg, view, cfg, im.new_tensor(.001))
        cfg.color_model = 'residual_mlp'
        cfg.lambda_res = .005
        new, residual = losses(pkg, view, cfg, im.new_tensor(.001))
        same_tree(self, original, residual)
        same_tree(self, old, new)
        torch.testing.assert_close(residual['dcolor'], .005 * delta.square().mean(), rtol=0, atol=0)
        cfg.lambda_res = 0.
        _, zero = losses(pkg, view, cfg, im.new_tensor(.001))
        self.assertEqual(zero['dcolor'].item(), 0.)
        for key in original:
            if key != 'dcolor':
                same_tree(self, original[key], zero[key])


if __name__ == '__main__':
    unittest.main(verbosity=2)
