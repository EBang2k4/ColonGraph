"""Appearance adapters for frozen-geometry refits only (not full training).

Legacy checkpoints have no appearance_config and are always HexPlane. Keep its
module names, initialization order and forward arithmetic unchanged.
"""
import copy
import math
import random

import numpy as np
import torch
from torch import nn

from scene.deformation import (NodeDeformationMLP, NodeConditionedResidualAppearance,
                               deform_network)


DEFAULTS = dict(color_model='hexplane', color_residual_scale=.1,
                color_residual_tanh=True, lambda_res=.005, appearance_seed=0,
                zero_hexplane_feature=False, node_appearance_dim=8)


def appearance_config(cfg):
    config = {key: getattr(cfg, key, value) for key, value in DEFAULTS.items()}
    if config['color_model'] not in ('hexplane', 'residual_mlp', 'node_residual_mlp'):
        raise ValueError('color_model must be hexplane, residual_mlp or node_residual_mlp')
    for key in ('color_residual_scale', 'lambda_res'):
        if not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    if not isinstance(config['node_appearance_dim'], int) or config['node_appearance_dim'] < 1:
        raise ValueError('node_appearance_dim must be a positive integer')
    return config


def checkpoint_config(cfg, state):
    # Metadata in the checkpoint is authoritative, never a caller's default.
    saved = state.get('appearance_config', {})
    for key, value in DEFAULTS.items():
        setattr(cfg, key, saved.get(key, value))
    return appearance_config(cfg)


class Appearance(nn.Module):
    """Unmodified legacy adapter, also the public compatibility default."""
    def __init__(self, cfg):
        super().__init__()
        original = deform_network(cfg).deformation_net
        self.grid = original.grid
        self.feature_out = original.feature_out
        self.color_deform = original.color_deform
        self.zero_hexplane_feature = appearance_config(cfg)['zero_hexplane_feature']
        self.grid.aabb.requires_grad_(False)

    def forward(self, points, times, colors, embedding, node_indices=None, node_weights=None):
        grid_feature = self.grid(points, times)
        if self.zero_hexplane_feature:
            grid_feature = torch.zeros_like(grid_feature)
        hidden = self.feature_out(grid_feature).float()
        delta = self.color_deform(torch.cat((hidden, embedding), -1))
        return colors * (1 + delta), delta


class ResidualAppearance(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        config = appearance_config(cfg)
        if cfg.color_emb_dim != 16:
            raise ValueError('Residual ablation requires the existing 16-D embedding')
        self.eta = config['color_residual_scale']
        self.use_tanh = config['color_residual_tanh']
        self.register_buffer('time_frequencies', 2.0 ** torch.arange(6))
        # Linear constructors consume RNG too: isolate construction, then use a
        # private generator for every meaningful hidden-layer initialization.
        with torch.random.fork_rng(devices=[]):
            self.residual_mlp = nn.Sequential(
                nn.Linear(29, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, 3))
            generator = torch.Generator(device='cpu').manual_seed(config['appearance_seed'])
            for layer in (self.residual_mlp[0], self.residual_mlp[2]):
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5), generator=generator)
                bound = 1 / math.sqrt(layer.weight.shape[1])
                nn.init.uniform_(layer.bias, -bound, bound, generator=generator)
            nn.init.zeros_(self.residual_mlp[-1].weight)
            nn.init.zeros_(self.residual_mlp[-1].bias)

    def forward(self, points, times, colors, embedding, node_indices=None, node_weights=None):
        times = torch.as_tensor(times, device=embedding.device, dtype=embedding.dtype).reshape(-1, 1)
        if len(times) == 1:
            times = times.expand(len(embedding), 1)
        if len(times) != len(embedding):
            raise ValueError('Expected one time or one time per Gaussian')
        gamma = NodeDeformationMLP.encode(times, self.time_frequencies)
        raw = self.residual_mlp(torch.cat((embedding, gamma), -1))
        delta = self.eta * (raw.tanh() if self.use_tanh else raw)
        return colors * (1 + delta), delta


def is_residual_model(cfg):
    return appearance_config(cfg)['color_model'] in ('residual_mlp', 'node_residual_mlp')


def build_appearance(cfg, num_nodes=None):
    model = appearance_config(cfg)['color_model']
    if model == 'residual_mlp':
        return ResidualAppearance(cfg)
    if model == 'node_residual_mlp':
        return NodeConditionedResidualAppearance(cfg, num_nodes)
    return Appearance(cfg)


def optimizer_groups(model, app, lrs):
    if isinstance(app, (ResidualAppearance, NodeConditionedResidualAppearance)):
        residual_params = list(app.residual_mlp.parameters())
        if isinstance(app, NodeConditionedResidualAppearance):
            residual_params.append(app.node_appearance)
        groups = [dict(name='deformation', params=residual_params, lr=lrs['deformation'])]
    else:
        groups = [dict(name='deformation', params=list(app.feature_out.parameters()) + list(app.color_deform.parameters()), lr=lrs['deformation']),
                  dict(name='grid', params=[p for p in app.grid.parameters() if p.requires_grad], lr=lrs['grid'])]
    return groups + [dict(name='f_dc', params=[model.features], lr=lrs['f_dc']),
                     dict(name='deformation_table', params=[model._deformation_table], lr=lrs['deformation_table'])]


def make_shared_initialization(model, cfg, box):
    """Match legacy seed0 exactly; base/embedding are identical across branches.

    Draw the embedding after the SAME legacy initialization for both choices,
    so changing model architecture cannot silently change embedding RNG draws.
    """
    config = appearance_config(cfg)
    with torch.random.fork_rng():
        torch.manual_seed(config['appearance_seed'])
        legacy_cfg = copy.copy(cfg)
        legacy_cfg.color_model = 'hexplane'
        app = Appearance(legacy_cfg)
        with torch.no_grad():
            app.grid.aabb.copy_(box)
        base = torch.full(tuple(model.features.shape), .5)
        embedding = .01 * torch.randn(tuple(model._deformation_table.shape))
        if config['color_model'] == 'residual_mlp':
            app = ResidualAppearance(cfg)
        elif config['color_model'] == 'node_residual_mlp':
            graph = getattr(model, 'deformation_graph', None)
            app = NodeConditionedResidualAppearance(cfg, len(graph.node_xyz) if graph is not None else None)
    return dict(appearance=app.state_dict(), base_color=base,
                color_embedding=embedding, appearance_config=config,
                appearance_format_version=1)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sampling_sequence(ids, steps, seed):
    rng = random.Random(seed)
    return [ids[rng.randint(0, len(ids) - 1)] for _ in range(steps)]


def residual_penalty_weight(cfg):
    return getattr(cfg, 'lambda_res', .005) if is_residual_model(cfg) else cfg.dcol_weight
