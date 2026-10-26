"""Residual graph FiLM with fixed K4 appearance and detached motion cues."""
import math
import torch
from torch import nn
from utils.r7_core import R7Core
from scene.deformation import NodeDeformationMLP


class R7V2(R7Core):
    def __init__(self, cfg, num_nodes):
        super().__init__(cfg, num_nodes)
        self.register_buffer('gaussian_time_frequencies', 2.0 ** torch.arange(6))
        with torch.random.fork_rng(devices=[]):
            self.micro_head = nn.Sequential(nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 6))
            gen = torch.Generator().manual_seed(int(cfg.appearance_seed))
            nn.init.kaiming_uniform_(self.micro_head[0].weight, a=math.sqrt(5), generator=gen)
            nn.init.uniform_(self.micro_head[0].bias, -1/math.sqrt(32), 1/math.sqrt(32), generator=gen)
            nn.init.zeros_(self.micro_head[-1].weight)
            nn.init.zeros_(self.micro_head[-1].bias)

    @torch.no_grad()
    def bind_nodes(self, appearance_ids, geometry_ids, appearance_xyz, geometry_xyz, motion_scale):
        n = len(geometry_ids)
        if appearance_ids.shape != (2000,) or n not in (2000, 2400):
            raise ValueError('Expected appearance2000, geometry2000 or2400')
        ids, order = geometry_ids.sort()
        loc = torch.searchsorted(ids, appearance_ids)
        if (loc >= n).any() or not torch.equal(ids[loc], appearance_ids):
            raise ValueError('Appearance source IDs absent from geometry')
        mapping = order[loc]
        if len(ids.unique()) != n or len(appearance_ids.unique()) != 2000:
            raise ValueError('Duplicate source IDs')
        if not torch.equal(appearance_xyz, geometry_xyz[mapping]):
            raise ValueError('Source IDs disagree with canonical centers')
        self.appearance_source_ids.copy_(appearance_ids)
        self.geometry_source_ids = geometry_ids.detach().clone()
        self.appearance_to_geometry.copy_(mapping)
        self.motion_scale.copy_(motion_scale.detach().clamp_min(1e-8))

    def _load_from_state_dict(self, state, prefix, *args, **kwargs):
        # Mapping size is checkpoint data, not a guessed runtime node order.
        key = prefix + 'geometry_source_ids'
        if key in state:
            n = state[key].numel()
            if n not in (2000, 2400):
                raise ValueError('Invalid saved geometry mapping size')
            self.geometry_source_ids = torch.empty_like(state[key], device=self.motion_scale.device)
        super()._load_from_state_dict(state, prefix, *args, **kwargs)

    def components(self, times, embedding, indices, weights, motion, gaussian_displacement):
        if indices.shape != weights.shape or indices.shape != (len(embedding), 4):
            raise ValueError('Original fixed appearance K4 support required')
        if (self.appearance_to_geometry < 0).any():
            raise ValueError('Unbound source-ID correspondence')
        t = torch.as_tensor(times, device=embedding.device, dtype=embedding.dtype).reshape(-1, 1)[:1]
        translation, rotation = motion[:2]
        mapping = self.appearance_to_geometry
        node_input = torch.cat((self.node_appearance,
            translation.detach()[mapping] / self.motion_scale, rotation.detach()[mapping],
            NodeDeformationMLP.encode(t.expand(2000, 1), self.time_frequencies)), -1)
        node = self.node_mlp(node_input)
        gain = .35 * node[:, :3].tanh()
        context = self.node_appearance + .50 * node[:, 3:].tanh()
        low = (gain[indices] * weights[..., None]).sum(1)
        h = (context[indices] * weights[..., None]).sum(1)
        z = embedding * (1 + h)
        x = torch.cat((z, NodeDeformationMLP.encode(t.expand(len(z), 1), self.gaussian_time_frequencies),
                       gaussian_displacement.detach()), -1)
        raw = self.micro_head(x)
        return low, .25 * raw[:, :3].tanh(), .15 * raw[:, 3:].tanh()

    def forward(self, points, times, colors, embedding, node_indices=None, node_weights=None,
                motion=None, gaussian_displacement=None):
        low, high, beta = self.components(times, embedding, node_indices, node_weights,
                                          motion, gaussian_displacement)
        gain = low + high
        return colors * (1 + gain) + beta, gain
