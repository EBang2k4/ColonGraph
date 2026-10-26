"""Fixed-node appearance variants. No geometry topology or loss changes.

R7 conditions on G3-corrected translation / fixed postcoarse bbox half-extent,
and the native scalar-first four-component node rotation residual (not an
absolute orientation). Both are detached; appearance uses its original K4.
"""
import math
import torch
from torch import nn
from scene.deformation import NodeConditionedResidualAppearance, NodeDeformationMLP


class R6Additive(NodeConditionedResidualAppearance):
    def __init__(self, cfg, num_nodes):
        super().__init__(cfg, num_nodes)
        self.additive_cap = float(getattr(cfg, "r6_additive_cap", .15))
        if not math.isfinite(self.additive_cap) or self.additive_cap <= 0:
            raise ValueError("R6 additive cap must be finite and positive")
        with torch.random.fork_rng(devices=[]):
            self.additive_head = nn.Linear(64, 3)
            nn.init.zeros_(self.additive_head.weight)
            nn.init.zeros_(self.additive_head.bias)

    def components(self, times, colors, embedding, indices, weights):
        t = torch.as_tensor(times, device=embedding.device, dtype=embedding.dtype).reshape(-1, 1)
        t = t.expand(len(embedding), 1)
        x = torch.cat((embedding, self.node_context(indices, weights),
                       NodeDeformationMLP.encode(t, self.time_frequencies)), -1)
        hidden = self.residual_mlp[:4](x)
        raw = self.residual_mlp[4](hidden)
        gain = self.eta * (raw.tanh() if self.use_tanh else raw)
        beta = self.additive_cap * self.additive_head(hidden).tanh()
        return gain, beta

    def forward(self, points, times, colors, embedding, node_indices=None, node_weights=None):
        gain, beta = self.components(times, colors, embedding, node_indices, node_weights)
        return colors * (1 + gain) + beta, gain


class R7Core(nn.Module):
    def __init__(self, cfg, num_nodes):
        super().__init__()
        if num_nodes != 2000 or getattr(cfg, "color_emb_dim", 16) != 16:
            raise ValueError("R7-Core requires exactly 2000 appearance nodes and 16D embeddings")
        self.register_buffer("time_frequencies", 2.0 ** torch.arange(4))
        # Persist explicit appearance-node -> geometry-node mapping and source IDs.
        self.register_buffer("appearance_source_ids", torch.full((num_nodes,), -1, dtype=torch.long))
        self.register_buffer("geometry_source_ids", torch.full((num_nodes,), -1, dtype=torch.long))
        self.register_buffer("appearance_to_geometry", torch.full((num_nodes,), -1, dtype=torch.long))
        self.register_buffer("motion_scale", torch.ones(()))
        with torch.random.fork_rng(devices=[]):
            self.node_appearance = nn.Parameter(torch.empty(num_nodes, 16))
            self.node_mlp = nn.Sequential(nn.Linear(32, 64), nn.ReLU(),
                                          nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 19))
            self.micro_head = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 6))
            gen = torch.Generator(device="cpu").manual_seed(int(cfg.appearance_seed))
            nn.init.normal_(self.node_appearance, std=.01, generator=gen)
            for layer in [self.node_mlp[0], self.node_mlp[2], self.micro_head[0]]:
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5), generator=gen)
                bound = 1 / math.sqrt(layer.weight.shape[1])
                nn.init.uniform_(layer.bias, -bound, bound, generator=gen)
            # All 19 node outputs and all six Gaussian outputs start at zero.
            for layer in [self.node_mlp[-1], self.micro_head[-1]]:
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

    @torch.no_grad()
    def bind_nodes(self, appearance_ids, geometry_ids, appearance_xyz, geometry_xyz, motion_scale):
        if appearance_ids.shape != (2000,) or geometry_ids.shape != (2000,):
            raise ValueError("Fixed2000 node IDs required")
        ids, order = geometry_ids.sort()
        loc = torch.searchsorted(ids, appearance_ids)
        if (loc >= 2000).any() or not torch.equal(ids[loc], appearance_ids):
            raise ValueError("Appearance nodes lack exact geometry-node source-ID correspondence")
        mapping = order[loc]
        if not torch.equal(appearance_xyz, geometry_xyz[mapping]):
            raise ValueError("Node source IDs and canonical centers disagree")
        if len(geometry_ids.unique()) != 2000 or len(appearance_ids.unique()) != 2000:
            raise ValueError("Duplicate node source IDs")
        self.appearance_source_ids.copy_(appearance_ids)
        self.geometry_source_ids.copy_(geometry_ids)
        self.appearance_to_geometry.copy_(mapping)
        self.motion_scale.copy_(motion_scale.detach().clamp_min(1e-8))

    def components(self, times, embedding, indices, weights, motion):
        if indices.shape != weights.shape or indices.shape != (len(embedding), 4):
            raise ValueError("Original fixed appearance K4 support required")
        if (self.appearance_to_geometry < 0).any():
            raise ValueError("R7 node correspondence has not been bound")
        translation, rotation = motion[:2]
        mapping = self.appearance_to_geometry
        t = torch.as_tensor(times, device=embedding.device, dtype=embedding.dtype).reshape(-1, 1)[:1].expand(2000, 1)
        x = torch.cat((self.node_appearance,
                       translation.detach()[mapping] / self.motion_scale,
                       rotation.detach()[mapping],
                       NodeDeformationMLP.encode(t, self.time_frequencies)), -1)
        node = self.node_mlp(x)
        interpolated = (node[indices] * weights[..., None]).sum(1)
        low, h = interpolated[:, :3], interpolated[:, 3:]
        raw = self.micro_head(embedding * h)
        high = .25 * raw[:, :3].tanh()
        beta = .35 * raw[:, 3:].tanh()
        return low, high, beta

    def forward(self, points, times, colors, embedding, node_indices=None, node_weights=None, motion=None):
        low, high, beta = self.components(times, embedding, node_indices, node_weights, motion)
        gain = low + high
        return colors * (1 + gain) + beta, gain


def create(cfg, count):
    mode = getattr(cfg, "appearance_mode", "r6")
    if mode == "r6_additive_micro":
        from utils.stage7 import R6AdditiveMicro
        return R6AdditiveMicro(cfg, count)
    if mode == "r6_additive_wide96":
        from utils.stage6 import R6AdditiveWide96
        return R6AdditiveWide96(cfg, count)
    if mode == "r6_speccam":
        from utils.stage3 import R6SpecCam
        return R6SpecCam(cfg, count)
    if mode == "r7_v2":
        from utils.r7_v2 import R7V2
        return R7V2(cfg, count)
    cls = {"r6": NodeConditionedResidualAppearance, "r6_hf": NodeConditionedResidualAppearance,
           "r6_additive": R6Additive, "r7_core": R7Core}.get(mode)
    if cls is None:
        raise ValueError("Unknown appearance_mode: " + mode)
    return cls(cfg, count)
