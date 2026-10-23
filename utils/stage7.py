"""Run O: shared R6+Additive trunk with a bounded world-space residual.

The scalar a is the saved NodeMLP bbox half-extent, not cameras_extent.
The tanh bound is per coordinate; the L2 norm is at most sqrt(3)*.005*a.
"""
import torch
from torch import nn
from scene.deformation import NodeDeformationMLP
from utils.r7_core import R6Additive


def micro_alpha(iteration):
    return min(1., max(0., (iteration - 6000) / 4000.))


def micro_penalty(raw):
    # raw already equals tanh(F_x); deliberately not multiplied by the ramp.
    return .001 * raw.square().sum(-1).mean()


class R6AdditiveMicro(R6Additive):
    def __init__(self, cfg, num_nodes):
        super().__init__(cfg, num_nodes)
        if num_nodes != 2000 or self.node_appearance_dim != 8 or self.additive_cap != .25:
            raise ValueError('Run O requires original2000/8D appearance and cap.25')
        with torch.random.fork_rng(devices=[]):
            self.micro_head = nn.Linear(64, 3)
            nn.init.zeros_(self.micro_head.weight)
            nn.init.zeros_(self.micro_head.bias)
        # Persist the ramp state for native cold loaders and read-only rendering.
        self.register_buffer('micro_iteration', torch.zeros((), dtype=torch.long))

    def forward_micro(self, times, colors, embedding, indices, weights, half_extent):
        t = torch.as_tensor(times, device=embedding.device, dtype=embedding.dtype).reshape(-1, 1)
        t = t.expand(len(embedding), 1)
        x = torch.cat((embedding, self.node_context(indices, weights),
                       NodeDeformationMLP.encode(t, self.time_frequencies)), -1)
        hidden = self.residual_mlp[:4](x)
        gain = self.eta * self.residual_mlp[4](hidden).tanh()
        beta = self.additive_cap * self.additive_head(hidden).tanh()
        raw = self.micro_head(hidden).tanh()
        # GPU scalar avoids a device synchronization at every render.
        alpha = ((self.micro_iteration - 6000) / 4000.).clamp(0., 1.)
        offset = alpha * (.005 * half_extent.detach()) * raw
        return (colors * (1 + gain) + beta, gain), offset, raw
