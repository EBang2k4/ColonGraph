"""Explicit Stage 3 schedules and detached camera-conditioned R6 appearance."""
import math
import torch
from torch import nn
from scene.deformation import NodeConditionedResidualAppearance, NodeDeformationMLP


def schedule(iteration):
    t = min(1., max(0., (iteration - 15000) / 15000))
    eta = .10 + .45 * (1 + math.cos(math.pi * t))
    prior = min(1., max(0., (iteration - 10000) / 15000))
    return eta, .25 - .20 * prior, .01 - .008 * prior


def update_training(model, opt, iteration, stage):
    if stage != 'fine' or not getattr(opt, 'stage3_dynamics', False):
        return
    from utils.stage5_schedule import dynamics
    eta, opt.depth_weight, opt.knn_weight = dynamics(iteration, getattr(opt, 'stage5_profile', 'J'))
    bases = dict(f_dc=opt.feature_lr, deformation_table=opt.feature_lr,
                 scaling=opt.scaling_lr, rotation=opt.rotation_lr,
                 opacity=opt.opacity_lr, skinning=model.args.ggas_lr)
    for group in model.optimizer.param_groups:
        if group['name'] in bases:
            group['lr'] = bases[group['name']] * eta


def camera_cues(xyz, world_view_transform, scale):
    """Camera matrices are transposed in Camera: row homogeneous xyz @ W2C.T.

    Detach both geometry and pose; normal geometry rendering remains live.
    The normalization is the frozen post-coarse scene bbox half-extent.
    """
    if world_view_transform is None:
        raise ValueError('R6-SpecCam requires active camera extrinsics')
    with torch.no_grad():
        matrix = world_view_transform.to(device=xyz.device, dtype=xyz.dtype)
        p = (xyz.detach() @ matrix[:3, :3] + matrix[3, :3]) / scale.detach()
        return torch.cat((p, p.norm(dim=-1, keepdim=True)), -1)


class R6SpecCam(NodeConditionedResidualAppearance):
    def __init__(self, cfg, num_nodes):
        super().__init__(cfg, num_nodes)
        if self.node_appearance_dim != 16 or num_nodes != 2000:
            raise ValueError('R6-SpecCam requires 2000 appearance nodes with 16D features')
        self.register_buffer('camera_scale', torch.ones(()))
        with torch.random.fork_rng(devices=[]):
            self.residual_mlp = nn.Sequential(nn.Linear(49, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))
            self.additive_head = nn.Linear(64, 3)
            generator = torch.Generator(device='cpu').manual_seed(int(cfg.appearance_seed))
            for layer in (self.residual_mlp[0], self.residual_mlp[2]):
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5), generator=generator)
                bound = 1 / math.sqrt(layer.weight.shape[1])
                nn.init.uniform_(layer.bias, -bound, bound, generator=generator)
            nn.init.normal_(self.node_appearance, std=.01, generator=generator)
            for head in (self.residual_mlp[4], self.additive_head):
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

    def components(self, times, colors, embedding, indices, weights, camera_input):
        if camera_input.shape != (len(embedding), 4) or camera_input.requires_grad:
            raise ValueError('Expected detached [N,4] camera cue')
        t = torch.as_tensor(times, device=embedding.device, dtype=embedding.dtype).reshape(-1, 1)
        t = t.expand(len(embedding), 1)
        x = torch.cat((embedding, self.node_context(indices, weights),
            NodeDeformationMLP.encode(t, self.time_frequencies), camera_input), -1)
        hidden = self.residual_mlp[:4](x)
        return .50 * self.residual_mlp[4](hidden).tanh(), .30 * self.additive_head(hidden).tanh()

    def forward(self, points, times, colors, embedding, node_indices=None, node_weights=None, camera_input=None):
        gain, beta = self.components(times, colors, embedding, node_indices, node_weights, camera_input)
        return colors * (1 + gain) + beta, gain
