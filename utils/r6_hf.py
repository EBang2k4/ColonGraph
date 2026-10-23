"""Bounded, DC-free degree-2 real SH residual for FULL-R6.

Directions are world-space camera-center minus deformed Gaussian center.
Only this direction construction detaches geometry; rasterization stays joint.
SH ordering/signs match utils.sh_utils.eval_sh (indices 1 through 8).
"""
import torch
from torch import nn
from torch.nn import functional as F
from utils.sh_utils import C1, C2


def basis(direction):
    x, y, z = direction.unbind(-1)
    return torch.stack((-C1*y, C1*z, -C1*x, C2[0]*x*y, C2[1]*y*z,
                        C2[2]*(2*z*z-x*x-y*y), C2[3]*x*z,
                        C2[4]*(x*x-y*y)), dim=-1)


class R6HF(nn.Module):
    def __init__(self, count, device, dtype=torch.float32, eta=0.15):
        super().__init__()
        self.eta = float(eta)
        self.coefficients = nn.Parameter(torch.zeros(count, 3, 8, device=device, dtype=dtype))

    def forward(self, deformed_xyz, camera_center):
        direction = F.normalize(camera_center.detach().to(deformed_xyz).reshape(1, 3)
                                - deformed_xyz.detach(), dim=-1, eps=1e-8)
        value = (self.coefficients * basis(direction)[:, None, :]).sum(-1)
        return self.eta * value.tanh()

    def payload(self):
        return dict(format_version=1, degree=2, exclude_DC=True, eta=self.eta,
                    detached_direction=True, state_dict=self.state_dict())


def initialize(model):
    mode = getattr(model.args, 'appearance_mode', 'r6')
    if mode not in ('r6', 'r6_hf', 'r6_additive', 'r7_core', 'r7_v2', 'r6_speccam', 'r6_additive_wide96', 'r6_additive_micro'):
        raise ValueError('Unknown appearance_mode')
    if mode == 'r6_hf' and model.hf_appearance is None:
        if model.color_model != 'node_residual_mlp' or model.texture_active:
            raise ValueError('R6-HF requires original R6 without texture')
        model.hf_appearance = R6HF(len(model._xyz), model._xyz.device,
                                  model._xyz.dtype, getattr(model.args, 'hf_eta', .15))


def load(model, payload):
    assert payload['format_version'] == 1 and payload['degree'] == 2
    assert payload['exclude_DC'] and payload['detached_direction']
    model.args.appearance_mode = 'r6_hf'
    model.args.hf_eta = payload['eta']
    initialize(model)
    model.hf_appearance.eta = float(payload['eta'])
    model.hf_appearance.load_state_dict(payload['state_dict'], strict=True)
