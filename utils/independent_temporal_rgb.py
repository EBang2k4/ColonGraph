"""Independent FP32 RGB controls; fixed open-uniform cubic B-spline basis.

Experiment-only adapter: the original renderer and model files stay unchanged.
"""
import types
import torch
from torch import nn


def knots(dtype=torch.float64, device='cpu'):
    # 32 controls, degree3: 28 interior knots and four copies of each endpoint.
    return torch.cat((torch.zeros(4, dtype=dtype, device=device),
                      torch.arange(1, 29, dtype=dtype, device=device)/29,
                      torch.ones(4, dtype=dtype, device=device)))


def active_basis(time, dtype=torch.float32, device='cpu'):
    """Four active indices/weights, computed once per camera timestamp."""
    t = float(time.reshape(-1)[0]) if torch.is_tensor(time) else float(time)
    if not 0 <= t <= 1:
        raise ValueError('Expected original scene time in [0,1]')
    span = min(31, int(t*29)+3)
    u = knots()
    basis = [1.0, 0.0, 0.0, 0.0]
    left, right = [0.0]*4, [0.0]*4
    for j in range(1, 4):
        left[j] = t-float(u[span+1-j])
        right[j] = float(u[span+j])-t
        saved = 0.0
        for r in range(j):
            temp = basis[r]/(right[r+1]+left[j-r])
            basis[r] = saved+right[r+1]*temp
            saved = left[j-r]*temp
        basis[j] = saved
    return (torch.arange(span-3, span+1, device=device),
            torch.tensor(basis, dtype=dtype, device=device))


class IndependentTemporalRGB(nn.Module):
    def __init__(self, canonical):
        super().__init__()
        if canonical.ndim != 2 or canonical.shape[1] != 3:
            raise ValueError('Expected canonical RGB N×3')
        self.controls = nn.Parameter(canonical.detach().float()[:, None].repeat(1, 32, 1))

    def forward(self, time=None, basis=None):
        indices, weights = basis if basis is not None else active_basis(
            time, self.controls.dtype, self.controls.device)
        return (self.controls[:, indices]*weights[None, :, None]).sum(1)


def freeze_source(model):
    for name in ('_xyz', '_scaling', '_rotation', '_opacity', 'features', '_deformation_table'):
        p = getattr(model, name); p.requires_grad_(False); p.grad = None
    for name in ('node_mlp', 'appearance_model', 'deformation_graph', 'appearance_graph',
                 'graph_motion_residual', '_deformation', 'surface_children', 'skinning'):
        module = getattr(model, name, None)
        if module is not None:
            module.requires_grad_(False)
            for p in module.parameters(): p.grad = None
    model.profile_deformation = False


def assert_source_frozen(model):
    params = [getattr(model, name) for name in
              ('_xyz', '_scaling', '_rotation', '_opacity', 'features', '_deformation_table')]
    for name in ('node_mlp', 'appearance_model', 'deformation_graph', 'appearance_graph',
                 'graph_motion_residual', '_deformation', 'surface_children', 'skinning'):
        module = getattr(model, name, None)
        if module is not None: params.extend(module.parameters())
    assert all(not p.requires_grad and p.grad is None for p in params)


def install(model, spline):
    """Bypass every pretrained appearance path, preserving native geometry math."""
    original = model.deform
    model.spline_basis = None

    def deform(self, points, scales, rotations, opacity, colors, time, embedding):
        with torch.no_grad():
            xyz, scale, rotation = self.deform_geometry_from_nodes(time)
        color = spline(time=time.reshape(-1)[0], basis=self.spline_basis)
        return xyz, scale, rotation, opacity, (color, torch.zeros_like(color))

    model.deform = types.MethodType(deform, model)
    return original
