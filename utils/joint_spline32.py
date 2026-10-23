"""Joint Spline32 appearance adapter; native geometry remains differentiable."""
import types
import torch
from utils.independent_temporal_rgb import IndependentTemporalRGB


def install(model, spline=None):
    model.spline = spline if spline is not None else IndependentTemporalRGB(model.features)
    model.spline_basis = None
    model.gats_joint = True  # Native instrumentation includes the terminal Adam update.
    model.features.requires_grad_(False); model.features.grad = None
    model._deformation_table.requires_grad_(False); model._deformation_table.grad = None
    model.appearance_model.requires_grad_(False)
    for parameter in model.appearance_model.parameters(): parameter.grad = None
    def deform(self, points, scales, rotations, opacity, colors, time, embedding):
        if self.profile_deformation:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        xyz, scale, rotation = self.deform_geometry_from_nodes(time)
        color = self.spline(time=time.reshape(-1)[0], basis=self.spline_basis)
        if self.profile_deformation:
            end.record(); end.synchronize()
            self.last_deformation_forward_ms = start.elapsed_time(end)
        return xyz, scale, rotation, opacity, (color, torch.zeros_like(color))
    model.deform = types.MethodType(deform, model)


def configure_optimizer(model):
    inactive = {id(model.features), id(model._deformation_table)}
    inactive.update(id(p) for p in model.appearance_model.parameters())
    for group in model.optimizer.param_groups:
        group['params'] = [p for p in group['params'] if id(p) not in inactive]
    model.optimizer.param_groups[:] = [g for g in model.optimizer.param_groups if g['params']]
    model.optimizer.add_param_group(dict(name='spline32', params=[model.spline.controls], lr=.0025))
    assert not any(id(p) in inactive for g in model.optimizer.param_groups for p in g['params'])
