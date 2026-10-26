"""Experiment-only graph-anchored, TRAIN-centered/RMS-normalized temporal RGB.

Node dictionary is evaluated before Gaussian interpolation. No N×R×K tensors.
"""
import types
import torch
from torch import nn
from utils.independent_temporal_rgb import active_basis, knots


def spline_matrix(times, dtype=torch.float64):
    result = torch.zeros(len(times), 32, dtype=dtype)
    for row, t in enumerate(times):
        ids, values = active_basis(float(t), dtype=dtype)
        result[row, ids] = values
    return result


def initialize_dictionary(train_times, nodes=2000, rank=20, seed=1729):
    times = torch.as_tensor(train_times, dtype=torch.float64)
    basis = spline_matrix(times)
    mean = basis.mean(0)
    centered = basis-mean
    target = torch.cos(torch.pi*times[:, None]*torch.arange(1, rank+1, dtype=torch.float64)[None])
    target -= target.mean(0)
    control = torch.linalg.lstsq(centered, target, rcond=1e-12, driver='gelsd').solution.T
    rng = torch.Generator().manual_seed(seed)
    noise = torch.randn(nodes*rank, 1, 32, generator=rng, dtype=torch.float64)
    noise = torch.nn.functional.avg_pool1d(torch.nn.functional.pad(noise, (2,2), mode='reflect'), 5, stride=1)
    dictionary = control[None].expand(nodes,-1,-1).clone() + .01*noise.reshape(nodes,rank,32)
    dictionary -= dictionary.mean(-1,keepdim=True)
    gram = centered.T @ centered / len(times)
    rms = ((dictionary @ gram)*dictionary).sum(-1).clamp_min(0).sqrt()
    dictionary /= rms[...,None]
    return dict(dictionary=dictionary.float(), mean=mean.float(), gram=gram.float(),
                train_times=times.float(), knots=knots().float(), seed=seed,
                initialization='TRAIN-centered cos(pi*r*t), r1..20, CPU float64 gelsd projection; seeded sigma.01 five-control-smoothed node perturbations; control DC removed; TRAIN RMS normalized')


class GATS20(nn.Module):
    def __init__(self, initial, indices, weights, epsilon=1e-8):
        super().__init__()
        assert indices.ndim == 2 and indices.shape[1] == 4 and indices.shape == weights.shape
        assert torch.allclose(weights.sum(-1), torch.ones_like(weights[:,0]), atol=2e-6, rtol=0)
        assert (weights >= 0).all()
        self.D = nn.Parameter(initial['dictionary'].to(weights.device).clone())
        self.A = nn.Parameter(weights.new_zeros((len(indices),20,3)))
        self.register_buffer('basis_mean', initial['mean'].to(weights.device).clone())
        self.register_buffer('gram', initial['gram'].to(weights.device).clone())
        self.register_buffer('train_times', initial['train_times'].to(weights.device).clone())
        self.register_buffer('indices', indices.detach().clone())
        self.register_buffer('weights', weights.detach().clone())
        self.epsilon = float(epsilon)
        self.basis_cache = {}

    def basis(self, t):
        key = float(t)
        if key not in self.basis_cache:
            ids, v = active_basis(key, device=self.D.device)
            b = self.D.new_zeros(32)
            b[ids] = v
            self.basis_cache[key] = b-self.basis_mean
        return self.basis_cache[key]

    def normalized_dictionary(self):
        variance = ((self.D @ self.gram)*self.D).sum(-1).clamp_min(0)
        return self.D / (variance+self.epsilon).sqrt()[...,None]

    def node_modes(self, t=None, basis=None):
        b = self.basis(t) if basis is None else basis
        return self.normalized_dictionary() @ b

    def forward(self, t=None, basis=None, gaussian_ids=None):
        node_modes = self.node_modes(t, basis)
        indices, weights, coeff = self.indices, self.weights, self.A
        if gaussian_ids is not None:
            indices, weights, coeff = indices[gaussian_ids], weights[gaussian_ids], coeff[gaussian_ids]
        # Index/weight detachment is confined to this path; geometry stays live.
        phi = (node_modes[indices.detach()] * weights.detach()[...,None]).sum(1)
        return (coeff * phi[...,None]).sum(1)

    def remap(self, parents, optimizer, new_rows=None):
        """Native ordering: survivors then clones/split children; new Adam moments zero."""
        parents = parents.to(self.A.device).long()
        old = self.A
        replacement = nn.Parameter(old.detach()[parents].clone())
        group = next(g for g in optimizer.param_groups if g['name']=='gats_A')
        assert group['params'][0] is old
        state = optimizer.state.pop(old, None)
        if state is not None:
            state = {k:(v.clone() if torch.is_tensor(v) else v) for k,v in state.items()}
            for key in ['exp_avg','exp_avg_sq','max_exp_avg_sq']:
                if key in state:
                    state[key] = state[key][parents].clone()
                    if new_rows is not None:
                        state[key][new_rows] = 0
            optimizer.state[replacement] = state
        group['params'][0] = replacement
        self.A = replacement
        self.indices = self.indices[parents].clone()
        self.weights = self.weights[parents].clone()


def install(model, gats, variant):
    assert variant in ('J_R6_GATS20', 'J_GATS20_ONLY')
    model.gats = gats
    model.gats_variant = variant
    model.gats_basis = None
    original = model.deform

    def deform(self, points, scales, rotations, opacity, colors, time, embedding):
        if self.profile_deformation:
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
        if variant == 'J_R6_GATS20':
            xyz, scales, rotations, opacity, (base, dcol) = original(points,scales,rotations,opacity,colors,time,embedding)
        else:
            xyz, scales, rotations = self.deform_geometry_from_nodes(time)
            base, dcol = colors, torch.zeros_like(colors)
        correction = gats(basis=self.gats_basis) if self.gats_basis is not None else gats(float(time.reshape(-1)[0]))
        self.gats_last_correction = correction
        if self.profile_deformation:
            end.record();end.synchronize()
            self.last_deformation_forward_ms=start.elapsed_time(end)
        return xyz, scales, rotations, opacity, (base+correction,dcol)

    model.deform = types.MethodType(deform, model)
    return original


def configure_optimizer(model):
    if model.gats_variant == 'J_GATS20_ONLY':
        removed = {id(model._deformation_table)} | {id(p) for p in model.appearance_model.parameters()}
        model._deformation_table.requires_grad_(False)
        model.appearance_model.requires_grad_(False)
        for group in model.optimizer.param_groups:
            group['params'] = [p for p in group['params'] if id(p) not in removed]
        model.optimizer.param_groups[:] = [g for g in model.optimizer.param_groups if g['params']]
    assert not any(g['name'].startswith('gats_') for g in model.optimizer.param_groups)
    model.optimizer.add_param_group(dict(name='gats_A', params=[model.gats.A], lr=.0025))
    model.optimizer.add_param_group(dict(name='gats_D', params=[model.gats.D], lr=.0001))


def install_lifecycle(model):
    """Extend original raw-Gaussian lifecycle operations without changing their selection.

The native fixed-node training protocol disables densification/pruning. These
hooks preserve GATS lineage and Adam state if those raw operations are invoked;
they do not enable an unsupported dynamic-node geometry policy.
"""
    original_prune = model._prune_optimizer
    original_cat = model.cat_tensors_to_optimizer
    original_clone = model.densify_and_clone
    original_split = model.densify_and_split

    def native_only(fn, arg):
        groups = model.optimizer.param_groups[:]
        model.optimizer.param_groups[:] = [g for g in groups if not g['name'].startswith('gats_')]
        inactive_embedding = not any(g['name']=='deformation_table' for g in groups)
        if inactive_embedding:
            # Native raw lifecycle expects this field even when it is inert in B.
            # Temporary bookkeeping only: never an optimizer step or Adam state.
            model.optimizer.param_groups.append(dict(name='deformation_table',params=[model._deformation_table]))
        try:
            result = fn(arg)
            if inactive_embedding:
                result['deformation_table'].requires_grad_(False)
            return result
        finally:
            model.optimizer.param_groups[:] = groups

    def prune(self, mask):
        result = native_only(original_prune, mask)
        self.gats.remap(mask.nonzero().flatten(), self.optimizer)
        return result

    def concatenate(self, tensors):
        parents = self._gats_pending_parents
        assert len(parents) == len(tensors['xyz'])
        old = len(self.gats.A)
        result = native_only(original_cat, tensors)
        ids = torch.cat((torch.arange(old,device=parents.device),parents))
        self.gats.remap(ids,self.optimizer,torch.arange(len(ids),device=parents.device)>=old)
        return result

    def clone(self, grads, threshold, extent):
        mask = (torch.norm(grads,dim=-1)>=threshold) & (self.get_scaling.max(1).values<=self.percent_dense*extent)
        self._gats_pending_parents = mask.nonzero().flatten()
        return original_clone(grads,threshold,extent)

    def split(self, grads, threshold, extent, N=2):
        padded = torch.zeros(len(self._xyz),device=self._xyz.device)
        padded[:len(grads)] = grads.squeeze()
        mask = (padded>=threshold) & (self.get_scaling.max(1).values>self.percent_dense*extent)
        self._gats_pending_parents = mask.nonzero().flatten().repeat(N)
        return original_split(grads,threshold,extent,N)

    model._prune_optimizer = types.MethodType(prune,model)
    model.cat_tensors_to_optimizer = types.MethodType(concatenate,model)
    model.densify_and_clone = types.MethodType(clone,model)
    model.densify_and_split = types.MethodType(split,model)
