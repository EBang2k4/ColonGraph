"""One-way, additive surface-local corrections. Never edits the G3 operator.

The original surface-repaired K4 remains intact. A disjoint dominant-backbone
support gets one extra corrective overlay, NOT a replacement fifth KNN slot.
Four linear temporal displacement knots (12 scalars/child), no new MLP/loss.
Each birth batch has its own Adam step counter and zero moments.
"""
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from utils.adaptive_deformation_graph import AdaptiveNodeManager
from utils.exact_split_branchpoint import digest
from utils.surface_graph_refine import distribution


class SurfaceChildren(nn.Module):
    FORMAT = 'surface_children_linear4_v1'

    def __init__(self, n, device, correction_enabled=True):
        super().__init__()
        self.correction_enabled = bool(correction_enabled)
        self.coefficients = nn.ParameterList()
        self.register_buffer('assignment', torch.full((n,), -1, dtype=torch.long, device=device))
        self.register_buffer('influence', torch.zeros(n, device=device))
        self.register_buffer('positions', torch.empty((0, 3), device=device))
        self.register_buffer('radii', torch.empty(0, device=device))
        self.register_buffer('parents', torch.empty(0, dtype=torch.long, device=device))
        self.register_buffer('births', torch.empty(0, dtype=torch.long, device=device))
        self.register_buffer('context_indices', torch.empty((0, 4), dtype=torch.long, device=device))
        self.register_buffer('context_weights', torch.empty((0, 4), device=device))

    def residual(self, time_value):
        if not len(self.coefficients):
            return self.positions.new_empty((0, 3))
        t = torch.as_tensor(time_value, device=self.positions.device).flatten()[0].clamp(0, 1)*3
        lo = t.floor().long().clamp(max=2)
        fraction = t-lo
        coefficients = torch.cat(list(self.coefficients), 0)
        return (1-fraction)*coefficients[:, lo] + fraction*coefficients[:, lo+1]

    def apply(self, translation, time_value):
        if not self.correction_enabled or not len(self.positions):
            return translation
        residual = self.residual(time_value)
        # Dense fixed-size mask, unique Gaussian rows: no atomic scatter sum.
        return translation + self.influence[:, None]*residual[self.assignment.clamp_min(0)]

    def child_motion(self, backbone_motion, time_value):
        return (backbone_motion[self.context_indices]*self.context_weights[..., None]).sum(1)+self.residual(time_value)

    @torch.no_grad()
    def insert(self, proposals, iteration):
        if not proposals:
            return None
        start = len(self.positions)
        for offset, p in enumerate(proposals):
            rows = p['rows']
            if (self.assignment[rows] >= 0).any():
                raise RuntimeError('Child support is immutable; never replace a trained correction')
            self.assignment[rows] = start+offset
            self.influence[rows] = p['weights']
        for key, values in (
            ('positions', torch.stack([p['position'] for p in proposals])),
            ('radii', self.radii.new_tensor([p.get('radius', 1.) for p in proposals])),
            ('parents', self.parents.new_tensor([p['parent'] for p in proposals])),
            ('births', self.births.new_full((len(proposals),), iteration)),
            ('context_indices', torch.stack([p['context_indices'] for p in proposals])),
            ('context_weights', torch.stack([p['context_weights'] for p in proposals]))):
            setattr(self, key, torch.cat((getattr(self, key), values), 0))
        parameter = nn.Parameter(self.positions.new_zeros((len(proposals), 4, 3)))
        if not self.correction_enabled:
            parameter.requires_grad_(False)
        self.coefficients.append(parameter)
        return parameter

    def payload(self):
        return dict(format=self.FORMAT, batch_sizes=[len(p) for p in self.coefficients],
                    correction_enabled=self.correction_enabled, state=self.state_dict())

    @classmethod
    def from_payload(cls, payload, device):
        if payload['format'] != cls.FORMAT:
            raise ValueError('Unsupported surface child representation')
        out = cls(len(payload['state']['assignment']), device, payload.get('correction_enabled', True))
        for key in out._buffers:
            setattr(out, key, payload['state'][key].to(device).clone())
        for n in payload['batch_sizes']:
            out.coefficients.append(nn.Parameter(torch.zeros((n, 4, 3), device=device),
                                                requires_grad=out.correction_enabled))
        out.load_state_dict(payload['state'], strict=True)
        return out


def register_optimizer(model, parameter=None):
    params = ([parameter] if parameter is not None else
              list(model.surface_children.parameters()) if model.surface_children is not None else [])
    if not params:
        return
    group = next((g for g in model.optimizer.param_groups if g['name'] == 'surface_children'), None)
    if group is None:
        model.optimizer.add_param_group(dict(params=params, lr=0., name='surface_children'))
    else:
        known = {id(p) for p in group['params']}
        group['params'].extend(p for p in params if id(p) not in known)


def backbone_state(model):
    # Parameters may train BETWEEN events; only structural tensors stay fixed.
    graph = model.graph_motion_residual
    return dict(k4=model.deformation_graph.state_dict(),
                appearance_graph=model.appearance_graph.state_dict(),
                topology={k:v for k,v in graph.state_dict().items() if not k.startswith('decoder.')})


def rng_state():
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
                python=random.getstate(), numpy=np.random.get_state())


class SurfaceChildManager:
    collect_online_gradient = False
    _depth_metrics = staticmethod(AdaptiveNodeManager._depth_metrics)

    def __init__(self, model, config, output_dir, *unused, mode='adaptive', **kwargs):
        if mode not in ('control', 'adaptive', 'shadow') or config.max_nodes != 2535:
            raise ValueError('Surface children require control/adaptive/shadow and fixed max2535')
        if len(model.deformation_graph.node_xyz) != 2000 or model.graph_motion_residual.mode != 'g3':
            raise ValueError('Requires immutable G3 fixed2000 backbone')
        if getattr(model.graph_motion_residual, 'initialize_transforms', None):
            raise ValueError('Must not compose with old edge refinement')
        self.model, self.config, self.mode = model, config, mode
        self.output_dir = str(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.initial_backbone_hash = digest(backbone_state(model))
        graph = model.deformation_graph
        self.anchor_weight, slots = graph.gaussian_node_weights.max(1)
        self.anchor = graph.gaussian_node_indices.gather(1, slots[:, None]).squeeze(1)
        self.events, self.bank = [], None

    def is_event(self, iteration):
        return self.config.start <= iteration <= self.config.end and (iteration-self.config.start)%self.config.interval == 0

    def begin_iteration(self, iteration):
        self.iteration = iteration

    def update_statistics(self, *args):
        raise RuntimeError('Surface children use depth contribution probes, not online RGB gradients')

    def collect_event_signals(self, iteration, cameras, render, pipe, background):
        started = time.perf_counter()
        rng = digest(rng_state())
        cameras = sorted(cameras, key=lambda c: float(torch.as_tensor(c.time).flatten()[0]))
        indices = torch.linspace(0, len(cameras)-1, 16).round().long().tolist()
        n = len(self.anchor)
        mass = self.anchor_weight.new_zeros(n)
        error_mass = torch.zeros_like(mass)
        seen = torch.zeros_like(self.anchor)
        checks = []
        with torch.enable_grad():
            for index in indices:
                c = cameras[index]
                output = render(c, self.model, pipe, background, stage='fine')
                depth = output['depth'].squeeze()
                gt = c.original_depth.to(depth).squeeze()
                valid = (gt > 0) & torch.isfinite(gt) & torch.isfinite(depth)
                if getattr(c, 'mask', None) is not None:
                    valid &= torch.as_tensor(c.mask, device=depth.device).bool().squeeze()
                if not valid.any():
                    raise RuntimeError('No valid depth in training probe')
                error = ((depth/(depth[valid].max()+1e-6)).clamp(0,1)-
                         (gt/(gt[valid].max()+1e-6)).clamp(0,1)).square().detach()
                probe = torch.stack((error*valid, valid.float(), torch.zeros_like(error)))
                adjoint, = torch.autograd.grad((output['render']*probe).sum(), output['transformed_color'])
                adjoint = adjoint.detach().reshape(n,3).clamp_min(0)
                if not torch.isfinite(adjoint).all():
                    raise RuntimeError('Nonfinite depth contribution probe')
                alpha_mass = float((output['alpha'].squeeze().detach()*valid).sum())
                discrepancy = abs(float(adjoint[:,1].sum())-alpha_mass)/max(alpha_mass,1e-12)
                if discrepancy > .002:
                    raise RuntimeError('Contribution adjoint does not reproduce alpha')
                checks.append(discrepancy)
                error_mass += adjoint[:,0]
                mass += adjoint[:,1]
                seen += adjoint[:,1] > 1e-6
                del output, adjoint
        self.bank = dict(mass=mass, error_mass=error_mass, seen=seen)
        self.probe_summary = dict(iteration=iteration, frames=[cameras[i].image_name for i in indices],
            max_adjoint_alpha_error=max(checks), seconds=time.perf_counter()-started)
        if digest(rng_state()) != rng:
            raise RuntimeError('Depth probes consumed training RNG')
        torch.save({k:v.cpu() for k,v in self.bank.items()}, Path(self.output_dir)/f'depth_probe_{iteration}.pth')

    @torch.no_grad()
    def proposals(self, quota=80):
        model = self.model
        mass, error_mass, seen = (self.bank[k] for k in ('mass','error_mass','seen'))
        children = model.surface_children
        occupied = torch.zeros_like(seen, dtype=torch.bool) if children is None else children.assignment >= 0
        visible = (seen >= 4) & ~occupied
        totals = mass.new_zeros(2000).index_add_(0, self.anchor, mass*self.anchor_weight*visible)
        errors = mass.new_zeros(2000).index_add_(0, self.anchor, error_mass*self.anchor_weight*visible)
        counts = mass.new_zeros(2000).index_add_(0, self.anchor, visible.float())
        score = errors/totals.clamp_min(1e-12)
        # At least 32 repeatedly observed Gaussians and one alpha-weighted
        # pixel-equivalent/frame. No RGB signal, motion product or quota forcing.
        eligible = (counts >= 32) & (totals/16 >= 1.) & (score > 0)
        if children is not None:
            eligible[children.parents] = False
        ids = eligible.nonzero().flatten()
        if not len(ids):
            return [], dict(eligible=0)
        floor = score[ids].median()
        ids = ids[score[ids] >= floor]
        ids = ids[torch.argsort(score[ids], descending=True, stable=True)]
        proposals = []
        graph = model.deformation_graph
        for parent in ids.tolist():
            if len(proposals) >= quota:
                break
            rows = ((self.anchor == parent) & ~occupied).nonzero().flatten()
            observed = rows[seen[rows] >= 4]
            gaussian_error = error_mass[observed]/mass[observed].clamp_min(1e-12)
            high = observed[gaussian_error >= gaussian_error.median()]
            q = error_mass[high]*self.anchor_weight[high]
            xyz = model._xyz.detach()[high]
            centroid = (xyz*q[:,None]).sum(0)/q.sum().clamp_min(1e-12)
            sample = high[(xyz-centroid).square().sum(1).argmin()]
            distance2=(model._xyz.detach()[rows]-model._xyz.detach()[sample]).square().sum(1)
            radius2=distance2.median().clamp_min(1e-8)
            influence=self.anchor_weight[rows]*torch.exp(-.5*distance2/radius2)
            if int(((seen[rows]>=4)&(influence>.05)).sum())<32 or float((mass[rows]*influence).sum()/16)<1.:
                continue
            proposals.append(dict(parent=parent, rows=rows, weights=influence, radius=float(radius2.sqrt()),
                position=model._xyz[sample].detach().clone(), sample=int(sample),
                context_indices=graph.gaussian_node_indices[sample].clone(),
                context_weights=graph.gaussian_node_weights[sample].clone(), score=float(score[parent])))
        return proposals, dict(eligible=int(eligible.sum()), quality_pool=len(ids), median_error_floor=float(floor))

    @torch.no_grad()
    def adapt(self, iteration, camera, render, pipe, background, *unused):
        model = self.model
        if digest(backbone_state(model)) != self.initial_backbone_hash:
            raise RuntimeError('Backbone topology/K4 mutated during training')
        count = 0 if model.surface_children is None else len(model.surface_children.positions)
        proposed, pool = self.proposals(min(80, 535-count))
        if self.mode == 'control':
            proposed = []
        base = model._query_base_node_motion(camera.time)[0]
        messages = model.graph_motion_residual.aggregate(base).clone()
        message_repeat = model.graph_motion_residual.aggregate(base).clone()
        fixed_motion = model.deformation_graph.blend(*model._query_node_motion(camera.time))[0]
        fixed_before = (fixed_motion if model.surface_children is None else
                        model.surface_children.apply(fixed_motion, camera.time)).clone()
        old_xyz = model.deform_geometry_from_nodes(camera.time)[0].clone()
        repeat_noise = float((model.deform_geometry_from_nodes(camera.time)[0]-old_xyz).abs().max())
        before_adam = digest(model.optimizer.state_dict())
        old_params = [id(p) for g in model.optimizer.param_groups for p in g['params']]
        if proposed:
            if model.surface_children is None:
                model.surface_children = SurfaceChildren(len(model._xyz), model._xyz.device,
                                                        correction_enabled=self.mode != 'shadow')
            parameter = model.surface_children.insert(proposed, iteration)
            # Check old optimizer first: insertion must not mutate any old state.
            assert digest(model.optimizer.state_dict()) == before_adam
            register_optimizer(model, parameter)
            assert parameter not in model.optimizer.state
            assert old_params == [id(p) for g in model.optimizer.param_groups if g['name'] != 'surface_children' for p in g['params']] + [id(p) for g in model.optimizer.param_groups if g['name'] == 'surface_children' for p in g['params'][:-1]]
        delta = (model.deform_geometry_from_nodes(camera.time)[0]-old_xyz).norm(dim=1)
        new_messages = model.graph_motion_residual.aggregate(base)
        fixed_after = (fixed_motion if model.surface_children is None else
                       model.surface_children.apply(fixed_motion, camera.time))
        message_error = float((new_messages-messages).abs().max())
        message_repeat_error = float((message_repeat-messages).abs().max())
        checks = dict(backbone_exact=digest(backbone_state(model)) == self.initial_backbone_hash,
            messages_numerically_unchanged=message_error <= max(1e-5,4*message_repeat_error),
            zero_birth_fixed_field_bit_exact=torch.equal(fixed_before, fixed_after),
            forward_continuity=float(delta.max()) <= max(5e-4,8*repeat_noise), old_adam_unchanged=True)
        if not all(checks.values()):
            raise RuntimeError(f'Surface child insertion audit failed: {checks}')
        row = dict(iteration=iteration, mode=self.mode, splits=len(proposed),
            total_nodes=2000+count+len(proposed), checks=checks, continuity=distribution(delta),
            message_max_error=message_error,message_repeat_noise=message_repeat_error,
            forward_repeat_noise=repeat_noise,
            pool=pool, probe=self.probe_summary,
            output_correction_enabled=self.mode != 'shadow',
            proposals=[dict(parent=p['parent'], sample=p['sample'], score=p['score']) for p in proposed])
        self.events.append(row)
        self._append('events.jsonl', row)
        self.log_children(iteration)
        print('[SURFACE_CHILD_EVENT] '+json.dumps(row), flush=True)
        return row

    @torch.no_grad()
    def log_children(self, iteration):
        child = self.model.surface_children
        if child is None:
            return
        mass, error, seen = (self.bank[k] for k in ('mass','error_mass','seen'))
        for i in range(len(child.positions)):
            rows = (child.assignment == i).nonzero().flatten()
            w = child.influence[rows]
            residual = torch.stack([child.residual(t)[i] for t in (0.,1/3,2/3,1.)])
            self._append('child_control.jsonl', dict(iteration=iteration, child_id=2000+i,
                parent=int(child.parents[i]), context_nodes=child.context_indices[i].tolist(),
                position=child.positions[i].tolist(), age=iteration-int(child.births[i]), activation=1.,
                output_correction_enabled=child.correction_enabled,
                support_radius=float(child.radii[i]), effective_support_gaussians=int((w>.05).sum()),
                support_gaussians=len(rows), visible_support_gaussians=int((seen[rows]>=4).sum()),
                sum_influence_weight=float(w.sum()), rendered_alpha_mass=float((mass[rows]*w).sum()/16),
                local_depth_error_mass=float((error[rows]*w).sum()/16),
                residual_norm=distribution(residual.norm(dim=1))))

    def record_motion_diagnostics(self, iteration):
        if iteration % 500 == 0 and self.model.surface_children is not None:
            child = self.model.surface_children
            if self.mode == 'shadow':
                for p in child.parameters():
                    if p.requires_grad or p.grad is not None or p.abs().max()>0 or p in self.model.optimizer.state:
                        raise RuntimeError('Shadow child received gradient/update/Adam moments')
                self._append('shadow_audit.jsonl',dict(iteration=iteration,output_disabled=True,
                    child_gradients_absent=True,coefficients_zero=True,child_adam_state_absent=True))
            with torch.no_grad():
                motions=[]
                for t in torch.linspace(0,1,16,device=child.positions.device):
                    motions.append(child.child_motion(self.model._query_node_motion(t)[0],t))
                magnitude=torch.stack(motions).norm(dim=-1)
                p95=float(torch.quantile(magnitude,.95))
                ref=json.loads(Path(self.model.args.graph_forward_motion_reference).read_text())['motion_reference_p95']
                keys=sorted(int(k) for k in ref)
                limit=float(np.interp(iteration,keys,[ref[str(k)] for k in keys]))
                if not torch.isfinite(magnitude).all() or p95>2*limit:
                    raise RuntimeError('Child motion exceeded matched fixed-G3 safety envelope')
            self._append('residual_learning.jsonl', dict(iteration=iteration,
                children=len(child.positions), child_motion_p95=p95, matched_motion_p95=limit,
                coefficient_norm=distribution(torch.cat(list(child.coefficients)).norm(dim=-1))))

    def _append(self, name, row):
        with (Path(self.output_dir)/name).open('a') as f:
            f.write(json.dumps(row, allow_nan=False)+'\n')

    def save(self, path, iteration):
        torch.save(dict(format='surface_child_manager_v1', iteration=iteration,
            backbone_hash=self.initial_backbone_hash, mode=self.mode, events=self.events), path)

    def load(self, path):
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state['format'] != 'surface_child_manager_v1' or state['backbone_hash'] != self.initial_backbone_hash or state['mode'] != self.mode:
            raise ValueError('Surface child continuation metadata mismatch')
        self.events = state['events']
