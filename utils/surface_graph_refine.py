"""Opt-in surface EDGE refinement. No new learned parameters or loss.

The graph used here is the existing G3-Surf graph, not a new surface proxy.
Insertion is followed by a fixed 250-update homotopy of local graph messages,
node initialization and old/target K4 interpolation. All buffers are checkpointed.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import torch

from scene.deformation_graph import SparseDeformationGraph
from utils.adaptive_deformation_graph import AdaptiveNodeManager
from utils.graph_forward import GraphMotionResidual


def distribution(x):
    x = x.detach().float().flatten()
    if not len(x):
        return dict(count=0, mean=0., median=0., p90=0., p95=0., max=0.)
    return dict(count=len(x), mean=float(x.mean()), median=float(x.median()),
                p90=float(torch.quantile(x, .90)), p95=float(torch.quantile(x, .95)), max=float(x.max()))


def robust_unit(x):
    low, high = torch.quantile(x.float(), x.new_tensor([.05, .95]))
    if high <= low:
        return torch.zeros_like(x)
    return ((x-low)/(high-low)).clamp(0, 1)


class SurfaceRefinedResidual(GraphMotionResidual):
    """Same G3 decoder; only local topology and nonlearned activation change."""
    FORMAT = 'surface_edge_refine_v1'
    TRANSITION = ('previous_edges', 'previous_weights', 'previous_denominator',
                  'endpoints', 'child_rows', 'parents', 'rows', 'previous_indices',
                  'previous_k4_weights')

    @classmethod
    def from_existing(cls, source):
        # Keep the exact decoder parameter objects AND their Adam moments.
        out = copy.copy(source)
        out.__class__ = cls
        out._buffers = source._buffers.copy()
        out._modules = source._modules.copy()
        out.iteration, out.birth_iteration, out.activation_steps = 0, -1, 250
        return out

    def activation(self):
        return min(1., max(0., (self.iteration-self.birth_iteration)/self.activation_steps))

    def set_iteration(self, iteration):
        self.iteration = int(iteration)
        if hasattr(self, 'previous_edges') and self.activation() >= 1:
            for name in self.TRANSITION:
                del self._buffers[name]

    @staticmethod
    def aggregate_on(base, xyz, edges, weights, denominator):
        src, dst = edges
        recipient, neighbor = torch.cat((src, dst)), torch.cat((dst, src))
        message = torch.cat((base[neighbor]-base[recipient], xyz[neighbor]-xyz[recipient]), -1)
        result = base.new_zeros((len(xyz), 6))
        result.index_add_(0, recipient, message*torch.cat((weights, weights))[:, None])
        return result/denominator[:, None]

    def initialize_transforms(self, values):
        if not hasattr(self, 'previous_edges'):
            return values
        a = self.activation()
        out = []
        for value in values:
            interpolated = .5*(value[self.parents[:, 0]]+value[self.parents[:, 1]])
            out.append(value.index_copy(0, self.child_rows, (1-a)*interpolated+a*value[self.child_rows]))
        return tuple(out)

    def forward(self, base):
        if not hasattr(self, 'previous_edges'):
            return super().forward(base)
        a = self.activation()
        target = super().aggregate(base)
        old_n = len(self.previous_denominator)
        previous = self.aggregate_on(base[:old_n], self.canonical_xyz[:old_n],
            self.previous_edges, self.previous_weights, self.previous_denominator)
        # Unrelated nodes keep the previous operator bit-for-bit during ramp.
        old_messages = previous.clone()
        old_messages = old_messages.index_copy(0, self.endpoints,
            (1-a)*previous[self.endpoints]+a*target[self.endpoints])
        aggregate = torch.cat((old_messages, target[old_n:]), 0)
        residual = self.decoder(aggregate)
        final = base+residual
        interpolated = .5*(final[self.parents[:, 0]]+final[self.parents[:, 1]])
        final = final.index_copy(0, self.child_rows, (1-a)*interpolated+a*final[self.child_rows])
        return final, final-base

    def blend_gaussians(self, graph, values):
        result = graph.blend(*values)
        if not hasattr(self, 'previous_edges'):
            return result
        a = self.activation()
        out = []
        for value, blended in zip(values, result):
            previous = (value[self.previous_indices]*self.previous_k4_weights[..., None]).sum(1)
            out.append(blended.index_copy(0, self.rows, (1-a)*previous+a*blended[self.rows]))
        return tuple(out)

    @torch.no_grad()
    def refine(self, positions, parents, rows, previous_indices, previous_weights):
        if hasattr(self, 'previous_edges'):
            raise RuntimeError('Previous local activation must finish before another event')
        old_n = len(self.canonical_xyz)
        src, dst = self.edge_indices
        remove = torch.zeros(len(src), device=src.device, dtype=torch.bool)
        old_edge_rows = []
        for left, right in parents.tolist():
            found = ((src == left) & (dst == right)) | ((src == right) & (dst == left))
            if int(found.sum()) != 1:
                raise ValueError('Refinement must replace exactly one existing surface edge')
            remove |= found
            old_edge_rows.append(int(found.nonzero()[0]))
        if int(remove.sum()) != len(parents):
            raise ValueError('Duplicate selected edge')
        self.register_buffer('previous_edges', self.edge_indices.clone())
        self.register_buffer('previous_weights', self.edge_weights.clone())
        self.register_buffer('previous_denominator', self.weight_denominator.clone())
        self.register_buffer('parents', parents.clone())
        self.register_buffer('child_rows', torch.arange(old_n, old_n+len(positions), device=src.device))
        self.register_buffer('endpoints', parents.flatten().unique())
        self.register_buffer('rows', rows.clone())
        self.register_buffer('previous_indices', previous_indices.clone())
        self.register_buffer('previous_k4_weights', previous_weights.clone())
        children = self.child_rows
        additions = torch.stack((parents.flatten(), children[:, None].expand(-1, 2).flatten()))
        # Series-conductance heuristic: equal halves receive 2*q_AB.
        # This DOES NOT preserve G3's row-normalized nonlinear message operator;
        # the endpoint denominator changes. Keep the completed experiment's
        # numerical behavior unchanged and audit this effect before a new run.
        added_weights = (2*self.edge_weights[torch.tensor(old_edge_rows, device=src.device)]).repeat_interleave(2)
        self.canonical_xyz = torch.cat((self.canonical_xyz, positions), 0)
        self.edge_indices = torch.cat((self.edge_indices[:, ~remove], additions), 1)
        self.edge_weights = torch.cat((self.edge_weights[~remove], added_weights))
        src, dst = self.edge_indices
        self.recipient, self.neighbor = torch.cat((src, dst)), torch.cat((dst, src))
        self.directed_weights = torch.cat((self.edge_weights, self.edge_weights))
        denominator = self.edge_weights.new_zeros(len(self.canonical_xyz))
        denominator.index_add_(0, self.recipient, self.directed_weights)
        unrelated = torch.ones(old_n, device=src.device, dtype=torch.bool)
        unrelated[self.endpoints] = False
        denominator[:old_n][unrelated] = self.previous_denominator[unrelated]
        self.weight_denominator = denominator
        self.birth_iteration = self.iteration
        return remove

    def state_dict_payload(self):
        out = super().state_dict_payload()
        out['surface_refine'] = dict(format=self.FORMAT, iteration=self.iteration,
            birth_iteration=self.birth_iteration, activation_steps=self.activation_steps,
            transition={k:self._buffers[k].detach().clone() for k in self.TRANSITION if k in self._buffers})
        return out

    @classmethod
    def from_surface_payload(cls, payload, device=None):
        standard = dict(payload)
        metadata = standard.pop('surface_refine')
        if metadata['format'] != cls.FORMAT or metadata['activation_steps'] != 250:
            raise ValueError('Unsupported surface-refinement activation protocol')
        out = cls.from_existing(GraphMotionResidual.from_state_dict_payload(standard, device))
        out.iteration = int(metadata['iteration'])
        out.birth_iteration = int(metadata['birth_iteration'])
        for k, value in metadata['transition'].items():
            out.register_buffer(k, value.to(out.canonical_xyz.device))
        return out


class SurfaceGraphRefineManager(AdaptiveNodeManager):
    """Edge score = robust(depth-position-gradient) * robust(node motion RMS)."""
    geometry_signal = 'depth'

    def __init__(self, model, config, output_dir, *unused, **kwargs):
        super().__init__(model, config, output_dir, allow_grown=True)
        if config.mode != 'split' or config.max_nodes != 2535 or not self.appearance_graph_fixed:
            raise ValueError('SurfRefine is split-only, max2535, fixed2000 appearance')
        if model.graph_motion_residual.mode != 'g3':
            raise ValueError('SurfRefine preserves G3 (not the old G4 motion guard)')
        if not isinstance(model.graph_motion_residual, SurfaceRefinedResidual):
            model.graph_motion_residual = SurfaceRefinedResidual.from_existing(model.graph_motion_residual)
        self.reference = model.graph_forward_reference_data
        self.backbone = self.graph.node_xyz[:2000].clone()
        self.appearance_indices = model.appearance_graph.gaussian_node_indices.clone()
        self.appearance_weights = model.appearance_graph.gaussian_node_weights.clone()
        self.quotas = dict(zip(range(5000, 20001, 2500), (58, 74, 85, 96, 120, 52, 50)))

    def begin_iteration(self, iteration):
        self.model.graph_motion_residual.set_iteration(iteration)

    @torch.no_grad()
    def collect_event_signals(self, iteration, cameras, render, pipe, bg):
        # No extra rendering or RGB probes: statistics come from depth loss only.
        self.begin_iteration(iteration)

    @torch.no_grad()
    def edge_candidates(self):
        residual = self.model.graph_motion_residual
        edges = residual.edge_indices
        motion = torch.stack([self.model._query_node_motion(t)[0]
                              for t in self.reference['trajectory_times']])
        self._event_motion = motion
        self._event_base = [self.model.node_mlp(self.graph.node_xyz, t)[0]
                            for t in self.reference['trajectory_times']]
        complexity = (motion[:, edges[0]]-motion[:, edges[1]]).square().sum(-1).mean(0).sqrt()
        idx, weights = self.graph.gaussian_node_indices, self.graph.gaussian_node_weights
        observations = self.gaussian_observations >= 5
        supports, errors, records = [], [], []
        # Reuse K4 memberships. CPU grouping avoids 6500 full GPU N scans.
        membership = [None]*len(self.graph.node_xyz)
        ids, slots = ((weights > self.config.weight_threshold) & observations[:, None]).nonzero(as_tuple=True)
        nodes = idx[ids, slots]
        order = torch.argsort(nodes, stable=True)
        nodes_cpu = nodes[order].cpu()
        bounds = torch.searchsorted(nodes_cpu, torch.arange(len(membership)+1))
        grouped_ids = ids[order]
        for n in range(len(membership)):
            membership[n] = grouped_ids[int(bounds[n]):int(bounds[n+1])]
        for e, (left, right) in enumerate(edges.T.tolist()):
            l, r = membership[left], membership[right]
            shared = l[torch.isin(l, r)]
            use_shared = len(shared) >= self.config.min_coverage
            support = shared if use_shared else torch.cat((l, r)).unique()
            if len(support) < self.config.min_coverage:
                supports.append(None); errors.append(0.); records.append(None); continue
            influence = (weights[support]*((idx[support] == left)|(idx[support] == right))).sum(1)
            difficulty = (influence*self.gaussian_score[support]).sum()/influence.sum().clamp_min(1e-8)
            supports.append(support)
            errors.append(float(difficulty))
            records.append(dict(node_A=left, node_B=right, support_size=len(support),
                support_method='shared_K4' if use_shared else 'union_significant_K4'))
        error = complexity.new_tensor(errors)
        score = robust_unit(error)*robust_unit(complexity)
        return edges, supports, error, complexity, score, records

    @torch.no_grad()
    def adapt(self, iteration, camera, render, pipe, background, perceptual=None):
        self.begin_iteration(iteration)
        residual = self.model.graph_motion_residual
        edges, supports, error, complexity, scores, records = self.edge_candidates()
        old_indices = self.graph.gaussian_node_indices.clone()
        old_weights = self.graph.gaussian_node_weights.clone()
        old_n = len(self.node_ids)
        nodes = self.graph.node_xyz
        target_idx, target_w = old_indices.clone(), old_weights.clone()
        positions, parents, accepted, rejected = [], [], [], []
        occupied_nodes, occupied_rows = set(), torch.zeros(len(old_indices), device=nodes.device, dtype=torch.bool)
        quota = min(self.quotas[int(iteration)], self.config.max_nodes-old_n)
        positive = (scores > 0).nonzero().flatten()
        ranked = positive[torch.argsort(scores[positive], descending=True, stable=True)]
        for e in ranked.tolist():
            if len(positions) >= quota: break
            left, right = edges[:, e].tolist()
            ids = supports[e]
            reason = None
            if ids is None: reason = 'insufficient_support'
            elif left in occupied_nodes or right in occupied_nodes or occupied_rows[ids].any(): reason = 'local_overlap'
            if reason:
                rejected.append(dict(edge_id=e, reason=reason)); continue
            influence = (old_weights[ids]*((old_indices[ids] == left)|(old_indices[ids] == right))).sum(1)
            q = influence*self.gaussian_score[ids].clamp_min(self.config.eps)
            centroid = (self.model._xyz.detach()[ids]*q[:, None]).sum(0)/q.sum()
            chosen = ids[(self.model._xyz.detach()[ids]-centroid).square().sum(1).argmin()]
            position = self.model._xyz.detach()[chosen].clone()
            spacing = (nodes[left]-nodes[right]).norm().clamp_min(1e-4)
            if (nodes-position).norm(dim=1).min() < self.config.min_child_separation*spacing:
                rejected.append(dict(edge_id=e, reason='min_child_distance')); continue
            child = old_n+len(positions)
            candidate_nodes = torch.cat((nodes, torch.stack(positions+[position])))
            candidate = torch.cat((old_indices[ids], old_indices.new_full((len(ids), 1), child)), 1)
            distance = (self.model._xyz.detach()[ids, None]-candidate_nodes[candidate]).norm(dim=-1)
            distance, choice = distance.topk(4, largest=False, sorted=True)
            new_idx = candidate.gather(1, choice)
            inverse = (distance+1e-8).reciprocal()
            new_w = inverse/inverse.sum(1, keepdim=True)
            controlled = (new_idx == child).any(1) & ((new_idx == left)|(new_idx == right)).any(1)
            child_mass = (new_w*(new_idx == child)).sum(1)
            if int(((child_mass > self.config.weight_threshold)&controlled).sum()) < self.config.min_coverage:
                rejected.append(dict(edge_id=e, reason='insufficient_child_control')); continue
            ids, new_idx, new_w = ids[controlled], new_idx[controlled], new_w[controlled]
            # Same existing deformation-shock limit, checked on the FULL target
            # operator, not hidden by activation=0. Preview never mutates model.
            preview = copy.copy(residual)
            preview._buffers = residual._buffers.copy()
            preview.set_iteration(iteration)
            preview.refine(position[None], nodes.new_tensor([[left, right]], dtype=torch.long),
                ids, old_indices[ids], old_weights[ids])
            preview.set_iteration(iteration+250)
            local_idx = new_idx.clone()
            local_idx[local_idx == child] = old_n
            touched = ((old_indices == left)|(old_indices == right)).any(1)
            touched[ids] = True
            touched_rows = touched.nonzero().flatten()
            ix, wt = old_indices[touched_rows].clone(), old_weights[touched_rows].clone()
            slot_map = torch.searchsorted(touched_rows, ids)
            ix[slot_map], wt[slot_map] = local_idx, new_w
            worst = 0.
            for time, base, old_motion in zip(self.reference['trajectory_times'], self._event_base, self._event_motion):
                child_base = self.model.node_mlp(position[None], time)[0]
                new_motion = preview(torch.cat((base, child_base)))[0]
                old_dx = (old_motion[old_indices[touched_rows]]*old_weights[touched_rows, :, None]).sum(1)
                new_dx = (new_motion[ix]*wt[..., None]).sum(1)
                worst = max(worst, float((new_dx-old_dx).norm(dim=1).max()))
            if worst > self.config.max_split_shock_ratio*self.shock_reference_scale:
                rejected.append(dict(edge_id=e, reason='full_target_shock', shock_max=worst)); continue
            occupied_nodes.update((left, right)); occupied_rows[ids] = True
            target_idx[ids], target_w[ids] = new_idx, new_w
            parents.append((left, right)); positions.append(position)
            accepted.append(dict(**records[e], edge_id=e, motion_complexity=float(complexity[e]),
                geometry_difficulty=float(error[e]), refine_score=float(scores[e]),
                edge_length=float(spacing), new_node_id=child, new_node_position=position.tolist(),
                placement_method='nearest_valid_Gaussian_to_depth_weighted_centroid',
                child_support=int(((child_mass > self.config.weight_threshold)&controlled).sum()),
                child_influence=float(child_mass[controlled].sum()), degree_of_X=2,
                full_target_shock_max=worst,
                removed_edge=[left, right], added_edges=[[left, child], [right, child]]))
        changed = (target_idx != old_indices).any(1) | (target_w != old_weights).any(1)
        rows = changed.nonzero().flatten()
        times = self.reference['trajectory_times']
        before = [self.model.deform_geometry_from_nodes(t) for t in times]
        before_motion = torch.stack([self.model._query_node_motion(t)[0] for t in times])
        before_edges, before_weights = residual.edge_indices.clone(), residual.edge_weights.clone()
        decoder_ids = [id(p) for p in residual.parameters()]
        if positions:
            parent_tensor = nodes.new_tensor(parents, dtype=torch.long)
            new_xyz = torch.stack(positions)
            removed = residual.refine(new_xyz, parent_tensor, rows, old_indices[rows], old_weights[rows])
            self.model.deformation_graph = SparseDeformationGraph(torch.cat((nodes, new_xyz)), target_idx, target_w)
            self.node_ids = torch.arange(old_n+len(positions), device=nodes.device)
            self.parent_ids = torch.cat((self.parent_ids, parent_tensor[:, 0]))
            self.birth_iterations = torch.cat((self.birth_iterations,
                self.birth_iterations.new_full((len(positions),), iteration)))
            self.prune_counters = torch.cat((self.prune_counters, self.prune_counters.new_zeros(len(positions))))
            self.next_node_id = len(self.node_ids)
            if not torch.equal(residual.edge_indices[:, :int((~removed).sum())], before_edges[:, ~removed]):
                raise AssertionError('Unrelated topology changed')
            if not torch.equal(residual.edge_weights[:int((~removed).sum())], before_weights[~removed]):
                raise AssertionError('Unrelated graph weights changed')
        displacement, scale_error, rotation_error = [], [], []
        for t, old_values in zip(times, before):
            new_values = self.model.deform_geometry_from_nodes(t)
            displacement.append((new_values[0]-old_values[0]).norm(dim=1))
            scale_error.append(float((new_values[1]-old_values[1]).abs().max()))
            rotation_error.append(float((new_values[2]-old_values[2]).abs().max()))
        delta = torch.stack(displacement)
        if float(delta.max()) > 1e-4*self.shock_reference_scale:
            raise RuntimeError('Insertion continuity failed; abort before more optimizer updates')
        if decoder_ids != [id(p) for p in residual.parameters()]: raise AssertionError('Decoder Adam ownership changed')
        if not torch.equal(target_idx[~changed], old_indices[~changed]) or not torch.equal(target_w[~changed], old_weights[~changed]):
            raise AssertionError('Unrelated K4 changed')
        if not torch.equal(self.model.appearance_graph.gaussian_node_indices, self.appearance_indices):
            raise AssertionError('Appearance graph changed')
        if not torch.equal(self.model.appearance_graph.gaussian_node_weights, self.appearance_weights):
            raise AssertionError('Appearance weights changed')
        if not torch.equal(self.graph.node_xyz[:2000], self.backbone): raise AssertionError('Backbone moved')
        event = dict(iteration=iteration, nodes_before=old_n, nodes_after=len(self.node_ids),
            num_surface_edges=int(before_edges.shape[1]), num_candidate_edges=len(positive),
            num_selected_edges=len(accepted), splits=len(accepted), prunes=0,
            requested_quota=quota, quota_shortfall=quota-len(accepted),
            motion_complexity=distribution(complexity), geometry_difficulty=distribution(error),
            refine_score=distribution(scores), accepted_edges=accepted, rejected_candidates=rejected,
            topology_edges_removed=len(accepted), topology_edges_added=2*len(accepted),
            changed_k4_percent=float(changed.float().mean())*100,
            outside_affected_physical_k4_exact=True, outside_affected_weights_exact=True,
            appearance_graph_fixed_2000=True, original_backbone_exact=True,
            continuity=distribution(delta), scale_max_error=max(scale_error), rotation_max_error=max(rotation_error),
            birth_motion='endpoint interpolation, then shared XYZ MLP over250 updates',
            activation_steps=250, shock_mean=float(delta.mean()), shock_max=float(delta.max()),
            geometry_signal='depth loss gradient w.r.t. transformed Gaussian centers; no RGB',
            previous_operator_motion_p95=float(torch.quantile(before_motion.norm(dim=-1), .95)))
        self._record(event)
        self.save(str(Path(self.output_dir)/'adaptive_state_latest.pth'), iteration)
        return event

    def state_dict(self, iteration):
        state = super().state_dict(iteration)
        state['surface_refine'] = dict(format=SurfaceRefinedResidual.FORMAT,
            residual=self.model.graph_motion_residual.state_dict_payload(), quotas=self.quotas)
        return state

    def load_state_dict(self, state):
        saved = state['surface_refine']
        if saved['format'] != SurfaceRefinedResidual.FORMAT or saved['quotas'] != self.quotas:
            raise ValueError('Surface refinement policy changed')
        active = self.model.graph_motion_residual
        super().load_state_dict(copy.deepcopy(state))
        restored = SurfaceRefinedResidual.from_surface_payload(saved['residual'], self.model._xyz.device)
        restored.decoder = active.decoder
        restored.decoder.load_state_dict(saved['residual']['decoder'])
        self.model.graph_motion_residual = restored
