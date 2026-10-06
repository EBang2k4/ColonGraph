"""Opt-in local node growth for G3, with matched score/placement ablations."""
from __future__ import annotations

import copy
import torch

from utils.adaptive_deformation_graph import AdaptiveNodeManager


def percentile_rank(values):
    """Tie-aware empirical ranks; constant blocks have rank zero."""
    if not len(values):
        return values.clone()
    unique, inverse, counts = torch.unique(values, sorted=True, return_inverse=True,
                                          return_counts=True)
    if len(unique) == 1:
        return torch.zeros_like(values)
    lower = counts.cumsum(0) - counts
    ranks = lower.float() + (counts.float() - 1) / 2
    return (ranks[inverse] / max(len(values) - 1, 1)).to(values)


class AdaptiveG3NodeManager(AdaptiveNodeManager):
    def __init__(self, model, config, output_dir, score="current", placement="centroid"):
        super().__init__(model, config, output_dir, allow_grown=True)
        if config.mode != "split" or model.graph_motion_residual.mode != "g3":
            raise ValueError("adaptive G3 supports split-only G3")
        if not self.appearance_graph_fixed:
            raise ValueError("appearance graph must remain fixed at its original 2000 nodes")
        if score not in ("current", "motion") or placement not in ("centroid", "motion_cluster"):
            raise ValueError("unsupported G3 adaptive score/placement")
        self.score_mode, self.placement_mode = score, placement
        self.direction_sum = torch.zeros_like(model._xyz)
        self.direction_count = torch.zeros(len(model._xyz), device=model._xyz.device)
        self.last_split = torch.full_like(self.node_ids, -100000)
        self.reference = model.graph_forward_reference_data
        self.surface_weights = self.reference["surface_edge_weights"].detach().clone()
        self._appearance_indices = model.appearance_graph.gaussian_node_indices.clone()
        self._appearance_weights = model.appearance_graph.gaussian_node_weights.clone()
        self._backbone = self.graph.node_xyz[:2000].clone()
        self._event_stats, self._proposal_logs, self._edge_logs = None, {}, []
        self._event_trajectory, self._preview_cache = None, {}

    @torch.no_grad()
    def update_statistics(self, gradient, visible):
        # Receives d L_RGB_reconstruction / d transformed Gaussian XYZ from
        # autograd.grad, never a node/graph/KNN parameter gradient.
        valid = visible & torch.isfinite(gradient).all(1)
        magnitude = gradient.norm(dim=1)
        valid &= torch.isfinite(magnitude) & (magnitude > 0)
        self.direction_sum[valid] += gradient[valid] / magnitude[valid, None]
        self.direction_count[valid] += 1
        super().update_statistics(gradient, visible)

    @torch.no_grad()
    def node_statistics(self, iteration):
        stats = super().node_statistics(iteration)
        m = len(self.graph.node_xyz)
        vector = self.direction_sum.new_zeros((m, 3))
        count = self.direction_sum.new_zeros(m)
        idx, weights = self.graph.gaussian_node_indices, self.graph.gaussian_node_weights
        for slot in range(4):
            vector.index_add_(0, idx[:, slot], weights[:, slot, None] * self.direction_sum)
            count.index_add_(0, idx[:, slot], weights[:, slot] * self.direction_count)
        heterogeneity = (1 - (vector / count[:, None].clamp_min(1)).norm(dim=1)).clamp(0, 1)
        heterogeneity = torch.where(count > 0, heterogeneity, torch.zeros_like(heterogeneity))
        eligible = stats["eligible"]
        ranks = []
        for value in (stats["geometry"], heterogeneity, stats["influence"]):
            rank = torch.zeros_like(value)
            rank[eligible] = percentile_rank(value[eligible])
            ranks.append(rank)
        composite = .50 * ranks[0] + .35 * ranks[1] + .15 * ranks[2]
        stats.update(heterogeneity=heterogeneity, observation_mass=count,
                     rank_a=ranks[0], rank_b=ranks[1], rank_c=ranks[2], composite=composite)
        self._event_stats, self._iteration = stats, iteration
        return stats

    def _split_candidates(self, stats):
        if self.score_mode == "current":
            return super()._split_candidates(stats)
        # Keep the original high-difficulty gate. H>=0.10 means >=10% loss of
        # directional agreement. Skip the next event after a successful split.
        mask = (stats["eligible"] & (stats["normalized"] >= self.config.split_z) &
                (stats["heterogeneity"] >= .10) &
                ((self._iteration - self.last_split) > self.config.interval))
        ids = mask.nonzero().flatten()
        return ids[torch.argsort(stats["composite"][ids], descending=True, stable=True)]

    def _node_translation(self, time):
        return self.model._query_node_motion(time)[0]

    @torch.no_grad()
    def _proposal(self, parent, stats):
        if self.placement_mode == "centroid":
            return super()._proposal(parent, stats)
        xyz = self.model._xyz.detach()
        membership = self.graph.gaussian_node_indices == parent
        ids = (membership.any(1) & (self.gaussian_observations >= 5)).nonzero().flatten()
        if len(ids) < self.config.min_coverage:
            return None
        slot = membership[ids].long().argmax(1)
        q = (self.graph.gaussian_node_weights[ids, slot] *
             self.gaussian_score[ids].clamp_min(self.config.eps))
        direction = self.direction_sum[ids] / self.direction_count[ids, None].clamp_min(1)
        direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(self.config.eps)
        blocks = []
        for block in (xyz[ids], direction):
            center = block.median(0).values
            scale = (block-center).abs().median(0).values.median().clamp_min(1e-5)
            blocks.append(((block-center) / scale).clamp(-5, 5))
        descriptor = torch.cat(blocks, dim=1)
        mean = (descriptor*q[:, None]).sum(0)/q.sum()
        first = int((descriptor-mean).square().sum(1).argmax())
        second = int((descriptor-descriptor[first]).square().sum(1).argmax())
        centers = descriptor[[first, second]].clone()
        for _ in range(12):
            label = torch.cdist(descriptor, centers).argmin(1)
            if not label.any() or label.all():
                return None
            for k in range(2):
                mask = label == k
                centers[k] = (descriptor[mask]*q[mask, None]).sum(0)/q[mask].sum()
        counts = [(label == k).sum() for k in range(2)]
        if min(int(n) for n in counts) < max(8, self.config.min_coverage // 4):
            return None
        motion = torch.stack([(direction[label == k]*q[label == k, None]).sum(0)/
                              q[label == k].sum() for k in range(2)])
        separation = (motion[0]-motion[1]).norm()
        if separation < .10:
            return None
        dominant = (direction*q[:, None]).sum(0)/q.sum()
        difficulty = torch.stack([(self.gaussian_score[ids[label == k]] * q[label == k]).sum()/
                                  q[label == k].sum() for k in range(2)])
        choice = int((.5 * percentile_rank(difficulty) +
                      .5 * percentile_rank((motion-dominant).norm(dim=1))).argmax())
        selected = label == choice
        centroid = (xyz[ids[selected]]*q[selected, None]).sum(0)/q[selected].sum()
        minimum = (self.config.min_child_separation *
                   (xyz[ids]-self.graph.node_xyz[parent]).norm(dim=1).median().clamp_min(1e-4))
        if torch.cdist(centroid[None], self.graph.node_xyz).min() < minimum:
            return None
        self._proposal_logs[parent] = dict(
            cluster_sizes=[int(n) for n in counts], cluster_motion_separation=float(separation),
            selected_cluster=choice, cluster_difficulty=difficulty.tolist(),
            child_to_parent_distance=float((centroid-self.graph.node_xyz[parent]).norm()),
            child_to_cluster_centroid_distance=0.0)
        return centroid, ids[selected], minimum

    @torch.no_grad()
    def _local_edges(self, parents, positions):
        residual = self.model.graph_motion_residual
        old_n = len(residual.canonical_xyz)
        times = self.reference["trajectory_times"]
        if self._event_trajectory is None:
            self._event_trajectory = torch.stack([self._node_translation(t) for t in times])
        new_xyz = torch.stack(positions)
        child_trajectory = torch.stack([self.model.node_mlp(new_xyz, t)[0] for t in times])
        edges, weights, surface_weights, records = [], [], [], []
        sigma = float(self.reference["audit"]["trajectory_sigma"])
        source, target = residual.edge_indices
        for offset, (parent, position) in enumerate(zip(parents, positions)):
            incident = (source == parent) | (target == parent)
            edge_rows = incident.nonzero().flatten()
            neighbors = torch.where(source[edge_rows] == parent, target[edge_rows], source[edge_rows])
            spacing = (residual.canonical_xyz[neighbors]-residual.canonical_xyz[parent]).norm(dim=1).median().clamp_min(1e-8)
            parent_cost = (position-residual.canonical_xyz[parent]).norm() / spacing
            # Frozen surface-path extension via the parent, never Euclidean
            # shortcuts between old nodes. Same exp(-cost²/2) surface kernel.
            old_cost = (-2*self.surface_weights[edge_rows].clamp_min(1e-12).log()).sqrt()
            costs = torch.cat((parent_cost[None], parent_cost + old_cost))
            candidates = torch.cat((neighbors.new_tensor([parent]), neighbors))
            ungated = torch.exp(-.5*costs.square()).clamp_min(1e-12)
            distances = (self._event_trajectory[:, candidates]-child_trajectory[:, offset, None]).square().sum(2).mean(0).sqrt()
            similarity = torch.exp(-distances.square()/(2*max(sigma, 1e-8)**2))
            gated = (ungated*similarity).clamp_min(1e-12)
            child = old_n + offset
            edges.append(torch.stack((candidates, torch.full_like(candidates, child))))
            weights.append(gated); surface_weights.append(ungated)
            records.append(dict(parent_id=int(self.node_ids[parent]), child_id=self.next_node_id+offset,
                                neighbors=candidates.tolist(), surface_weights=ungated.tolist(),
                                trajectory_distance=distances.tolist(), similarity=similarity.tolist(),
                                gated_weights=gated.tolist(), trajectory_sigma=sigma))
        return torch.cat(edges, 1), torch.cat(weights), torch.cat(surface_weights), records

    @torch.no_grad()
    def _candidate_shock(self, parent, position, time, old_node_dx):
        edges, weights, surface, records = self._local_edges([parent], [position])
        # Shallow module copy with independent buffers: no mutation, no random
        # initialization, and the exact same trained decoder/parameters.
        preview = copy.copy(self.model.graph_motion_residual)
        preview._buffers = preview._buffers.copy()
        preview.append_local_nodes(position[None], edges, weights)
        nodes = preview.canonical_xyz
        base = self.model.node_mlp(nodes, time)[0]
        new_motion, _ = preview(base)
        affected, rows, indices, local_weights = self._local_child_update(
            self.graph.gaussian_node_indices, self.graph.gaussian_node_weights,
            nodes, parent, position, len(nodes)-1)
        # New local messages also alter old incident nodes, even when a
        # Gaussian keeps its K4. Include those Gaussians in the shock test.
        incident_nodes = edges[0].unique()
        motion_affected = affected | torch.isin(self.graph.gaussian_node_indices, incident_nodes).any(1)
        old_dx = (old_node_dx[self.graph.gaussian_node_indices[motion_affected]] *
                  self.graph.gaussian_node_weights[motion_affected, :, None]).sum(1)
        all_indices = self.graph.gaussian_node_indices[motion_affected].clone()
        all_weights = self.graph.gaussian_node_weights[motion_affected].clone()
        local_rows = affected[motion_affected]
        all_indices[local_rows], all_weights[local_rows] = indices, local_weights
        new_dx = (new_motion[all_indices]*all_weights[:, :, None]).sum(1)
        shock = (new_dx-old_dx).norm(dim=1)
        return affected, float(shock.mean()), float(shock.max())

    @torch.no_grad()
    def _append_children(self, parents, positions):
        edges, weights, surface, records = self._local_edges(parents, positions)
        residual = self.model.graph_motion_residual
        original_edges, original_weights = residual.edge_indices.clone(), residual.edge_weights.clone()
        decoder_ids = [id(p) for p in residual.parameters()]
        old_n = len(self.node_ids)
        affected, old_ids = super()._append_children(parents, positions)
        residual.append_local_nodes(torch.stack(positions), edges, weights)
        if not (torch.equal(residual.edge_indices[:, :len(original_weights)], original_edges) and
                torch.equal(residual.edge_weights[:len(original_weights)], original_weights)):
            raise AssertionError("existing G3 graph edges/weights changed")
        if decoder_ids != [id(p) for p in residual.parameters()]:
            raise AssertionError("G3 decoder/optimizer parameter identity changed")
        self.surface_weights = torch.cat((self.surface_weights, surface))
        self.last_split = torch.cat((self.last_split, self.last_split.new_full((len(parents),), -100000)))
        self.last_split[parents] = self._iteration
        self._edge_logs = records
        self._birth_parents, self._birth_old_n = parents, old_n
        return affected, old_ids

    def _record(self, event):
        stats = self._event_stats
        event.update(score_mode=self.score_mode, placement_mode=self.placement_mode,
                     graph_edges=int(self.model.graph_motion_residual.edge_indices.shape[1]),
                     old_graph_edges_weights_exact=True, graph_local_edge_additions=self._edge_logs,
                     appearance_graph_fixed_2000=True,
                     birth_initialization="existing shared XYZ MLP; no per-node motion parameters or Adam rows",
                     cluster_proposals={str(int(self.node_ids[p])): value for p, value in self._proposal_logs.items()})
        if stats is not None:
            for name, value in (("score_A", stats["geometry"]), ("score_B", stats["heterogeneity"]),
                                ("score_C", stats["influence"]), ("composite_score", stats["composite"])):
                event[name] = self._distribution(value)
                event[name]["p90"] = float(torch.quantile(value.float(), .90))
            for selected in event.get("selected_parent_statistics", []):
                p = int((self.node_ids == selected["node_id"]).nonzero()[0])
                selected["motion_heterogeneity"] = float(stats["heterogeneity"][p])
                selected["composite_score"] = float(stats["composite"][p])
        if not (torch.equal(self.model.appearance_graph.gaussian_node_indices, self._appearance_indices) and
                torch.equal(self.model.appearance_graph.gaussian_node_weights, self._appearance_weights) and
                torch.equal(self.graph.node_xyz[:len(self._backbone)], self._backbone)):
            raise AssertionError("appearance graph or FPS backbone changed")
        return super()._record(event)

    @torch.no_grad()
    def adapt(self, *args, **kwargs):
        self._event_trajectory, self._proposal_logs, self._edge_logs = None, {}, []
        event = super().adapt(*args, **kwargs)
        # Legacy save occurs before _record; persist the complete event metadata
        # as well, so restoring the latest state retains the just-applied split.
        self.save(self.output_dir + "/adaptive_state_latest.pth", event["iteration"])
        return event

    def state_dict(self, iteration):
        state = super().state_dict(iteration)
        state["adaptive_g3"] = dict(score=self.score_mode, placement=self.placement_mode,
            direction_sum=self.direction_sum.cpu(), last_split=self.last_split.cpu(),
            direction_count=self.direction_count.cpu(),
            surface_weights=self.surface_weights.cpu(),
            residual=self.model.graph_motion_residual.state_dict_payload())
        return state

    @torch.no_grad()
    def load_state_dict(self, state):
        saved = state["adaptive_g3"]
        if (saved["score"], saved["placement"]) != (self.score_mode, self.placement_mode):
            raise ValueError("adaptive G3 score/placement changed on resume")
        super().load_state_dict(state)
        from utils.graph_forward import GraphMotionResidual
        restored = GraphMotionResidual.from_state_dict_payload(saved["residual"], self.model._xyz.device)
        active = self.model.graph_motion_residual
        # Keep optimizer-owned decoder parameter objects, restore only buffers
        # and parameter values. This avoids silently orphaning Adam state.
        active.decoder.load_state_dict(restored.decoder.state_dict())
        active._buffers = restored._buffers.copy()
        self.direction_sum = saved["direction_sum"].to(self.model._xyz)
        self.direction_count = saved["direction_count"].to(self.direction_count)
        self.last_split = saved["last_split"].to(self.node_ids)
        self.surface_weights = saved["surface_weights"].to(active.edge_weights)
        if not torch.equal(active.canonical_xyz, self.graph.node_xyz):
            raise ValueError("restored graph residual/node positions mismatch")
