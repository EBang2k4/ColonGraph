"""Local, forward-audited adaptation of a sparse deformation graph."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import torch
from torch import nn

from scene.deformation_graph import SparseDeformationGraph
from utils.image_utils import psnr
from utils.loss_utils import lpips_loss, ssim


@dataclass
class AdaptiveNodeConfig:
    mode: str = "none"
    start: int = 5000
    end: int = 20000
    interval: int = 2500
    split_end: int = 20000
    prune_start: int = 22500
    max_nodes: int = 2400
    min_nodes: int = 1600
    max_splits: int = 60
    max_prunes: int = 40
    prune_windows: int = 2
    prune_quantile: float = 0.20
    prune_redundancy_distance_ratio: float = 1.0
    prune_redundancy_overlap: float = 0.50
    prune_motion_error_ratio: float = 0.01
    prune_safe_error_ratio: float = 0.002
    score_decay: float = 0.99
    running_decay: float = 0.9
    split_z: float = 1.0
    min_coverage: int = 32
    min_age: int = 5000
    weight_threshold: float = 0.05
    min_child_separation: float = 0.1
    max_split_shock_ratio: float = 0.01
    eps: float = 1e-8

    def validate(self):
        if self.mode not in ("none", "stats", "split_once", "split", "prune_once",
                             "split_prune", "split_prune_conservative",
                             "split_prune_late"):
            raise ValueError("unsupported adaptive node mode")
        if not (0 < self.start <= self.end):
            raise ValueError("invalid adaptive interval")
        if self.interval < 1 or self.max_splits < 1 or self.max_prunes < 1:
            raise ValueError("interval, max_splits, and max_prunes must be positive")
        if not (self.min_nodes <= 2000 <= self.max_nodes):
            raise ValueError("node budget must contain the 2,000-node backbone")
        if not (0 <= self.score_decay < 1 and 0 <= self.running_decay < 1):
            raise ValueError("EMA decays must be in [0,1)")
        if self.max_split_shock_ratio <= 0:
            raise ValueError("max_split_shock_ratio must be positive")
        if self.prune_windows < 2 or not (0 < self.prune_quantile < 0.5):
            raise ValueError("pruning needs >=2 windows and a quantile in (0,0.5)")
        if self.prune_redundancy_distance_ratio <= 0:
            raise ValueError("prune redundancy distance ratio must be positive")
        if not (0 < self.prune_redundancy_overlap <= 1):
            raise ValueError("prune redundancy overlap must be in (0,1]")
        if self.prune_motion_error_ratio <= 0 or self.prune_safe_error_ratio <= 0:
            raise ValueError("prune motion/safe error ratios must be positive")
        if self.mode == "split_prune_late" and not (
                self.start <= self.split_end < self.prune_start <= self.end):
            raise ValueError("late prune requires start <= split_end < prune_start <= end")


def _replace_parameter_rows(optimizer, module, name, source_rows=None, keep_rows=None):
    """Resize a row-parameter while preserving old Adam state.

    Appended rows copy parameter values from ``source_rows`` and receive zero
    optimizer moments.  Pruning support is reserved for S3 through keep_rows.
    """
    old = getattr(module, name)
    if (source_rows is None) == (keep_rows is None):
        raise ValueError("specify exactly one of source_rows or keep_rows")
    if source_rows is not None:
        source_rows = torch.as_tensor(source_rows, device=old.device, dtype=torch.long)
        value = torch.cat((old.detach(), old.detach()[source_rows]), dim=0)
        old_rows = len(old)
    else:
        keep_rows = torch.as_tensor(keep_rows, device=old.device, dtype=torch.long)
        value = old.detach()[keep_rows]
        old_rows = None
    new = nn.Parameter(value.clone(), requires_grad=old.requires_grad)
    found = False
    for group in optimizer.param_groups:
        for index, parameter in enumerate(group["params"]):
            if parameter is old:
                group["params"][index] = new
                found = True
    if not found:
        raise RuntimeError(f"optimizer does not own {name}")
    state = optimizer.state.pop(old, {})
    resized = {}
    for key, item in state.items():
        if not torch.is_tensor(item) or item.ndim == 0:
            resized[key] = item.clone() if torch.is_tensor(item) else item
        elif item.shape == old.shape:
            if source_rows is not None:
                resized[key] = torch.cat((item, torch.zeros_like(item[source_rows])), dim=0)
            else:
                resized[key] = item[keep_rows].clone()
        else:
            resized[key] = item.clone()
    optimizer.state[new] = resized
    setattr(module, name, new)
    return old, new, old_rows


def _replace_parameter_value(optimizer, module, name, value):
    """Replace one optimizer-owned parameter without changing group ordering."""
    old = getattr(module, name)
    new = nn.Parameter(value.detach().to(old).clone(), requires_grad=old.requires_grad)
    found = False
    for group in optimizer.param_groups:
        for index, parameter in enumerate(group["params"]):
            if parameter is old:
                group["params"][index] = new
                found = True
    if not found:
        raise RuntimeError(f"optimizer does not own {name}")
    optimizer.state.pop(old, None)
    setattr(module, name, new)
    return new


class AdaptiveNodeManager:
    def __init__(self, model, config: AdaptiveNodeConfig, output_dir: str):
        config.validate()
        if model.deformation_graph.gaussian_node_indices.shape[1] != 4:
            raise ValueError("adaptive graph requires K=4")
        if model.appearance_model is None or not hasattr(model.appearance_model, "node_appearance"):
            raise ValueError("adaptive graph requires R6 node appearance")
        if (len(model.deformation_graph.node_xyz) != 2000 and
                config.mode != "prune_once"):
            raise ValueError("adaptive graph must start from exactly 2,000 FPS nodes")
        self.model = model
        self.config = config
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        n_gaussians = len(model._xyz)
        device = model._xyz.device
        self.gaussian_score = torch.zeros(n_gaussians, device=device)
        self.gaussian_observations = torch.zeros(n_gaussians, device=device, dtype=torch.long)
        initial_nodes = len(model.deformation_graph.node_xyz)
        self.node_ids = torch.arange(initial_nodes, device=device, dtype=torch.long)
        self.parent_ids = torch.full((initial_nodes,), -1, device=device, dtype=torch.long)
        self.birth_iterations = torch.zeros(initial_nodes, device=device, dtype=torch.long)
        self.prune_counters = torch.zeros(initial_nodes, device=device, dtype=torch.long)
        self.next_node_id = initial_nodes
        self.running_log_mean = None
        self.running_log_var = None
        self.events = []
        self._single_split_done = False
        self._single_prune_done = False
        self.log_path = os.path.join(output_dir, "adaptive_events.jsonl")
        self.motion_log_path = os.path.join(output_dir, "parent_child_motion.jsonl")
        self.split_pairs = []
        self.motion_diagnostics = []
        self.motion_diagnostic_offsets = (500, 1000, 2500)
        self.motion_diagnostic_times = (0.0, 0.25, 0.5, 0.75, 1.0)

    @property
    def graph(self):
        return self.model.deformation_graph

    @property
    def appearance_graph_fixed(self):
        return getattr(self.model, "appearance_graph", None) is not None

    @property
    def shock_reference_scale(self):
        scale = getattr(getattr(self.model, "node_mlp", None), "position_scale", None)
        if scale is not None and float(scale) > 1:
            return float(scale)
        extent = self.model._xyz.detach().amax(0) - self.model._xyz.detach().amin(0)
        return float((extent.amax() / 2).clamp_min(1e-8))

    def is_event(self, iteration):
        cfg = self.config
        return cfg.start <= iteration <= cfg.end and (iteration - cfg.start) % cfg.interval == 0

    @torch.no_grad()
    def update_statistics(self, gradient, visible):
        if gradient.shape != self.model._xyz.shape or visible.shape != (len(gradient),):
            raise ValueError("invalid adaptive gradient shapes")
        finite = visible & torch.isfinite(gradient).all(dim=1)
        magnitude = gradient.norm(dim=1)
        finite &= torch.isfinite(magnitude)
        old = self.gaussian_score[finite]
        decay = self.config.score_decay
        first = self.gaussian_observations[finite] == 0
        updated = old * decay + magnitude[finite] * (1 - decay)
        self.gaussian_score[finite] = torch.where(first, magnitude[finite], updated)
        self.gaussian_observations[finite] += 1

    @torch.no_grad()
    def node_statistics(self, iteration):
        graph = self.graph
        m = len(graph.node_xyz)
        indices, weights = graph.gaussian_node_indices, graph.gaussian_node_weights
        influence = torch.zeros(m, device=weights.device)
        weighted_error = torch.zeros_like(influence)
        coverage = torch.zeros(m, device=weights.device, dtype=torch.long)
        observed = self.gaussian_observations > 0
        for column in range(4):
            node = indices[:, column]
            weight = weights[:, column]
            influence.scatter_add_(0, node, weight)
            weighted_error.scatter_add_(0, node, weight * self.gaussian_score)
            coverage.scatter_add_(0, node, ((weight > self.config.weight_threshold) & observed).long())
        geometry = weighted_error / influence.clamp_min(self.config.eps)
        log_geometry = torch.log(geometry.clamp_min(self.config.eps))
        eligible = (coverage >= self.config.min_coverage) & ((iteration - self.birth_iterations) >= self.config.min_age)
        values = log_geometry[eligible]
        if len(values):
            mean, var = values.mean(), values.var(unbiased=False).clamp_min(1e-12)
            if self.running_log_mean is None:
                self.running_log_mean, self.running_log_var = mean, var
            else:
                d = self.config.running_decay
                self.running_log_mean = d * self.running_log_mean + (1-d) * mean
                self.running_log_var = d * self.running_log_var + (1-d) * var
        mean = log_geometry.new_tensor(0.) if self.running_log_mean is None else self.running_log_mean
        var = log_geometry.new_tensor(1.) if self.running_log_var is None else self.running_log_var
        normalized = (log_geometry - mean) / var.sqrt().clamp_min(1e-6)
        return dict(geometry=geometry, influence=influence, coverage=coverage,
                    normalized=normalized, eligible=eligible)

    @torch.no_grad()
    def _proposal(self, parent, stats):
        graph, xyz = self.graph, self.model._xyz.detach()
        membership = graph.gaussian_node_indices == parent
        support = membership.any(dim=1) & (self.gaussian_observations > 0)
        ids = support.nonzero(as_tuple=False).flatten()
        if len(ids) < self.config.min_coverage:
            return None
        slot = membership[ids].float().argmax(dim=1)
        parent_weight = graph.gaussian_node_weights[ids, slot]
        error = self.gaussian_score[ids]
        q = parent_weight * error.clamp_min(self.config.eps)
        centroid = (xyz[ids] * q[:, None]).sum(0) / q.sum().clamp_min(self.config.eps)
        parent_xyz = graph.node_xyz[parent]
        centered = xyz[ids] - centroid
        covariance = (centered * q[:, None]).T @ centered / q.sum().clamp_min(self.config.eps)
        _, eigenvectors = torch.linalg.eigh(covariance)
        axis = eigenvectors[:, -1]
        spread = ((centered @ axis).square() * q).sum().sqrt() / q.sum().sqrt().clamp_min(self.config.eps)
        local_scale = (xyz[ids] - parent_xyz).norm(dim=1).median().clamp_min(1e-4)
        minimum = self.config.min_child_separation * local_scale
        if (centroid - parent_xyz).norm() < minimum:
            sign = torch.sign(((xyz[ids] - parent_xyz) @ axis * q).sum())
            if sign == 0:
                sign = axis.new_tensor(1.)
            centroid = parent_xyz + sign * axis * torch.maximum(spread, minimum)
        nearest = torch.cdist(centroid[None], graph.node_xyz).min()
        if nearest < minimum:
            return None
        return centroid, ids, minimum

    @torch.no_grad()
    def _local_child_update(self, indices, weights, nodes, parent, child_xyz, child):
        support = (indices == parent).any(dim=1)
        current_radius = (self.model._xyz.detach()[:, None, :] - nodes[indices]).norm(dim=2).amax(dim=1)
        child_distance = (self.model._xyz.detach() - child_xyz).norm(dim=1)
        affected = support | (child_distance < current_radius)
        rows = affected.nonzero(as_tuple=False).flatten()
        candidates = torch.cat((indices[rows], torch.full(
            (len(rows), 1), child, device=indices.device, dtype=torch.long)), dim=1)
        distance = (self.model._xyz.detach()[rows, None, :] - nodes[candidates]).norm(dim=2)
        distance, choice = distance.topk(4, largest=False, sorted=True)
        new_indices = candidates.gather(1, choice)
        inverse = (distance + 1e-8).reciprocal()
        new_weights = inverse / inverse.sum(dim=1, keepdim=True)
        return affected, rows, new_indices, new_weights

    @torch.no_grad()
    def _candidate_shock(self, parent, position, time, old_node_dx):
        graph = self.graph
        child = len(graph.node_xyz)
        nodes = torch.cat((graph.node_xyz, position[None]), dim=0)
        affected, rows, indices, weights = self._local_child_update(
            graph.gaussian_node_indices, graph.gaussian_node_weights,
            nodes, parent, position, child)
        child_dx = self.model.node_mlp(position[None], time)[0]
        motion = torch.cat((old_node_dx, child_dx), dim=0)
        old_dx = (old_node_dx[graph.gaussian_node_indices[rows]] *
                  graph.gaussian_node_weights[rows, :, None]).sum(1)
        new_dx = (motion[indices] * weights[:, :, None]).sum(1)
        shock = (new_dx - old_dx).norm(dim=1)
        return affected, float(shock.mean()), float(shock.max())

    @torch.no_grad()
    def _append_children(self, parents, positions):
        graph = self.graph
        old_indices = graph.gaussian_node_indices.clone()
        old_weights = graph.gaussian_node_weights.clone()
        old_ids_per_gaussian = self.node_ids[old_indices]
        nodes = torch.cat((graph.node_xyz, torch.stack(positions)), dim=0)
        indices, weights = old_indices.clone(), old_weights.clone()
        affected_total = torch.zeros(len(indices), device=indices.device, dtype=torch.bool)
        for offset, (parent, child_xyz) in enumerate(zip(parents, positions)):
            child = len(graph.node_xyz) + offset
            affected, rows, new_indices, new_weights = self._local_child_update(
                indices, weights, nodes, parent, child_xyz, child)
            affected_total |= affected
            indices[rows] = new_indices
            weights[rows] = new_weights
        outside = ~affected_total
        if not torch.equal(self.node_ids[indices[outside]], old_ids_per_gaussian[outside]):
            raise AssertionError("Gaussian outside split affected-set changed physical K4")
        if not torch.equal(weights[outside], old_weights[outside]):
            raise AssertionError("Gaussian outside split affected-set changed K4 weights")
        self.model.deformation_graph = SparseDeformationGraph(nodes, indices, weights)
        parent_node_ids = self.node_ids[torch.as_tensor(parents, device=self.node_ids.device)]
        count = len(parents)
        new_ids = torch.arange(self.next_node_id, self.next_node_id + count, device=self.node_ids.device)
        self.next_node_id += count
        self.node_ids = torch.cat((self.node_ids, new_ids))
        self.parent_ids = torch.cat((self.parent_ids, parent_node_ids))
        return affected_total, old_ids_per_gaussian

    def _resize_appearance(self, parents):
        if self.appearance_graph_fixed:
            return
        _replace_parameter_rows(self.model.optimizer, self.model.appearance_model,
                                "node_appearance", source_rows=parents)

    def _record_split_pairs(self, iteration, parents, positions):
        """Persist canonical parent-child separation at the split event."""
        if not parents:
            return []
        child_ids = self.node_ids[-len(parents):]
        records = []
        for parent_row, child_id, child_xyz in zip(parents, child_ids, positions):
            parent_id = int(self.parent_ids[-len(parents) + len(records)])
            parent_xyz = self.graph.node_xyz[parent_row].detach()
            child_row = len(self.graph.node_xyz) - len(parents) + len(records)
            appearance_distance = None if self.appearance_graph_fixed else float((
                self.model.appearance_model.node_appearance[child_row] -
                self.model.appearance_model.node_appearance[parent_row]).detach().norm())
            record = {
                "parent_id": parent_id,
                "child_id": int(child_id),
                "birth_iteration": int(iteration),
                "parent_xyz": parent_xyz.cpu().tolist(),
                "child_xyz": child_xyz.detach().cpu().tolist(),
                "d_pos": float((child_xyz - parent_xyz).norm()),
                "d_appearance_at_split": appearance_distance,
            }
            self.split_pairs.append(record)
            records.append(record)
        return records

    @torch.no_grad()
    def record_motion_diagnostics(self, iteration):
        """Measure learned parent-child motion separation at fixed offsets.

        Physical node IDs make this robust to row compaction after a late
        prune. Missing pairs are reported explicitly instead of silently
        measuring the wrong rows.
        """
        due = [pair for pair in self.split_pairs
               if iteration - pair["birth_iteration"] in self.motion_diagnostic_offsets and
               not any(item["child_id"] == pair["child_id"] and
                       item["iteration"] == iteration for item in self.motion_diagnostics)]
        if not due:
            return []
        row_by_id = {int(node_id): row for row, node_id in enumerate(self.node_ids.tolist())}
        sample_times = self.graph.node_xyz.new_tensor(self.motion_diagnostic_times)
        motion_bank = torch.stack([
            self.model.node_mlp(self.graph.node_xyz, time)[0] for time in sample_times])
        records = []
        for pair in due:
            parent_row = row_by_id.get(pair["parent_id"])
            child_row = row_by_id.get(pair["child_id"])
            if parent_row is None or child_row is None:
                record = {**pair, "iteration": int(iteration),
                          "offset": int(iteration - pair["birth_iteration"]),
                          "available": False, "reason": "parent_or_child_pruned"}
            else:
                difference = (motion_bank[:, child_row] -
                              motion_bank[:, parent_row]).norm(dim=1)
                appearance_difference = (None if self.appearance_graph_fixed else
                    (self.model.appearance_model.node_appearance[child_row] -
                     self.model.appearance_model.node_appearance[parent_row]).detach())
                record = {
                    **pair,
                    "iteration": int(iteration),
                    "offset": int(iteration - pair["birth_iteration"]),
                    "available": True,
                    "sample_times": list(self.motion_diagnostic_times),
                    "d_motion": difference.cpu().tolist(),
                    "d_motion_mean": float(difference.mean()),
                    "d_motion_p95": float(torch.quantile(difference, 0.95)),
                    "d_motion_max": float(difference.max()),
                    "d_appearance": (None if appearance_difference is None else
                                     float(appearance_difference.norm())),
                    "d_appearance_abs_mean": (None if appearance_difference is None else
                                              float(appearance_difference.abs().mean())),
                    "d_appearance_abs_max": (None if appearance_difference is None else
                                             float(appearance_difference.abs().max())),
                }
            self.motion_diagnostics.append(record)
            records.append(record)
            with open(self.motion_log_path, "a") as stream:
                stream.write(json.dumps(record) + "\n")
        if records:
            print("[PARENT_CHILD_MOTION] " + json.dumps({
                "iteration": int(iteration), "pairs": len(records),
                "available": sum(record["available"] for record in records),
            }), flush=True)
        return records

    @torch.no_grad()
    def _prune_candidates(self, iteration, stats):
        """Find nodes whose usage stayed low for multiple adaptation windows."""
        age_ok = (iteration - self.birth_iterations) >= self.config.min_age
        if not age_ok.any():
            self.prune_counters.zero_()
            return torch.empty(0, device=self.node_ids.device, dtype=torch.long)
        influence_cut = torch.quantile(
            stats["influence"][age_ok], self.config.prune_quantile)
        coverage_cut = torch.quantile(
            stats["coverage"][age_ok].float(), self.config.prune_quantile)
        low = age_ok & (stats["influence"] <= influence_cut) & (
            stats["coverage"].float() <= coverage_cut)
        if self.config.mode in ("split_prune_conservative", "split_prune_late"):
            # Physical IDs 0..1999 are the immutable Standard-FPS backbone.
            low &= (self.node_ids >= 2000) & (self.parent_ids >= 0)

        # Protect parents and spatial neighborhoods created in the grace period.
        recent = (iteration - self.birth_iterations) < self.config.min_age
        if recent.any():
            low &= ~torch.isin(self.node_ids, self.parent_ids[recent])
            nodes = self.graph.node_xyz
            if len(nodes) > 1:
                nearest = torch.cdist(nodes, nodes).fill_diagonal_(float("inf")).amin(1)
                exclusion_radius = nearest.median()
                low &= torch.cdist(nodes, nodes[recent]).amin(1) > exclusion_radius

        self.prune_counters = torch.where(
            low, self.prune_counters + 1, torch.zeros_like(self.prune_counters))
        candidates = torch.where(self.prune_counters >= self.config.prune_windows)[0]
        if len(candidates):
            usage = stats["influence"][candidates] * (
                stats["coverage"][candidates].float() + 1)
            candidates = candidates[torch.argsort(usage, stable=True)]
        return candidates

    @torch.no_grad()
    def _redundancy_diagnostic(self, candidate, motion_bank):
        """Check whether a nearby, overlapping, motion-compatible node exists."""
        graph = self.graph
        support = (graph.gaussian_node_indices == candidate).any(1)
        rows = support.nonzero(as_tuple=False).flatten()
        base = {
            "support_size": int(len(rows)),
            "nearest_replacement_node": None,
            "canonical_distance": None,
            "support_overlap": 0.0,
            "motion_difference": None,
            "redundancy_pass": False,
        }
        if not len(rows):
            return {**base, "redundancy_reject_reason": "empty_support"}
        replacements = torch.unique(graph.gaussian_node_indices[rows])
        replacements = replacements[replacements != candidate]
        if not len(replacements):
            return {**base, "redundancy_reject_reason": "no_local_replacement"}

        row_indices = graph.gaussian_node_indices[rows]
        overlap = torch.stack([(row_indices == replacement).any(1).float().mean()
                               for replacement in replacements])
        distance = (graph.node_xyz[replacements] - graph.node_xyz[candidate]).norm(dim=1)
        if motion_bank.ndim == 2:
            motion_bank = motion_bank.unsqueeze(0)
        motion_difference = (
            motion_bank[:, replacements] - motion_bank[:, candidate, None, :]
        ).norm(dim=2).amax(0)
        support_radius = (self.model._xyz.detach()[rows] -
                          graph.node_xyz[candidate]).norm(dim=1).median().clamp_min(1e-6)
        distance_limit = self.config.prune_redundancy_distance_ratio * support_radius
        motion_limit = self.config.prune_motion_error_ratio * self.shock_reference_scale
        valid = ((overlap >= self.config.prune_redundancy_overlap) &
                 (distance <= distance_limit) &
                 (motion_difference <= motion_limit))
        # Report the strongest support replacement, breaking ties by distance.
        score = overlap - distance / distance_limit.clamp_min(1e-8) * 1e-3
        report_index = int(torch.argmax(score))
        result = {
            **base,
            "nearest_replacement_node": int(self.node_ids[replacements[report_index]]),
            "canonical_distance": float(distance[report_index]),
            "support_overlap": float(overlap[report_index]),
            "motion_difference": float(motion_difference[report_index]),
            "distance_limit": float(distance_limit),
            "motion_limit": float(motion_limit),
            "redundancy_pass": bool(valid.any()),
        }
        if valid.any():
            valid_ids = valid.nonzero(as_tuple=False).flatten()
            best = valid_ids[torch.argmax(score[valid_ids])]
            result.update({
                "nearest_replacement_node": int(self.node_ids[replacements[best]]),
                "canonical_distance": float(distance[best]),
                "support_overlap": float(overlap[best]),
                "motion_difference": float(motion_difference[best]),
                "redundancy_reject_reason": None,
            })
        else:
            failures = []
            if float(overlap.max()) < self.config.prune_redundancy_overlap:
                failures.append("support_overlap")
            if float(distance.min()) > float(distance_limit):
                failures.append("canonical_distance")
            if float(motion_difference.min()) > motion_limit:
                failures.append("motion_difference")
            result["redundancy_reject_reason"] = "+".join(failures) or "no_joint_match"
        return result

    @torch.no_grad()
    def _preview_prune(self, prune_rows, old_node_dx):
        graph = self.graph
        prune_rows = torch.as_tensor(
            prune_rows, device=self.node_ids.device, dtype=torch.long)
        prune_mask = torch.zeros(
            len(graph.node_xyz), device=self.node_ids.device, dtype=torch.bool)
        prune_mask[prune_rows] = True
        affected = prune_mask[graph.gaussian_node_indices].any(1)
        rows = affected.nonzero(as_tuple=False).flatten()
        if not len(rows):
            return affected, rows, None, None, 0.0, 0.0

        # Local repair pool: surviving members of the old K4 plus 16 spatial
        # neighbors of each removed node. This never rebuilds global K4.
        surviving = (~prune_mask).nonzero(as_tuple=False).flatten()
        local_count = min(16, len(surviving))
        local_by_pruned = surviving[
            torch.cdist(graph.node_xyz[prune_rows], graph.node_xyz[surviving]).topk(
                local_count, largest=False, sorted=True).indices]
        current = graph.gaussian_node_indices[rows].clone()
        current[prune_mask[current]] = -1
        relevant = (graph.gaussian_node_indices[rows, :, None] ==
                    prune_rows[None, None, :]).any(1)
        local = local_by_pruned[None].expand(len(rows), -1, -1).clone()
        local[~relevant[:, :, None].expand_as(local)] = -1
        candidates = torch.cat((current, local.flatten(1)), dim=1)
        valid = candidates >= 0
        for column in range(1, candidates.shape[1]):
            duplicate = (candidates[:, :column] == candidates[:, column, None]).any(1)
            valid[:, column] &= ~duplicate
        safe = candidates.clamp_min(0)
        distance = (self.model._xyz.detach()[rows, None, :] -
                    graph.node_xyz[safe]).norm(dim=2)
        distance[~valid] = float("inf")
        distance, choice = distance.topk(4, largest=False, sorted=True)
        new_indices = candidates.gather(1, choice)
        if (new_indices < 0).any() or not torch.isfinite(distance).all():
            raise RuntimeError("local prune repair has fewer than four candidates")
        inverse = (distance + self.config.eps).reciprocal()
        new_weights = inverse / inverse.sum(1, keepdim=True)
        old_dx = (old_node_dx[graph.gaussian_node_indices[rows]] *
                  graph.gaussian_node_weights[rows, :, None]).sum(1)
        new_dx = (old_node_dx[new_indices] * new_weights[:, :, None]).sum(1)
        shock = (new_dx - old_dx).norm(dim=1)
        return (affected, rows, new_indices, new_weights,
                float(shock.mean()), float(shock.max()))

    @torch.no_grad()
    def _prune_nodes(self, prune_rows, old_node_dx=None):
        graph = self.graph
        old_indices = graph.gaussian_node_indices.clone()
        old_weights = graph.gaussian_node_weights.clone()
        old_ids_per_gaussian = self.node_ids[old_indices]
        if old_node_dx is None:
            old_node_dx = torch.zeros(
                (len(graph.node_xyz), 3), device=graph.node_xyz.device)
        affected, rows, repaired_indices, repaired_weights, _, _ = \
            self._preview_prune(prune_rows, old_node_dx)
        prune_mask = torch.zeros(
            len(graph.node_xyz), device=self.node_ids.device, dtype=torch.bool)
        prune_mask[torch.as_tensor(prune_rows, device=self.node_ids.device)] = True
        keep = (~prune_mask).nonzero(as_tuple=False).flatten()
        remap = torch.full(
            (len(graph.node_xyz),), -1, device=self.node_ids.device, dtype=torch.long)
        remap[keep] = torch.arange(len(keep), device=self.node_ids.device)
        indices = old_indices.clone()
        weights = old_weights.clone()
        indices[rows] = repaired_indices
        weights[rows] = repaired_weights
        indices = remap[indices]
        if (indices < 0).any():
            raise AssertionError("pruned node remains in repaired graph")
        self.model.deformation_graph = SparseDeformationGraph(
            graph.node_xyz[keep], indices, weights)
        if not self.appearance_graph_fixed:
            _replace_parameter_rows(self.model.optimizer, self.model.appearance_model,
                                    "node_appearance", keep_rows=keep)
        self.node_ids = self.node_ids[keep]
        self.parent_ids = self.parent_ids[keep]
        self.birth_iterations = self.birth_iterations[keep]
        self.prune_counters = self.prune_counters[keep]
        outside = ~affected
        if not torch.equal(self.node_ids[indices[outside]], old_ids_per_gaussian[outside]):
            raise AssertionError("Gaussian outside prune affected-set changed physical K4")
        if not torch.equal(weights[outside], old_weights[outside]):
            raise AssertionError("Gaussian outside prune affected-set changed K4 weights")
        return affected, old_ids_per_gaussian, keep

    @staticmethod
    def _metrics(image, gt, lpips_model):
        return dict(PSNR=float(psnr(image, gt).mean()), SSIM=float(ssim(image, gt)),
                    LPIPS=float(lpips_loss(image, gt, lpips_model)))

    @staticmethod
    def _distribution(value):
        value = value.detach().float().flatten()
        if not len(value):
            return dict(min=None, p05=None, median=None, p95=None, max=None, mean=None)
        quantiles = torch.quantile(value, value.new_tensor([.05, .5, .95]))
        return dict(min=float(value.min()), p05=float(quantiles[0]),
                    median=float(quantiles[1]), p95=float(quantiles[2]),
                    max=float(value.max()), mean=float(value.mean()))

    def adapt(self, iteration, camera, render_fn, pipe, background, lpips_model):
        stats = self.node_statistics(iteration)
        running_mean = 0.0 if self.running_log_mean is None else float(self.running_log_mean)
        running_std = (1.0 if self.running_log_var is None
                       else float(self.running_log_var.sqrt()))
        base = dict(iteration=iteration, mode=self.config.mode, nodes_before=len(self.graph.node_xyz),
                    G_mean=float(stats["geometry"].mean()), G_max=float(stats["geometry"].max()),
                    influence_mean=float(stats["influence"].mean()), influence_min=float(stats["influence"].min()),
                    coverage_mean=float(stats["coverage"].float().mean()), coverage_min=int(stats["coverage"].min()),
                    eligible_nodes=int(stats["eligible"].sum()),
                    G_distribution=self._distribution(stats["geometry"]),
                    influence_distribution=self._distribution(stats["influence"]),
                    coverage_distribution=self._distribution(stats["coverage"]),
                    running_log_mean=running_mean, running_log_std=running_std)
        if self.config.mode == "stats" or (self.config.mode == "split_once" and self._single_split_done):
            event = self._record({**base, "nodes_after": len(self.graph.node_xyz),
                                  "splits": 0, "splits_suppressed_by_budget": 0})
            self.save(os.path.join(self.output_dir, "adaptive_state_latest.pth"), iteration)
            return event
        before = render_fn(camera, self.model, pipe, background, stage="fine")
        before_xyz = before["transformed_points"].detach().clone()
        before_metrics = self._metrics(before["render"].detach()[None], camera.original_image.cuda().float()[None], lpips_model)
        split_enabled = self.config.mode in (
            "split_once", "split", "split_prune", "split_prune_conservative") or (
                self.config.mode == "split_prune_late" and
                iteration <= self.config.split_end)
        candidates = torch.where(stats["eligible"] & (stats["normalized"] >= self.config.split_z))[0]
        if not split_enabled:
            candidates = candidates[:0]
        candidates = candidates[torch.argsort(stats["normalized"][candidates], descending=True, stable=True)]
        room = max(0, self.config.max_nodes - len(self.graph.node_xyz))
        limit = min(self.config.max_splits, room)
        if self.config.mode == "split_once":
            limit = min(limit, 1)
        if limit == 0:
            candidates = candidates[:0]
        parents, positions, proposal_separations = [], [], []
        rejected = 0
        rejected_shock = 0
        rejected_shock_maxima = []
        predicted_shocks = []
        occupied = torch.zeros(len(self.model._xyz), device=self.model._xyz.device,
                               dtype=torch.bool)
        old_node_dx = self.model.node_mlp(self.graph.node_xyz, camera.time)[0]
        shock_limit = self.config.max_split_shock_ratio * self.shock_reference_scale
        for parent in candidates.tolist():
            proposal = self._proposal(parent, stats)
            if proposal is None:
                rejected += 1
                continue
            position, _, minimum_separation = proposal
            if positions:
                proposal_distance = (position[None] - torch.stack(positions)).norm(dim=1)
                required = torch.minimum(
                    minimum_separation.expand_as(proposal_distance),
                    torch.stack(proposal_separations))
                if (proposal_distance < required).any():
                    rejected += 1
                    continue
            affected_candidate, shock_mean, shock_max = self._candidate_shock(
                parent, position, camera.time, old_node_dx)
            if shock_max > shock_limit or (affected_candidate & occupied).any():
                rejected_shock += 1
                rejected_shock_maxima.append(shock_max)
                continue
            parents.append(parent); positions.append(position)
            proposal_separations.append(minimum_separation)
            occupied |= affected_candidate
            predicted_shocks.append({"mean": shock_mean, "max": shock_max})
            if len(parents) == limit:
                break
        suppressed = max(0, len(candidates) - len(parents)) if room == 0 else max(0, min(len(candidates), self.config.max_splits) - limit)
        parent_node_ids = [int(self.node_ids[p]) for p in parents]
        selected_parent_statistics = [
            {"node_id": int(self.node_ids[parent]),
             "G": float(stats["geometry"][parent]),
             "influence": float(stats["influence"][parent]),
             "coverage": int(stats["coverage"][parent]),
             "normalized_G": float(stats["normalized"][parent])}
            for parent in parents]
        if parents:
            affected, old_ids = self._append_children(parents, positions)
            self._resize_appearance(parents)
            births = torch.full((len(parents),), iteration, device=self.birth_iterations.device, dtype=torch.long)
            self.birth_iterations = torch.cat((self.birth_iterations, births))
            self.prune_counters = torch.cat((self.prune_counters, torch.zeros_like(births)))
            self._single_split_done = True
            split_pair_records = self._record_split_pairs(iteration, parents, positions)
        else:
            affected = torch.zeros(len(self.model._xyz), device=self.model._xyz.device, dtype=torch.bool)
            old_ids = self.node_ids[self.graph.gaussian_node_indices]
            split_pair_records = []
        pruned_node_ids = []
        prune_affected = torch.zeros_like(affected)
        prune_shock_predictions = []
        prune_enabled = self.config.mode in (
            "prune_once", "split_prune", "split_prune_conservative") or (
                self.config.mode == "split_prune_late" and
                self.config.prune_start <= iteration <= self.config.end)
        conservative_prune = self.config.mode in (
            "split_prune_conservative", "split_prune_late")
        prune_diagnostics = []
        rejected_low_redundancy = 0
        rejected_high_prune_error = 0
        if prune_enabled and not (self.config.mode == "prune_once" and self._single_prune_done):
            prune_stats = self.node_statistics(iteration)
            prune_candidates = self._prune_candidates(iteration, prune_stats)
            prune_limit = min(self.config.max_prunes,
                              max(0, len(self.graph.node_xyz) - self.config.min_nodes))
            if self.config.mode == "prune_once":
                prune_limit = min(prune_limit, 1)
            current_node_dx = self.model.node_mlp(self.graph.node_xyz, camera.time)[0]
            if conservative_prune:
                sample_times = current_node_dx.new_tensor([0.0, 0.5, 1.0])
                motion_bank = torch.stack([
                    self.model.node_mlp(self.graph.node_xyz, sample_time)[0]
                    for sample_time in sample_times])
            else:
                motion_bank = current_node_dx
            selected = []
            prune_occupied = affected.clone()
            for candidate in prune_candidates.tolist():
                diagnostic = {
                    "node_id": int(self.node_ids[candidate]),
                    "parent_id": int(self.parent_ids[candidate]),
                    "age": int(iteration - self.birth_iterations[candidate]),
                    "influence": float(prune_stats["influence"][candidate]),
                    "coverage": int(prune_stats["coverage"][candidate]),
                    "safe_prune_error": None,
                    "safe_prune_error_mean": None,
                    "safe_prune_error_max": None,
                    "accepted": False,
                    "reject_reason": None,
                }
                if len(selected) >= prune_limit:
                    diagnostic["reject_reason"] = "event_limit"
                    prune_diagnostics.append(diagnostic)
                    continue
                if conservative_prune:
                    redundancy = self._redundancy_diagnostic(candidate, motion_bank)
                    diagnostic.update(redundancy)
                    if not redundancy["redundancy_pass"]:
                        diagnostic["reject_reason"] = "low_redundancy"
                        rejected_low_redundancy += 1
                        prune_diagnostics.append(diagnostic)
                        continue
                candidate_affected, _, _, _, shock_mean, shock_max = self._preview_prune(
                    [candidate], current_node_dx)
                diagnostic["safe_prune_error"] = shock_mean
                diagnostic["safe_prune_error_mean"] = shock_mean
                diagnostic["safe_prune_error_max"] = shock_max
                safe_limit = ((self.config.prune_safe_error_ratio * self.shock_reference_scale)
                              if conservative_prune else shock_limit)
                diagnostic["safe_prune_error_limit"] = safe_limit
                if shock_mean > safe_limit or shock_max > shock_limit:
                    diagnostic["reject_reason"] = "high_prune_error"
                    rejected_high_prune_error += 1
                    prune_diagnostics.append(diagnostic)
                    continue
                if (candidate_affected & prune_occupied).any():
                    diagnostic["reject_reason"] = "overlapping_affected_region"
                    prune_diagnostics.append(diagnostic)
                    continue
                selected.append(candidate)
                prune_occupied |= candidate_affected
                prune_shock_predictions.append({"mean": shock_mean, "max": shock_max})
                diagnostic["accepted"] = True
                prune_diagnostics.append(diagnostic)
            if selected:
                pruned_node_ids = [int(self.node_ids[p]) for p in selected]
                prune_affected, _, _ = self._prune_nodes(selected, current_node_dx)
                self._single_prune_done = True
        affected |= prune_affected
        after = render_fn(camera, self.model, pipe, background, stage="fine")
        after_xyz = after["transformed_points"].detach()
        after_metrics = self._metrics(after["render"].detach()[None], camera.original_image.cuda().float()[None], lpips_model)
        changed = (old_ids != self.node_ids[self.graph.gaussian_node_indices]).any(dim=1)
        if (changed & ~affected).any():
            raise AssertionError("adaptation changed K4 outside affected set")
        shock = (after_xyz - before_xyz).norm(dim=1)
        if float(shock.max()) > shock_limit:
            raise RuntimeError(
                f"adaptive deformation shock {float(shock.max()):.6g} exceeds "
                f"limit {shock_limit:.6g}")
        event = {**base, "nodes_after": len(self.graph.node_xyz), "splits": len(parents),
                 "parents": parent_node_ids, "prunes": len(pruned_node_ids),
                 "pruned_node_ids": pruned_node_ids,
                 "predicted_prune_shocks": prune_shock_predictions,
                 "prune_candidates": len(prune_diagnostics),
                 "prune_candidate_diagnostics": prune_diagnostics,
                 "rejected_low_redundancy": rejected_low_redundancy,
                 "rejected_high_prune_error": rejected_high_prune_error,
                 "prunes_applied": len(pruned_node_ids),
                 "rejected_proposals": rejected,
                 "rejected_by_shock_or_overlap": rejected_shock,
                 "shock_reference_scale": self.shock_reference_scale,
                 "shock_limit_world": shock_limit,
                 "rejected_shock_max_distribution": self._distribution(
                     torch.as_tensor(rejected_shock_maxima, device=self.model._xyz.device)),
                 "predicted_split_shocks": predicted_shocks,
                 "parent_child_splits": split_pair_records,
                 "selected_parent_statistics": selected_parent_statistics,
                 "splits_suppressed_by_budget": suppressed,
                 "affected_percent": 100 * float(affected.float().mean()),
                 "changed_k4_percent": 100 * float(changed.float().mean()),
                 "outside_affected_physical_k4_exact": True,
                 "outside_affected_weights_exact": True,
                 "shock_mean": float(shock.mean()), "shock_max": float(shock.max()),
                 "metrics_before": before_metrics, "metrics_after": after_metrics}
        self.save(os.path.join(self.output_dir, "adaptive_state_latest.pth"), iteration)
        return self._record(event)

    def _record(self, event):
        self.events.append(event)
        with open(self.log_path, "a") as stream:
            stream.write(json.dumps(event) + "\n")
        print("[ADAPTIVE_NODE] " + json.dumps(event), flush=True)
        return event

    def state_dict(self, iteration):
        return dict(format="adaptive_deformation_graph_v1", iteration=iteration,
                    config=asdict(self.config), node_xyz=self.graph.node_xyz.detach().cpu(),
                    node_ids=self.node_ids.detach().cpu(), parent_ids=self.parent_ids.detach().cpu(),
                    birth_iterations=self.birth_iterations.detach().cpu(),
                    prune_counters=self.prune_counters.detach().cpu(),
                    gaussian_node_indices=self.graph.gaussian_node_indices.detach().cpu(),
                    gaussian_node_weights=self.graph.gaussian_node_weights.detach().cpu(),
                    gaussian_score=self.gaussian_score.detach().cpu(),
                    gaussian_observations=self.gaussian_observations.detach().cpu(),
                    running_log_mean=None if self.running_log_mean is None else self.running_log_mean.detach().cpu(),
                    running_log_var=None if self.running_log_var is None else self.running_log_var.detach().cpu(),
                    next_node_id=self.next_node_id, events=self.events,
                    split_pairs=self.split_pairs,
                    motion_diagnostics=self.motion_diagnostics,
                    single_split_done=self._single_split_done,
                    single_prune_done=self._single_prune_done,
                    node_appearance=self.model.appearance_model.node_appearance.detach().cpu(),
                    optimizer=self.model.optimizer.state_dict())

    def save(self, path, iteration):
        torch.save(self.state_dict(iteration), path)

    @torch.no_grad()
    def load_state_dict(self, state):
        """Restore adaptive topology, metadata, R6 rows, and optimizer exactly."""
        if state.get("format") != "adaptive_deformation_graph_v1":
            raise ValueError("unsupported adaptive checkpoint format")
        saved_config = AdaptiveNodeConfig(**state["config"])
        if asdict(saved_config) != asdict(self.config):
            raise ValueError("adaptive checkpoint config differs from the active config")
        device = self.model._xyz.device
        indices = state["gaussian_node_indices"].to(device=device, dtype=torch.long)
        weights = state["gaussian_node_weights"].to(device=device, dtype=self.model._xyz.dtype)
        nodes = state["node_xyz"].to(device=device, dtype=self.model._xyz.dtype)
        if indices.shape != (len(self.model._xyz), 4) or int(indices.max()) >= len(nodes):
            raise ValueError("invalid adaptive checkpoint graph")
        self.model.deformation_graph = SparseDeformationGraph(nodes, indices, weights)
        _replace_parameter_value(
            self.model.optimizer, self.model.appearance_model, "node_appearance",
            state["node_appearance"])
        self.model.optimizer.load_state_dict(state["optimizer"])
        for optimizer_state in self.model.optimizer.state.values():
            for key, value in optimizer_state.items():
                if torch.is_tensor(value):
                    optimizer_state[key] = value.to(device)
        self.node_ids = state["node_ids"].to(device=device, dtype=torch.long)
        self.parent_ids = state["parent_ids"].to(device=device, dtype=torch.long)
        self.birth_iterations = state["birth_iterations"].to(device=device, dtype=torch.long)
        self.prune_counters = state["prune_counters"].to(device=device, dtype=torch.long)
        self.gaussian_score = state["gaussian_score"].to(device=device, dtype=self.model._xyz.dtype)
        self.gaussian_observations = state["gaussian_observations"].to(device=device, dtype=torch.long)
        self.running_log_mean = (None if state["running_log_mean"] is None
                                 else state["running_log_mean"].to(device))
        self.running_log_var = (None if state["running_log_var"] is None
                                else state["running_log_var"].to(device))
        self.next_node_id = int(state["next_node_id"])
        self.events = list(state["events"])
        self.split_pairs = list(state.get("split_pairs", []))
        self.motion_diagnostics = list(state.get("motion_diagnostics", []))
        self._single_split_done = bool(state.get("single_split_done", False))
        self._single_prune_done = bool(state.get("single_prune_done", False))
        expected = len(nodes)
        for name, value in (("node IDs", self.node_ids), ("parent IDs", self.parent_ids),
                            ("birth iterations", self.birth_iterations),
                            ("prune counters", self.prune_counters)):
            if len(value) != expected:
                raise ValueError(f"adaptive checkpoint {name} length mismatch")

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
