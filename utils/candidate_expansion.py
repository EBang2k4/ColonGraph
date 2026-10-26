"""Admission-only ablation: same A20 score, proposals, local graph and shock limit."""
import copy
from collections import Counter

import torch

from utils.adaptive_g3_graph import AdaptiveG3NodeManager


class CandidateExpansionManager(AdaptiveG3NodeManager):
    # Approximate upper targets 420/630/840, not quotas to be filled at any cost.
    PROTOCOL = {"D1": (60, .25), "D2": (90, .50), "D3": (120, .75)}

    def __init__(self, *args, admission, **kwargs):
        super().__init__(*args, **kwargs)
        if admission not in self.PROTOCOL or self.score_mode != "current" or self.placement_mode != "centroid":
            raise ValueError("Candidate expansion requires current A20 score and centroid")
        self.admission = admission
        quota, self.overlap_fraction = self.PROTOCOL[admission]
        if self.config.max_splits != quota or self.config.split_z != 1.0:
            raise ValueError("Preregistered quota/quality floor mismatch")

    @torch.no_grad()
    def _batch_preview(self, parents, positions, indices, weights, time, old_dx):
        edges, edge_weights, _, _ = self._local_edges(parents, positions)
        preview = copy.copy(self.model.graph_motion_residual)
        preview._buffers = preview._buffers.copy()
        preview.append_local_nodes(torch.stack(positions), edges, edge_weights)
        nodes = preview.canonical_xyz
        affected, rows, local_idx, local_w = self._local_child_update(
            indices, weights, nodes, parents[-1], positions[-1], len(nodes)-1)
        new_indices, new_weights = indices.clone(), weights.clone()
        new_indices[rows], new_weights[rows] = local_idx, local_w
        motion, _ = preview(self.model.node_mlp(nodes, time)[0])
        new_dx = (motion[new_indices] * new_weights[..., None]).sum(1)
        shock = (new_dx-old_dx).norm(dim=1)
        return float(shock.max()), new_indices, new_weights

    @torch.no_grad()
    def _rank_candidates(self, stats):
        return super()._split_candidates(stats)

    def _ranking_values(self, stats):
        return stats['normalized']

    def _event_quota(self):
        return self.config.max_splits

    @torch.no_grad()
    def _split_candidates(self, stats):
        candidates = self._rank_candidates(stats)
        ranking = self._ranking_values(stats)
        quota = min(self._event_quota(), self.config.max_nodes-len(self.node_ids))
        parents, positions, separations = [], [], []
        idx, weights = self.graph.gaussian_node_indices.clone(), self.graph.gaussian_node_weights.clone()
        occupied = torch.zeros(len(idx), dtype=torch.bool, device=idx.device)
        old_motion = self._node_translation(self._admission_time)
        old_dx = (old_motion[idx]*weights[..., None]).sum(1)
        limit = self.config.max_split_shock_ratio*self.shock_reference_scale
        ledger = []
        for parent in candidates.tolist():
            row = dict(node_id=int(self.node_ids[parent]), score=float(ranking[parent]),
                       geometry_score=float(stats["geometry"][parent]))
            ledger.append(row)
            if len(parents) >= quota:
                row["reason"] = "event_quota"
                continue
            proposal = self._proposal(parent, stats)
            # Optional Rcap gate diagnostics; absent for every original D/R run.
            details = getattr(self, '_admission_proposal_details', {}).get(parent, {})
            row.update(details)
            if proposal is None:
                membership = (self.graph.gaussian_node_indices == parent).any(1)
                support = int((membership & (self.gaussian_observations > 0)).sum())
                row["reason"] = details.get('capacity_reject_reason') or ("insufficient_support" if support < self.config.min_coverage else "minimum_existing_node_distance")
                continue
            position, _, minimum = proposal
            if positions:
                distances = (torch.stack(positions)-position).norm(dim=1)
                if (distances < torch.minimum(torch.stack(separations), minimum)).any():
                    row["reason"] = "spatial_nms"
                    continue
                # At most two births within one parent-support radius per event.
                # Same spatial-diversity gate in all D branches, independent of score.
                if int((distances < minimum/self.config.min_child_separation).sum()) >= 2:
                    row["reason"] = "local_region_quota"
                    continue
            affected, shock_mean, shock_max = self._candidate_shock(parent, position, self._admission_time, old_motion)
            overlap = int((affected & occupied).sum()) / max(1, int(affected.sum()))
            row.update(individual_shock=shock_max, overlap_fraction=overlap)
            if not torch.isfinite(torch.tensor(shock_max)) or shock_max > limit:
                row["reason"] = "individual_shock"
                continue
            if overlap > self.overlap_fraction:
                row["reason"] = "overlap"
                continue
            batch_shock, next_idx, next_w = self._batch_preview(
                parents+[parent], positions+[position], idx, weights, self._admission_time, old_dx)
            row["batch_shock"] = batch_shock
            if not torch.isfinite(torch.tensor(batch_shock)) or batch_shock > limit:
                row["reason"] = "cumulative_shock"
                continue
            row["reason"] = "accepted"
            parents.append(parent); positions.append(position); separations.append(minimum)
            occupied |= affected
            idx, weights = next_idx, next_w
        self._admission_audit = dict(
            branch=self.admission, score_floor=1.0, overlap_fraction_limit=self.overlap_fraction,
            shock_limit=limit, considered_nodes=len(self.node_ids), eligible_nodes=int(stats["eligible"].sum()),
            above_score_floor=len(candidates), below_score_floor=int(stats["eligible"].sum())-len(candidates),
            cooldown_rejected=0, counts=dict(Counter(r["reason"] for r in ledger)), candidates=ledger,
            below_floor_scores=self._distribution(stats["normalized"][stats["eligible"] & (stats["normalized"] < 1.0)]),
            ineligible_scores=self._distribution(stats["normalized"][~stats["eligible"]]),
            accepted_scores=self._distribution(ranking[parents]),
            rejected_scores=self._distribution(ranking[[p for p in candidates.tolist() if p not in parents]]))
        return candidates.new_tensor(parents)

    @staticmethod
    def _split_rejection_flags(shock_max, shock_limit, affected, occupied):
        # Overlap has already passed the fractional AND cumulative shock gates.
        return shock_max > shock_limit, False, int((affected & occupied).sum())

    def adapt(self, iteration, camera, *args, **kwargs):
        self._admission_time = camera.time
        return super().adapt(iteration, camera, *args, **kwargs)

    def _record(self, event):
        event["candidate_admission"] = self._admission_audit
        if event["splits"] != self._admission_audit["counts"].get("accepted", 0):
            raise AssertionError("Admission preview and committed split count differ")
        return super()._record(event)

    def state_dict(self, iteration):
        state = super().state_dict(iteration)
        state["candidate_admission"] = self.admission
        return state

    def load_state_dict(self, state):
        if state.get("candidate_admission") != self.admission:
            raise ValueError("Cannot change admission on resume")
        return super().load_state_dict(state)
