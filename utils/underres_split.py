"""Single R_underres ablation: persistent priority times directional 2-means gain.

Only event-time allocation changes. No forward, placement, graph or loss changes.
The vector is the EXISTING accumulated normalized RGB reconstruction gradient
w.r.t. deformed Gaussian XYZ; it is not a measured motion trajectory.
"""
import numpy as np
import torch
from utils.capacity_split import CapacitySplitManager


def directional_heterogeneity(vectors, weights, min_support=32, eps=1e-8):
    """Deterministic CPU float64 weighted two-means, at most five Lloyd updates."""
    x = np.asarray(vectors, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3 or w.shape != (len(x),):
        raise ValueError('Expected [N,3] directions and [N] weights')
    norms = np.linalg.norm(x, axis=1)
    valid = np.isfinite(x).all(1) & np.isfinite(w) & (w > 0) & (norms > eps)
    x, w, norms = x[valid], w[valid], norms[valid]
    row = dict(H=0., support_size=len(x), cluster_weight_ratio=[0., 0.],
               sse1=0., sse2=0., heterogeneity_reason='insufficient_direction_support')
    if len(x) < min_support:
        return row
    x = x / norms[:, None]
    w = w / w.sum()
    mean = (x * w[:, None]).sum(0)
    sse1 = float((w * ((x - mean)**2).sum(1)).sum())
    row['sse1'] = sse1
    if sse1 <= eps:
        row['heterogeneity_reason'] = 'coherent'
        return row
    # Stable first-index tie breaks, then farthest from that endpoint.
    first = int(((x - mean)**2).sum(1).argmax())
    second = int(((x - x[first])**2).sum(1).argmax())
    centers = x[[first, second]].copy()
    for _ in range(5):
        labels = ((x[:, None] - centers[None])**2).sum(2).argmin(1)
        masses = np.array([w[labels == k].sum() for k in (0, 1)])
        if (masses <= 0).any():
            row['heterogeneity_reason'] = 'empty_cluster'
            return row
        centers = np.stack([(x[labels == k] * w[labels == k, None]).sum(0) / masses[k]
                            for k in (0, 1)])
    row['cluster_weight_ratio'] = masses.tolist()
    row['sse2'] = float((w * ((x - centers[labels])**2).sum(1)).sum())
    if masses.min() < .20 - 1e-12:
        row['heterogeneity_reason'] = 'outlier_cluster'
        return row
    row['H'] = float(np.clip((sse1 - row['sse2']) / (sse1 + eps), 0, 1))
    row['heterogeneity_reason'] = 'valid'
    return row


def h_distribution(rows):
    values = np.asarray([r['H'] for r in rows], dtype=np.float64)
    return dict(count=len(values), **({k: float(v) for k, v in zip(
        ('mean', 'median', 'p75', 'p90'),
        (values.mean(), *np.quantile(values, [.5, .75, .90])))} if len(values) else
        dict(mean=None, median=None, p75=None, p90=None)))


class UnderresolutionSplitManager(CapacitySplitManager):
    FORMAT = 'persistent_directional_two_means_v1'

    def __init__(self, *args, **kwargs):
        if kwargs.get('signal') != 'persistent_underres':
            raise ValueError('R_underres needs its explicit feature flag')
        super().__init__(*args, **kwargs)
        if not self.quota_schedule or sum(self.quota_schedule.values()) != 535:
            raise ValueError('Use completed Rcap accepted-event quotas (535 births total)')
        self._underres_rows = {}
        self._priority = None

    @torch.no_grad()
    def _rank_candidates(self, stats):
        if self._bank is None:
            raise RuntimeError('Persistent training probe bank required')
        idx = self.graph.gaussian_node_indices.detach().cpu().numpy()
        influence = self.graph.gaussian_node_weights.detach().cpu().numpy()
        observations = self.gaussian_observations.detach().cpu().numpy()
        counts = self.direction_count.detach().cpu().numpy()
        vectors = self.direction_sum.detach().cpu().numpy() / np.maximum(counts[:, None], 1)
        difficulty = self.gaussian_score.detach().cpu().numpy()
        meaningful = ((influence > self.config.weight_threshold) &
                      (observations[:, None] > 0) & (counts[:, None] > 0))
        gaussian_ids, slots = np.nonzero(meaningful)
        parents = idx[gaussian_ids, slots]
        order = np.argsort(parents, kind='stable')
        parents, gaussian_ids, slots = parents[order], gaussian_ids[order], slots[order]
        bounds = np.searchsorted(parents, np.arange(len(self.node_ids) + 1))
        score = self._bank['score'].detach().cpu().numpy()
        allowed = self._bank['allowed'].detach().cpu().numpy()
        priority = np.zeros(len(self.node_ids), dtype=np.float32)
        self._underres_rows = {}
        for p in np.flatnonzero(allowed):
            ids, ss = gaussian_ids[bounds[p]:bounds[p+1]], slots[bounds[p]:bounds[p+1]]
            # Same influence * EMA-gradient-magnitude weighting as placement.
            weights = influence[ids, ss] * np.maximum(difficulty[ids], self.config.eps)
            row = directional_heterogeneity(vectors[ids], weights,
                min_support=self.config.min_coverage, eps=self.config.eps)
            priority[p] = score[p] * row['H']
            row.update(parent_id=int(self.node_ids[p]), difficulty_score=float(score[p]),
                       underres_priority=float(priority[p]))
            self._underres_rows[int(p)] = row
        self._priority = self._bank['score'].new_tensor(priority)
        # A zero improvement/coherent or invalid two-cluster fit cannot justify
        # a split. No tuned H threshold; positive candidates fill fixed quotas.
        ids = torch.as_tensor(np.flatnonzero(allowed & (priority > 0)), device=self.node_ids.device)
        return ids[torch.argsort(self._priority[ids], descending=True, stable=True)]

    def _ranking_values(self, stats):
        return self._priority

    def _record(self, event):
        rows_by_id = {r['parent_id']: r for r in self._underres_rows.values()}
        ledger = self._admission_audit['candidates']
        present = set()
        for row in ledger:
            row.update(rows_by_id[row['node_id']])
            present.add(row['node_id'])
        excluded = [dict(r, node_id=r['parent_id'], reason=r['heterogeneity_reason'])
                    for r in rows_by_id.values() if r['parent_id'] not in present]
        accepted = [r for r in ledger if r['reason'] == 'accepted']
        rejected = [r for r in ledger if r['reason'] != 'accepted'] + excluded
        for pair in event.get('parent_child_splits', []):
            pair.update(rows_by_id[pair['parent_id']])
        event['underresolution'] = dict(format=self.FORMAT,
            vector_source='mean accumulated unit RGB gradient w.r.t. deformed Gaussian XYZ',
            num_candidates=len(rows_by_id), num_accepted=len(accepted), num_rejected=len(rejected),
            current_nodes=event['nodes_after'], requested_quota=self._event_quota(),
            quota_shortfall=self._event_quota()-len(accepted),
            H=h_distribution(list(rows_by_id.values())), accepted_H=h_distribution(accepted),
            rejected_H=h_distribution(rejected), invalid_candidates=excluded,
            fold_membership=None,
            fold_note='Existing Fold mask is per-view pixels; no canonical parent Fold mapping. No new heuristic.')
        return super()._record(event)

    def state_dict(self, iteration):
        state = super().state_dict(iteration)
        state['underresolution'] = dict(format=self.FORMAT, max_lloyd_steps=5,
            min_cluster_weight=.20, min_support=self.config.min_coverage)
        return state

    def load_state_dict(self, state):
        expected = dict(format=self.FORMAT, max_lloyd_steps=5,
            min_cluster_weight=.20, min_support=self.config.min_coverage)
        if state.get('underresolution') != expected:
            raise ValueError('Underresolution policy changed on reload')
        return super().load_state_dict(state)
