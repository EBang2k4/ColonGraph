"""Feature-flagged persistent-impact ranking on unchanged D3 split infrastructure."""
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from utils.candidate_expansion import CandidateExpansionManager
from utils.motion_capacity_audit import node_reduce, fit_correction
from utils.split_signal_audit import make_signals, observed_median


def capacity_tiebreak(score, capacity, allowed, relative_bin=.05):
    """Capacity only reorders scores within fixed <=5% multiplicative bins."""
    ids = allowed.nonzero().flatten()
    ids = ids[torch.argsort(score[ids], descending=True, stable=True)]
    ids = ids[torch.argsort(capacity[ids], descending=True, stable=True)]
    bins = torch.floor(torch.log(score[ids].clamp_min(1e-30))/math.log1p(relative_bin))
    return ids[torch.argsort(bins, descending=True, stable=True)]


def state_hash(model):
    digest = hashlib.sha256()
    for name in ('_xyz', '_scaling', '_rotation', '_opacity', 'features', '_deformation_table'):
        value = getattr(model, name)
        digest.update(value.detach().contiguous().cpu().numpy().tobytes())
    for name in ('node_mlp', 'appearance_model', 'deformation_graph', 'appearance_graph', 'graph_motion_residual'):
        for key, value in getattr(model, name).state_dict().items():
            digest.update((name+'.'+key).encode())
            digest.update(value.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


class PersistentSplitManager(CandidateExpansionManager):
    MODES = ('rgb_matched', 'persistent_impact', 'persistent_tie', 'persistent_tie_motion', 'persistent_capacity', 'persistent_underres')

    def __init__(self, *args, signal, quota_reference='', depth_weight=.25, **kwargs):
        super().__init__(*args, **kwargs)
        if signal not in self.MODES or self.admission != 'D3':
            raise ValueError('Persistent study requires explicit D3 signal mode')
        self.signal = signal
        self.depth_weight = depth_weight
        self.quota_schedule = {}
        if quota_reference:
            rows = [json.loads(s) for s in Path(quota_reference).read_text().splitlines()]
            self.quota_schedule = {int(r['iteration']): int(r['splits']) for r in rows}
            if sorted(self.quota_schedule) != [5000, 7500, 10000, 12500, 15000, 17500, 20000]:
                raise ValueError('Quota reference must contain all seven matched events')
            if any(n < 0 or n > self.config.max_splits for n in self.quota_schedule.values()):
                raise ValueError('Quota outside D3 budget')
        self._bank = None
        self._signal_summary = {}
        self._motion_placements = {}

    def _event_quota(self):
        return self.quota_schedule.get(self._iteration, self.config.max_splits)

    def collect_event_signals(self, iteration, cameras, render_fn, pipe, background):
        started = time.perf_counter()
        before = state_hash(self.model)
        rng_cpu, rng_cuda = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        rng_np, rng_py = np.random.get_state(), random.getstate()
        # Sorted camera bank is independent of the shuffled training stack/RNG.
        cameras = sorted(cameras, key=lambda c: (float(c.time), str(c.image_name)))
        indices = np.rint(np.linspace(0, len(cameras)-1, 32)).astype(int)[::2]
        if len(set(indices.tolist())) != 16:
            raise ValueError('Need 16 distinct temporal probe frames')
        idx, w = self.graph.gaussian_node_indices, self.graph.gaussian_node_weights
        n, m = len(idx), len(self.graph.node_xyz)
        seen = w.new_zeros(n)
        control_mass = w.new_zeros(n) if self.signal in ('persistent_capacity', 'persistent_underres') else None
        rows, capacity, motion = [], [], []
        need_capacity = self.signal in ('persistent_tie', 'persistent_tie_motion')
        solver_reports = []
        with torch.enable_grad():
            for camera_index in indices:
                camera = cameras[int(camera_index)]
                output = render_fn(camera, self.model, pipe, background, stage='fine')
                depth, gt = output['depth'].squeeze(), camera.original_depth.cuda().float().squeeze()
                valid = torch.isfinite(depth) & torch.isfinite(gt) & (gt > 0)
                if getattr(camera, 'mask', None) is not None:
                    valid &= torch.as_tensor(camera.mask, device=depth.device).bool().squeeze()
                if not valid.any():
                    raise RuntimeError('Probe camera has no valid depth')
                predicted = (depth/(depth[valid].max()+1e-6)).clamp(0, 1)
                target = (gt/(gt[valid].max()+1e-6)).clamp(0, 1)
                error = torch.where(valid, (predicted-target).square(), 0.).detach()
                probes = torch.stack((error, valid.float(), torch.zeros_like(error)))
                adjoint, = torch.autograd.grad((output['render']*probes).sum(), output['transformed_color'], retain_graph=need_capacity)
                adjoint = adjoint.reshape(-1, 3).detach().clamp_min(0)
                mass = adjoint[:, 1]
                if control_mass is not None:
                    control_mass += mass
                observed = mass > 1e-6
                if not observed.any():
                    raise RuntimeError('No contributing Gaussians in probe')
                confidence = (mass/torch.quantile(mass[observed], .95).clamp_min(1e-12)).clamp(0, 1)*observed
                seen += observed
                totals = node_reduce(adjoint[:, :2], idx, w, m)
                confidence_node = node_reduce(confidence, idx, w, m)
                alpha_sum = float((output['alpha'].squeeze().detach()*valid).sum())
                adjoint_error = abs(float(mass.sum())-alpha_sum)/max(alpha_sum, 1e-12)
                if adjoint_error > .002:
                    raise RuntimeError('Probe contribution/alpha mismatch')
                rows.append(dict(
                    error=(totals[:, 0]/totals[:, 1].clamp_min(1e-30)).cpu().numpy(),
                    impact=(totals[:, 0]/totals[:, 0].sum().clamp_min(1e-30)).cpu().numpy(),
                    observed=(confidence_node >= 1).cpu().numpy(), name=str(camera.image_name),
                    adjoint_error=adjoint_error))
                if need_capacity:
                    loss = self.depth_weight*(predicted-target).abs()[valid].mean()
                    gradient, = torch.autograd.grad(loss, output['transformed_points'])
                    if not torch.isfinite(gradient).all():
                        raise RuntimeError('Nonfinite capacity gradient')
                    residual, fit = fit_correction(idx.cpu().numpy(), w.cpu().numpy(),
                        gradient.detach().cpu().numpy(), confidence.cpu().numpy(), m)
                    g2 = gradient.detach().square().sum(1)
                    r2 = torch.from_numpy(residual).to(w).square().sum(1)
                    numerator = node_reduce(r2*confidence, idx, w, m)
                    denominator = node_reduce(g2*confidence, idx, w, m)
                    capacity.append((numerator/denominator.clamp_min(1e-30)).sqrt().cpu().numpy())
                    solver_reports.append(fit)
                if self.signal == 'persistent_tie_motion':
                    motion.append((output['transformed_points'].detach()-self.model._xyz.detach()).cpu())
                del output
        coverage = node_reduce((seen >= 4).float(), idx, (w > .05).float(), m).cpu().numpy()
        observed = np.stack([r['observed'] for r in rows])
        eligible = ((coverage >= self.config.min_coverage) & (observed.sum(0) >= 4) &
                    ((iteration-self.birth_iterations.cpu().numpy()) >= self.config.min_age))
        if eligible.sum() < 1:
            raise RuntimeError('No eligible support in persistent probe bank')
        bank = dict(depth_error=np.stack([r['error'] for r in rows]), observed=observed,
                    error_impact=np.stack([r['impact'] for r in rows]))
        zeros = np.zeros(m)
        scores, detail = make_signals(bank, eligible, zeros, zeros, zeros, zeros)
        score, allowed = scores['persistent_impact']
        U = observed_median(np.stack(capacity), observed) if capacity else zeros
        self._bank = dict(score=w.new_tensor(score), allowed=torch.as_tensor(allowed, device=w.device),
                          U=w.new_tensor(U), detail=detail, motion=torch.stack(motion) if motion else None,
                          times=[float(cameras[int(i)].time) for i in indices])
        if control_mass is not None:
            self._bank['control_mass'] = control_mass.detach()
        if state_hash(self.model) != before:
            raise AssertionError('Probe mutated model/geometry/appearance/graph')
        if not (torch.equal(rng_cpu, torch.get_rng_state()) and torch.equal(rng_cuda, torch.cuda.get_rng_state()) and
                random.getstate() == rng_py and np.array_equal(np.random.get_state()[1], rng_np[1]) and
                np.random.get_state()[2:] == rng_np[2:]):
            raise AssertionError('Probe consumed training RNG')
        self._signal_summary = dict(iteration=iteration, mode=self.signal,
            frames=[r['name'] for r in rows], eligible=int(eligible.sum()), qualified=int(allowed.sum()),
            persistent_score=self._distribution(w.new_tensor(score[eligible])),
            capacity_U=self._distribution(w.new_tensor(U[eligible])) if capacity else None,
            max_adjoint_error=max(r['adjoint_error'] for r in rows), solver=solver_reports,
            model_hash_unchanged=True, training_rng_unchanged=True,
            probe_seconds=time.perf_counter()-started)
        np.savez_compressed(Path(self.output_dir)/f'probe_{iteration}.npz', **bank,
                            score=score, allowed=allowed, U=U, coverage=coverage, **detail)
        print('[PERSISTENT_PROBE] '+json.dumps(self._signal_summary), flush=True)

    def _rank_candidates(self, stats):
        if self._bank is None:
            raise RuntimeError('Must collect matched probe bank before a split event')
        if self.signal == 'rgb_matched':
            return super()._rank_candidates(stats)
        score, allowed = self._bank['score'], self._bank['allowed']
        if self.signal in ('persistent_tie', 'persistent_tie_motion'):
            return capacity_tiebreak(score, self._bank['U'], allowed)
        ids = allowed.nonzero().flatten()
        return ids[torch.argsort(score[ids], descending=True, stable=True)]

    def _ranking_values(self, stats):
        return stats['normalized'] if self.signal == 'rgb_matched' else self._bank['score']

    @torch.no_grad()
    def _proposal(self, parent, stats):
        baseline = super()._proposal(parent, stats)
        if self.signal != 'persistent_tie_motion' or baseline is None:
            return baseline
        centroid, ids, minimum = baseline
        membership = self.graph.gaussian_node_indices[ids] == parent
        slot = membership.long().argmax(1)
        q = self.graph.gaussian_node_weights[ids, slot]*self.gaussian_score[ids].clamp_min(self.config.eps)
        xyz = self.model._xyz.detach()[ids]
        trajectories = self._bank['motion'][:, ids.cpu()].to(xyz)
        trajectories -= trajectories.mean(0, keepdim=True)
        trajectory_block = trajectories.permute(1, 0, 2).reshape(len(ids), -1)
        blocks = []
        for block in (xyz, trajectory_block):
            centered = block-block.median(0).values
            scale = centered.abs().median().clamp_min(1e-5)
            blocks.append((centered/scale).clamp(-5, 5)/math.sqrt(block.shape[1]))
        descriptor = torch.cat(blocks, 1)
        mean = (descriptor*q[:, None]).sum(0)/q.sum()
        first = int((descriptor-mean).square().sum(1).argmax())
        second = int((descriptor-descriptor[first]).square().sum(1).argmax())
        centers = descriptor[[first, second]].clone()
        for _ in range(12):
            label = torch.cdist(descriptor, centers).argmin(1)
            if min(int((label == k).sum()) for k in (0, 1)) < max(8, self.config.min_coverage//4):
                self._motion_placements[parent] = 'fallback_small_cluster'
                return baseline
            for k in (0, 1):
                mask = label == k
                centers[k] = (descriptor[mask]*q[mask, None]).sum(0)/q[mask].sum()
        cluster_motion = torch.stack([(trajectory_block[label == k]*q[label == k, None]).sum(0)/q[label == k].sum() for k in (0, 1)])
        separation = (cluster_motion[0]-cluster_motion[1]).square().mean().sqrt()
        variation = trajectory_block.square().mean().sqrt().clamp_min(1e-5)
        if separation < .1*variation:
            self._motion_placements[parent] = 'fallback_motion_similar'
            return baseline
        parent_motion = torch.stack([self._node_translation(t)[parent] for t in self._bank['times']])
        parent_motion = (parent_motion-parent_motion.mean(0)).flatten()
        choice = int((cluster_motion-parent_motion).square().mean(1).argmax())
        selected = label == choice
        position = (xyz[selected]*q[selected, None]).sum(0)/q[selected].sum()
        if torch.cdist(position[None], self.graph.node_xyz).min() < minimum:
            self._motion_placements[parent] = 'fallback_child_distance'
            return baseline
        self._motion_placements[parent] = dict(cluster_counts=[int((label == k).sum()) for k in (0, 1)],
            normalized_motion_separation=float(separation/variation), chosen_cluster=choice)
        return position, ids[selected], minimum

    def _record(self, event):
        audit = event['candidate_admission'] if 'candidate_admission' in event else self._admission_audit
        if self.signal != 'rgb_matched':
            audit['score_floor'] = None
            audit['quality_floor'] = 'P>=0.25; >=4 observed frames; >=32 supported Gaussians; age>=5000'
            for key in ('below_floor_scores', 'below_score_floor', 'above_score_floor'):
                audit.pop(key, None)
        event['split_signal'] = dict(**self._signal_summary, event_quota=self._event_quota(),
                                    matched_quota_filled=event['splits'] == self._event_quota(),
                                    motion_placement=self._motion_placements)
        return super()._record(event)

    def adapt(self, *args, **kwargs):
        self._motion_placements = {}
        result = super().adapt(*args, **kwargs)
        self._bank = None
        return result

    def state_dict(self, iteration):
        result = super().state_dict(iteration)
        result['persistent_split'] = dict(signal=self.signal, quota_schedule=self.quota_schedule,
                                           depth_weight=self.depth_weight, last_probe=self._signal_summary)
        return result

    def load_state_dict(self, state):
        saved = state['persistent_split']
        if (saved['signal'] != self.signal or saved['quota_schedule'] != self.quota_schedule or
                saved['depth_weight'] != self.depth_weight):
            raise ValueError('Signal/quota schedule changed on resume')
        super().load_state_dict(state)
        self._signal_summary = saved['last_probe']
