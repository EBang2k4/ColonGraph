"""J8 lifecycle adapter for the original B1/A20/D3 split selectors.

Selection, K4 score support, placement, quotas and proposal safety are inherited.
Only committed topology is connected to J's active K8 operator. The original
post-commit rendered-position shock check remains mandatory and aborts on failure.
"""
import torch

from utils.adaptive_deformation_graph import AdaptiveNodeConfig, AdaptiveNodeManager
from utils.adaptive_g3_graph import AdaptiveG3NodeManager
from utils.candidate_expansion import CandidateExpansionManager
from utils.exact_split_branchpoint import digest
from utils.geometry_blending import graph_spacing
from utils.ggas_v2 import connected_growth, warm_start_prior
from utils.graph_forward import _dense_adjacency


def config(variant):
    return AdaptiveNodeConfig(mode="split", max_nodes=3000 if variant == "D0" else 2400,
                              max_splits=120 if variant == "D0" else 60)


@torch.no_grad()
def refresh_j8(model, affected):
    """Reapply original J spatial candidate/prior rule only on affected K4 rows.

Retained node logits and Adam moments follow physical node IDs across slots.
New candidate slots start with zero correction/moments; the tensor's scalar
Adam step is retained. Parameter identity and every unaffected row stay exact.
"""
    skin = model.skinning
    if skin is None:
        return dict(active=False)
    rows = affected.nonzero().flatten()
    if not len(rows):
        return dict(active=True, affected_rows=0)
    g = model.deformation_graph
    edges = model.graph_motion_residual.edge_indices
    n = len(g.node_xyz)
    adjacency = _dense_adjacency(edges, n)
    two = adjacency | (adjacency.float() @ adjacency.float() > 0)
    two.fill_diagonal_(True)
    spacing, _ = graph_spacing(g.node_xyz, edges)
    ids, weights = g.gaussian_node_indices[rows], g.gaussian_node_weights[rows]
    anchor = ids.gather(1, weights.argmax(1)[:, None])[:, 0]
    new_ids, targets = [], []
    for lo in range(0, len(rows), 512):
        distance = torch.cdist(model._xyz.detach()[rows[lo:lo+512]], g.node_xyz,
                               compute_mode="donot_use_mm_for_euclid_dist")
        score = distance.square() / (2 * spacing[anchor[lo:lo+512], None].square())
        selected = connected_growth(ids[lo:lo+512], adjacency, two, score)
        new_ids.append(selected)
        targets.append(warm_start_prior(score, selected, weights[lo:lo+512]))
    new_ids, target = torch.cat(new_ids), torch.cat(targets)
    assert (new_ids.sort(1).values.diff(dim=1) > 0).all()
    assert torch.isfinite(target).all() and (target > 0).all()
    assert torch.allclose(target.sum(1), torch.ones_like(target[:, 0]), atol=2e-6, rtol=0)
    # Connected-growth requires a connected starting support. A native adaptive
    # K4 repair can break it; report and abort instead of silently changing K4.
    reach = adjacency[new_ids[:, :, None], new_ids[:, None, :]]
    reach |= torch.eye(8, dtype=torch.bool, device=new_ids.device)[None]
    for k in range(8):
        reach |= reach[:, :, k:k+1] & reach[:, k:k+1, :]
    if not reach.all():
        raise RuntimeError("Original split K4 produces disconnected J8; no candidate fallback authorized")
    old_ids = skin.indices[rows].clone()
    match = new_ids[:, :, None] == old_ids[:, None, :]
    keep = match.any(-1)
    slots = match.long().argmax(-1)
    outside = ~affected
    before_outside = digest({k: v[outside] for k, v in skin.state_dict().items()
                             if v.ndim == 2})
    parameter_id = id(skin.delta)
    adam = model.optimizer.state.get(skin.delta, {})
    step_before = digest(adam.get("step"))
    for value in [skin.delta, *[adam[k] for k in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq") if k in adam]]:
        old = value[rows].clone()
        value[rows] = torch.where(keep, old.gather(1, slots), torch.zeros_like(old))
        assert torch.equal(value[rows][keep], old.gather(1, slots)[keep])
    skin.indices[rows] = new_ids
    skin.initial[rows] = torch.cat((weights, torch.zeros_like(weights)), 1)
    skin.target[rows] = target
    assert id(skin.delta) == parameter_id and digest(adam.get("step")) == step_before
    assert before_outside == digest({k: v[outside] for k, v in skin.state_dict().items() if v.ndim == 2})
    return dict(active=True, affected_rows=len(rows),
                changed_candidate_rows=int((new_ids != old_ids).any(1).sum()),
                retained_slots=int(keep.sum()), new_slots=int((~keep).sum()),
                retained_logits_and_Adam_by_node_ID=True, new_slot_logits_Adam_zero=True,
                scalar_Adam_step_unchanged=True, parameter_identity_unchanged=True,
                outside_rows_exact=True, connected=True, fallbacks=0,
                candidate_max_node=int(skin.indices.max()),
                directly_used_nodes=int(skin.indices.unique().numel()),
                phase=float(skin.phase))


class J8Commit:
    def _append_children(self, parents, positions):
        result = super()._append_children(parents, positions)
        self._j8_audit = refresh_j8(self.model, result[0])
        return result

    def _record(self, event):
        event["J8_commit"] = getattr(self, "_j8_audit", {"splits": 0})
        event["selection_protocol"] = self.variant
        event["selection_support"] = "original local K4; active rendered operator is J8"
        return super()._record(event)

    def adapt(self, *args, **kwargs):
        self._j8_audit = {"splits": 0}
        return super().adapt(*args, **kwargs)


class B1SelectionJ8(J8Commit, AdaptiveG3NodeManager):
    """B1 score/proposal/individual safety, with common G3 extension on commit."""
    variant = "C0_B1_selection_on_G3_J8"

    def update_statistics(self, gradient, visible):
        return AdaptiveNodeManager.update_statistics(self, gradient, visible)

    def _candidate_shock(self, parent, position, time, old_node_dx):
        old_base = self.model.node_mlp(self.graph.node_xyz, time)[0]
        return AdaptiveNodeManager._candidate_shock(self, parent, position, time, old_base)


class A20J8(J8Commit, AdaptiveG3NodeManager):
    variant = "A0_A20_selection_on_G3_J8"


class D3J8(J8Commit, CandidateExpansionManager):
    variant = "D0_D3_selection_on_G3_J8"


def make_manager(model, variant, output):
    cls = {"C0": B1SelectionJ8, "A0": A20J8, "D0": D3J8}[variant]
    kw = {"admission": "D3"} if variant == "D0" else {}
    return cls(model, config(variant), str(output), score="current", placement="centroid", **kw)


@torch.no_grad()
def import_pre_split_statistics(manager, state):
    """Transfer observer history only at the common, unsplit 5k source."""
    assert state["iteration"] == 5000 and not state["events"]
    assert len(state["node_xyz"]) == 2000
    for key, current in [("node_xyz", manager.graph.node_xyz),
                         ("gaussian_node_indices", manager.graph.gaussian_node_indices),
                         ("gaussian_node_weights", manager.graph.gaussian_node_weights)]:
        assert torch.equal(state[key].to(current), current), key
    for key in ("gaussian_score", "gaussian_observations", "running_log_mean", "running_log_var"):
        value = state[key]
        setattr(manager, key, None if value is None else value.to(manager.model._xyz.device).clone())
    if "adaptive_g3" in state:
        for key in ("direction_sum", "direction_count", "last_split", "surface_weights"):
            value = state["adaptive_g3"][key]
            setattr(manager, key, value.to(getattr(manager, key)).clone())
