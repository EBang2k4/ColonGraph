"""Surface-graph-aware Gaussian control and node-motion message passing."""

import math

import torch
from torch import nn


def _dense_adjacency(edge_indices, node_count, device=None):
    device = device or edge_indices.device
    adjacency = torch.zeros((node_count, node_count), dtype=torch.bool, device=device)
    source, target = edge_indices.to(device)
    adjacency[source, target] = True
    adjacency[target, source] = True
    return adjacency


@torch.no_grad()
def k4_topology_tensors(indices, edge_indices, node_count):
    """Return induced-subgraph connectivity diagnostics for every K4."""
    if indices.ndim != 2 or indices.shape[1] != 4:
        raise ValueError("indices must be [N,4]")
    adjacency = _dense_adjacency(edge_indices, node_count, indices.device)
    sub = adjacency[indices[:, :, None], indices[:, None, :]]
    eye = torch.eye(4, dtype=torch.bool, device=indices.device)[None]
    reach = sub | eye
    for pivot in range(4):
        reach = reach | (reach[:, :, pivot:pivot+1] & reach[:, pivot:pivot+1, :])
    connected = reach.all(dim=(1, 2))
    positions = torch.arange(4, device=indices.device)[None, None, :]
    labels = torch.where(reach, positions, positions.new_full((), 4)).amin(-1)
    component_count = torch.ones(len(indices), dtype=torch.long, device=indices.device)
    for column in range(1, 4):
        component_count += (labels[:, column:column+1] != labels[:, :column]).all(1)
    internal_edges = sub.triu(1).sum(dim=(1, 2))

    distance = torch.where(sub, torch.ones_like(sub, dtype=torch.float32),
                           torch.full_like(sub, float("inf"), dtype=torch.float32))
    distance = torch.where(eye, torch.zeros_like(distance), distance)
    for pivot in range(4):
        distance = torch.minimum(
            distance, distance[:, :, pivot:pivot+1] + distance[:, pivot:pivot+1, :])
    diameter = distance.amax(dim=(1, 2))
    return {
        "connected": connected,
        "component_count": component_count,
        "internal_edges": internal_edges,
        "diameter": diameter,
    }


def _summary(values):
    values = torch.as_tensor(values).float().flatten()
    finite = values[torch.isfinite(values)]
    result = {
        "mean": float(finite.mean()) if len(finite) else None,
        "median": float(finite.median()) if len(finite) else None,
        "p90": float(torch.quantile(finite, .9)) if len(finite) else None,
        "p95": float(torch.quantile(finite, .95)) if len(finite) else None,
        "max": float(finite.max()) if len(finite) else None,
        "finite_fraction": float(torch.isfinite(values).float().mean()),
    }
    return result


@torch.no_grad()
def k4_topology_report(indices, edge_indices, node_count):
    values = k4_topology_tensors(indices, edge_indices, node_count)
    return {
        "gaussians": len(indices),
        "connected_ratio": float(values["connected"].float().mean()),
        "component_count": _summary(values["component_count"]),
        "internal_edge_count_of_6": _summary(values["internal_edges"]),
        "diameter": _summary(values["diameter"]),
    }


@torch.no_grad()
def repair_gaussian_k4_minimal(gaussian_xyz, node_xyz, original_indices,
                               original_weights, edge_indices):
    """Repair only disconnected K4s with a connected anchor-local selection."""
    if original_indices.shape != original_weights.shape or original_indices.shape[1] != 4:
        raise ValueError("original K4 indices/weights must match [N,4]")
    if gaussian_xyz.shape != (len(original_indices), 3) or node_xyz.ndim != 2:
        raise ValueError("canonical coordinates do not match K4")
    device = original_indices.device
    node_count = len(node_xyz)
    before = k4_topology_tensors(original_indices, edge_indices, node_count)
    disconnected = (~before["connected"]).nonzero().flatten().tolist()

    edges_cpu = edge_indices.detach().cpu()
    adjacency = [set() for _ in range(node_count)]
    for left, right in edges_cpu.T.tolist():
        adjacency[left].add(right)
        adjacency[right].add(left)
    xyz_cpu = gaussian_xyz.detach().cpu()
    nodes_cpu = node_xyz.detach().cpu()
    indices_cpu = original_indices.detach().cpu().clone()
    weights_cpu = original_weights.detach().cpu().clone()
    fallback = torch.zeros(len(original_indices), dtype=torch.bool)

    for gaussian_index in disconnected:
        old = original_indices[gaussian_index].detach().cpu().tolist()
        old_weights = original_weights[gaussian_index].detach().cpu()
        anchor = old[int(old_weights.argmax())]

        old_set = set(old)
        component = {anchor}
        frontier = [anchor]
        while frontier:
            current = frontier.pop()
            for neighbor in adjacency[current] & old_set:
                if neighbor not in component:
                    component.add(neighbor)
                    frontier.append(neighbor)
        selected = [node for node in old if node in component]

        one_hop = set(adjacency[anchor]) - set(selected)
        two_hop = set()
        for node in adjacency[anchor]:
            two_hop.update(adjacency[node])
        two_hop -= set(selected)
        pools = (one_hop, two_hop)
        point = xyz_cpu[gaussian_index]
        for pool in pools:
            while len(selected) < 4:
                eligible = [node for node in pool if node not in selected and
                            any(node in adjacency[current] for current in selected)]
                if not eligible:
                    break
                candidate_xyz = nodes_cpu[torch.tensor(eligible, dtype=torch.long)]
                distance = (candidate_xyz-point).square().sum(-1)
                selected.append(eligible[int(distance.argmin())])
        if len(selected) != 4:
            fallback[gaussian_index] = True
            continue
        indices_cpu[gaussian_index] = torch.tensor(selected, dtype=torch.long)
        distance = (nodes_cpu[indices_cpu[gaussian_index]]-point).norm(dim=-1)
        inverse = (distance+1e-8).reciprocal()
        weights_cpu[gaussian_index] = inverse/inverse.sum()

    repaired_indices = indices_cpu.to(device)
    repaired_weights = weights_cpu.to(device=device, dtype=original_weights.dtype)
    after = k4_topology_tensors(repaired_indices, edge_indices, node_count)
    changed = (repaired_indices != original_indices).any(1)
    if not after["connected"].all():
        bad = (~after["connected"]).nonzero().flatten()
        if not torch.equal(bad.cpu(), fallback.nonzero().flatten()):
            raise RuntimeError("Graph-aware K4 repair produced an unexpected disconnected set")
    unchanged = ~changed
    if (not torch.equal(repaired_indices[unchanged], original_indices[unchanged]) or
            not torch.equal(repaired_weights[unchanged], original_weights[unchanged])):
        raise RuntimeError("Minimal repair changed an already-compatible K4")
    audit = {
        "before": k4_topology_report(original_indices, edge_indices, node_count),
        "after": k4_topology_report(repaired_indices, edge_indices, node_count),
        "changed_gaussians": int(changed.sum()),
        "changed_ratio": float(changed.float().mean()),
        "fallback_gaussians": int(fallback.sum()),
        "fallback_ratio": float(fallback.float().mean()),
        "unchanged_indices_bit_exact": True,
        "unchanged_weights_bit_exact": True,
    }
    return repaired_indices, repaired_weights, changed, fallback.to(device), audit


def preserve_base_motion(base, residual, alpha=0.10, eps=1e-8):
    """Limit cancellation along the base direction; preserve perpendicular motion.

    Near-zero base motion has no reliable direction, so leave its residual alone.
    Adding only the correction preserves the existing result bit-exactly when
    the guard is inactive, including the zero-initialized decoder.
    """
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("motion guard alpha must be finite in [0,1]")
    norm = base.norm(dim=-1, keepdim=True)
    unit = base / norm.clamp_min(eps)
    parallel = (residual * unit).sum(dim=-1, keepdim=True)
    correction = (-alpha * norm - parallel).clamp_min(0)
    correction = torch.where(norm > eps, correction, torch.zeros_like(correction))
    return residual + correction * unit


def graph_index_add_(target, dim, indices, source, deterministic=False):
    """Opt-in deterministic built-in reduction, scoped to this operation only.

    Never leave the process-wide deterministic/warn-only setting changed. This
    does not promise deterministic custom CUDA rasterization or full training.
    """
    if not deterministic:
        return target.index_add_(dim, indices, source)
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        return target.index_add_(dim, indices, source)
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


class GraphMotionResidual(nn.Module):
    """One-hop relative-motion message passing with a zero-init residual head."""

    def __init__(self, canonical_xyz, edge_indices, edge_weights,
                 hidden_dim=64, seed=0, mode="g2", cancellation_alpha=0.10,
                 deterministic_aggregation=False):
        super().__init__()
        if mode not in ("g2", "g3", "g4"):
            raise ValueError("GraphMotionResidual mode must be g2, g3 or g4")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        self.mode = mode
        self.hidden_dim = int(hidden_dim)
        self.seed = int(seed)
        self.deterministic_aggregation = bool(deterministic_aggregation)
        if not math.isfinite(cancellation_alpha) or not 0 <= cancellation_alpha <= 1:
            raise ValueError("cancellation_alpha must be finite in [0,1]")
        self.cancellation_alpha = float(cancellation_alpha)
        self.register_buffer("canonical_xyz", canonical_xyz.detach().clone())
        self.register_buffer("edge_indices", edge_indices.detach().long().clone())
        self.register_buffer("edge_weights", edge_weights.detach().float().clone())
        source, target = self.edge_indices
        self.register_buffer("recipient", torch.cat((source, target)))
        self.register_buffer("neighbor", torch.cat((target, source)))
        self.register_buffer("directed_weights", torch.cat((self.edge_weights,
                                                              self.edge_weights)))
        denominator = torch.zeros(len(canonical_xyz), device=edge_weights.device,
                                  dtype=edge_weights.dtype)
        graph_index_add_(denominator, 0, self.recipient, self.directed_weights,
                         self.deterministic_aggregation)
        if (denominator <= 0).any():
            raise ValueError("GraphMotionResidual graph contains an isolated node")
        self.register_buffer("weight_denominator", denominator)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.decoder = nn.Sequential(
                nn.Linear(6, self.hidden_dim), nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
                nn.Linear(self.hidden_dim, 3),
            )
            nn.init.zeros_(self.decoder[-1].weight)
            nn.init.zeros_(self.decoder[-1].bias)

    def aggregate(self, base_translation):
        if base_translation.shape != self.canonical_xyz.shape:
            raise ValueError("base_translation must be [M,3]")
        relative_motion = (base_translation[self.neighbor] -
                           base_translation[self.recipient])
        relative_position = (self.canonical_xyz[self.neighbor] -
                             self.canonical_xyz[self.recipient])
        message = torch.cat((relative_motion, relative_position), dim=-1)
        aggregate = torch.zeros((len(self.canonical_xyz), 6),
                                device=message.device, dtype=message.dtype)
        graph_index_add_(aggregate, 0, self.recipient,
                         message*self.directed_weights[:, None],
                         self.deterministic_aggregation)
        return aggregate/self.weight_denominator[:, None]

    @torch.no_grad()
    def append_local_nodes(self, new_xyz, new_edges, new_weights):
        """Append local edges/nodes without rebuilding or changing old edges.

        The shared decoder and its optimizer state are untouched.  Denominators
        change only for recipients incident to an appended edge.
        """
        old_count, old_edges = len(self.canonical_xyz), self.edge_indices.shape[1]
        if new_xyz.ndim != 2 or new_xyz.shape[1] != 3 or not len(new_xyz):
            raise ValueError("new_xyz must be nonempty [C,3]")
        count = old_count + len(new_xyz)
        if new_edges.shape != (2, len(new_weights)) or not torch.isfinite(new_weights).all():
            raise ValueError("invalid appended edges/weights")
        if ((new_edges < 0).any() or (new_edges >= count).any() or
                (new_edges[0] == new_edges[1]).any() or (new_weights <= 0).any()):
            raise ValueError("invalid appended graph")
        if not (new_edges >= old_count).any(dim=0).all():
            raise ValueError("every added edge must touch a new child")
        keys = new_edges.amin(0) * count + new_edges.amax(0)
        if len(keys.unique()) != len(keys):
            raise ValueError("duplicate appended edge")
        denominator = torch.cat((self.weight_denominator,
                                 new_weights.new_zeros(len(new_xyz))))
        recipients = torch.cat((new_edges[0], new_edges[1]))
        graph_index_add_(denominator, 0, recipients,
                         torch.cat((new_weights, new_weights)),
                         self.deterministic_aggregation)
        if (denominator <= 0).any():
            raise ValueError("isolated child")
        self.canonical_xyz = torch.cat((self.canonical_xyz, new_xyz))
        self.edge_indices = torch.cat((self.edge_indices, new_edges), dim=1)
        self.edge_weights = torch.cat((self.edge_weights, new_weights))
        source, target = self.edge_indices
        self.recipient = torch.cat((source, target))
        self.neighbor = torch.cat((target, source))
        self.directed_weights = torch.cat((self.edge_weights, self.edge_weights))
        self.weight_denominator = denominator
        return {"old_nodes": old_count, "new_nodes": count,
                "old_edges": old_edges, "new_edges": self.edge_indices.shape[1]}

    def forward(self, base_translation):
        residual = self.decoder(self.aggregate(base_translation))
        if self.mode == "g4":
            residual = preserve_base_motion(
                base_translation, residual, alpha=self.cancellation_alpha)
        return base_translation+residual, residual

    def state_dict_payload(self):
        return {
            "format_version": 1,
            "mode": self.mode,
            "hidden_dim": self.hidden_dim,
            "seed": self.seed,
            "cancellation_alpha": self.cancellation_alpha,
            "deterministic_aggregation": self.deterministic_aggregation,
            "canonical_xyz": self.canonical_xyz.detach().clone(),
            "edge_indices": self.edge_indices.detach().clone(),
            "edge_weights": self.edge_weights.detach().clone(),
            # CUDA index_add can differ by a few ulps across reconstruction.
            # Persist this derived normalization buffer so a checkpoint replay
            # uses the exact same graph operator.
            "weight_denominator": self.weight_denominator.detach().clone(),
            "decoder": self.decoder.state_dict(),
        }

    @classmethod
    def from_state_dict_payload(cls, payload, device=None):
        if 'surface_refine' in payload:
            from utils.surface_graph_refine import SurfaceRefinedResidual
            return SurfaceRefinedResidual.from_surface_payload(payload, device)
        device = device or payload["canonical_xyz"].device
        result = cls(payload["canonical_xyz"].to(device),
                     payload["edge_indices"].to(device),
                     payload["edge_weights"].to(device),
                     payload["hidden_dim"], payload["seed"], payload["mode"],
                     payload.get("cancellation_alpha", 0.10),
                     payload.get("deterministic_aggregation", False)).to(device)
        if "weight_denominator" in payload:
            result.weight_denominator.copy_(payload["weight_denominator"].to(device))
        result.decoder.load_state_dict(payload["decoder"], strict=True)
        return result


@torch.no_grad()
def residual_statistics(base, residual, eps=1e-8):
    base_norm = base.norm(dim=-1).flatten()
    residual_norm = residual.norm(dim=-1).flatten()
    ratio = residual_norm/(base_norm+eps)
    return {
        "base_norm": _summary(base_norm),
        "graph_residual_norm": _summary(residual_norm),
        "graph_residual_ratio": _summary(ratio),
    }
