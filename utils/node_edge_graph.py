"""Fixed deformation-node graphs and short-term edge-length consistency."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class NodeEdgeDiagnostics:
    absolute_relative_change: torch.Tensor


class NodeEdgeConsistencyGraph(nn.Module):
    """An undirected weighted graph with fixed canonical normalization."""

    def __init__(self, canonical_xyz, edge_indices, edge_weights,
                 canonical_lengths, topology, metadata=None):
        super().__init__()
        self.topology = str(topology)
        self.metadata = dict(metadata or {})
        self.register_buffer("canonical_xyz", canonical_xyz.detach().clone())
        self.register_buffer("edge_indices", edge_indices.detach().long().clone())
        self.register_buffer("edge_weights", edge_weights.detach().float().clone())
        self.register_buffer("canonical_lengths", canonical_lengths.detach().float().clone())
        self._validate()

    def _validate(self):
        node_count = len(self.canonical_xyz)
        if self.canonical_xyz.shape != (node_count, 3):
            raise ValueError("canonical_xyz must be [M,3]")
        if self.edge_indices.ndim != 2 or self.edge_indices.shape[0] != 2:
            raise ValueError("edge_indices must be [2,E]")
        edge_count = self.edge_indices.shape[1]
        if self.edge_weights.shape != (edge_count,):
            raise ValueError("edge_weights must be [E]")
        if self.canonical_lengths.shape != (edge_count,):
            raise ValueError("canonical_lengths must be [E]")
        if edge_count == 0:
            raise ValueError("node-edge graph cannot be empty")
        source, target = self.edge_indices
        if (source >= target).any() or source.min() < 0 or target.max() >= node_count:
            raise ValueError("edges must be unique canonical undirected pairs")
        keys = source * node_count + target
        if len(torch.unique(keys)) != edge_count:
            raise ValueError("duplicate node edges")
        if (not torch.isfinite(self.edge_weights).all() or
                (self.edge_weights <= 0).any()):
            raise ValueError("edge weights must be finite and positive")
        if (not torch.isfinite(self.canonical_lengths).all() or
                (self.canonical_lengths <= 0).any()):
            raise ValueError("canonical lengths must be finite and positive")
        degree = torch.bincount(torch.cat((source, target)), minlength=node_count)
        if (degree == 0).any():
            raise ValueError("node-edge graph contains isolated nodes")

    @property
    def degree(self):
        source, target = self.edge_indices
        return torch.bincount(torch.cat((source, target)),
                              minlength=len(self.canonical_xyz))

    def forward(self, positions_t1, positions_t2, eps=1e-12,
                return_diagnostics=False):
        if positions_t1.shape != self.canonical_xyz.shape or positions_t2.shape != positions_t1.shape:
            raise ValueError("node positions must match canonical shape")
        source, target = self.edge_indices
        d1 = (positions_t1[source] - positions_t1[target]).norm(dim=1)
        d2 = (positions_t2[source] - positions_t2[target]).norm(dim=1)
        relative = (d1 - d2) / self.canonical_lengths.clamp_min(eps)
        loss = ((self.edge_weights * relative.square()).sum() /
                self.edge_weights.sum().clamp_min(eps))
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite node-edge consistency loss")
        if not return_diagnostics:
            return loss
        return loss, NodeEdgeDiagnostics(relative.detach().abs())

    def state_dict_payload(self):
        return {
            "format_version": 1,
            "topology": self.topology,
            "metadata": self.metadata,
            "canonical_xyz": self.canonical_xyz.detach().clone(),
            "edge_indices": self.edge_indices.detach().clone(),
            "edge_weights": self.edge_weights.detach().clone(),
            "canonical_lengths": self.canonical_lengths.detach().clone(),
        }

    @classmethod
    def from_state_dict_payload(cls, payload, device=None):
        device = device or payload["canonical_xyz"].device
        return cls(payload["canonical_xyz"].to(device),
                   payload["edge_indices"].to(device),
                   payload["edge_weights"].to(device),
                   payload["canonical_lengths"].to(device),
                   payload["topology"], payload.get("metadata", {}))

    @torch.no_grad()
    def motion_disagreement(self, positions):
        """Mean temporal disagreement of endpoint displacement trajectories."""
        if positions.ndim != 3 or positions.shape[1:] != self.canonical_xyz.shape:
            raise ValueError("positions must be [T,M,3]")
        displacement = positions - self.canonical_xyz[None]
        source, target = self.edge_indices
        values = (displacement[:, source] - displacement[:, target]).norm(
            dim=-1).mean(dim=0)
        return values
