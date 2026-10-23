"""Canonical deformation-node ARAP graph and loss.

The graph topology, edge weights, and normalization scales are derived only
from canonical node positions and remain fixed during optimization.  Local
Procrustes rotations are refit for every temporal pair, then detached so the
SVD is not part of the backward graph.
"""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ARAPLossDiagnostics:
    rotations: torch.Tensor
    singular_values: torch.Tensor
    reflection_ratio: float
    per_node_loss: torch.Tensor


class CanonicalNodeARAPGraph(nn.Module):
    """Symmetric union of canonical KNN neighborhoods."""

    def __init__(self, canonical_xyz, neighbor_indices, neighbor_mask,
                 edge_weights, local_scale, k):
        super().__init__()
        self.k = int(k)
        self.register_buffer("canonical_xyz", canonical_xyz.detach().clone())
        self.register_buffer("neighbor_indices", neighbor_indices.detach().clone())
        self.register_buffer("neighbor_mask", neighbor_mask.detach().clone())
        self.register_buffer("edge_weights", edge_weights.detach().clone())
        self.register_buffer("local_scale", local_scale.detach().clone())
        self._validate()

    def _validate(self):
        n = len(self.canonical_xyz)
        if self.canonical_xyz.shape != (n, 3) or n <= self.k:
            raise ValueError("canonical_xyz must be [N,3] with N > K")
        shape = self.neighbor_indices.shape
        if self.neighbor_indices.ndim != 2 or shape[0] != n:
            raise ValueError("neighbor_indices must be [N,D]")
        if self.neighbor_mask.shape != shape or self.edge_weights.shape != shape:
            raise ValueError("ARAP neighbor tensors must have matching shapes")
        if self.local_scale.shape != (n,):
            raise ValueError("local_scale must be [N]")
        if self.neighbor_indices.dtype != torch.long:
            raise ValueError("neighbor_indices must be int64")
        if self.neighbor_mask.dtype != torch.bool:
            raise ValueError("neighbor_mask must be bool")
        if not torch.isfinite(self.canonical_xyz).all():
            raise ValueError("canonical_xyz must be finite")
        if not torch.isfinite(self.edge_weights).all() or (self.edge_weights < 0).any():
            raise ValueError("edge_weights must be finite and nonnegative")
        if not torch.isfinite(self.local_scale).all() or (self.local_scale <= 0).any():
            raise ValueError("local_scale must be finite and positive")
        if (self.neighbor_mask.sum(1) < self.k).any():
            raise ValueError("symmetric graph must retain at least K neighbors per node")

    @classmethod
    @torch.no_grad()
    def build(cls, canonical_xyz, k=6):
        if canonical_xyz.ndim != 2 or canonical_xyz.shape[1] != 3:
            raise ValueError("canonical_xyz must be [N,3]")
        n = len(canonical_xyz)
        if k < 1 or n <= k:
            raise ValueError("ARAP K must satisfy 1 <= K < N")
        distances = torch.cdist(canonical_xyz, canonical_xyz)
        distances.fill_diagonal_(float("inf"))
        knn_distance, knn = distances.topk(k, largest=False, dim=1)
        local_scale = knn_distance.median(dim=1).values.clamp_min(1e-8)

        # Undirected union of directed KNN edges.  Every union edge is stored in
        # both directions, giving identical topology for the two endpoints.
        adjacency = torch.zeros((n, n), dtype=torch.bool, device=canonical_xyz.device)
        rows = torch.arange(n, device=canonical_xyz.device)[:, None].expand_as(knn)
        adjacency[rows, knn] = True
        adjacency = adjacency | adjacency.transpose(0, 1)
        adjacency.fill_diagonal_(False)
        degree = adjacency.sum(1)
        max_degree = int(degree.max().item())
        indices = torch.arange(n, device=canonical_xyz.device)[:, None].expand(n, max_degree).clone()
        mask = torch.zeros((n, max_degree), dtype=torch.bool, device=canonical_xyz.device)
        for node in range(n):
            neighbors = adjacency[node].nonzero(as_tuple=False).flatten()
            indices[node, :len(neighbors)] = neighbors
            mask[node, :len(neighbors)] = True

        neighbor_distance = (canonical_xyz[:, None] - canonical_xyz[indices]).norm(dim=-1)
        sigma_i = local_scale[:, None]
        sigma_j = local_scale[indices]
        raw_i = torch.exp(-neighbor_distance.square() / (2 * sigma_i.square()))
        raw_j = torch.exp(-neighbor_distance.square() / (2 * sigma_j.square()))
        weights = 0.5 * (raw_i + raw_j)
        weights = torch.where(mask, weights, torch.zeros_like(weights))
        return cls(canonical_xyz, indices.long(), mask, weights, local_scale, k)

    @classmethod
    def from_state_dict_payload(cls, payload, device=None):
        device = device or payload["canonical_xyz"].device
        return cls(
            payload["canonical_xyz"].to(device),
            payload["neighbor_indices"].to(device),
            payload["neighbor_mask"].to(device),
            payload["edge_weights"].to(device),
            payload["local_scale"].to(device),
            int(payload["k"]),
        )

    def state_dict_payload(self):
        return {
            "format_version": 1,
            "k": self.k,
            "canonical_xyz": self.canonical_xyz.detach().clone(),
            "neighbor_indices": self.neighbor_indices.detach().clone(),
            "neighbor_mask": self.neighbor_mask.detach().clone(),
            "edge_weights": self.edge_weights.detach().clone(),
            "local_scale": self.local_scale.detach().clone(),
        }

    @staticmethod
    def _fit_rotations(edges_source, edges_target, weights):
        # Find R such that target ~= R @ source.  Padding has zero weight.
        with torch.no_grad():
            covariance = torch.einsum(
                "nki,nk,nkj->nij", edges_source, weights, edges_target)
            u, singular, vh = torch.linalg.svd(covariance, full_matrices=False)
            v = vh.transpose(-1, -2)
            candidate = v @ u.transpose(-1, -2)
            reflection = torch.det(candidate) < 0
            correction = torch.eye(3, device=covariance.device,
                                   dtype=covariance.dtype).expand(len(covariance), 3, 3).clone()
            correction[:, -1, -1] = torch.where(
                reflection, correction.new_tensor(-1), correction.new_tensor(1))
            rotation = v @ correction @ u.transpose(-1, -2)
        return rotation.detach(), singular.detach(), reflection

    def forward(self, positions_t1, positions_t2, eps=1e-12, return_diagnostics=False):
        if positions_t1.shape != self.canonical_xyz.shape or positions_t2.shape != positions_t1.shape:
            raise ValueError("ARAP positions must match canonical node shape")
        if not torch.isfinite(positions_t1).all() or not torch.isfinite(positions_t2).all():
            raise FloatingPointError("ARAP received non-finite node positions")
        idx = self.neighbor_indices
        edges_t1 = positions_t1[:, None, :] - positions_t1[idx]
        edges_t2 = positions_t2[:, None, :] - positions_t2[idx]
        weights = self.edge_weights
        rotation, singular, reflection = self._fit_rotations(edges_t2, edges_t1, weights)
        rotated_t2 = torch.einsum("nij,nkj->nki", rotation, edges_t2)
        residual_sq = (edges_t1 - rotated_t2).square().sum(-1)
        numerator = (weights * residual_sq).sum(1)
        denominator = (self.local_scale.square() + eps) * (weights.sum(1) + eps)
        per_node = numerator / denominator
        loss = per_node.mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite node ARAP loss")
        if not return_diagnostics:
            return loss
        diagnostics = ARAPLossDiagnostics(
            rotations=rotation,
            singular_values=singular,
            reflection_ratio=float(reflection.float().mean()),
            per_node_loss=per_node.detach(),
        )
        return loss, diagnostics

    @torch.no_grad()
    def stretch_statistics(self, positions):
        if positions.ndim == 2:
            positions = positions[None]
        if positions.ndim != 3 or positions.shape[1:] != self.canonical_xyz.shape:
            raise ValueError("positions must be [T,N,3] or [N,3]")
        idx = self.neighbor_indices
        canonical_length = (self.canonical_xyz[:, None] - self.canonical_xyz[idx]).norm(dim=-1)
        current_length = (positions[:, :, None] - positions[:, idx]).norm(dim=-1)
        ratio_error = (current_length / canonical_length.clamp_min(1e-12) - 1).abs()
        values = ratio_error[:, self.neighbor_mask].flatten()
        return {
            "median": float(values.median()),
            "p90": float(torch.quantile(values, 0.90)),
            "p95": float(torch.quantile(values, 0.95)),
            "max": float(values.max()),
        }
