"""Fixed surface-aware Gaussian graph and short-term local isometry loss."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class GaussianIsometryDiagnostics:
    relative_length_change: torch.Tensor


class LocalGaussianIsometryGraph(nn.Module):
    """Undirected canonical KNN graph filtered by shared deformation K4."""

    def __init__(self, edge_indices, edge_weights, canonical_lengths,
                 overlap_count, degree, fallback_directed_edges, candidate_k,
                 minimum_degree, sigma):
        super().__init__()
        self.candidate_k = int(candidate_k)
        self.minimum_degree = int(minimum_degree)
        self.fallback_directed_edges = int(fallback_directed_edges)
        self.sigma = float(sigma)
        self.register_buffer("edge_indices", edge_indices.detach().long().clone())
        self.register_buffer("edge_weights", edge_weights.detach().clone())
        self.register_buffer("canonical_lengths", canonical_lengths.detach().clone())
        self.register_buffer("overlap_count", overlap_count.detach().long().clone())
        self.register_buffer("degree", degree.detach().long().clone())
        self._validate()

    def _validate(self):
        if self.edge_indices.ndim != 2 or self.edge_indices.shape[0] != 2:
            raise ValueError("edge_indices must be [2,E]")
        edge_count = self.edge_indices.shape[1]
        if self.edge_weights.shape != (edge_count,) or self.canonical_lengths.shape != (edge_count,):
            raise ValueError("edge attributes must be [E]")
        if self.overlap_count.shape != (edge_count,):
            raise ValueError("overlap_count must be [E]")
        if edge_count == 0 or len(self.degree) == 0:
            raise ValueError("isometry graph cannot be empty")
        if (self.edge_indices[0] >= self.edge_indices[1]).any():
            raise ValueError("edges must be unique canonical undirected pairs")
        if not torch.isfinite(self.edge_weights).all() or (self.edge_weights <= 0).any():
            raise ValueError("edge weights must be finite and positive")
        if not torch.isfinite(self.canonical_lengths).all() or (self.canonical_lengths <= 0).any():
            raise ValueError("canonical lengths must be finite and positive")
        if (self.degree < self.minimum_degree).any():
            raise ValueError("minimum degree target was not satisfied")

    @staticmethod
    def _shared_k4_count(left, right):
        return (left[:, :, None] == right[:, None, :]).sum(dim=(1, 2))

    @classmethod
    @torch.no_grad()
    def build(cls, canonical_xyz, candidate_indices, gaussian_node_indices,
              candidate_k=12, minimum_degree=4, chunk_size=65536):
        if canonical_xyz.ndim != 2 or canonical_xyz.shape[1] != 3:
            raise ValueError("canonical_xyz must be [N,3]")
        n = len(canonical_xyz)
        if candidate_indices.shape[0] != n or gaussian_node_indices.shape != (n, 4):
            raise ValueError("candidate/K4 tensors do not match Gaussian count")
        if candidate_k < minimum_degree or candidate_indices.shape[1] < candidate_k:
            raise ValueError("candidate_k must cover minimum_degree")

        xyz_cpu = canonical_xyz.detach().float().cpu()
        candidates = candidate_indices[:, :candidate_k].detach().long().cpu()
        k4 = gaussian_node_indices.detach().long().cpu()
        selected_keys, fallback = [], 0
        for start in range(0, n, chunk_size):
            end = min(n, start + chunk_size)
            rows = torch.arange(start, end)[:, None].expand(-1, candidate_k)
            cols = candidates[start:end]
            overlap = (k4[start:end, None, :, None] ==
                       k4[cols][:, :, None, :]).sum(dim=(2, 3))
            keep = overlap > 0
            shortage = (minimum_degree - keep.sum(1)).clamp_min(0)
            if shortage.any():
                for offset in range(end - start):
                    needed = int(shortage[offset])
                    if needed:
                        available = (~keep[offset]).nonzero(as_tuple=False).flatten()[:needed]
                        keep[offset, available] = True
                        fallback += needed
            row = rows[keep]
            col = cols[keep]
            lower, upper = torch.minimum(row, col), torch.maximum(row, col)
            valid = lower != upper
            selected_keys.append(lower[valid] * n + upper[valid])
        keys = torch.unique(torch.cat(selected_keys), sorted=True)
        source, target = keys.div(n, rounding_mode="floor"), keys.remainder(n)
        edge_indices = torch.stack((source, target))
        degree = torch.bincount(torch.cat((source, target)), minlength=n)
        if (degree < minimum_degree).any():
            raise RuntimeError("symmetrized graph missed minimum degree")

        lengths = (xyz_cpu[source] - xyz_cpu[target]).norm(dim=1).clamp_min(1e-8)
        sigma = float(lengths.median())
        weights = torch.exp(
            -lengths.square() / (2 * max(sigma, 1e-8) ** 2)).clamp_min(1e-12)
        overlap_parts = []
        for start in range(0, len(source), chunk_size):
            end = min(len(source), start + chunk_size)
            overlap_parts.append(cls._shared_k4_count(
                k4[source[start:end]], k4[target[start:end]]))
        overlap_count = torch.cat(overlap_parts)
        device = canonical_xyz.device
        return cls(edge_indices.to(device), weights.to(device), lengths.to(device),
                   overlap_count.to(device), degree.to(device), fallback,
                   candidate_k, minimum_degree, sigma)

    @classmethod
    def from_state_dict_payload(cls, payload, device=None):
        device = device or payload["edge_indices"].device
        return cls(payload["edge_indices"].to(device),
                   payload["edge_weights"].to(device),
                   payload["canonical_lengths"].to(device),
                   payload["overlap_count"].to(device),
                   payload["degree"].to(device),
                   int(payload["fallback_directed_edges"]),
                   int(payload["candidate_k"]),
                   int(payload["minimum_degree"]),
                   float(payload["sigma"]))

    def state_dict_payload(self):
        return {
            "format_version": 1,
            "candidate_k": self.candidate_k,
            "minimum_degree": self.minimum_degree,
            "fallback_directed_edges": self.fallback_directed_edges,
            "sigma": self.sigma,
            "edge_indices": self.edge_indices.detach().clone(),
            "edge_weights": self.edge_weights.detach().clone(),
            "canonical_lengths": self.canonical_lengths.detach().clone(),
            "overlap_count": self.overlap_count.detach().clone(),
            "degree": self.degree.detach().clone(),
        }

    def forward(self, positions_t1, positions_t2, eps=1e-12,
                return_diagnostics=False):
        if positions_t1.shape != positions_t2.shape or positions_t1.ndim != 2 or positions_t1.shape[1] != 3:
            raise ValueError("positions must be matching [N,3] tensors")
        source, target = self.edge_indices
        length_t1 = (positions_t1[source] - positions_t1[target]).norm(dim=1)
        length_t2 = (positions_t2[source] - positions_t2[target]).norm(dim=1)
        relative = (length_t1 - length_t2) / self.canonical_lengths.clamp_min(eps)
        loss = (self.edge_weights * relative.square()).sum() / self.edge_weights.sum().clamp_min(eps)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite local Gaussian isometry loss")
        if not return_diagnostics:
            return loss
        return loss, GaussianIsometryDiagnostics(relative.detach().abs())

    def graph_statistics(self):
        overlap_histogram = {
            str(value): float((self.overlap_count == value).float().mean())
            for value in range(5)
        }
        return {
            "edges": int(self.edge_indices.shape[1]),
            "candidate_k": self.candidate_k,
            "minimum_degree": self.minimum_degree,
            "fallback_directed_edges": self.fallback_directed_edges,
            "fallback_edge_ratio": self.fallback_directed_edges /
                                   max(self.candidate_k * len(self.degree), 1),
            "sigma": self.sigma,
            "degree": {
                "min": int(self.degree.min()),
                "median": float(self.degree.float().median()),
                "p90": float(torch.quantile(self.degree.float(), 0.90)),
                "p95": float(torch.quantile(self.degree.float(), 0.95)),
                "max": int(self.degree.max()),
                "mean": float(self.degree.float().mean()),
            },
            "overlap_fraction": overlap_histogram,
        }
