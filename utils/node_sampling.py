"""Deterministic FPS and exact, cached canonical-space associations."""
import torch


@torch.no_grad()
def farthest_point_sampling(xyz: torch.Tensor, num_samples: int) -> torch.LongTensor:
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not torch.isfinite(xyz).all():
        raise ValueError("xyz must be finite [N, 3]")
    if not 0 < num_samples <= len(xyz):
        raise ValueError("num_samples must be in [1, N]")
    selected = torch.empty(num_samples, dtype=torch.long, device=xyz.device)
    distances = torch.full((len(xyz),), float("inf"), device=xyz.device)
    current = torch.zeros((), dtype=torch.long, device=xyz.device)
    for j in range(num_samples):
        selected[j] = current
        distance = (xyz - xyz[current]).square().sum(-1)
        torch.minimum(distances, distance, out=distances)
        # Exclude selected indices, including coincident points.
        distances[current] = -1
        current = distances.argmax()
    return selected


@torch.no_grad()
def build_gaussian_node_knn(gaussian_xyz: torch.Tensor, node_xyz: torch.Tensor,
                            k: int = 4) -> tuple:
    if not 0 < k <= len(node_xyz):
        raise ValueError("k must be in [1, M]")
    if gaussian_xyz.ndim != 2 or node_xyz.ndim != 2 or gaussian_xyz.shape[1] != 3 or node_xyz.shape[1] != 3:
        raise ValueError("positions must be [N, 3] and [M, 3]")
    if not torch.isfinite(gaussian_xyz).all() or not torch.isfinite(node_xyz).all():
        raise ValueError("positions must be finite")
    indices, weights = [], []
    for chunk in gaussian_xyz.split(4096):
        # Direct distances avoid cancellation for coincident points.
        distance = torch.cdist(chunk, node_xyz, compute_mode="donot_use_mm_for_euclid_dist")
        distance, index = distance.topk(k, largest=False, sorted=True)
        weight = (distance + 1e-8).reciprocal()
        indices.append(index)
        weights.append(weight / weight.sum(-1, keepdim=True))
    return torch.cat(indices), torch.cat(weights)
