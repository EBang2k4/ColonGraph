"""Deterministic FPS and exact, cached canonical-space associations."""
import torch
import math


class CoarseGradientImportance:
    """Visible-observation XYZ gradient moments, before optimizer clipping.

    Float64 moments avoid negative variance from cancellation. Invisible
    Gaussians are not observations, even if the global KNN loss has a gradient.
    """
    def __init__(self, xyz, gradient_source="total"):
        if gradient_source not in ("total", "depth"):
            raise ValueError("gradient_source must be total or depth")
        self.gradient_source = gradient_source
        self.grad_sum = torch.zeros_like(xyz, dtype=torch.float64)
        self.grad_sq_sum = torch.zeros(len(xyz), device=xyz.device, dtype=torch.float64)
        self.grad_count = torch.zeros(len(xyz), device=xyz.device, dtype=torch.int64)
        self.steps = 0

    @torch.no_grad()
    def update(self, gradient, visible):
        self.steps += 1
        if gradient is None:
            return
        if gradient is None or gradient.shape != self.grad_sum.shape:
            raise ValueError("Coarse XYZ gradients missing or Gaussian topology changed")
        if visible.shape != self.grad_count.shape:
            raise ValueError("Visibility must be [N]")
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("Non-finite coarse XYZ gradient")
        g = gradient.detach().to(torch.float64)
        mask = visible.to(torch.bool)
        self.grad_sum[mask] += g[mask]
        self.grad_sq_sum[mask] += g[mask].square().sum(-1)
        self.grad_count[mask] += 1

    @torch.no_grad()
    def finalize(self):
        n = self.grad_count.clamp_min(1)
        mean = self.grad_sum / n[:, None]
        variance = (self.grad_sq_sum / n - mean.square().sum(-1)).clamp_min(0)
        deviation = variance.sqrt()
        confidence = (self.grad_count.to(torch.float64) / 5).clamp(max=1)
        raw = confidence * deviation
        # P5/P95 over all canonical Gaussians, including unobserved zero scores.
        limits = torch.quantile(raw, raw.new_tensor([.05, .95]))
        width = limits[1] - limits[0]
        importance = ((raw - limits[0]) / width).clamp(0, 1) if width > 0 else torch.zeros_like(raw)
        return {"grad_sum": self.grad_sum, "grad_sq_sum": self.grad_sq_sum,
                "grad_count": self.grad_count, "mean_gradient": mean,
                "deviation": deviation, "confidence": confidence, "raw_score": raw,
                "percentile_limits": limits, "importance": importance.float(),
                "coarse_steps": self.steps,
                "gradient_source": self.gradient_source,
                "observation_rule": f"radii > 0; {self.gradient_source} loss XYZ gradient before clipping",
                "percentile_population": "all canonical Gaussians; unseen score=0"}


def position_loss_gradient(loss: torch.Tensor, xyz: torch.Tensor):
    """Return one loss component's XYZ gradient without touching ``xyz.grad``."""
    if not loss.requires_grad:
        return None
    return torch.autograd.grad(loss, xyz, retain_graph=True, allow_unused=True)[0]


@torch.no_grad()
def validate_index_aligned_reference(observed_xyz: torch.Tensor,
                                     reference_xyz: torch.Tensor,
                                     point_atol: float = 1e-3,
                                     max_mismatched_fraction: float = 2e-2,
                                     max_bbox_relative_difference: float = 1e-3,
                                     max_bbox_relative_mean_difference: float = 1e-5) -> dict:
    """Guard transfer of coarse per-Gaussian statistics to a paired snapshot.

    Historical CUDA runs need not produce bitwise-identical post-coarse XYZ.
    With topology changes disabled, indices remain stable, so sparse numerical
    drift is acceptable. Broad drift or a large displacement is rejected. The
    paired snapshot is then used to keep every fine-stage attribute identical
    to the standard-FPS control except for the selected nodes and K4 graph.
    """
    if observed_xyz.shape != reference_xyz.shape or observed_xyz.ndim != 2 or observed_xyz.shape[1] != 3:
        raise ValueError("Coarse/reference Gaussian topology does not match")
    if not torch.isfinite(observed_xyz).all() or not torch.isfinite(reference_xyz).all():
        raise ValueError("Coarse/reference XYZ must be finite")
    if (point_atol < 0 or not 0 <= max_mismatched_fraction <= 1 or
            max_bbox_relative_difference < 0 or max_bbox_relative_mean_difference < 0):
        raise ValueError("Invalid reference-alignment thresholds")

    difference = (observed_xyz - reference_xyz).abs()
    mismatched = difference.amax(dim=1) > point_atol
    bbox_extent = (reference_xyz.amax(dim=0) - reference_xyz.amin(dim=0)).amax()
    bbox_extent = bbox_extent.clamp_min(torch.finfo(reference_xyz.dtype).eps)
    max_difference = difference.max()
    mismatch_fraction = mismatched.float().mean()
    bbox_relative_difference = max_difference / bbox_extent
    bbox_relative_mean_difference = difference.mean() / bbox_extent
    passed = (mismatch_fraction <= max_mismatched_fraction and
              bbox_relative_difference <= max_bbox_relative_difference and
              bbox_relative_mean_difference <= max_bbox_relative_mean_difference)
    result = {
        "point_atol": float(point_atol),
        "max_mismatched_fraction_allowed": float(max_mismatched_fraction),
        "max_bbox_relative_difference_allowed": float(max_bbox_relative_difference),
        "max_bbox_relative_mean_difference_allowed": float(max_bbox_relative_mean_difference),
        "mismatched_gaussians": int(mismatched.sum().item()),
        "mismatched_fraction": float(mismatch_fraction.item()),
        "max_abs_difference": float(max_difference.item()),
        "mean_abs_difference": float(difference.mean().item()),
        "bbox_extent": float(bbox_extent.item()),
        "max_bbox_relative_difference": float(bbox_relative_difference.item()),
        "bbox_relative_mean_difference": float(bbox_relative_mean_difference.item()),
        "passed": bool(passed),
    }
    if not passed:
        raise ValueError(f"Coarse/reference XYZ drift is too large for index-aligned importance transfer: {result}")
    return result


@torch.no_grad()
def weighted_farthest_point_sampling(xyz, num_samples, importance, alpha=1.0):
    """Exact weighted Euclidean FPS; index 0 first, same as standard FPS.

    Compare d^2 * (1+alpha*w)^2: this has exactly the requested ranking
    of d * (1+alpha*w), avoiding an N-element sqrt at every FPS step.
    """
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not torch.isfinite(xyz).all():
        raise ValueError("xyz must be finite [N,3]")
    if not 0 < num_samples <= len(xyz):
        raise ValueError("num_samples must be in [1,N]")
    if importance.shape != (len(xyz),) or importance.device != xyz.device:
        raise ValueError("importance must be [N] on the same device as xyz")
    if not torch.isfinite(importance).all() or (importance < 0).any() or (importance > 1).any():
        raise ValueError("importance must be finite in [0,1]")
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    selected = torch.empty(num_samples, dtype=torch.long, device=xyz.device)
    min_dist_sq = torch.full((len(xyz),), float("inf"), device=xyz.device)
    multiplier_sq = (1 + alpha * importance).square()
    current = torch.zeros((), dtype=torch.long, device=xyz.device)
    for j in range(num_samples):
        selected[j] = current
        distance = (xyz - xyz[current]).square().sum(-1)
        torch.minimum(min_dist_sq, distance, out=min_dist_sq)
        min_dist_sq[current] = -1  # also excludes duplicate-coordinate nodes
        current = (min_dist_sq * multiplier_sq).argmax()
    return selected


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
