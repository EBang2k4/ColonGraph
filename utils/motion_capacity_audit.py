"""Read-only geometry-difficulty statistics and a free-node K4 correction fit."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
import torch


def fit_correction(indices, weights, gradient, confidence, node_count,
                   ridge_relative=1e-6):
    """Weighted ridge fit W Q ~= -gradient; no model parameters are changed.

    Solve in float64, scaling the target by its weighted RMS for numerical
    accuracy. Ridge is relative to the mean observed normal-matrix diagonal.
    A smaller-ridge solve reports whether ridge, rather than K4, drives residual.
    """
    idx = np.asarray(indices, dtype=np.int64)
    w = np.asarray(weights, dtype=np.float64)
    g = np.asarray(gradient, dtype=np.float64)
    omega = np.asarray(confidence, dtype=np.float64)
    valid = np.isfinite(g).all(1) & np.isfinite(omega) & (omega > 0)
    selected = np.flatnonzero(valid)
    if not len(selected):
        raise ValueError("No observed gradients for correction fit")
    W = sparse.csr_matrix((w[selected].ravel(), idx[selected].ravel(),
                           np.arange(0, 4*len(selected)+1, 4)),
                          shape=(len(selected), node_count))
    o = omega[selected]
    target = -g[selected]
    scale = np.sqrt(np.sum(o[:, None]*target**2)/max(o.sum(), 1e-30))
    if scale <= 1e-30:
        raise ValueError("Zero geometry gradient")
    target /= scale
    normal = (W.T @ W.multiply(o[:, None])).tocsc()
    rhs = W.T @ (o[:, None]*target)
    diagonal = normal.diagonal()
    reference = diagonal[diagonal > 0].mean()
    identity = sparse.eye(node_count, format="csc")
    out = []
    for factor in (ridge_relative, ridge_relative*.01):
        regularized = normal + factor*reference*identity
        coefficients = splu(regularized).solve(rhs)
        residual = target-W @ coefficients
        normal_error = np.linalg.norm(regularized @ coefficients-rhs)/max(np.linalg.norm(rhs), 1e-30)
        fraction = np.sum(o[:, None]*residual**2)/np.sum(o[:, None]*target**2)
        if not np.isfinite(coefficients).all() or normal_error > 1e-7:
            raise RuntimeError("Correction solver failed normal-equation residual audit")
        out.append((coefficients*scale, residual*scale, float(fraction), float(normal_error)))
    all_residual = np.zeros_like(g)
    all_residual[selected] = out[0][1]
    report = dict(weighted_unexplained_energy=out[0][2],
                  lower_ridge_unexplained_energy=out[1][2],
                  ridge_sensitivity=abs(out[0][2]-out[1][2]),
                  relative_normal_residual=out[0][3], ridge_relative=ridge_relative,
                  target_rms=float(scale), observed_gaussians=len(selected),
                  observed_nodes=int((diagonal > 0).sum()))
    return all_residual, report


def node_reduce(value, indices, weights, node_count):
    out = value.new_zeros((node_count,)+value.shape[1:])
    for slot in range(4):
        factor = weights[:, slot].reshape((-1,)+(1,)*(value.ndim-1))
        out.index_add_(0, indices[:, slot], value*factor)
    return out


def frame_geometry_stats(gradient, confidence, residual, indices, weights, node_count):
    norm = gradient.norm(dim=1)
    positive = norm[(confidence > 0) & (norm > 0) & torch.isfinite(norm)]
    floor = torch.quantile(positive, .10) if len(positive) else norm.new_tensor(float("inf"))
    direction_confidence = confidence * ((norm >= floor) & (norm > 0))
    unit = gradient/norm[:, None].clamp_min(1e-30)
    mass = node_reduce(confidence, indices, weights, node_count)
    direction_mass = node_reduce(direction_confidence, indices, weights, node_count)
    direction_sum = node_reduce(unit*direction_confidence[:, None], indices, weights, node_count)
    disagreement = 1-(direction_sum/direction_mass[:, None].clamp_min(1e-30)).norm(dim=1)
    disagreement = torch.where(direction_mass > 0, disagreement.clamp(0, 1), 0.)
    energy = node_reduce(norm.square()*confidence, indices, weights, node_count)
    residual_energy = node_reduce(residual.square().sum(1)*confidence, indices, weights, node_count)
    difficulty = node_reduce(norm*confidence, indices, weights, node_count)/mass.clamp_min(1e-30)
    return dict(E=difficulty, H=disagreement, confidence=mass,
                U=(residual_energy/energy.clamp_min(1e-30)).sqrt(),
                residual_rms=(residual_energy/mass.clamp_min(1e-30)).sqrt(),
                direction_sum=direction_sum, direction_mass=direction_mass,
                gradient_floor=float(floor), gradient_energy=energy)
