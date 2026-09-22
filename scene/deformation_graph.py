import torch
from torch import nn
from utils.node_sampling import farthest_point_sampling, build_gaussian_node_knn


def node_rotation_matrices(rotation_residual):
    """Scalar-first quaternion R(normalize([1,0,0,0] + residual))."""
    identity = rotation_residual.new_tensor([1., 0., 0., 0.])
    q = torch.nn.functional.normalize(identity + rotation_residual, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack((
        1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y),
        2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x),
        2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y),
    ), dim=-1).reshape(-1, 3, 3)


class SparseDeformationGraph(nn.Module):
    def __init__(self, node_xyz, indices, weights):
        super().__init__()
        self.register_buffer("node_xyz", node_xyz.detach().clone())
        self.register_buffer("gaussian_node_indices", indices.detach().clone())
        self.register_buffer("gaussian_node_weights", weights.detach().clone())

    @classmethod
    def from_gaussians(cls, xyz, num_nodes=2000, k=4):
        nodes = xyz.detach()[farthest_point_sampling(xyz.detach(), num_nodes)]
        indices, weights = build_gaussian_node_knn(xyz.detach(), nodes, k)
        return cls(nodes, indices, weights)

    def blend(self, node_d_xyz, node_d_rotation, node_d_scale):
        indices = self.gaussian_node_indices
        weights = self.gaussian_node_weights.unsqueeze(-1)
        return tuple((value[indices] * weights).sum(1)
                     for value in (node_d_xyz, node_d_rotation, node_d_scale))

    def deform_positions_local(self, gaussian_xyz, node_d_xyz, node_d_rotation):
        """Blend R_j (x_i - c_j) + c_j + dt_j with fixed canonical weights.

        Node rotations are used before the separate Gaussian rotation-residual
        clamp. Canonical xyz remains differentiable; node centers/weights do not.
        """
        indices = self.gaussian_node_indices
        centers = self.node_xyz[indices]
        rotations = node_rotation_matrices(node_d_rotation)[indices]
        local = gaussian_xyz[:, None, :] - centers
        rotated = torch.matmul(rotations, local.unsqueeze(-1)).squeeze(-1)
        per_node = rotated + centers + node_d_xyz[indices]
        return (self.gaussian_node_weights.unsqueeze(-1) * per_node).sum(dim=1)
