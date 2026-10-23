"""Static, Gaussian-weighted zero-mean RGB textures and 3D ray coordinates.

The measure is independent N(0,1) coordinates truncated to [-3,3]^2.
Grid knots are -3,0,3, with bilinear interpolation and clamped borders.
The two texture axes are fixed canonical axis identities. The ray is mapped
to its closest point in the deformed Gaussian's Mahalanobis metric, rather
than replacing the 3D opacity footprint by a 2DGS plane intersection.
"""
import math
import torch
from utils.general_utils import build_rotation


def integration_coefficients(device=None, dtype=torch.float32):
    z = math.erf(3 / math.sqrt(2))
    side = (1 - math.exp(-4.5)) / (3 * math.sqrt(2 * math.pi) * z)
    a = torch.tensor([side, 1-2*side, side], device=device, dtype=dtype)
    return torch.outer(a,a).reshape(9)


def centered_texture(texels):
    a = integration_coefficients(texels.device,texels.dtype)
    return texels - (texels*a[None,:,None]).sum(1,keepdim=True)


def integration_gram(device=None,dtype=torch.float32):
    """Exact continuous second moments E[b_j b_k] under the same measure."""
    z=math.erf(3/math.sqrt(2))
    side=(1-math.exp(-4.5))/(3*math.sqrt(2*math.pi)*z)
    square=1/18-math.exp(-4.5)/(3*math.sqrt(2*math.pi)*z)
    cross=side-square
    g=torch.tensor([[square,cross,0.],[cross,1-4*side+2*square,cross],
                    [0.,cross,square]],device=device,dtype=dtype)
    return torch.kron(g,g)


def canonical_axis_order(scales):
    normal = scales.detach().argmin(1)
    orders = torch.tensor([[1,2,0],[0,2,1],[0,1,2]],device=scales.device)
    return orders[normal]


def ray_parameters(means, scales, rotations, axis_order, settings):
    # R maps local Gaussian axes to world axes, matching the baseline covariance.
    inverse_frame = build_rotation(rotations).transpose(1,2)/scales[:,:,None]
    inverse_frame = inverse_frame.gather(1,axis_order[:,:,None].expand(-1,-1,3))
    center = settings.campos.to(means)
    inverse_projection = torch.linalg.inv(settings.projmatrix.to(means).T)
    ray = inverse_projection[:3] - center[:,None]*inverse_projection[3:4]
    width,height = settings.image_width,settings.image_height
    linear = torch.stack((ray[:,0]*(2/width),ray[:,1]*(2/height),
        ray[:,0]*(1/width-1)+ray[:,1]*(1/height-1)+ray[:,2]+ray[:,3]),1)
    origin = torch.einsum('nij,nj->ni',inverse_frame,center-means)
    direction = inverse_frame @ linear
    return torch.cat((origin,direction.reshape(-1,9)),1).contiguous()


def sample_reference(textures, packed, pixels):
    """Small differentiable CPU/GPU reference for gradient and orientation tests."""
    ray = torch.einsum('nij,pj->npi',packed[:,3:].reshape(-1,3,3),
                       torch.cat((pixels,torch.ones_like(pixels[:,:1])),1))
    origin = packed[:,:3,None].transpose(1,2)
    denominator=ray.square().sum(-1,keepdim=True)
    dot=(ray*origin).sum(-1,keepdim=True)
    valid=(denominator>1e-12)&torch.isfinite(denominator)&torch.isfinite(dot)
    t = -dot/denominator.clamp_min(1e-12)
    local = origin+t*ray
    grid = torch.where(valid,local[...,:2]/3,torch.zeros_like(local[...,:2])).reshape(len(packed),-1,1,2)
    image = textures.reshape(-1,3,3,3).permute(0,3,1,2)
    rgb = torch.nn.functional.grid_sample(image,grid,mode='bilinear',
             padding_mode='border',align_corners=True).squeeze(-1).transpose(1,2)
    return torch.where(valid,rgb,torch.zeros_like(rgb)),local


@torch.no_grad()
def texture_statistics(model):
    if model.texture_texels is None:
        return {'enabled':False}
    q=model.texture_texels
    t=centered_texture(q)
    mean=(t*integration_coefficients(t.device)[None,:,None]).sum(1)
    magnitude=t.square().mean((1,2)).sqrt()
    continuous_variance=(t*torch.einsum('jk,nkc->njc',integration_gram(t.device,t.dtype),t)).sum(1)
    return dict(enabled=True,active=model.texture_active,gaussians=len(q),
        parameters=q.numel(),raw_min=float(q.min()),raw_max=float(q.max()),
        centered_min=float(t.min()),centered_max=float(t.max()),
        centered_rms=float(t.square().mean().sqrt()),mean_abs=float(t.abs().mean()),
        weighted_mean_max_error=float(mean.abs().max()),
        weighted_mean_rms_error=float(mean.square().mean().sqrt()),
        spatial_variance=float(t.var(1,unbiased=False).mean()),
        continuous_weighted_spatial_variance=float(continuous_variance.mean()),
        continuous_weighted_rms=float(continuous_variance.mean().clamp_min(0).sqrt()),
        nontrivial_fraction=float((magnitude>1e-4).float().mean()),
        magnitude_quantiles=torch.quantile(magnitude,torch.tensor([.5,.9,.99],device=q.device)).tolist())
