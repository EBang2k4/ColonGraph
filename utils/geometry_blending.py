"""Opt-in, fixed canonical geometry weights. No changes to graph forward/R6."""
from pathlib import Path
import json
import torch
from utils.exact_split_branchpoint import digest


def validate_weights(weights, indices=None, node_count=None):
    if weights.ndim != 2 or weights.shape[1] != 4 or not weights.is_floating_point():
        raise ValueError('Geometry weights must be floating N x 4')
    if weights.requires_grad or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Fixed geometry weights must be finite and nonnegative')
    if not torch.allclose(weights.sum(-1), torch.ones_like(weights[:, 0]), atol=2e-6, rtol=0):
        raise ValueError('Geometry weight rows do not sum to one')
    if indices is not None:
        if indices.shape != weights.shape or indices.dtype != torch.long:
            raise ValueError('Invalid geometry indices shape/dtype')
        if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= node_count):
            raise ValueError('Invalid geometry node index')


@torch.no_grad()
def graph_spacing(node_xyz, edges):
    """Median over unique symmetrized neighbors, including diagnosed zero edges.

    True statistical median (average two middle values for even degree).
    Zero spacing is not clamped: such nodes have zero support and use fallback.
    """
    if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype != torch.long:
        raise ValueError('Invalid node edges')
    m = len(node_xyz)
    if int(edges.min()) < 0 or int(edges.max()) >= m:
        raise ValueError('Invalid edge index')
    if not torch.isfinite(node_xyz).all():
        raise ValueError('Invalid canonical node coordinates')
    pairs = torch.cat((edges, edges.flip(0)), dim=1)
    pairs = torch.unique(pairs.t(), dim=0).t()
    if (pairs[0] == pairs[1]).any():
        raise ValueError('Self edges are not valid spacing samples')
    lengths = (node_xyz[pairs[0]].double() - node_xyz[pairs[1]].double()).norm(dim=-1)
    h = torch.zeros(m, device=node_xyz.device, dtype=torch.float64)
    for j in range(m):  # Only once per node during precompute, never in training.
        row = lengths[pairs[0] == j]
        if row.numel():
            h[j] = torch.quantile(row, .5)
    return h, dict(directed_unique_edges=pairs.shape[1],
        zero_length_directed_edges=int((lengths == 0).sum()),
        zero_spacing_nodes=int((h == 0).sum()),
        isolated_nodes=int((torch.bincount(pairs[0], minlength=m) == 0).sum()),
        duplicate_coordinate_nodes=m-len(torch.unique(node_xyz, dim=0)),
        radius_clamping=False)


@torch.no_grad()
def compute_weights(distances, baseline_weights, mode='idw', indices=None,
                    spacing=None, beta=2.5, epsilon=1e-8):
    """Retain A exactly; precompute B/C with float64 and row-max normalization.

    Even very small positive support is normalized, not mistaken for no support.
    Float32 rounding of the normalized result is the only precision reduction.
    """
    validate_weights(baseline_weights)
    if distances.shape != baseline_weights.shape:
        raise ValueError('Distance/weight shape mismatch')
    if not torch.isfinite(distances).all() or (distances < 0).any():
        raise ValueError('Invalid canonical distances')
    if mode == 'idw':
        return baseline_weights.detach().clone(), dict(mode=mode, zero_support=0,
            numerical_fallback=0, fallback=torch.zeros(len(distances), dtype=torch.bool, device=distances.device))
    d = distances.detach().double()
    if mode == 'idw_squared':
        raw = (d + epsilon).reciprocal().square()
        support = torch.ones_like(d, dtype=torch.bool)
    elif mode == 'wendland_c2':
        if beta <= 0 or spacing is None or indices is None:
            raise ValueError('Wendland needs positive beta, spacing and indices')
        radius = (spacing.detach().double() * beta)[indices]
        if not torch.isfinite(radius).all() or (radius < 0).any():
            raise ValueError('Invalid fixed radius')
        q = torch.where(radius > 0, d / radius, torch.full_like(d, float('inf')))
        support = (radius > 0) & (q < 1)
        safe_q = torch.where(support, q, torch.ones_like(q))
        raw = (1-safe_q).pow(4) * (4*safe_q+1)
    else:
        raise ValueError('Unknown geometry_blend_mode: '+mode)
    maximum = raw.max(-1, keepdim=True).values
    zero_support = ~support.any(-1)
    numerical = (~torch.isfinite(raw).all(-1) | (maximum[:, 0] <= 0)) & ~zero_support
    fallback = zero_support | numerical
    scaled = raw / torch.where(maximum > 0, maximum, torch.ones_like(maximum))
    denom = scaled.sum(-1, keepdim=True)
    weights = (scaled / torch.where(denom > 0, denom, torch.ones_like(denom))).to(baseline_weights)
    weights[fallback] = baseline_weights[fallback]
    validate_weights(weights)
    if mode == 'wendland_c2' and (weights[~fallback][~support[~fallback]] != 0).any():
        raise RuntimeError('Nonzero weight outside support')
    return weights, dict(mode=mode, beta=beta, zero_support=int(zero_support.sum()),
        numerical_fallback=int(numerical.sum()), fallback=fallback,
        active_count=support.sum(-1), raw_denominator=raw.sum(-1))


def validate_cache(model, payload, require_weights=False):
    if payload.get('format_version') != 1:
        raise ValueError('Unsupported geometry blending cache')
    graph = model.deformation_graph
    app = model.appearance_graph
    if app is None or app.gaussian_node_weights.data_ptr() == graph.gaussian_node_weights.data_ptr():
        raise ValueError('Geometry blending requires independently cached appearance K4')
    invariants = dict(nodes=digest(graph.node_xyz), indices=digest(graph.gaussian_node_indices),
        appearance=digest(app.state_dict()), g3_graph=digest({k:v for k,v in model.graph_motion_residual.state_dict().items()
            if not k.startswith('decoder.')}))
    for key, value in invariants.items():
        if payload['invariants'][key] != value:
            raise ValueError('Geometry cache invariant changed: '+key)
    validate_weights(payload['weights'], graph.gaussian_node_indices.cpu(), len(graph.node_xyz))
    if digest(payload['weights']) != payload['weights_sha256']:
        raise ValueError('Geometry weight cache digest mismatch')
    if require_weights and digest(graph.gaussian_node_weights) != payload['weights_sha256']:
        raise ValueError('Reloaded geometry weights differ from saved cache')


@torch.no_grad()
def install_cache(model, path, output=None):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    validate_cache(model, payload)
    if getattr(model.args, 'geometry_blend_mode', 'idw') != payload['mode']:
        raise ValueError('Requested mode differs from cache')
    if payload['mode'] == 'wendland_c2' and model.args.geometry_blend_beta != payload['beta']:
        raise ValueError('Requested beta differs from cache')
    if digest(model.deformation_graph.gaussian_node_weights) != payload['baseline_weights_sha256']:
        raise ValueError('Initial geometry cache is not the source baseline')
    before = dict(appearance=digest(model.appearance_graph.state_dict()),
        g3=digest(model.graph_motion_residual.state_dict()), node_mlp=digest(model.node_mlp.state_dict()))
    model.deformation_graph.gaussian_node_weights.copy_(payload['weights'].to(model._xyz))
    # Serialized metadata is not a trainable buffer and does not alter model_digest.
    model.geometry_blending_cache = payload
    after = dict(appearance=digest(model.appearance_graph.state_dict()),
        g3=digest(model.graph_motion_residual.state_dict()), node_mlp=digest(model.node_mlp.state_dict()))
    if before != after:
        raise RuntimeError('Geometry weight installation changed other modules')
    validate_cache(model, payload, require_weights=True)
    if output is not None:
        row = dict(mode=payload['mode'], beta=payload['beta'], cache=str(path),
            weights_sha256=payload['weights_sha256'], baseline_weights_sha256=payload['baseline_weights_sha256'],
            invariants=payload['invariants'], other_modules_unchanged=True,
            trainable_geometry=all(getattr(model,k).requires_grad for k in ('_xyz','_rotation','_scaling','_opacity')))
        (Path(output)/'geometry_blending_initial.json').write_text(json.dumps(row, indent=2)+'\n')

