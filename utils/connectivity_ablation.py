"""Connectivity-only fixed-G3 references and opt-in matched initialization audit."""
import copy
import random
from pathlib import Path
import numpy as np
import torch
from utils.exact_split_branchpoint import digest, RAW_NAMES


def restore_optimizer_independent(optimizer, source_state):
    """Adam may alias source CPU step counters; isolate every branch restore."""
    optimizer.load_state_dict(copy.deepcopy(source_state))


@torch.no_grad()
def euclidean_reference(surface, displacement, spatial_distances, spatial_sigma, k=6):
    nodes=surface['canonical_node_xyz']
    if displacement.ndim!=3 or displacement.shape[1:]!=nodes.shape:
        raise ValueError('Need common source node displacements at fixed trajectory timestamps')
    distances=torch.cdist(nodes,nodes)
    distances.fill_diagonal_(float('inf'))
    neighbors=torch.argsort(distances,dim=1,stable=True)[:,:k]
    rows=torch.arange(len(nodes))[:,None].expand_as(neighbors)
    keys=torch.unique((torch.minimum(rows,neighbors)*len(nodes)+torch.maximum(rows,neighbors)).flatten(),sorted=True)
    edges=torch.stack((keys//len(nodes),keys%len(nodes)))
    if spatial_distances.shape!=distances.shape or spatial_sigma<=0:
        raise ValueError('Need the same frozen Surface per-pair spatial weighting kernel')
    lengths=spatial_distances[edges[0],edges[1]]
    spatial=torch.exp(-lengths.square()/(2*spatial_sigma**2)).clamp_min(1e-12)
    motion=(displacement[:,edges[0]]-displacement[:,edges[1]]).square().sum(-1).mean(0).sqrt()
    trajectory_sigma=float(surface['audit']['trajectory_sigma'])
    if trajectory_sigma<=0:raise ValueError('Original trajectory-gate sigma must be positive')
    similarity=torch.exp(-motion.square()/(2*trajectory_sigma**2))
    result=copy.deepcopy(surface)
    result.update(surface_edge_indices=edges,surface_edge_weights=spatial,
                  trajectory_gated_edge_weights=(spatial*similarity).clamp_min(1e-12),
                  trajectory_distance=motion,trajectory_similarity=similarity)
    result['audit']['connectivity_ablation']=dict(variant='G3-Euc',directed_k=k,
        symmetrization='union, same as surface graph; degree may exceed 6',
        distance='canonical Euclidean for neighbor selection ONLY',spatial_kernel='frozen Surface geodesic RBF, same per-pair function in both branches',
        spatial_sigma=float(spatial_sigma),trajectory_sigma=trajectory_sigma,trajectory_sigma_recalibrated=False,
        geometry_k4_recomputed=False,edge_count=len(keys))
    return result


@torch.no_grad()
def record_initial(model,state,cameras,render_fn,pipe,bg,output):
    if (state is None or state['iteration']!=2000 or state.get('pending_split') or
            len(model.deformation_graph.node_xyz)!=2000 or model.graph_motion_residual.mode!='g3'):
        raise ValueError('Connectivity ablation requires the fixed-node pre-G3 2k source')
    if [c.image_name for c in cameras]!=state['train_camera_order']:
        raise ValueError('Connectivity camera stack mismatch')
    values={k:getattr(model,k) for k in RAW_NAMES}
    values.update({k:getattr(model,k).state_dict() for k in
                   ('node_mlp','appearance_model','deformation_graph','appearance_graph','_deformation')})
    values['closest_point_indices']=model.closest_point_indices
    rng=dict(torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
             python_rng=random.getstate(),numpy_rng=np.random.get_state())
    if digest(rng)!=digest({k:state[k] for k in rng}):raise ValueError('Connectivity source RNG mismatch')
    clone=random.Random();clone.setstate(state['python_rng'])
    out=render_fn(cameras[0],model,pipe,bg,stage='fine')
    row=dict(iteration=2000,non_connectivity_sha256=digest(values),
        decoder_sha256=digest(model.graph_motion_residual.decoder.state_dict()),
        optimizer_sha256=digest(model.optimizer.state_dict()),rng_sha256=digest(rng),
        camera_names=[c.image_name for c in cameras],
        next_camera_names=[cameras[clone.randint(0,len(cameras)-1)].image_name for _ in range(5)],
        geometry_k4_sha256=digest(model.deformation_graph.state_dict()),
        appearance_k4_sha256=digest(model.appearance_graph.state_dict()),
        **{k:out[k].detach().cpu() for k in ('render','depth','alpha')})
    if digest(rng)!=digest(dict(torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                                python_rng=random.getstate(),numpy_rng=np.random.get_state())):
        raise RuntimeError('Initial connectivity audit consumed RNG')
    torch.save(row,Path(output)/'connectivity_initial_audit.pth')
