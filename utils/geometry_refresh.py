"""Opt-in fixed-node cache refresh and 20k freeze for the replicated full study."""
import copy
import json
import random
from pathlib import Path
import numpy as np
import torch
from utils.node_sampling import build_gaussian_node_knn
from utils.graph_forward import repair_gaussian_k4_minimal, k4_topology_tensors
from utils.geometry_blending import validate_cache, validate_weights
from utils.exact_split_branchpoint import digest, model_digest
from utils.texture_training import freeze_geometry, audit_frozen_geometry
from utils.adaptive_deformation_graph import AdaptiveNodeManager


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def stats(value):
    x=value.detach().float().flatten();finite=x[torch.isfinite(x)]
    if not finite.numel():return dict(count=len(x),finite_fraction=0.)
    q=torch.quantile(finite,finite.new_tensor([0.,.5,.9,.95,.99,1.]))
    return dict(count=len(x),finite_fraction=float(torch.isfinite(x).float().mean()),mean=float(finite.mean()),
        **dict(zip(('min','median','p90','p95','p99','max'),q.tolist())))


@torch.no_grad()
def refreshed_k4(xyz,nodes,old_indices,old_weights,edges,policy):
    """No new assignment heuristic: exact canonical Euclidean K4 + existing repair."""
    if policy not in ('weights_only','neighbors_and_weights'):raise ValueError('Invalid refresh policy')
    if policy=='weights_only':
        indices=old_indices.detach().clone()
        distances=(xyz.detach()[:,None]-nodes[indices]).norm(dim=-1)
        inverse=(distances+1e-8).reciprocal();weights=inverse/inverse.sum(-1,keepdim=True)
        repaired=torch.zeros(len(xyz),device=xyz.device,dtype=torch.bool)
        candidate_indices=indices
        fallback=torch.zeros_like(repaired)
    else:
        candidate_indices,candidate_weights=build_gaussian_node_knn(xyz.detach(),nodes.detach(),4)
        indices,weights,repaired,fallback,_=repair_gaussian_k4_minimal(
            xyz.detach(),nodes.detach(),candidate_indices,candidate_weights,edges)
        # Repair recomputes only changed rows; now all selected rows use current IDW.
        distances=(xyz.detach()[:,None]-nodes[indices]).norm(dim=-1)
        inverse=(distances+1e-8).reciprocal();weights=inverse/inverse.sum(-1,keepdim=True)
    valid=k4_topology_tensors(indices,edges,len(nodes))['connected']
    if fallback.any() or not valid.all():
        raise RuntimeError('Existing surface repair cannot satisfy refreshed K4: fallback={}, disconnected={}'.format(
            int(fallback.sum()),int((~valid).sum())))
    validate_weights(weights,indices,len(nodes))
    return indices,weights,dict(candidate_indices=candidate_indices,repair_mask=repaired,fallback=fallback)


@torch.no_grad()
def refresh(model,iteration,cameras,render,pipe,bg,output):
    import time
    policy=model.args.geometry_study_refresh
    if model.args.geometry_study_frozen or iteration>=model.args.geometry_study_freeze_iteration:
        raise RuntimeError('Cannot refresh frozen geometry')
    path=Path(output)/'geometry_refresh';path.mkdir(exist_ok=True)
    graph=model.deformation_graph;xyz=model._xyz.detach();nodes=graph.node_xyz
    ids=graph.gaussian_node_indices.clone();weights=graph.gaussian_node_weights.clone()
    original=model._query_node_motion
    diagnostic=[next(c for c in cameras if str(c.image_name)==str(f)) for f in (0,144,264)]
    before=[];motions=[]
    for c in diagnostic:
        shared=original(float(c.time));motions.append(shared);model._query_node_motion=lambda *a,**k:shared
        p=render(c,model,pipe,bg,stage='fine')
        before.append({key:p[key].detach().clone() for key in ('render','depth','alpha')})
    model._query_node_motion=original
    app_before=digest(model.appearance_graph.state_dict());g3_before=digest(model.graph_motion_residual.state_dict())
    torch.cuda.synchronize();started=time.monotonic()
    ni,nw,repair=refreshed_k4(xyz,nodes,ids,weights,model.graph_motion_residual.edge_indices,policy)
    torch.cuda.synchronize();seconds=time.monotonic()-started
    old_d=(xyz[:,None]-nodes[ids]).norm(dim=-1);new_d=(xyz[:,None]-nodes[ni]).norm(dim=-1)
    old_dom=ids.gather(1,weights.argmax(-1)[:,None])[:,0];new_dom=ni.gather(1,nw.argmax(-1)[:,None])[:,0]
    changed=(ids!=ni).any(-1);set_changed=~(ids[:,:,None]==ni[:,None,:]).any(-1).all(-1)
    slot_changes=(ids!=ni).sum(-1)
    # Align weights by node identity so reordering is not mistaken for weight change.
    aligned=(nw[:,None,:]*(ids[:,:,None]==ni[:,None,:])).sum(-1)
    removed_mass=(nw*~(ni[:,:,None]==ids[:,None,:]).any(-1)).sum(-1)
    graph.gaussian_node_indices.copy_(ni);graph.gaussian_node_weights.copy_(nw)
    cache=copy.deepcopy(model.geometry_blending_cache)
    cache.update(mode='idw',weights=nw.cpu(),weights_sha256=digest(nw),refresh_policy=policy,refresh_iteration=iteration)
    cache['invariants']['indices']=digest(ni)
    model.geometry_blending_cache=cache
    validate_cache(model,cache,require_weights=True)
    if app_before!=digest(model.appearance_graph.state_dict()) or g3_before!=digest(model.graph_motion_residual.state_dict()):
        raise RuntimeError('Refresh changed appearance/G3')
    views=[]
    try:
        for c,pre,shared in zip(diagnostic,before,motions):
            model._query_node_motion=lambda *a,**k:shared
            post=render(c,model,pipe,bg,stage='fine')
            pd=pre['depth']/pre['depth'].max().clamp_min(1e-6);qd=post['depth']/post['depth'].max().clamp_min(1e-6)
            gt=c.original_image[:3].cuda().float()
            pre_error=(pre['render'].clamp(0,1)-gt).abs().mean()
            post_error=(post['render'].clamp(0,1)-gt).abs().mean()
            pre_depth=AdaptiveNodeManager._depth_metrics(pre['depth'],c.original_depth.cuda().float(),getattr(c,'mask',None))
            post_depth=AdaptiveNodeManager._depth_metrics(post['depth'],c.original_depth.cuda().float(),getattr(c,'mask',None))
            views.append(dict(frame=c.image_name,RGB_L1_before=float(pre_error),RGB_L1_after=float(post_error),
                RGB_L1_jump=float(post_error-pre_error),normalized_depth_RMS_change=float((pd-qd).square().mean().sqrt()),
                RGB_RMS_change=float((pre['render']-post['render']).square().mean().sqrt()),
                alpha_RMS_change=float((pre['alpha']-post['alpha']).square().mean().sqrt()),
                depth_float_valid_before=pre_depth['normalized_mse'],depth_float_valid_after=post_depth['normalized_mse']))
    finally:model._query_node_motion=original
    old_top=k4_topology_tensors(ids,model.graph_motion_residual.edge_indices,len(nodes))
    candidate_top=k4_topology_tensors(repair['candidate_indices'],model.graph_motion_residual.edge_indices,len(nodes))
    new_top=k4_topology_tensors(ni,model.graph_motion_residual.edge_indices,len(nodes))
    row=dict(iteration=iteration,policy=policy,phase='after_optimizer_before_next_step' if iteration>5000 else 'source_before_first_step',
        changed_rows_count=int(changed.sum()),changed_rows_fraction=float(changed.float().mean()),
        changed_sets_fraction=float(set_changed.float().mean()),changed_neighbor_slots=int(slot_changes.sum()),
        dominant_reassignment_fraction=float((old_dom!=new_dom).float().mean()),
        identity_aligned_weight_L1=stats((weights-aligned).abs().sum(-1)+removed_mass),
        fraction_requiring_repair=float(repair['repair_mask'].float().mean()),repair_fallback=int(repair['fallback'].sum()),
        candidate_connected_fraction=float(candidate_top['connected'].float().mean()),old_induced_hops=stats(old_top['diameter']),
        new_induced_hops=stats(new_top['diameter']),distances_before=stats(old_d),distances_after=stats(new_d),
        compute_seconds=seconds,indices_sha256=digest(ni),weights_sha256=digest(nw),appearance_G3_exact=True,diagnostic_views=views)
    np.savez_compressed(path/f'iteration_{iteration}.npz',changed_gaussian_ids=changed.nonzero().flatten().cpu().numpy(),
        changed_xyz=xyz[changed].cpu().numpy(),old_indices=ids[changed].cpu().numpy(),new_indices=ni[changed].cpu().numpy(),
        distances_before=old_d.cpu().numpy(),distances_after=new_d.cpu().numpy(),
        weight_L1=((weights-aligned).abs().sum(-1)+removed_mass).cpu().numpy())
    write(path/f'iteration_{iteration}.json',row)
    with (Path(output)/'geometry_refresh.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
    model.geometry_study_meta['last_refresh']=iteration
    print('[GEOMETRY_REFRESH]',iteration,'policy',policy,'changed_sets',row['changed_sets_fraction'],
        'repair',row['fraction_requiring_repair'],'seconds',seconds,flush=True)


def initialize_study(model,state,cameras,render,pipe,bg,output,scene=None):
    if state is None or not getattr(model.args,'matched_texture_audit',False):
        raise ValueError('Full geometry study requires exact full-state source audit')
    if model.args.geometry_study_refresh not in ('none','weights_only','neighbors_and_weights'):
        raise ValueError('Invalid study refresh policy')
    if model.appearance_graph is None or len(model._xyz)!=364500 or len(model.deformation_graph.node_xyz)!=2000:
        raise ValueError('Full study requires fixed Gaussian/node counts and independent appearance graph')
    resumed=state.get('geometry_study_meta')
    if resumed:
        model.geometry_study_meta=copy.deepcopy(resumed)
        if (resumed['camera_seed']!=model.args.geometry_study_camera_seed or resumed['refresh_policy']!=model.args.geometry_study_refresh):
            raise ValueError('Resume changed study policy/seed')
    else:
        if state['iteration']!=5000:raise ValueError('New study must start at common fine5k')
        model.geometry_study_meta=dict(format_version=1,camera_seed=model.args.geometry_study_camera_seed,
            refresh_policy=model.args.geometry_study_refresh,last_refresh=None,frozen=False,freeze_iteration=model.args.geometry_study_freeze_iteration)
        random.seed(model.args.geometry_study_camera_seed)  # Only camera-order RNG changes across blocks.
    model.args.geometry_study_frozen=False
    row=dict(source_iteration=state['iteration'],source_model_sha256=state['model_sha256'],
        model_post_kernel_sha256=model_digest(model),optimizer_sha256=digest(model.optimizer.state_dict()),
        post_camera_seed_python_rng_sha256=digest(random.getstate()),camera_seed=model.args.geometry_study_camera_seed,
        refresh_policy=model.args.geometry_study_refresh,resume=bool(resumed))
    write(Path(output)/'geometry_study_initial.json',row)
    if state['iteration']>=model.args.geometry_study_freeze_iteration:
        freeze_geometry(model,output);model.args.geometry_study_frozen=True
        if resumed and model.frozen_geometry_sha256!=resumed['frozen_geometry_sha256']:
            raise ValueError('Resumed frozen geometry differs from 20k snapshot')
    elif not resumed and model.args.geometry_study_refresh!='none':
        refresh(model,5000,cameras,render,pipe,bg,output)
    if not resumed and scene is not None:
        from utils.texture_training import save_full_state
        save_full_state(scene,model,cameras,5000)


def after_optimizer(model,iteration,cameras,render,pipe,bg,output):
    args=model.args
    if (args.geometry_study_refresh!='none' and iteration<args.geometry_study_freeze_iteration and
            iteration%args.geometry_study_refresh_interval==0):
        refresh(model,iteration,cameras,render,pipe,bg,output)
    if iteration==args.geometry_study_freeze_iteration:
        freeze_geometry(model,output);args.geometry_study_frozen=True
        model.geometry_study_meta.update(frozen=True,frozen_geometry_sha256=model.frozen_geometry_sha256,
            frozen_adam_sha256=model.frozen_adam_sha256)
        print('[GEOMETRY_FROZEN_AFTER_OPTIMIZER]',iteration,flush=True)
    if args.geometry_study_frozen and (iteration%1000==0 or iteration==args.fine_stop_iteration):
        audit_frozen_geometry(model,iteration,output)
