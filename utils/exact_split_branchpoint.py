"""Exact PRE-split model/Adam/RNG/manager/camera state, feature flagged only."""
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch

RAW_NAMES=('_xyz','_scaling','_rotation','_opacity','features','_deformation_table')


def digest(value):
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            h.update(str((x.dtype,tuple(x.shape))).encode());h.update(x.detach().contiguous().cpu().numpy().tobytes())
        elif isinstance(x,np.ndarray):h.update(x.tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(tuple,list)):
            for v in x:visit(v)
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()


def model_digest(model):
    values={k:getattr(model,k) for k in RAW_NAMES if hasattr(model,k)}
    for name in ('node_mlp','appearance_model','deformation_graph','appearance_graph','graph_motion_residual','_deformation','surface_children','skinning','hf_appearance'):
        module=getattr(model,name,None)
        if module is not None:values[name]=module.state_dict()
    for name in ('closest_point_indices','active_sh_degree','spatial_lr_scale'):
        if hasattr(model,name):values[name]=getattr(model,name)
    return digest(values)


def save_branchpoint(path,scene,model,manager,cameras,event_camera,iteration,source_render=None):
    if iteration!=5000 or manager.events or len(manager.graph.node_xyz)!=2000:
        raise ValueError('Only the exact first pre-split 5k point may branch')
    scene.save(iteration,'fine')
    state=dict(format_version=2,phase='pre_first_split_after_optimizer',iteration=iteration,
        pending_split=True,pending_camera_name=event_camera.image_name,
        optimizer=model.optimizer.state_dict(),manager_state=manager.state_dict(iteration),
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
        python_rng=random.getstate(),numpy_rng=np.random.get_state(),
        train_camera_order=[c.image_name for c in cameras],
        exact_raw={k:getattr(model,k).detach().cpu().clone() for k in RAW_NAMES},
        active_sh_degree=model.active_sh_degree,spatial_lr_scale=model.spatial_lr_scale,
        model_sha256=model_digest(model),optimizer_sha256=digest(model.optimizer.state_dict()))
    state['rng_sha256']=digest({k:state[k] for k in ('torch_rng','cuda_rng','python_rng','numpy_rng')})
    if source_render is not None:
        state['source_render']={k:source_render[k].detach().cpu() for k in ('render','depth','alpha')}
    clone=random.Random();clone.setstate(state['python_rng'])
    state['next_camera_names']=[cameras[clone.randint(0,len(cameras)-1)].image_name for _ in range(5)]
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp');torch.save(state,temporary);temporary.replace(path)
    (path.parent/'branchpoint_manifest.json').write_text(json.dumps({k:state[k] for k in
        ('phase','iteration','pending_camera_name','model_sha256','optimizer_sha256','rng_sha256','next_camera_names')},indent=2)+'\n')


def restore_raw(model,state):
    if not state.get('pending_split'):return
    if state.get('phase')!='pre_first_split_after_optimizer' or state['iteration']!=5000:
        raise ValueError('Not an exact first pre-split branchpoint')
    with torch.no_grad():
        for k,value in state['exact_raw'].items():getattr(model,k).copy_(value.to(getattr(model,k)))
    model.active_sh_degree=state['active_sh_degree'];model.spatial_lr_scale=state['spatial_lr_scale']


def restore_knn_reference(model,path):
    """Reload the immutable external KNN-99, not a recomputed spatial graph."""
    reference=torch.load(path,map_location=model._xyz.device,weights_only=True)
    if (not torch.is_tensor(reference) or reference.dtype!=torch.long or
            reference.shape!=(len(model._xyz),99) or
            (reference.numel() and (int(reference.min())<0 or int(reference.max())>=len(model._xyz)))):
        raise ValueError('Invalid saved Gaussian KNN-99 reference')
    existing=getattr(model,'closest_point_indices',None)
    if existing is not None and not torch.equal(existing,reference):
        raise ValueError('Loaded Gaussian KNN-99 differs from matched reference')
    model.closest_point_indices=reference


def restore_manager(manager,state):
    from utils.candidate_expansion import CandidateExpansionManager
    core=state['manager_state']
    if core['events'] or len(core['node_xyz'])!=2000 or core['iteration']!=5000:
        raise ValueError('Cannot change strategy after a split has occurred')
    # Intentionally bypass ONLY the strategy/quota validators at the authorized
    # common pre-event point. All topology/config/Adam/G3 validators still run.
    CandidateExpansionManager.load_state_dict(manager,core)
    if model_digest(manager.model)!=state['model_sha256']:
        raise RuntimeError('Pre-split model reload is not tensor-exact')
    if digest(manager.model.optimizer.state_dict())!=state['optimizer_sha256']:
        raise RuntimeError('Pre-split Adam reload is not exact')
