"""Full joint-training state and matched A/B audits; opt-in experiment helpers."""
import copy
import json
import random
from pathlib import Path
import numpy as np
import torch
from utils.exact_split_branchpoint import RAW_NAMES, digest, model_digest


def save_full_state(scene, model, cameras, iteration):
    scene.save(iteration,'fine')
    state=dict(format_version=3,phase='after_optimizer',iteration=iteration,
        exact_raw={k:getattr(model,k).detach().cpu().clone() for k in RAW_NAMES},
        active_sh_degree=model.active_sh_degree,spatial_lr_scale=model.spatial_lr_scale,
        optimizer=copy.deepcopy(model.optimizer.state_dict()),
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
        python_rng=random.getstate(),numpy_rng=np.random.get_state(),
        train_camera_order=[c.image_name for c in cameras],model_sha256=model_digest(model))
    if model.texture_texels is not None:
        state['exact_texture']=model.texture_texels.detach().cpu().clone()
    if getattr(model,'geometry_study_meta',None) is not None:
        state['geometry_study_meta']=copy.deepcopy(model.geometry_study_meta)
    path=Path(scene.model_path)/f'fine_state_{iteration}.pth'
    torch.save(state,path)
    return path


@torch.no_grad()
def matched_initial(model,state,cameras,render_fn,pipe,bg,output):
    expected=getattr(model.args,'matched_texture_source_iteration',5000)
    if state is None or state['iteration'] != expected or state.get('pending_split'):
        raise ValueError(f'Matched texture experiment must start from complete fine {expected}')
    rng=dict(torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
             python_rng=random.getstate(),numpy_rng=np.random.get_state())
    if digest(rng)!=digest({k:state[k] for k in rng}):
        raise ValueError('Matched branch did not restore source RNG')
    if model_digest(model)!=state['model_sha256']:
        raise ValueError('Matched branch did not restore every non-texture source parameter')
    row=dict(iteration=expected,non_texture_model_sha256=model_digest(model),
             optimizer_sha256=digest(model.optimizer.state_dict()),rng_sha256=digest(rng),
             camera_order=[c.image_name for c in cameras],texture_zero=(model.texture_texels is None
                 or bool((model.texture_texels==0).all())))
    sample=render_fn(cameras[0],model,pipe,bg,stage='fine')
    torch.save(dict(**row,**{k:sample[k].detach().cpu() for k in ('render','depth','alpha')}),
               Path(output)/'matched_initial.pth')
    if digest(rng)!=digest(dict(torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                               python_rng=random.getstate(),numpy_rng=np.random.get_state())):
        raise RuntimeError('Initial audit unexpectedly consumed RNG')
    (Path(output)/'matched_initial.json').write_text(json.dumps(row,indent=2)+'\n')


def frozen_geometry_values(model):
    values={name:getattr(model,name) for name in ('_xyz','_scaling','_rotation','_opacity')}
    for name in ('node_mlp','graph_motion_residual','deformation_graph','appearance_graph','_deformation','surface_children','skinning'):
        module=getattr(model,name,None)
        if module is not None:values[name]=module.state_dict()
    for name in ('closest_point_indices','spatial_lr_scale'):
        if hasattr(model,name):values[name]=getattr(model,name)
    return values


def freeze_geometry(model,output):
    """Freeze every geometry path while preserving restored Adam groups/moments."""
    allowed=[model.features,model._deformation_table,*model.appearance_model.parameters()]
    if model.texture_texels is not None:allowed.append(model.texture_texels)
    allowed_ids={id(p) for p in allowed}
    frozen=[]
    for name in ('_xyz','_scaling','_rotation','_opacity'):
        p=getattr(model,name);p.requires_grad_(False);p.grad=None;frozen.append(p)
    for name in ('node_mlp','graph_motion_residual','_deformation','surface_children','skinning'):
        module=getattr(model,name,None)
        if module is not None:
            for p in module.parameters():p.requires_grad_(False);p.grad=None;frozen.append(p)
    for p in allowed:p.requires_grad_(True)
    for group in model.optimizer.param_groups:
        for p in group['params']:
            if p.requires_grad and id(p) not in allowed_ids:raise RuntimeError('Unexpected trainable parameter in '+group['name'])
        if not any(p.requires_grad for p in group['params']):group['lr']=0.
    model.frozen_geometry_sha256=digest(frozen_geometry_values(model))
    model.frozen_parameters=frozen
    model.frozen_adam_sha256=digest([model.optimizer.state.get(p,{}) for p in frozen])
    row=dict(geometry_sha256=model.frozen_geometry_sha256,frozen_adam_sha256=model.frozen_adam_sha256,
        frozen_parameter_count=sum(p.numel() for p in frozen),
        trainable={'canonical_color':model.features.numel(),'Gaussian_embedding':model._deformation_table.numel(),
                   'node_appearance_features':model.appearance_model.node_appearance.numel(),
                   'appearance_decoder':sum(p.numel() for p in model.appearance_model.parameters())-model.appearance_model.node_appearance.numel(),
                   'texture':0 if model.texture_texels is None else model.texture_texels.numel()})
    (Path(output)/'frozen_geometry_initial.json').write_text(json.dumps(row,indent=2)+'\n')


def audit_frozen_geometry(model,iteration,output):
    checks=dict(geometry_exact=digest(frozen_geometry_values(model))==model.frozen_geometry_sha256,
        frozen_gradients_absent=all(p.grad is None and not p.requires_grad for p in model.frozen_parameters),
        frozen_adam_unchanged=digest([model.optimizer.state.get(p,{}) for p in model.frozen_parameters])==model.frozen_adam_sha256)
    with (Path(output)/'frozen_geometry_audit.jsonl').open('a') as stream:stream.write(json.dumps(dict(iteration=iteration,**checks))+'\n')
    if not all(checks.values()):raise RuntimeError('Frozen geometry changed: '+str(checks))
