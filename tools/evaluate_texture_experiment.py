"""Final matched evaluation, native exports, texture-off control and research report."""
import argparse
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from gaussian_renderer import render
from tools.experiment_common import load_scene,tensor_digest
from utils.canonical_texture import texture_statistics,ray_parameters
from utils.exact_split_branchpoint import RAW_NAMES,digest,model_digest
from utils.image_utils import psnr,ssim,lpips_score
from diff_gaussian_rasterization import GaussianRasterizationSettings


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def quantize(tensor):
    return (tensor.clamp(0,1)*255+.5).to(torch.uint8).cpu().numpy()


def all_cameras(scene):
    cameras=scene.getVideoCameras()
    if cameras is None:cameras=scene.getTrainCameras()+scene.getTestCameras()
    by_name={camera.image_name:camera for camera in cameras}
    return [by_name[key] for key in sorted(by_name,key=int)]


def distribution(tensor):
    tensor=torch.as_tensor(tensor).detach().float().flatten()
    finite=torch.isfinite(tensor);valid=tensor[finite]
    if not valid.numel():return dict(count=len(tensor),finite_fraction=0.)
    q=torch.quantile(valid,torch.tensor([0.,.01,.5,.9,.95,.99,1.],device=valid.device))
    return dict(count=len(tensor),finite_fraction=float(finite.float().mean()),
        mean=float(valid.mean()),rms=float(valid.square().mean().sqrt()),
        **dict(zip(('min','p01','median','p90','p95','p99','max'),q.tolist())))


def metrics(prediction,target,perceptual=True):
    prediction=prediction.contiguous()[None];target=target.contiguous()[None]
    row=dict(PSNR=float(psnr(prediction,target).mean()),SSIM=float(ssim(prediction,target)))
    if perceptual:
        row.update(LPIPS_A=float(lpips_score(prediction,target,net='alex').mean()),
                   LPIPS_V=float(lpips_score(prediction,target,net='vgg').mean()))
    return row


def camera_settings(camera,model,bg):
    return GaussianRasterizationSettings(image_height=camera.image_height,image_width=camera.image_width,
        tanfovx=math.tan(camera.FoVx/2),tanfovy=math.tan(camera.FoVy/2),bg=bg,
        scale_modifier=1.,viewmatrix=camera.world_view_transform.cuda(),
        projmatrix=camera.full_proj_transform.cuda(),sh_degree=model.active_sh_degree,
        campos=camera.camera_center.cuda(),prefiltered=False,debug=False,
        require_coord=True,require_depth=True,kernel_size=0.)


@torch.no_grad()
def uv_diagnostics(model,camera,bg):
    n=len(model._xyz);times=torch.full((n,1),float(camera.time),device='cuda')
    means,scales,rotations,_,_=model.deform(model._xyz,model._scaling,model._rotation,
        model._opacity,model.features,times,model._deformation_table)
    scales=model.scaling_activation(scales).clamp_max(.05*model.spatial_lr_scale)
    rotations=model.rotation_activation(rotations)
    s=camera_settings(camera,model,bg)
    packed=ray_parameters(means,scales,rotations,model.texture_axis_order,s)
    ids=torch.arange(0,n,max(1,n//8192),device='cuda')
    points=means[ids];p=torch.cat((points,torch.ones_like(points[:,:1])),1)@s.projmatrix
    screen=p[:,:2]/p[:,3:].clamp_min(1e-8)
    screen=(screen+1)*points.new_tensor([s.image_width,s.image_height])/2-.5
    in_frame=(p[:,3]>0)&(screen[:,0]>=0)&(screen[:,0]<s.image_width)&(screen[:,1]>=0)&(screen[:,1]<s.image_height)
    ids=ids[in_frame];screen=screen[in_frame]
    offsets=points.new_tensor([[0.,0.],[2.,0.],[-2.,0.],[0.,2.],[0.,-2.]])
    pixels=screen[:,None]+offsets[None]
    pixel_h=torch.cat((pixels,torch.ones_like(pixels[:,:,:1])),2)
    ray=torch.einsum('nij,npj->npi',packed[ids,3:].reshape(-1,3,3),pixel_h)
    denominator=ray.square().sum(-1);origin=packed[ids,:3,None].transpose(1,2)
    dot=(ray*origin).sum(-1)
    local=origin-(dot/denominator.clamp_min(1e-12))[:,:,None]*ray
    valid=torch.isfinite(local).all(-1)&torch.isfinite(denominator)&(denominator>1e-12)
    return dict(camera=camera.image_name,gaussians_sampled=len(ids),rays_sampled=valid.numel(),
        invalid_fraction=float((~valid).float().mean()) if valid.numel() else None,
        near_degenerate_fraction=float((denominator<=1e-12).float().mean()) if valid.numel() else None,
        uv_clamped_fraction=float((local[...,:2].abs()>3).any(-1).float().mean()) if valid.numel() else None,
        ray_denominator=distribution(denominator),
        note='Sampled projected Gaussian centers and +/-2px offsets, including culled support; not an exact contribution-weighted UV census.')


@torch.no_grad()
def benchmark(model,scene,pipe,bg):
    cameras=scene.getTestCameras()
    for camera in cameras[:5]:render(camera,model,pipe,bg)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    samples=[];wall=time.monotonic()
    for _ in range(3):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        for camera in cameras:render(camera,model,pipe,bg)
        end.record();end.synchronize()
        samples.append(start.elapsed_time(end)/len(cameras))
    native_wall=(time.monotonic()-wall)/(3*len(cameras))
    times=torch.full((len(model._xyz),1),.5,device='cuda')
    graph=model.appearance_graph if model.appearance_graph is not None else model.deformation_graph
    def color():
        return model.appearance_model(model._xyz,times,model.features,model._deformation_table,
            graph.gaussian_node_indices,graph.gaussian_node_weights)
    for _ in range(5):color()
    color_samples=[]
    for _ in range(30):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record();color();end.record();end.synchronize()
        color_samples.append(start.elapsed_time(end))
    return dict(device=torch.cuda.get_device_name(0),device_index=0,test_views=len(cameras),
        native_render_ms=float(np.mean(samples)),native_fps=1000/float(np.mean(samples)),
        native_wall_ms=native_wall*1000,color_forward_ms=float(np.median(color_samples)),
        eval_peak_vram_mib=torch.cuda.max_memory_allocated()/2**20,repeats=3,
        note='Same GPU, sequential models, CUDA events, no_grad, original profiling disabled, no exports/LPIPS. Color-forward is R6 only; per-pixel texture cost is included in native FPS.')


@torch.no_grad()
def checkpoint_audit(path,model,scene,pipe,bg,config,iteration=30000):
    from utils.exact_split_branchpoint import restore_knn_reference
    restore_knn_reference(model,config.gaussian_knn_reference)
    state=torch.load(path/f'fine_state_{iteration}.pth',map_location='cpu',weights_only=False)
    checks=dict(iteration_expected=state['iteration']==iteration,
        full_raw_state_exact=all(torch.equal(getattr(model,k).detach().cpu(),state['exact_raw'][k]) for k in RAW_NAMES),
        non_texture_model_digest_exact=model_digest(model)==state['model_sha256'],
        gaussian_count_364500=len(model._xyz)==364500,nodes_2000=len(model.deformation_graph.node_xyz)==2000,
        texture_state_exact=model.texture_texels is None or torch.equal(model.texture_texels.detach().cpu(),state['exact_texture']))
    reference=torch.load(config.graph_forward_reference,map_location='cpu',weights_only=True)
    checks.update(geometry_k4_exact=torch.equal(model.deformation_graph.gaussian_node_indices.cpu(),reference['graph_k4_indices']),
        geometry_weights_exact=torch.equal(model.deformation_graph.gaussian_node_weights.cpu(),reference['graph_k4_weights']),
        appearance_k4_exact=torch.equal(model.appearance_graph.gaussian_node_indices.cpu(),reference['original_k4_indices']),
        appearance_weights_exact=torch.equal(model.appearance_graph.gaussian_node_weights.cpu(),reference['original_k4_weights']),
        surface_edges_exact=torch.equal(model.graph_motion_residual.edge_indices.cpu(),reference['surface_edge_indices']),
        trajectory_gates_exact=torch.equal(model.graph_motion_residual.edge_weights.cpu(),reference['trajectory_gated_edge_weights']))
    optimizer=[]
    for group in state['optimizer']['param_groups']:
        moments=[state['optimizer']['state'].get(index,{}) for index in group['params']]
        steps=[float(x['step']) for x in moments if 'step' in x]
        row=dict(name=group['name'],lr=group['lr'],parameters=len(group['params']),states=len(steps),
            step_min=min(steps) if steps else None,step_max=max(steps) if steps else None,
            finite=all(bool(torch.isfinite(value).all()) for x in moments for value in x.values() if torch.is_tensor(value)))
        if group['name']=='texture':
            row['moment_rms']=float(moments[0]['exp_avg'].square().mean().sqrt())
            row['moment_shape']=list(moments[0]['exp_avg'].shape)
        optimizer.append(row)
    checks['optimizer_finite']=all(x['finite'] for x in optimizer)
    model.training_setup(config)
    model.add_graph_residual_optimizer_group()
    if model.texture_texels is not None:model.add_texture_optimizer_group()
    model.optimizer.load_state_dict(state['optimizer'])
    checks['optimizer_roundtrip_exact']=digest(model.optimizer.state_dict())==digest(state['optimizer'])
    model.optimizer=None
    if model.texture_texels is not None:
        texture_group=next(g for g in optimizer if g['name']=='texture')
        checks.update(texture_count_aligned=len(model.texture_texels)==len(model._xyz),
            texture_optimizer_steps_exact=texture_group['step_min']==iteration-config.texture_activate_iteration-(1 if iteration==30000 else 0),
            texture_axes_shape=model.texture_axis_order.shape==(len(model._xyz),3))
    # Native checkpoint reloaded independently; CUDA graph scatter is nondeterministic.
    second,second_scene,second_pipe,second_bg,_=load_scene(path,iteration)
    restore_knn_reference(second,config.gaussian_knn_reference)
    second.profile_deformation=False
    checks['native_reload_state_exact']=model_digest(second)==model_digest(model)
    checks['native_reload_texture_exact']=(model.texture_texels is None or
        (torch.equal(model.texture_texels,second.texture_texels) and torch.equal(model.texture_axis_order,second.texture_axis_order)))
    first=render(scene.getTestCameras()[17],model,pipe,bg)
    repeat=render(second_scene.getTestCameras()[17],second,second_pipe,second_bg)
    differences={k:float((first[k]-repeat[k]).abs().max()) for k in ('render','depth','alpha')}
    rms_difference={k:float((first[k]-repeat[k]).square().mean().sqrt()) for k in ('render','depth','alpha')}
    if getattr(config,'freeze_geometry',False):
        checks['native_reload_rgb_RMS_within_atomic_tolerance']=rms_difference['render']<1e-5
        camera=scene.getTestCameras()[17]
        original_first,original_second=model.deform,second.deform
        time=torch.full((len(model._xyz),1),float(camera.time),device='cuda')
        shared=original_first(model._xyz,model._scaling,model._rotation,model._opacity,model.features,time,model._deformation_table)
        try:
            model.deform=lambda *args,**kwargs:shared
            second.deform=lambda *args,**kwargs:shared
            isolated_first=render(camera,model,pipe,bg);isolated_second=render(camera,second,pipe,bg)
            checks['native_reload_shared_inputs_exact']=all(torch.equal(isolated_first[k],isolated_second[k]) for k in ('render','depth','alpha'))
        finally:
            model.deform,second.deform=original_first,original_second
        del shared,isolated_first,isolated_second
    else:
        checks['native_reload_rgb_within_atomic_tolerance']=differences['render']<1e-3
    graph_diagnostics=model.graph_forward_diagnostics(reference['trajectory_times'])
    result=dict(pass_=all(checks.values()),checks=checks,optimizer=optimizer,
        reload_max_difference=differences,reload_RMS_difference=rms_difference,graph_forward=graph_diagnostics,
        checkpoint_bytes=sum(p.stat().st_size for p in (path/f'point_cloud/iteration_{iteration}').glob('*') if p.is_file()),
        full_state_sha256=file_hash(path/f'fine_state_{iteration}.pth'),
        note=('Exact serialized parameters and full Adam state; independent RGB replay has original graph-scatter variation. Frozen-geometry reload also requires exact shared-input RGB/depth/alpha identity.'
              if getattr(config,'freeze_geometry',False) else
              'Exact serialized parameters and full Adam state; independent RGB replay has original graph-scatter variation. Final A/B geometry differs because both train jointly.'))
    del second,second_scene,state,reference,first,repeat;gc.collect();torch.cuda.empty_cache()
    write_json(path/'checkpoint_integrity.json',result)
    if not result['pass_']:raise RuntimeError('Checkpoint integrity failed: '+str(checks))
    return result


@torch.no_grad()
def deformation_diagnostics(model,scene):
    xyz=model._xyz
    reference=torch.load(model.args.graph_forward_reference,map_location='cuda',weights_only=True)
    edges=reference['surface_edge_indices']
    a,b=edges[0],edges[1]
    canonical=model.deformation_graph.node_xyz
    length=(canonical[a]-canonical[b]).norm(dim=-1).clamp_min(1e-8)
    stretch=[];fold=[];eligible=[]
    ids=torch.arange(0,len(xyz),max(1,len(xyz)//4096),device='cuda')
    neighbors=model.closest_point_indices.to('cuda')[ids,:12]
    delta=xyz[neighbors]-xyz[ids,None]
    covariance=delta.transpose(1,2)@delta
    eigen=torch.linalg.eigvalsh(covariance)
    well_conditioned=eigen[:,0]>eigen[:,-1]*1e-3
    for camera in [all_cameras(scene)[i] for i in (0,72,144,216,264,287)]:
        positions=model.node_positions_at_time(camera.time)
        stretch.append(((positions[a]-positions[b]).norm(dim=-1)/length-1).abs())
        transformed=model.deform_geometry_from_nodes(camera.time)[0]
        moved=transformed[neighbors]-transformed[ids,None]
        jacobian=torch.linalg.pinv(covariance)@(delta.transpose(1,2)@moved)
        determinants=torch.linalg.det(jacobian)
        valid=well_conditioned&torch.isfinite(determinants)
        fold.append(float((determinants[valid]<0).float().mean()) if bool(valid.any()) else None)
        eligible.append(float(valid.float().mean()))
    return dict(node_edge_absolute_relative_stretch=distribution(torch.cat(stretch)),
        sampled_local_fold_fraction=fold,fold_eligible_fraction=eligible,
        fold_times=[0,72,144,216,264,287],
        note='Fold proxy fits local 3D least-squares Jacobians to 12 fixed KNN99 neighbors on ~4096 Gaussians, excluding condition ratio <=1e-3; surface-like rank-deficient neighborhoods excluded. It is not calibrated tissue geometry.')


@torch.no_grad()
def evaluate_branch(root,name):
    off=name=='B_texture_off';branch='B' if off else name;path=root/branch
    output=root/'evaluation'/name;output.mkdir(parents=True,exist_ok=True)
    model,scene,pipe,bg,config=load_scene(path,30000)
    model.profile_deformation=False
    integrity=None if off else checkpoint_audit(path,model,scene,pipe,bg,config)
    if off:model.texture_active=False
    # The fixed KNN99 is an audited training regularizer, unused by inference.
    # Keep it off GPU for all native benchmarks, including B texture-off.
    if hasattr(model,'closest_point_indices'):
        model.closest_point_indices=model.closest_point_indices.cpu()
    for key in ('xyz_gradient_accum','denom'):
        if hasattr(model,key):setattr(model,key,torch.empty(0))
    from utils.image_utils import lpips_alex,lpips_vgg
    lpips_alex.to('cpu');lpips_vgg.to('cpu');gc.collect();torch.cuda.empty_cache()
    timing=benchmark(model,scene,pipe,bg)
    primary={};alpha_rows=[];normalized_depths=[];alphas=[];temporal=[];gacv=[]
    frames=output/'frames';frames.mkdir(exist_ok=True)
    targets=root/'evaluation/gt';targets.mkdir(parents=True,exist_ok=True)
    tests={c.image_name:i for i,c in enumerate(scene.getTestCameras())}
    native=path/'test/ours_30000' if not off else output/'test/ours_30000'
    for directory in ('renders','gt','depth','gt_depth','alpha'):(native/directory).mkdir(parents=True,exist_ok=True)
    depth_by_name={};alpha_by_name={};previous=None
    for camera in all_cameras(scene):
        result=render(camera,model,pipe,bg)
        rgb=result['render'].clamp(0,1);gt=camera.original_image[:3].cuda().float()
        byte=quantize(rgb).transpose(1,2,0);gtbyte=quantize(gt).transpose(1,2,0)
        Image.fromarray(byte).save(frames/(camera.image_name+'.png'))
        if not (targets/(camera.image_name+'.png')).exists():Image.fromarray(gtbyte).save(targets/(camera.image_name+'.png'))
        residual=rgb-gt
        if previous is not None:
            delta=residual-previous['residual']
            temporal.append(dict(frame=int(camera.image_name),residual_temporal_l1=float(delta.abs().mean()),
                residual_temporal_rms=float(delta.square().mean().sqrt()),
                predicted_temporal_l1=float((rgb-previous['rgb']).abs().mean()),
                gt_temporal_l1=float((gt-previous['gt']).abs().mean())))
            if int(camera.image_name) in (144,145,264,265):
                shared=result['visibility_filter']&previous['visible']
                changes=(result['transformed_color']-previous['color'])[shared].abs()
                gacv.append(dict(pair=[int(camera.image_name)-1,int(camera.image_name)],
                    shared_visible_gaussians=int(shared.sum()),intrinsic_mean_rgb_l1=float(changes.mean())))
        previous=dict(rgb=rgb,gt=gt,residual=residual,visible=result['visibility_filter'],color=result['transformed_color'])
        if camera.image_name in tests:
            index=tests[camera.image_name];filename=f'{index:05d}.png'
            Image.fromarray(byte).save(native/'renders'/filename)
            Image.fromarray(gtbyte).save(native/'gt'/filename)
            metric_rgb=torch.as_tensor(byte.transpose(2,0,1),device='cuda').float()/255
            metric_gt=torch.as_tensor(gtbyte.transpose(2,0,1),device='cuda').float()/255
            row=metrics(metric_rgb,metric_gt)
            row.update(camera=camera.image_name,index=index,time=float(camera.time),PSNR_float=float(psnr(rgb[None],gt[None]).mean()))
            depth=result['depth'].squeeze();prior=camera.depth.cuda().float().squeeze()
            depth=depth/depth.max().clamp_min(1e-8);prior=prior/prior.max().clamp_min(1e-8)
            qdepth=quantize(depth);qprior=quantize(prior)
            row.update(DEPTH_MSE=float(np.mean(((qdepth.astype(np.float32)-qprior.astype(np.float32))/255)**2)),
                DEPTH_MSE_float=float((depth-prior).square().mean()),alpha_mean=float(result['alpha'].mean()),
                alpha_coverage_0_5=float((result['alpha']>.5).float().mean()),alpha_coverage_0_95=float((result['alpha']>.95).float().mean()),
                rgb_unclamped_negative_fraction=float((result['render']<0).float().mean()),
                rgb_unclamped_above_one_fraction=float((result['render']>1).float().mean()))
            Image.fromarray(qdepth).save(native/'depth'/filename);Image.fromarray(qprior).save(native/'gt_depth'/filename)
            Image.fromarray(quantize(result['alpha'].squeeze())).save(native/'alpha'/filename)
            primary[camera.image_name]=row;depth_by_name[camera.image_name]=depth.cpu().numpy();alpha_by_name[camera.image_name]=result['alpha'].squeeze().cpu().numpy()
        if int(camera.image_name)%48==0:print(name,'frame',camera.image_name,flush=True)
    sorted_names=sorted(primary,key=int)
    np.savez_compressed(output/'depth_alpha.npz',names=np.array(sorted_names),
        depth=np.stack([depth_by_name[n] for n in sorted_names]),alpha=np.stack([alpha_by_name[n] for n in sorted_names]))
    means={key:float(np.mean([primary[n][key] for n in sorted_names])) for key in primary[sorted_names[0]] if key not in ('camera','index','time')}
    texture=texture_statistics(model)
    gaussian_stats=dict(scale=distribution(model.get_scaling),opacity=distribution(model.get_opacity),
        effective_canonical_scale=distribution(model.get_scaling.clamp_max(.05*model.spatial_lr_scale)),
        scale_bound=.05*model.spatial_lr_scale,
        canonical_scale_clamped_fraction=float((model.get_scaling>.05*model.spatial_lr_scale).float().mean()),
        canonical_rgb=distribution(model.features),rotation_norm=distribution(model._rotation.norm(dim=-1)))
    appearance_params=model.features.numel()+model._deformation_table.numel()+sum(p.numel() for p in model.appearance_model.parameters())
    if model.texture_texels is not None:appearance_params+=model.texture_texels.numel()
    summary=dict(name=name,test_views=len(primary),metrics=means,performance=timing,
        gaussian_count=len(model._xyz),appearance_parameters=appearance_params,
        texture=texture,gaussian_statistics=gaussian_stats,checkpoint_integrity=integrity,
        GACV=float(np.mean([r['intrinsic_mean_rgb_l1'] for r in gacv])),GACV_pairs=gacv,
        temporal_residual_l1=float(np.mean([r['residual_temporal_l1'] for r in temporal])),
        temporal_residual_rms=float(np.mean([r['residual_temporal_rms'] for r in temporal])),
        temporal_note='Consecutive 288-frame camera sequence; unwarped temporal error change relative to GT, not optical-flow compensated reflectance flicker.',
        GACV_note='Mean intrinsic RGB variation on shared-visible Gaussians in 143->144->145 and 263->264->265. Zero-mean texture contribution integrates to zero canonically; GACV measures R6 mean, not per-pixel texture flicker.')
    if not off:
        summary['training']=json.loads((path/'training_summary.json').read_text())
        summary['deformation']=deformation_diagnostics(model,scene)
        if model.texture_texels is not None:
            summary['uv']=[uv_diagnostics(model,all_cameras(scene)[i],bg) for i in (0,144,264,287)]
    write_json(output/'summary.json',summary);write_json(output/'per_view.json',primary);write_json(output/'temporal.json',temporal)
    write_json(path/'results.json' if not off else output/'results.json',{'ours_30000':means})
    print(name,means,flush=True)
    del model,scene,previous,result;gc.collect();torch.cuda.empty_cache()
    return summary


def panel(images,labels,title):
    width=sum(img.width for img in images);height=max(img.height for img in images)+56
    canvas=Image.new('RGB',(width,height),(25,25,25));draw=ImageDraw.Draw(canvas)
    try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    except OSError:font=ImageFont.load_default()
    draw.text((8,5),title,fill='white',font=font)
    x=0
    for image,label in zip(images,labels):
        draw.text((x+8,30),label,fill='white',font=font);canvas.paste(image,(x,56));x+=image.width
    return canvas


def visual_artifacts(root):
    evaluation=root/'evaluation';visual=root/'visuals';visual.mkdir(exist_ok=True)
    if not (root/'roi_coordinates.json').exists():
        write_json(root/'roi_coordinates.json',dict(source='Selected from source RGB; historical crop coordinates were absent from supplied files.',
            coordinates={'144':{'texture_left':[20,100,180,300],'highlight_right':[440,20,215,290]},
                         '264':{'texture_left':[20,40,180,260],'highlight_right':[515,245,145,250]}}))
    roi=json.loads((root/'roi_coordinates.json').read_text());roi_results={}
    def paths(frame):
        return [evaluation/'gt'/f'{frame}.png']+[evaluation/name/'frames'/f'{frame}.png' for name in ('A','B','B_texture_off')]
    for frame,boxes in roi['coordinates'].items():
        full=[Image.open(path).convert('RGB') for path in paths(frame)]
        panel(full,['GT','A: R6','B: R6 + texture','B: texture off'],f'Frame {frame}: training diagnostic').save(visual/f'frame_{frame}.png')
        for name,(x,y,w,h) in boxes.items():
            images=[img.crop((x,y,x+w,y+h)) for img in full]
            panel(images,['GT','A: R6','B: R6 + texture','B: texture off'],f'Frame {frame} {name}: x={x} y={y} w={w} h={h}').save(visual/f'frame_{frame}_{name}.png')
            target=torch.as_tensor(np.array(images[0]).transpose(2,0,1),device='cuda').float()/255
            row={}
            for label,img in zip(('A','B','B_texture_off'),images[1:]):
                pred=torch.as_tensor(np.array(img).transpose(2,0,1),device='cuda').float()/255
                row[label]=metrics(pred,target)
                residual=(pred-target)
                row[label]['mean_rgb_error']=residual.mean((1,2)).tolist()
                row[label]['bias_removed_rmse']=float((residual-residual.mean((1,2),keepdim=True)).square().mean().sqrt())
                # Fine-detail diagnostic after subtracting a local 5x5 box average.
                residual_high=residual[:,2:-2,2:-2]-torch.nn.functional.avg_pool2d(residual[None],5,stride=1,padding=0)[0]
                row[label]['highpass_residual_rmse']=float(residual_high.square().mean().sqrt())
            roi_results[f'{frame}_{name}']=row
            contact=[]
            for neighboring in (int(frame)-1,int(frame),int(frame)+1):
                crops=[Image.open(path).convert('RGB').crop((x,y,x+w,y+h)) for path in paths(str(neighboring))]
                contact.append(panel(crops,['GT','A: R6','B: R6 + texture','B: texture off'],f'Consecutive frame {neighboring}, {name}'))
            sheet=Image.new('RGB',(contact[0].width,sum(image.height for image in contact)),(25,25,25))
            offset=0
            for image in contact:sheet.paste(image,(0,offset));offset+=image.height
            sheet.save(visual/f'consecutive_{frame}_{name}.png')
    test_names=sorted(json.loads((evaluation/'A/per_view.json').read_text()),key=int)
    for filename,names,fps,title in [('heldout_comparison.mp4',test_names,6,'Held-out sampled views; not a consecutive timeline'),
            ('consecutive_texture_stability.mp4',[str(i) for i in range(288)],30,'Consecutive original camera frames')]:
        with imageio.get_writer(visual/filename,fps=fps,codec='libx264',quality=8,macro_block_size=2) as writer:
            for frame in names:
                images=[Image.open(path).convert('RGB') for path in paths(frame)]
                writer.append_data(np.array(panel(images,['GT','A: R6','B: R6 + texture','B: texture off'],f'{title}; frame {frame}')))
    write_json(evaluation/'roi.json',dict(definition=roi,metrics=roi_results,note='144/264 are training frames. Highpass is an evaluation diagnostic only; it was not a training loss.'))
    return roi_results


def learning_curves(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    destination=root/'learning_curves';destination.mkdir(exist_ok=True)
    summaries={}
    common_config=json.loads((root/'configs/common.json').read_text())
    initializer=Path(common_config['ModelHiddenParams']['fine_warmup_reference']).parents[1]
    prefix=[]
    for directory in (initializer,root/'common_5000'):
        with (directory/'joint_r6_diagnostics.csv').open() as stream:prefix+=list(csv.DictReader(stream))
    for name in ('A','B'):
        with (root/name/'joint_r6_diagnostics.csv').open() as stream:rows=list(csv.DictReader(stream))
        complete=prefix+rows
        plots=[('loss','Joint loss'),('rgb_l1','Training RGB L1'),('depth_loss','Depth loss'),('iteration_ms','Update compute ms')]
        figure,axes=plt.subplots(2,2,figsize=(12,7))
        for ax,(key,label) in zip(axes.flat,plots):
            ax.plot([int(r['iteration']) for r in complete],[float(r[key]) for r in complete],linewidth=1)
            ax.axvspan(0,5000,color='lightgray',alpha=.3,label='Shared prefix')
            ax.axvline(25000,color='gray',linestyle='--');ax.set(xlabel='Iteration',ylabel=label);ax.grid(alpha=.2)
        figure.suptitle(name+': matched joint G3-Surf + R6'+(' + texture' if name=='B' else ''))
        figure.tight_layout();figure.savefig(destination/f'{name}.png',dpi=150);plt.close(figure)
        with (destination/f'{name}.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=complete[0].keys());writer.writeheader();writer.writerows(complete)
        times=[float(r['iteration_ms']) for r in rows]
        summaries[name]=dict(samples=len(rows),update_ms_median=float(np.median(times)),update_ms_mean=float(np.mean(times)),
            note='Original iteration CUDA-event timing includes render/loss/backward, excludes optimizer and CPU diagnostics; training branches used separate identical GPUs concurrently.')
    diagnostics=[json.loads(line) for line in (root/'B/texture_diagnostics.jsonl').read_text().splitlines()]
    shutil.copy2(root/'B/texture_diagnostics.jsonl',destination/'texture_statistics.jsonl')
    figure,axes=plt.subplots(2,2,figsize=(12,7))
    for ax,(key,label) in zip(axes.flat,[('centered_rms','Texture RMS'),('texture_grad_norm','Texture gradient norm'),
            ('update_rms','Sampled texture update RMS'),('weighted_mean_max_error','Weighted mean max error')]):
        ax.plot([r['iteration'] for r in diagnostics],[r[key] for r in diagnostics]);ax.set(xlabel='Iteration',ylabel=label);ax.grid(alpha=.2)
    figure.tight_layout();figure.savefig(destination/'texture.png',dpi=150);plt.close(figure)
    optimizer_audit=dict(first_iteration=diagnostics[0]['iteration'],initial_texture_zero=diagnostics[0]['centered_rms']==0,
        first_grad_norm=diagnostics[0]['texture_grad_norm'],first_update_rms=diagnostics[0]['update_rms'],
        lr_before_decay=next(r['texture_lr'] for r in diagnostics if r['iteration']==25000),
        lr_after_decay=next(r['texture_lr'] for r in diagnostics if r['iteration']==25100),
        finite=all(math.isfinite(value) for row in diagnostics for value in row.values() if isinstance(value,(int,float))),
        r6_gradient_nonzero_fraction=float(np.mean([r['r6_grad_norm']>0 for r in diagnostics])),
        canonical_rgb_gradient_nonzero_fraction=float(np.mean([r['canonical_rgb_grad_norm']>0 for r in diagnostics])),
        texture_gradient_nonzero_fraction=float(np.mean([r['texture_grad_norm']>0 for r in diagnostics])),
        final_iteration_update_rms=diagnostics[-1]['update_rms'])
    write_json(root/'gradient_optimizer_audit.json',optimizer_audit)
    return summaries,optimizer_audit


def service_elapsed(unit):
    query=subprocess.run(['systemctl','--user','show',unit,'-p','ActiveEnterTimestampMonotonic','--value'],
        capture_output=True,text=True)
    try:start=int(query.stdout.strip())/1e6
    except ValueError:return None
    return time.monotonic()-start if start>0 else None


def report(root,summaries,rois,learning,optimizer):
    evaluation=root/'evaluation';a,b,off=[summaries[n] for n in ('A','B','B_texture_off')]
    validation=root/'validation';validation.mkdir(exist_ok=True)
    for path in (ROOT/'output/texture_validation').glob('*.json'):shutil.copy2(path,validation/path.name)
    implementation={}
    code_paths=[ROOT/name for name in ('arguments/__init__.py','gaussian_renderer/__init__.py','scene/gaussian_model.py',
        'train.py','utils/canonical_texture.py','utils/texture_training.py','tools/run_texture_experiment.py',
        'tools/evaluate_texture_experiment.py','tools/validate_canonical_texture.py')]
    fork=ROOT/'submodules/diff-gaussian-rasterization-texture'
    code_paths+=list((fork/'cuda_rasterizer').glob('*'))+list((fork/'diff_gaussian_rasterization_texture').glob('*.py'))
    code_paths+=[fork/name for name in ('setup.py','ext.cpp','rasterize_points.cu','rasterize_points.h')]
    for path in code_paths:
        if path.is_file():implementation[str(path.relative_to(ROOT))]=file_hash(path)
    import diff_gaussian_rasterization_texture._C as extension
    write_json(root/'implementation_manifest.json',dict(files=implementation,
        extension_path=extension.__file__,extension_sha256=file_hash(extension.__file__),
        torch=torch.__version__,cuda=torch.version.cuda,python=sys.version,
        validation_timing='Core renderer gates passed before full training; additional Jacobian, variance and trained-10k reload checks were performed during training.'))
    initial=json.loads((root/'matched_initial_audit.json').read_text())
    source=json.loads((root/'source_manifest.json').read_text())
    source_checks={path:file_hash(path)==row['sha256'] for path,row in source.items()}
    cameras_a=(root/'A/connectivity_camera_sequence.jsonl').read_bytes()
    cameras_b=(root/'B/connectivity_camera_sequence.jsonl').read_bytes()
    camera_rows=[json.loads(line) for line in cameras_a.splitlines()]
    camera_check=cameras_a==cameras_b and len(camera_rows)==25000 and camera_rows[0]['iteration']==5001 and camera_rows[-1]['iteration']==30000
    config_a=json.loads((root/'configs/A.json').read_text());config_b=json.loads((root/'configs/B.json').read_text())
    for config in (config_a,config_b):
        config['ModelParams'].pop('model_path');config['ModelHiddenParams'].pop('texture_enabled')
    config_check=config_a==config_b
    integrity=dict(source_artifacts_unchanged=all(source_checks.values()),source_files=source_checks,
        baseline_config_exact=config_check,
        camera_sequence_exact=camera_check,camera_sequence_sha256=hashlib.sha256(cameras_a).hexdigest(),
        camera_samples=len(camera_rows),initial=initial,branches={n:summaries[n]['checkpoint_integrity']['pass_'] for n in ('A','B')})
    integrity['pass']=all([integrity['source_artifacts_unchanged'],camera_check,config_check,*integrity['branches'].values()])
    write_json(root/'integrity_report.json',integrity)
    views={n:json.loads((evaluation/n/'per_view.json').read_text()) for n in ('A','B','B_texture_off')}
    deltas={key:{metric:views['B'][key][metric]-views['A'][key][metric] for metric in ('PSNR','SSIM','LPIPS_A','LPIPS_V','DEPTH_MSE')}
        for key in views['A']}
    write_json(root/'per_view_deltas.json',deltas)
    depth_a=np.load(evaluation/'A/depth_alpha.npz');depth_b=np.load(evaluation/'B/depth_alpha.npz')
    geometry=dict(normalized_depth_AB_mean_abs=float(np.mean(np.abs(depth_a['depth']-depth_b['depth']))),
        alpha_AB_mean_abs=float(np.mean(np.abs(depth_a['alpha']-depth_b['alpha']))),raw_parameters={})
    off_geometry=np.load(evaluation/'B_texture_off/depth_alpha.npz')
    texture_effect=dict(depth_normalized_max_difference=float(np.max(np.abs(depth_b['depth']-off_geometry['depth']))),
        alpha_max_difference=float(np.max(np.abs(depth_b['alpha']-off_geometry['alpha']))),rgb_mean_abs=[],rgb_max_abs=[])
    for key in views['B']:
        on_rgb=np.asarray(Image.open(evaluation/'B/frames'/f'{key}.png')).astype(np.float32)/255
        off_rgb=np.asarray(Image.open(evaluation/'B_texture_off/frames'/f'{key}.png')).astype(np.float32)/255
        change=np.abs(on_rgb-off_rgb)
        texture_effect['rgb_mean_abs'].append(float(change.mean()));texture_effect['rgb_max_abs'].append(float(change.max()))
    texture_effect['mean_test_rgb_abs_contribution']=float(np.mean(texture_effect['rgb_mean_abs']))
    texture_effect['note']='Same B checkpoint; RGB difference on quantized held-out images. Independent graph scatter replay can introduce tiny geometry variation; renderer equivalence was tested with shared geometry.'
    state_a=torch.load(root/'A/fine_state_30000.pth',map_location='cpu',weights_only=False)
    state_b=torch.load(root/'B/fine_state_30000.pth',map_location='cpu',weights_only=False)
    for key in ('_xyz','_scaling','_rotation','_opacity'):
        geometry['raw_parameters'][key]=distribution((state_a['exact_raw'][key]-state_b['exact_raw'][key]).abs())
    del state_a,state_b
    common=json.loads((root/'common_5000/training_summary.json').read_text())
    common_config=json.loads((root/'configs/common.json').read_text())
    initializer=Path(common_config['ModelHiddenParams']['fine_warmup_reference']).parents[1]
    initializer_summary=json.loads((initializer/'training_summary.json').read_text())
    result=dict(scene='c1_sigmoid2_t4_v4',target_PSNR=28.,target_reached=b['metrics']['PSNR']>=28,
        models=summaries,delta_B_minus_A={k:b['metrics'][k]-a['metrics'][k] for k in a['metrics']},
        texture_off_PSNR_drop=b['metrics']['PSNR']-off['metrics']['PSNR'],
        texture_off_image_contribution=texture_effect,
        texture_render_overhead_percent=100*(b['performance']['native_render_ms']/off['performance']['native_render_ms']-1),
        per_view_improved=int(sum(r['PSNR']>0 for r in deltas.values())),test_view_count=36,
        geometry_comparison=geometry,roi=rois,learning=learning,gradient_optimizer=optimizer,
        shared_prefix_time=common,integrity=integrity,protocol=json.loads((root/'protocol.json').read_text()),
        existing_initializer_time=initializer_summary,
        matched_pipeline_wall_s=service_elapsed('colongraph-texture-matched.service'),
        new_experiment_wall_including_common_s=service_elapsed('colongraph-texture-common5k.service'),
        zero_mean_validation=json.loads((ROOT/'output/texture_validation/preflight.json').read_text()),
        conclusion='Matched experiment completed; inspect RGB, depth and temporal tradeoffs together. No automatic promotion or follow-up variants.')
    write_json(root/'comparison.json',result)
    lines=['# Zero-mean intra-Gaussian RGB texture: matched G3-Surf/R6 experiment','',
        'Scene `c1_sigmoid2_t4_v4`; all 36 held-out views. A/B share the exact complete fine-5000 model, Adam, RNG and camera order. Both continue the established joint geometry/deformation/R6 objective to fine iteration 30000.','',
        '| Model | PSNR ↑ | SSIM ↑ | LPIPS-A ↓ | LPIPS-V ↓ | Normalized depth-prior MSE ↓ |',
        '|---|---:|---:|---:|---:|---:|']
    for name in ('A','B','B_texture_off'):
        m=summaries[name]['metrics'];lines.append(f"| {name} | {m['PSNR']:.5f} | {m['SSIM']:.6f} | {m['LPIPS_A']:.6f} | {m['LPIPS_V']:.6f} | {m['DEPTH_MSE']:.9f} |")
    delta=result['delta_B_minus_A']
    analysis=[]
    if delta['PSNR']>0:
        analysis.append(f"B improves held-out PSNR by {delta['PSNR']:.5f} dB on this matched pair.")
    else:analysis.append(f"B does not improve held-out PSNR (B−A {delta['PSNR']:+.5f} dB).")
    analysis.append(f"SSIM change is {delta['SSIM']:+.6f}; LPIPS-A/V changes are {delta['LPIPS_A']:+.6f}/{delta['LPIPS_V']:+.6f} (lower LPIPS is better).")
    analysis.append(f"Normalized depth-prior MSE changes by {100*delta['DEPTH_MSE']/max(a['metrics']['DEPTH_MSE'],1e-12):+.2f}%; consecutive residual temporal L1 changes by {100*(b['temporal_residual_l1']-a['temporal_residual_l1'])/max(a['temporal_residual_l1'],1e-12):+.2f}%.")
    roi_psnr=np.mean([r['B']['PSNR']-r['A']['PSNR'] for r in rois.values()])
    detail_improved=sum(r['B']['highpass_residual_rmse']<r['A']['highpass_residual_rmse'] for r in rois.values())
    analysis.append(f"The four adopted training ROIs have mean PSNR change {roi_psnr:+.5f} dB; local highpass residual improves in {detail_improved}/4. These crops are not held-out evidence.")
    if roi_psnr>0 and delta['PSNR']<=0:
        analysis.append('Training-crop gains without held-out PSNR gains are consistent with possible overfitting or view-correspondence limitations; this pair does not isolate their cause.')
    analysis.append(f"Turning texture off changes held-out PSNR by {-result['texture_off_PSNR_drop']:+.5f} dB; mean absolute RGB contribution is {texture_effect['mean_test_rgb_abs_contribution']:.6f}.")
    analysis.append('This is one matched pair with the original CUDA nondeterminism, without replicated seeds. It does not establish robustness or a strict frequency separation. No model was promoted.')
    result['interpretation']=' '.join(analysis)
    write_json(root/'comparison.json',result)
    lines+=['',' '.join(analysis)]
    lines+=['',f"B−A PSNR: **{delta['PSNR']:+.5f} dB**; improved test views: {result['per_view_improved']}/36. B texture-off PSNR decrease: {result['texture_off_PSNR_drop']:.5f} dB. Target ≥28 dB {'reached' if result['target_reached'] else 'not reached'}.",
        '', 'Primary RGB metrics reproduce the existing PNG-export convention: clamp output only, round to 8 bits, whole image without mask, SSIM 11×11 Gaussian window, LPIPS `normalize=True`. Depth is independently normalized by each image maximum and quantized; `*_float` diagnostics are also in JSON. The learned depth prior is not calibrated physical GT.',
        '', '## Implementation and protocol', '',
        'The upstream `textured_gaussians` code uses 2DGS plane intersections and RGB/alpha textures. This implementation extends ColonGraph’s original RaDe-GS rasterizer with RGB sampling inside each Gaussian-pixel contribution. Projection, covariance, sorting, culling, alpha, depth and transmittance remain unchanged. Geometry is still trained jointly; final A/B depth and alpha are allowed to differ.',
        '', 'Each Gaussian has a static 3×3 RGB texture. Canonical grid knots are −3, 0, +3 standardized local units; sampling is bilinear and clamps at the border. The measure is normalized independent N(0,1) truncated on [−3,3]². Analytic bilinear integration coefficients center the texture continuously, rather than averaging nine texels. CUDA RGB is `(base + centered_texture) * (1 + R6_delta)`; no extra loss, bias, clipping or alpha texture was added.',
        '', 'UV uses the ray’s closest point in the deformed 3D Gaussian Mahalanobis metric. Axis identity is fixed at initialization, choosing the smallest canonical scale as the excluded normal axis. It is transported with the baseline’s deformed scale and quaternion. A reliable tissue tangent plane is unavailable; this volume chart is an explicit alternative, not the upstream 2DGS surface chart. Degenerate/nonfinite rays disable only the texture correction and have zero UV gradient.',
        '', 'Textures are zero/inactive through iteration 5000; a new Adam group is added at 5001 without resetting existing states. Texture LR is 0.0025 through 25000 and 0.00025 at 25001–30000. All established parameter LRs/constraints/losses are preserved. The common prefix reuses the fresh baseline’s original fine-2k initializer and then trains G3 to 5k, retaining the historical warmup protocol. No extra 30k texture refinement was run.',
        '', 'Protocol deviations/limitations: the actual baseline has no frame affine; none was invented. Its R6 uses original fixed appearance K4 while geometry uses repaired surface K4; both were preserved exactly. Gaussian count is fixed at 364500 because baseline append/split/prune start at 500000; active topology mutation is rejected for texture rather than implementing unneeded inheritance. Nodes stay at 2000. Original graph scatter-add and rasterizer atomic backward are nondeterministic; this prevents bit-exact trajectories across separate branches.',
        '', 'The objective is the unchanged original RGB L1 plus its existing RGB TV (0.001), intrinsic color smoothness (0.0001), learned depth-prior term (0.25), and KNN99 geometry term (0.01 with the existing constraint schedule). No perceptual or additional texture loss was added. Canonical scale/opacity/rotation remain trainable at baseline LRs 0.005/0.05/0.001, and R6 remains in the scheduled deformation Adam group.',
        '', 'The existing convention skips the final Adam update at fine iteration 2000 in warmup and at 30000. This convention is preserved for both models: the common 5000 state is saved after its optimizer update; each continuation executes 25000 sampled fine iterations and 24999 Adam updates. Texture therefore has 24999 Adam steps. The native entrypoint also runs its historical 100-coarse-iteration initializer before restoring the fine source; that transient state is discarded by full model/Adam/RNG restoration. Source-initial checks take no fine optimization step after restoration.',
        '', '## Validation and integrity', '',
        f"All preflight CUDA checks passed: continuous mean/invariance, zero-texture RGB/depth/alpha, reference and finite-difference UV/RGB gradients, dynamic rigid transport, delayed Adam group creation, real R6/geometry gradients, degenerate-ray fallback and native checkpoint roundtrip. Overall final integrity: **{integrity['pass']}**. Camera sequence exact: {camera_check}; source artifacts unchanged: {integrity['source_artifacts_unchanged']}.",
        '', 'Zero-texture identity and reload rasterizer isolation use shared deformed inputs to separate renderer behavior from upstream atomic graph variation. Separate-process initial and reload image errors are recorded numerically; all serialized parameters are checked exactly. Full native textures, axes and full Adam/RNG/camera checkpoint are saved in B.',
        '', '## Timing and texture diagnostics', '',
        '| Model | Native FPS ↑ | R6 color ms ↓ | Train CUDA update ms median | Branch wall min | Train peak MiB | Appearance params |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name in ('A','B'):
        s=summaries[name];lines.append(f"| {name} | {s['performance']['native_fps']:.2f} | {s['performance']['color_forward_ms']:.3f} | {learning[name]['update_ms_median']:.3f} | {s['training']['total_training_duration_s']/60:.2f} | {s['training']['peak_vram_mb']:.1f} | {s['appearance_parameters']} |")
    lines+=['', 'Native inference is measured sequentially on GPU 0 after both training jobs finish, using all 36 cameras and three repeats. Color-forward measures R6 only; texture packing and per-pixel cost are included in native FPS. Training uses two identical RTX A4000 GPUs concurrently, so wall-time differences include CPU contention and individual device load. Reported branch wall time excludes the reused fine-2k source and shared 3k prefix; common prefix duration is recorded separately. Training CUDA update times exclude optimizer and CPU diagnostics.',
        '',f"B texture-on render time overhead relative to the same B checkpoint with texture off: {result['texture_render_overhead_percent']:+.2f}%. B final texel RMS: {b['texture']['centered_rms']:.6f}; continuous weighted RMS: {b['texture']['continuous_weighted_rms']:.6f}; weighted mean max error: {b['texture']['weighted_mean_max_error']:.3g}; nontrivial fraction: {b['texture']['nontrivial_fraction']:.4f}. See `gradient_optimizer_audit.json`, `learning_curves/texture_statistics.jsonl`, and `evaluation/B/summary.json` for distributions and UV statistics.",
        '', '## Geometry and temporal interpretation', '',
        f"Normalized depth-prior MSE change B−A: {delta['DEPTH_MSE']:+.9f}; alpha mean change: {delta['alpha_mean']:+.6f}. Final A/B normalized-depth mean absolute difference: {geometry['normalized_depth_AB_mean_abs']:.6f}; alpha difference: {geometry['alpha_AB_mean_abs']:.6f}.",
        '', f"GACV A/B: {a['GACV']:.7f}/{b['GACV']:.7f}; consecutive temporal residual L1 A/B: {a['temporal_residual_l1']:.7f}/{b['temporal_residual_l1']:.7f}. GACV measures intrinsic R6 mean RGB on shared-visible Gaussians in pairs 143–145/263–265. It does not measure texture swimming. The dense-frame residual metric measures unwarped prediction-error changes relative to GT; it is not optical-flow compensated reflectance flicker.",
        '', 'Scale/opacity distributions, fixed node-edge stretch, graph residual/motion statistics, and conditioned local 3D fold proxies are reported. Rank-deficient surface neighborhoods are excluded from the fold proxy; no calibrated geometric reconstruction claim is made.',
        '', '## Visual artifacts and training ROIs', '',
        'Videos: `visuals/heldout_comparison.mp4` (36 sampled test views at 6 fps) and `visuals/consecutive_texture_stability.mp4` (all 288 consecutive original camera frames at 30 fps). Both show GT/A/B/B-texture-off. Frames 144/264 are training diagnostics, not held-out evidence.',
        '', 'Historical crop coordinates were absent from the supplied local files. The selected crops and their source are recorded in `roi_coordinates.json`; side-by-side images are in `visuals/frame_*_texture_left.png` and `visuals/frame_*_highlight_right.png`. ROI metrics include bias-removed RMSE and a local highpass residual diagnostic to distinguish fine-detail changes from brightness shifts. These are evaluation diagnostics, not training losses.',
        '', 'Inspect the held-out metrics, depth and temporal tradeoffs together before deciding whether the texture representation is useful. Zero mean removes a same-Gaussian constant texture offset; it does not prove strict frequency separation or eliminate overlap-related color cancellation. No model promotion or follow-up variants were performed.',
        '', 'Reproduction: source `/home/robot/colongraph/activate_colongraph.sh`, then use `tools/run_texture_experiment.py`; the completed outputs are preserved against overwrite. See `protocol.json`, `environment.txt`, source manifest, preflight validation and integrity reports for exact provenance.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    if not integrity['pass']:raise RuntimeError('Final experiment integrity failed')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    parser.add_argument('--branch',choices=['A','B','B_texture_off']);parser.add_argument('--summarize-only',action='store_true')
    args=parser.parse_args();torch.cuda.set_device(0);torch.set_num_threads(8)
    if args.branch:return evaluate_branch(args.root,args.branch)
    summaries={}
    for name in ('A','B','B_texture_off'):
        summaries[name]=json.loads((args.root/'evaluation'/name/'summary.json').read_text()) if args.summarize_only else evaluate_branch(args.root,name)
    rois=visual_artifacts(args.root);learning,optimizer=learning_curves(args.root)
    result=report(args.root,summaries,rois,learning,optimizer)
    print(json.dumps(dict(state='COMPLETED',PSNR_A=result['models']['A']['metrics']['PSNR'],
        PSNR_B=result['models']['B']['metrics']['PSNR'],integrity=result['integrity']['pass']),indent=2),flush=True)


if __name__=='__main__':main()
