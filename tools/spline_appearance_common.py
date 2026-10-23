"""Isolated spline experiment I/O and the unchanged native metric convention."""
import argparse,gc,json,random,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image
from gaussian_renderer import render
from tools.experiment_common import load_scene
from tools.run_j_baseline30k import write,sha
from tools.evaluate_texture_experiment import metrics,quantize,panel,all_cameras
from tools.evaluate_geometry_blending import masks,depth_metrics
from utils.exact_split_branchpoint import model_digest,restore_knn_reference,digest
from utils.independent_temporal_rgb import IndependentTemporalRGB,active_basis,freeze_source,assert_source_frozen,install,knots
from utils.image_utils import psnr
SCENE='c1_sigmoid2_t4_v4'
SOURCES=dict(K4=ROOT.parent/'ColonGraph/output/g3_surf_fresh'/SCENE,
    J=ROOT/'output/experiments/j_baseline30k/fresh_20261011'/SCENE/'runs/J_GGAS_V2')
STEPS=[5000,10000,20000,30000]

def source_files(source):
    from tools.audit_sparse_run import read_config
    cfg=read_config(source/'cfg_args')
    paths=[source/'cfg_args',*sorted((source/'point_cloud/iteration_30000').iterdir())]
    for key in ['gaussian_knn_reference','graph_forward_reference']:
        paths.append(Path(cfg[key]))
    for name in ['fine_state_30000.pth','training_summary.json','integrity.json','graph_forward_checkpoint_audit.json']:
        p=source/name
        if p.is_file():paths.append(p)
    return {str(p.resolve()):sha(p) for p in paths if p.is_file()}

def load(branch):
    model,scene,pipe,bg,cfg=load_scene(SOURCES[branch],30000)
    restore_knn_reference(model,cfg.gaussian_knn_reference)
    assert len(model._xyz)==364500 and len(model.deformation_graph.node_xyz)==2000
    assert cfg.graph_forward_mode=='g3' and cfg.color_model=='node_residual_mlp' and not cfg.use_do
    assert not pipe.compute_cov3D_python and not model.texture_active
    assert (model.skinning is None) if branch=='K4' else (model.skinning is not None and cfg.ggas_variant=='J_NO_MOTION_SELECTION')
    assert len(scene.getTrainCameras())==252 and len(scene.getTestCameras())==36
    if branch=='J':
        full=torch.load(SOURCES[branch]/'fine_state_30000.pth',map_location='cpu',weights_only=False)
        assert model_digest(model)==full['model_sha256'];del full
        assert json.loads((SOURCES[branch]/'integrity.json').read_text())['pass_']
    else:assert json.loads((SOURCES[branch]/'graph_forward_checkpoint_audit.json').read_text())['pass']
    freeze_source(model);return model,scene,pipe,bg,cfg

def bases(scene):
    return {str(c.image_name):active_basis(c.time,device='cuda') for c in all_cameras(scene)}

def checkpoint(dest,model,spline,optimizer,rng,iteration,camera_order,source_hash):
    assert_source_frozen(model)
    state=dict(format_version=1,iteration=iteration,phase='after_update' if iteration<30000 else 'final_label_without_Adam',
        optimizer_steps=iteration if iteration<30000 else 29999,controls=spline.controls.detach().cpu(),optimizer=optimizer.state_dict(),
        geometry_source=str(SOURCES[dest.name]),source_manifest_sha256=source_hash,frozen_model_sha256=model_digest(model),
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),
        python_rng=random.getstate(),camera_sampling_rng=rng.getstate(),train_camera_order=camera_order,
        basis=dict(degree=3,controls=32,knots=knots(),time_domain=[0,1],scene_time_convention='original idx/288, no split renormalization'))
    path=dest/'checkpoints'/f'iteration_{iteration}.pth';tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)
    return path

@torch.no_grad()
def evaluate(dest,label,model,scene,pipe,bg,spline=None,save_frames=False):
    output=dest/'evaluation'/str(label);output.mkdir(parents=True,exist_ok=False)
    frames=output/'frames'
    if save_frames:frames.mkdir()
    test={str(c.image_name) for c in scene.getTestCameras()};rows={};previous=None;temporal=[];depthsum=[]
    expected=model_digest(model);cache=bases(scene)
    for c in all_cameras(scene):
        name=str(c.image_name)
        if spline is not None:model.spline_basis=cache[name]
        p=render(c,model,pipe,bg);rgb=p['render'].clamp(0,1);gt=c.original_image[:3].cuda().float()
        byte=quantize(rgb);gtbyte=quantize(gt);prediction=torch.as_tensor(byte,device='cuda').float()/255;target=torch.as_tensor(gtbyte,device='cuda').float()/255
        row=metrics(prediction,target,perceptual=name in test)
        row.update(PSNR_float=float(psnr(rgb[None],gt[None]).mean()),frame=name,time=float(c.time),split='TEST' if name in test else 'TRAIN',
            gaussian_color_below0=float((p['transformed_color']<0).float().mean()),gaussian_color_above1=float((p['transformed_color']>1).float().mean()),
            rendered_RGB_below0=float((p['render']<0).float().mean()),rendered_RGB_above1=float((p['render']>1).float().mean()))
        if name in test:
            _,valid,*_=masks(c);d,_=depth_metrics(p['depth'],c,p['depth'].squeeze()[valid].max())
            row['DEPTH_FLOAT_VALID_MSE']=d['DEPTH_FLOAT_VALID_MSE'];row['alpha_mean']=float(p['alpha'].mean());depthsum.append(d['DEPTH_FLOAT_VALID_MSE'])
        rows[name]=row
        if save_frames:rgb_image=Image.fromarray(byte.transpose(1,2,0));rgb_image.save(frames/(name+'.png'))
        if str(label)=='30000' and name in ['1','145','281']:
            torch.save({k:p[k].detach().cpu() for k in ['render','depth','alpha']},output/('float_replay_'+name+'.pth'))
        if label=='R6_source':
            gtpath=dest/'evaluation/GT';gtpath.mkdir(exist_ok=True);Image.fromarray(gtbyte.transpose(1,2,0)).save(gtpath/(name+'.png'))
        residual=rgb-gt
        if previous is not None:temporal.append(dict(frame=name,residual_L1=float((residual-previous).abs().mean())))
        previous=residual
    train=[r for r in rows.values() if r['split']=='TRAIN'];held=[r for r in rows.values() if r['split']=='TEST']
    keys=['PSNR','PSNR_float','SSIM','LPIPS_A','LPIPS_V','DEPTH_FLOAT_VALID_MSE','alpha_mean','gaussian_color_below0','gaussian_color_above1','rendered_RGB_below0','rendered_RGB_above1']
    means={k:float(np.mean([r[k] for r in held])) for k in keys}
    trainmean={k:float(np.mean([r[k] for r in train])) for k in ['PSNR','PSNR_float','SSIM']}
    assert expected==model_digest(model)
    result=dict(label=str(label),test_views=36,train_views=252,test=means,train=trainmean,train_test_PSNR_gap=trainmean['PSNR']-means['PSNR'],
        temporal_residual_L1=float(np.mean([r['residual_L1'] for r in temporal])),
        temporal_note='Consecutive unwarped prediction-error change relative to GT, not motion-compensated reflectance flicker',
        metrics_provenance='recomputed on current source and this experiment, no imported historical score')
    write(output/'summary.json',result);write(output/'per_view.json',rows);write(output/'temporal.json',temporal)
    from utils.image_utils import lpips_alex,lpips_vgg
    lpips_alex.cpu();lpips_vgg.cpu();gc.collect();torch.cuda.empty_cache()
    print('EVALUATED',dest.name,label,result['test'],flush=True);return result

@torch.no_grad()
def timing(model,scene,pipe,bg,spline):
    from tools.evaluate_geometry_blending import benchmark
    cache=bases(scene);original=model.deform
    def timed_deform(*args,**kwargs):
        # Basis selection is outside the timed color computation; geometry is native.
        time_tensor=args[5]; t=float(time_tensor.reshape(-1)[0]);ids,w=active_basis(t,device='cuda');model.spline_basis=(ids,w)
        return original(*args,**kwargs)
    # Benchmark loop sets basis by timestamp without CPU synchronization via wrapper camera.
    times=[];color_times=[]
    cams=scene.getTestCameras()
    for c in cams[:3]:model.spline_basis=cache[str(c.image_name)];render(c,model,pipe,bg)
    for repeat in range(3):
        total=0.;colors=0.
        for c in cams:
            model.spline_basis=cache[str(c.image_name)]
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();render(c,model,pipe,bg);b.record();b.synchronize();total+=a.elapsed_time(b)
            a.record();spline(basis=model.spline_basis);b.record();b.synchronize();colors+=a.elapsed_time(b)
        times.append(total/len(cams));color_times.append(colors/len(cams))
    ms=float(np.median(times));return dict(FPS=1000/ms,native_gpu_ms=ms,appearance_forward_ms=float(np.median(color_times)),repeats_gpu_ms=times,
        benchmark_note='Both training branches complete before sequential isolated benchmark')

@torch.no_grad()
def temporal_controls(dest,spline):
    controls=spline.controls;flat=controls.flatten();sample=flat[::max(1,len(flat)//500000)]
    controlstats=dict(mean=float(flat.mean()),std=float(flat.std()),min=float(flat.min()),max=float(flat.max()),
        below0=float((flat<0).float().mean()),above1=float((flat>1).float().mean()),
        quantiles_sampled=torch.quantile(sample,sample.new_tensor([0,.01,.5,.99,1])).tolist(),sample_size=len(sample))
    times=np.linspace(0,1,65);lo=torch.full((len(controls),3),torch.inf,device='cuda');hi=-lo;delta=[];prev=None
    ids=torch.linspace(0,len(controls)-1,12,device='cuda').long();trajectories=[]
    for t in times:
        color=spline(float(t));lo=torch.minimum(lo,color);hi=torch.maximum(hi,color);trajectories.append(color[ids].cpu().numpy())
        if prev is not None:delta.append(float((color-prev).abs().mean()))
        prev=color
    amplitude=(hi-lo).mean(1);result=dict(control_statistics=controlstats,mean_RGB_range=float(amplitude.mean()),
        nearly_constant_fraction_range_lt_001=float((amplitude<.001).float().mean()),substantial_variation_fraction_range_gt_005=float((amplitude>.05).float().mean()),
        sampled_dt=1/64,mean_adjacent_RGB_L1=float(np.mean(delta)),mean_temporal_derivative_L1=float(np.mean(delta)*64),
        trajectory_ids=ids.cpu().tolist(),trajectory_selection='12 evenly spaced immutable IDs, independent of held-out targets')
    write(dest/'evaluation/temporal_controls.json',result)
    np.savez_compressed(dest/'visuals/temporal_trajectories.npz',times=times,ids=ids.cpu().numpy(),rgb=np.stack(trajectories))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,3,figsize=(12,10));array=np.stack(trajectories)
    for j,ax in enumerate(axes.flat):
        for ch,color in enumerate(['r','g','b']):ax.plot(times,array[:,j,ch],color=color)
        ax.set_title('Gaussian '+str(int(ids[j])));ax.set_xlabel('Original normalized time');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(dest/'visuals/RGB_trajectories.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));ax.hist(sample.cpu().numpy(),bins=100);ax.set_xlabel('Unclamped RGB control value');ax.set_ylabel('Sample count')
    fig.tight_layout();fig.savefig(dest/'visuals/control_statistics.png',dpi=150);plt.close(fig);return result
