"""Paired geometry metrics; original rasterizer and quantized RGB/depth protocol."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import copy
import gc
import json
import time
import numpy as np
import torch
from PIL import Image
from gaussian_renderer import render
from tools.experiment_common import load_scene
from tools.evaluate_texture_experiment import quantize, metrics, write_json, panel, file_hash
from tools.audit_geometry_blending import OUT, BASE, VARIANTS, stats, correlation
from utils.geometry_blending import install_cache, validate_cache
from utils.exact_split_branchpoint import digest, model_digest, restore_knn_reference
from utils.adaptive_deformation_graph import AdaptiveNodeManager


def gradient(x):
    return torch.stack((x[1:,1:]-x[1:,:-1],x[1:,1:]-x[:-1,1:]),0)


def masks(camera):
    gt = camera.original_depth.cuda().float().squeeze()
    valid = torch.isfinite(gt) & (gt>0)
    mask = getattr(camera,'mask',None)
    if mask is not None:
        valid &= torch.as_tensor(mask,device=gt.device).bool().reshape_as(gt)
    normalized = (gt/gt[valid].max().clamp_min(1e-6)).clamp(0,1)
    gv = valid[1:,1:] & valid[1:,:-1] & valid[:-1,1:]
    grad = gradient(normalized); magnitude = grad.square().sum(0).sqrt()
    p80 = torch.quantile(magnitude[gv],.8)
    fold = gv & (magnitude>=p80)
    return gt,valid,normalized,gv,grad,fold,magnitude


def depth_metrics(pred, camera, baseline_max):
    pred = pred.squeeze().float()
    gt,valid,ngt,gv,ggrad,fold,mag = masks(camera)
    if not torch.isfinite(pred).all():
        raise FloatingPointError('Nonfinite rendered geometry')
    # Direct call to existing float-valid diagnostic, then verify implementation.
    official = AdaptiveNodeManager._depth_metrics(pred,gt,getattr(camera,'mask',None))
    norm = (pred/pred[valid].max().clamp_min(1e-6)).clamp(0,1)
    err = (norm-ngt).square()
    fvalid = float(err[valid].mean())
    if abs(fvalid-official['normalized_mse'])>1e-10:
        raise RuntimeError('Float-valid diagnostic mismatch')
    # Existing PNG protocol uses whole-image maximum, followed by uint8 rounding.
    prior = camera.depth.cuda().float().squeeze()
    pngpred = quantize(pred/pred.max().clamp_min(1e-8))
    pnggt = quantize(prior/prior.max().clamp_min(1e-8))
    shared = (pred/baseline_max.clamp_min(1e-6)).clamp(0,1)
    shared_err = (shared-ngt).square()
    graderr = (gradient(norm)-ggrad).square().sum(0)
    row = dict(DEPTH_PNG_MSE=float(np.mean(((pngpred.astype(np.float32)-pnggt.astype(np.float32))/255)**2)),
        DEPTH_FLOAT_VALID_MSE=fvalid,FOLD_GT_P80_DIAGNOSTIC_MSE=float(err[1:,1:][fold].mean()),
        DEPTH_SHARED_A_MAX_MSE=float(shared_err[valid].mean()),
        FOLD_SHARED_A_MAX_DIAGNOSTIC_MSE=float(shared_err[1:,1:][fold].mean()),
        DEPTH_GRADIENT_ERROR=float(graderr[gv].mean()), FOLD_DEPTH_GRADIENT_ERROR=float(graderr[fold].mean()),
        valid_fraction=float(valid.float().mean()), alpha_mean=None,
        shared_max_clipped_fraction=float((pred[valid]>baseline_max).float().mean()),
        prior_original_depth_exact=bool(torch.equal(prior,gt)))
    return row, dict(depth=norm,shared=shared,error=err.sqrt(),fold=fold,png=pngpred,
        gradient_error=graderr.sqrt(),gt=ngt,gt_magnitude=mag)


def rgb_image(x):
    return Image.fromarray(quantize(x).transpose(1,2,0))


def heatmap(x,maximum=None,signed=False):
    from matplotlib import colormaps
    a=torch.as_tensor(x).detach().cpu().numpy()
    maximum = maximum if maximum is not None else max(float(np.quantile(np.abs(a),.99)),1e-8)
    value=np.clip(.5+.5*a/maximum,0,1) if signed else np.clip(a/maximum,0,1)
    return Image.fromarray((colormaps['coolwarm' if signed else 'magma'](value)[...,:3]*255+.5).astype(np.uint8))


def benchmark(model,scene,pipe,bg):
    for camera in scene.getTestCameras()[:3]:render(camera,model,pipe,bg)
    torch.cuda.synchronize(); times=[]; wall=time.monotonic()
    for _ in range(3):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record()
        for c in scene.getTestCameras():render(c,model,pipe,bg)
        b.record();b.synchronize();times.append(a.elapsed_time(b)/len(scene.getTestCameras()))
    ms=float(np.median(times))
    return dict(native_gpu_ms=ms,FPS=1000/ms,wall_ms=(time.monotonic()-wall)*1000/(3*len(scene.getTestCameras())),
        repeats_gpu_ms=times,peak_memory_mb=torch.cuda.max_memory_allocated()/2**20)


def summarize(directory, rows, timings):
    keys=[key for key in next(iter(next(iter(rows.values())).values())) if key not in ('frame','time','prior_original_depth_exact')]
    means={name:{key:float(np.mean([v[key] for v in values.values()])) for key in keys} for name,values in rows.items()}
    a=rows['A_idw']; paired={}
    for name,values in rows.items():
        if name=='A_idw':continue
        pairs={key:stats([values[f][key]-a[f][key] for f in a]) for key in keys}
        for key in keys:
            pairs[key]['negative_delta_views']=sum(values[f][key]<a[f][key] for f in a)
            pairs[key]['positive_delta_views']=sum(values[f][key]>a[f][key] for f in a)
        paired[name]=pairs
    write_json(directory/'summary.json',dict(means=means,paired_delta_variant_minus_A=paired,timing=timings,
        fold_note='New common GT P80 gradient-region depth-error diagnostic; historical official implementation absent. Not negative Jacobian fraction.'))
    write_json(directory/'per_view.json',rows)
    import csv
    with (directory/'per_view.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=['variant','frame','time','prior_original_depth_exact']+keys);writer.writeheader()
        for name,values in rows.items():
            for f,row in values.items():writer.writerow(dict(variant=name,**row))
    print('EVALUATION_SUMMARY',means,flush=True)
    return means


@torch.no_grad()
def correctness():
    model,scene,pipe,bg,args=load_scene(BASE,20000);model.profile_deformation=False
    camera=scene.getTestCameras()[17]
    indices=model.deformation_graph.gaussian_node_indices.clone()
    baseline=model.deformation_graph.gaussian_node_weights.clone()
    app=digest(model.appearance_graph.state_dict());g3=digest(model.graph_motion_residual.state_dict())
    original=model._query_node_motion
    motion=original(float(camera.time))
    model._query_node_motion=lambda *args,**kwargs:motion
    source_geometry=model.deform_geometry_from_nodes(float(camera.time));source_render=render(camera,model,pipe,bg)
    results={}
    directory=OUT/'audit/roundtrip';directory.mkdir(exist_ok=True)
    for name,(mode,beta) in VARIANTS.items():
        model.deformation_graph.gaussian_node_weights.copy_(baseline)
        model.args.geometry_blend_mode=mode;model.args.geometry_blend_beta=beta
        install_cache(model,OUT/'caches'/(name+'.pth'))
        xyz,scale,rot=model.deform_geometry_from_nodes(float(camera.time))
        w=model.deformation_graph.gaussian_node_weights
        dx,dr,ds=model.deformation_graph.blend(*motion)
        checked=dict(indices_exact=torch.equal(indices,model.deformation_graph.gaussian_node_indices),
            appearance_exact=app==digest(model.appearance_graph.state_dict()),
            g3_exact=g3==digest(model.graph_motion_residual.state_dict()),
            finite_geometry=all(torch.isfinite(v).all().item() for v in (xyz,scale,rot)),
            dtype_device_shape=w.shape==baseline.shape and w.dtype==baseline.dtype and w.device==baseline.device,
            xyz_blends_geometry_weights=torch.equal(xyz,model._xyz+dx),
            logscale_blends_geometry_weights=torch.equal(scale,model._scaling+ds.clamp(-.1*model._scaling.abs(),.1*model._scaling.abs())),
            rotation_blends_geometry_weights=torch.equal(rot,model._rotation+dr.clamp(-.1*model._rotation.abs(),.1*model._rotation.abs())),
            quaternion_unit_length=bool(torch.allclose(model.rotation_activation(rot).norm(dim=-1),torch.ones(len(rot),device='cuda'),atol=2e-6,rtol=0)))
        if name=='A_idw':
            repeat=render(camera,model,pipe,bg)
            checked['baseline_weights_exact']=torch.equal(w,baseline)
            checked['baseline_geometry_exact']=all(torch.equal(a,b) for a,b in zip((xyz,scale,rot),source_geometry))
            checked['baseline_RGB_depth_alpha_exact_shared_motion']=all(torch.equal(source_render[k],repeat[k]) for k in ('render','depth','alpha'))
        # Serialize actual native checkpoint modules, then strict reload all modules.
        temp=directory/name;temp.mkdir(exist_ok=True);model.save_deformation(str(temp))
        before_modules=digest({k:getattr(model,k).state_dict() for k in ('deformation_graph','appearance_graph','node_mlp','appearance_model','graph_motion_residual','_deformation')})
        model.load_model(str(temp))
        checked['cached_weights_reload_exact']=torch.equal(w,model.deformation_graph.gaussian_node_weights)
        checked['native_modules_strict_reload_exact']=before_modules==digest({k:getattr(model,k).state_dict() for k in ('deformation_graph','appearance_graph','node_mlp','appearance_model','graph_motion_residual','_deformation')})
        validate_cache(model,model.geometry_blending_cache,require_weights=True)
        # All load_state_dict calls default strict or explicitly strict=True.
        checked['strict_parameter_loads_succeeded']=True
        results[name]=checked
    model._query_node_motion=original
    write_json(OUT/'audit/correctness.json',dict(pass_=all(all(v.values()) for v in results.values()),checks=results))
    if not all(all(v.values()) for v in results.values()):raise RuntimeError('Geometry correctness checks failed: '+str(results))
    print('REAL_CHECKPOINT_CORRECTNESS_PASSED',flush=True)


def influence_overlays(camera,result,model,arrays):
    """Canonical Gaussian diagnostics projected to the current deformed centers.

    Point projection overlay (no opacity weighting), explicitly a location proxy.
    """
    from PIL import ImageDraw
    xyz=result['transformed_points'].detach()
    p=torch.cat((xyz,torch.ones_like(xyz[:,:1])),1)@camera.full_proj_transform.cuda()
    screen=p[:,:2]/p[:,3:]
    screen=(screen+1)*screen.new_tensor([camera.image_width,camera.image_height])/2-.5
    keep=(p[:,3]>0)&(screen[:,0]>=0)&(screen[:,0]<camera.image_width)&(screen[:,1]>=0)&(screen[:,1]<camera.image_height)
    labels={
        'distant_rho_gt_1_5':np.sum(arrays['A_idw_weights']*(arrays['cache_distances']/arrays['node_spacing'][arrays['indices']]>1.5),axis=1)>.01,
        'fallback_beta2':arrays['C_beta2_fallback'], 'fallback_beta2_5':arrays['C_beta2_5_fallback']}
    overlays={}
    for label,mask in labels.items():
        chosen=keep & torch.as_tensor(mask,device='cuda')
        points=screen[chosen].detach().cpu().numpy()
        im=rgb_image(result['render'].clamp(0,1));draw=ImageDraw.Draw(im)
        if len(points)>15000:points=points[::max(1,len(points)//15000)]
        for x,y in points:draw.ellipse((x-1,y-1,x+1,y+1),fill=(255,30,30))
        draw.text((8,8),f'{label}: {int(chosen.sum())} projected points; location proxy',fill=(255,255,0))
        overlays[label]=im
    return overlays


@torch.no_grad()
def render_only():
    directory=OUT/'render_only';directory.mkdir(exist_ok=True)
    model,scene,pipe,bg,args=load_scene(BASE,20000);model.profile_deformation=False
    source_digest=model_digest(model)
    base=model.deformation_graph.gaussian_node_weights.clone();original=model._query_node_motion
    camera_scores={c.image_name:float(masks(c)[-1][masks(c)[3]].mean()) for c in scene.getTestCameras()}
    selected=sorted(camera_scores,key=lambda f:(-camera_scores[f],int(f)))[:3]
    write_json(directory/'visual_selection.json',dict(frames=selected,GT_mean_gradient=camera_scores,selection='GT-only before any variant render'))
    rows={name:{} for name in VARIANTS};timings={};arrays=np.load(OUT/'audit/distributions.npz')
    normalized={name:[] for name in VARIANTS};alpha={name:[] for name in VARIANTS}
    for name in VARIANTS:
        for child in ('rgb','depth','alpha','error'):(directory/name/child).mkdir(parents=True,exist_ok=True)
    (directory/'visuals').mkdir(exist_ok=True)
    appearance_exact=True; motion_match=True
    for c in sorted(scene.getTestCameras(),key=lambda c:int(c.image_name)):
        model._query_node_motion=original;shared=original(float(c.time));model._query_node_motion=lambda *a,**k:shared
        # Native appearance evaluated once; only geometry blends are different.
        n=len(model._xyz);times=torch.full((n,1),float(c.time),device='cuda')
        appearance=model.appearance_model(model._xyz,times,model.features,model._deformation_table,
            model.appearance_graph.gaussian_node_indices,model.appearance_graph.gaussian_node_weights)
        original_app=model.appearance_model.forward
        model.appearance_model.forward=lambda *a,**k:appearance
        visual={};amax=None;basenorm=None;basecolor=None
        for name,(mode,beta) in VARIANTS.items():
            model.deformation_graph.gaussian_node_weights.copy_(base)
            model.args.geometry_blend_mode=mode;model.args.geometry_blend_beta=beta
            install_cache(model,OUT/'caches'/(name+'.pth'))
            result=render(c,model,pipe,bg)
            if name=='A_idw':
                _,valid,*_=masks(c);amax=result['depth'].squeeze()[valid].max();basecolor=result.get('transformed_color')
            row,diagnostic=depth_metrics(result['depth'],c,amax)
            rgb=result['render'].clamp(0,1);gt=c.original_image[:3].cuda().float()
            # Match historical PNG RGB evaluation, including target quantization.
            qrgb=torch.as_tensor(quantize(rgb),device='cuda').float()/255
            qgt=torch.as_tensor(quantize(gt),device='cuda').float()/255
            row.update(metrics(qrgb,qgt),frame=c.image_name,time=float(c.time),alpha_mean=float(result['alpha'].mean()))
            rows[name][c.image_name]=row
            if basecolor is not None:appearance_exact &= torch.equal(basecolor,result['transformed_color'])
            if name=='A_idw':basenorm=diagnostic['shared'].clone()
            normalized[name].append(diagnostic['depth'].cpu().numpy());alpha[name].append(result['alpha'].squeeze().cpu().numpy())
            rgb_image(rgb).save(directory/name/'rgb'/(c.image_name+'.png'))
            Image.fromarray(diagnostic['png']).save(directory/name/'depth'/(c.image_name+'.png'))
            Image.fromarray(quantize(result['alpha'].squeeze())).save(directory/name/'alpha'/(c.image_name+'.png'))
            heatmap(diagnostic['error'],maximum=.25).save(directory/name/'error'/(c.image_name+'.png'))
            if c.image_name in selected:
                visual[name]=dict(rgb=rgb_image(rgb),depth=heatmap(diagnostic['shared'],maximum=1.),
                    error=heatmap(diagnostic['error'],maximum=.25),delta=heatmap(diagnostic['shared']-basenorm,maximum=.05,signed=True))
                if name=='A_idw':
                    for label,im in influence_overlays(c,result,model,arrays).items():im.save(directory/'visuals'/f'{c.image_name}_{label}.png')
                    gtimage=rgb_image(gt);gtdepth=heatmap(diagnostic['gt'],maximum=1.)
                    fold=Image.fromarray((diagnostic['fold'].cpu().numpy()*255).astype(np.uint8))
                    fold.save(directory/'visuals'/f'{c.image_name}_common_gt_p80_mask.png')
        model.appearance_model.forward=original_app
        if c.image_name in selected:
            labels=list(VARIANTS)
            for kind in ('rgb','depth','error','delta'):
                images=[visual[name][kind] for name in labels]
                prefix=[gtimage] if kind=='rgb' else [gtdepth] if kind=='depth' else []
                panel(prefix+images,(['GT'] if prefix else [])+labels,f'Frame {c.image_name} / {kind}; selected by GT gradient').save(directory/'visuals'/f'{c.image_name}_{kind}.png')
        print('PAIRED_RENDER',c.image_name,flush=True)
    model._query_node_motion=original
    if not appearance_exact:raise RuntimeError('Render-only appearance changed')
    # Timing uses native path, no shared motion or appearance substitution.
    from utils.image_utils import lpips_alex,lpips_vgg
    lpips_alex.cpu();lpips_vgg.cpu();gc.collect();torch.cuda.empty_cache()
    for name,(mode,beta) in VARIANTS.items():
        model.deformation_graph.gaussian_node_weights.copy_(base)
        model.args.geometry_blend_mode=mode;model.args.geometry_blend_beta=beta
        install_cache(model,OUT/'caches'/(name+'.pth'))
        torch.cuda.reset_peak_memory_stats();timings[name]=benchmark(model,scene,pipe,bg)
        np.savez_compressed(directory/name/'depth_alpha.npz',names=np.array(sorted(rows[name],key=int)),
            depth=np.stack(normalized[name]),alpha=np.stack(alpha[name]))
    model.deformation_graph.gaussian_node_weights.copy_(base)
    if model_digest(model)!=source_digest:raise RuntimeError('Paired rendering changed source model')
    write_json(directory/'integrity.json',dict(pass_=True,model_before_after_exact=True,appearance_shared_exact=True,
        node_motion_shared_per_view=True,model_sha256=source_digest,geometry_only_weights_changed=True))
    summarize(directory,rows,timings)


@torch.no_grad()
def short_evaluation():
    directory=OUT/'short_train/evaluation';directory.mkdir(parents=True,exist_ok=True)
    rows={};timings={};names=('A_idw','C_selected')
    basemax={};basepred={}
    selected=json.loads((OUT/'render_only/visual_selection.json').read_text())['frames']
    (directory/'visuals').mkdir(exist_ok=True)
    visual={}
    for name in names:
        path=OUT/'short_train'/name
        model,scene,pipe,bg,args=load_scene(path,10000);model.profile_deformation=False
        restore_knn_reference(model,args.gaussian_knn_reference)
        state=torch.load(path/'fine_state_10000.pth',map_location='cpu',weights_only=False)
        exact=model_digest(model)==state['model_sha256']
        if not exact:raise RuntimeError('Short checkpoint native/full state mismatch: '+name)
        cache=model.geometry_blending_cache;validate_cache(model,cache,require_weights=True)
        checks=dict(native_full_model_exact=exact,cache_weights_exact=True,
            gaussian_count=len(model._xyz)==364500,node_count=len(model.deformation_graph.node_xyz)==2000,
            appearance_same_reference=cache['invariants']['appearance']==digest(model.appearance_graph.state_dict()),
            G3_graph_same_reference=cache['invariants']['g3_graph']==digest({k:v for k,v in model.graph_motion_residual.state_dict().items() if not k.startswith('decoder.')}),
            final_iteration=state['iteration']==10000)
        if not all(checks.values()):raise RuntimeError('Short integrity checks failed')
        write_json(path/'checkpoint_integrity.json',dict(pass_=True,checks=checks,full_state_sha256=file_hash(path/'fine_state_10000.pth')))
        del state
        (directory/name/'rgb').mkdir(parents=True,exist_ok=True);(directory/name/'depth').mkdir(exist_ok=True)
        rows[name]={};depths=[];alphas=[]
        for camera in sorted(scene.getTestCameras(),key=lambda c:int(c.image_name)):
            result=render(camera,model,pipe,bg);frame=camera.image_name
            if name=='A_idw':
                _,valid,*_=masks(camera);basemax[frame]=result['depth'].squeeze()[valid].max().clone()
                basepred[frame]=result['depth'].squeeze().cpu().numpy()
            row,diag=depth_metrics(result['depth'],camera,basemax[frame])
            rgb=result['render'].clamp(0,1);gt=camera.original_image[:3].cuda().float()
            row.update(metrics(torch.as_tensor(quantize(rgb),device='cuda').float()/255,
                torch.as_tensor(quantize(gt),device='cuda').float()/255),frame=frame,time=float(camera.time),alpha_mean=float(result['alpha'].mean()))
            rows[name][frame]=row;depths.append(diag['depth'].cpu().numpy());alphas.append(result['alpha'].squeeze().cpu().numpy())
            rgb_image(rgb).save(directory/name/'rgb'/(frame+'.png'));Image.fromarray(diag['png']).save(directory/name/'depth'/(frame+'.png'))
            if frame in selected:
                if name=='A_idw':
                    visual[frame]=dict(GT=rgb_image(gt),GT_depth=heatmap(diag['gt'],maximum=1.),variants={})
                visual[frame]['variants'][name]=dict(rgb=rgb_image(rgb),depth=heatmap(diag['shared'],maximum=1.),
                    error=heatmap(diag['error'],maximum=.25),
                    delta=heatmap(diag['shared']-torch.as_tensor(basepred[frame],device='cuda')/basemax[frame],maximum=.05,signed=True))
        np.savez_compressed(directory/name/'depth_alpha.npz',names=np.array(sorted(rows[name],key=int)),depth=np.stack(depths),alpha=np.stack(alphas))
        model.closest_point_indices=model.closest_point_indices.cpu()
        from utils.image_utils import lpips_alex,lpips_vgg
        lpips_alex.cpu();lpips_vgg.cpu();gc.collect();torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats();timings[name]=benchmark(model,scene,pipe,bg)
        del model,scene;gc.collect();torch.cuda.empty_cache()
        print('SHORT_EVALUATED',name,flush=True)
    summarize(directory,rows,timings)
    for frame,v in visual.items():
        for kind in ('rgb','depth','error','delta'):
            prefix=[v['GT']] if kind=='rgb' else [v['GT_depth']] if kind=='depth' else []
            panel(prefix+[v['variants'][name][kind] for name in names],(['GT'] if prefix else [])+list(names),
                f'Short continuation 10k / frame {frame} / {kind}; same GT-selected views').save(directory/'visuals'/f'{frame}_{kind}.png')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['correctness','render','short'])
    action=parser.parse_args().action
    {'correctness':correctness,'render':render_only,'short':short_evaluation}[action]()
