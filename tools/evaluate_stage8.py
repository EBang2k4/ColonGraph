"""Post-training C3VD RGB evaluation at fixed epochs; no depth accuracy metric."""
import gc
import time
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image,ImageDraw
import imageio.v2 as imageio
from tools import run_stage8 as run
from utils.exact_split_branchpoint import digest,model_digest
from gaussian_renderer import render

STEPS=[2000,5000,10000,20000,30000]


def snapshot(study,i):
    return study/'setup/warmup' if i==2000 else study/'common_5000' if i==5000 else study/'J'


def clean_depth(pred,cam):
    from utils.stage8_depth import depth_statistics
    name=getattr(cam,'original_image_name',str(cam.image_name).removeprefix('test__'))
    path=run.DATASETS['transverse']/'test/img'/(name+'.png')
    alpha=torch.as_tensor(np.array(Image.open(path).convert('RGBA'))[...,3],device='cuda')
    return depth_statistics(pred,cam.original_depth.cuda().float().squeeze(),alpha)


def evaluate(root,scene,key):
    from tools.evaluate_texture_experiment import metrics,quantize
    from tools.evaluate_geometry_blending import benchmark
    run.configure(scene)
    study=root/scene/'runs'/key;out=study/'J';dest=out/'evaluation';dest.mkdir(exist_ok=False)
    for folder in ['frames','GT']:(dest/folder).mkdir()
    curves=[];integrity=[]
    for i in STEPS:
        path=snapshot(study,i)
        m,s,p,b,c,st=run.restore(path,i);del st
        before=(model_digest(m),digest(m.optimizer.state_dict()),run.sha(path/f'fine_state_{i}.pth'))
        expected=json.loads((study/'initial_integrity.json').read_text())
        assert digest(m.appearance_graph.state_dict())==expected['appearance_graph_sha256']
        assert len(m.appearance_graph.node_xyz)==2000
        assert m.appearance_model.additive_cap==.25
        assert c.loss_rgb_mode=='charbonnier_psnr_ssim_handoff'
        assert len(m.deformation_graph.node_xyz)==(2400)
        assert c.stage5_profile==key and c.position_lr_final==4e-6
        assert c.node_appearance_dim==(16 if key=='N' else 8)
        assert int(m.appearance_model.micro_iteration)==i
        rows=[];app_rows=[];micro_magnitudes=[]
        with torch.no_grad():
            for cam in s.getTestCameras():
                q=render(cam,m,p,b);rgb=q['render'].clamp(0,1);gt=cam.original_image[:3].cuda().float()
                byte,gtbyte=quantize(rgb),quantize(gt)
                row=metrics(torch.as_tensor(byte,device='cuda').float()/255,
                            torch.as_tensor(gtbyte,device='cuda').float()/255,True)
                dx=(rgb[:,:,1:]-rgb[:,:,:-1])-(gt[:,:,1:]-gt[:,:,:-1])
                dy=(rgb[:,1:,:]-rgb[:,:-1,:])-(gt[:,1:,:]-gt[:,:-1,:])
                row.update(frame=str(cam.image_name),time=float(cam.time),RGB_L1=float((rgb-gt).abs().mean()),RGB_MSE=float((rgb-gt).square().mean()),
                    image_gradient_L1=.5*float(dx.abs().mean()+dy.abs().mean()),
                    alpha_mean=float(q['alpha'].mean()),
                    RGB_out_of_range=float(((q['render']<0)|(q['render']>1)).float().mean()))
                if scene=='transverse':
                    from tools.evaluate_geometry_blending import masks,depth_metrics
                    _,valid,*_=masks(cam)
                    d,_=depth_metrics(q['depth'],cam,q['depth'].squeeze()[valid].max())
                    row['depth_MSE']=d['DEPTH_FLOAT_VALID_MSE']
                    raw_depth=q['depth'].squeeze();gt_depth=cam.original_depth.cuda().float().squeeze()
                    row['raw_depth_MSE']=float((raw_depth[valid]-gt_depth[valid]).square().mean())
                    row.update(clean_depth(q['depth'],cam))
                    row['legacy_depth_MSE']=row['depth_MSE']
                rows.append(row)
                if i==30000:
                    micro_magnitudes.append((q['micro_offset'].norm(dim=-1)/m.node_mlp.position_scale).cpu())
                    Image.fromarray(byte.transpose(1,2,0)).save(dest/'frames'/(row['frame']+'.png'))
                    Image.fromarray(gtbyte.transpose(1,2,0)).save(dest/'GT'/(row['frame']+'.png'))
                    app_rows.append(dict(frame=row['frame'],**run.appearance_statistics(m,cam.time,cam)))
            mean={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ['frame','time']}
            mean.update(RMSE=float(np.sqrt(mean["RGB_MSE"])))
            mean["RMSE_MAE_ratio"]=mean["RMSE"]/mean["RGB_L1"]
            curves.append(dict(iteration=i,**mean))
            run.write(dest/f'per_view_{i}.json',rows)
            if i==30000:
                magnitudes=np.concatenate([v.numpy() for v in micro_magnitudes])
                micro_stats=dict(mean=float(magnitudes.mean()),p95=float(np.quantile(magnitudes,.95)),
                    maximum=float(magnitudes.max()),normalization='norm(delta_world)/NodeMLP.position_scale',
                    aggregation='pooled all Gaussian IDs x all native TEST cameras, equal weights',
                    sample_count=len(magnitudes),alpha=1.,half_extent=float(m.node_mlp.position_scale))
                assert micro_stats['maximum']<=np.sqrt(3)*.005+1e-7
                del magnitudes,micro_magnitudes
                # Warmup/benchmark before temporal diagnostics; no other compute jobs on this GPU.
                assert torch.cuda.is_available()
                perf=benchmark(m,s,p,b)
                temporal=[];previous=None
                for cam in sorted(s.getTrainCameras()+([] if scene=='transverse' else s.getTestCameras()),key=lambda x:float(x.time)):
                    q=render(cam,m,p,b)
                    err=q['render'].clamp(0,1)-cam.original_image[:3].cuda().float()
                    if previous is not None:
                        temporal.append(dict(frame=str(cam.image_name),time=float(cam.time),
                                             residual_delta_L1=float((err-previous).abs().mean())))
                    previous=err
                run.write(dest/'per_view.json',rows);run.write(dest/'appearance_statistics.json',app_rows)
                run.write(dest/'temporal.json',temporal)
                run.write(dest/'summary.json',dict(metrics=mean,TEST_frames=len(rows),checkpoint=30000,
                    micro_displacement=micro_stats,performance=perf,runtime=json.loads((out/'runtime.json').read_text()),
                    temporal_residual_delta_L1=float(np.mean([r['residual_delta_L1'] for r in temporal])),
                    temporal_note='Unwarped successive prediction-error changes, TRAIN only on DynamicColon and all native times on C3VD; not motion-compensated flicker.',
                    depth_note='Normalized depth follows existing per-render-max convention; raw depth MSE also logged; CH/HD95 evaluated separately using active micro centers'))
        assert before==(model_digest(m),digest(m.optimizer.state_dict()),run.sha(path/f'fine_state_{i}.pth'))
        integrity.append(dict(iteration=i,pass_=True,model_Adam_checkpoint_unchanged=True,optimizer_updates=0))
        del m,s,p,b,c;gc.collect();torch.cuda.empty_cache()
    run.write(dest/'learning_curve.json',curves)
    run.write(out/'reload_evaluation_integrity.json',dict(pass_=True,rows=integrity))
    with imageio.get_writer(dest/'reconstruction.mp4',fps=20,codec='libx264',macro_block_size=1,
                            ffmpeg_params=['-pix_fmt','yuv444p']) as writer:
        for row in rows:writer.append_data(np.asarray(Image.open(dest/'frames'/(row['frame']+'.png'))))
    reader=imageio.get_reader(dest/'reconstruction.mp4');count=sum(1 for _ in reader);reader.close()
    assert count==(200 if scene=='transverse' else 36)
    run.write(out/'status.json',dict(state='EVALUATED',iteration=30000,TEST_frames=count))


def surface_state(m,t):
    """Renderer-consistent O centers including micro, with native scale/rotation."""
    xyz,scales,rotations=m.deform_geometry_from_nodes(t)
    graph=m.appearance_graph
    _,delta,_=m.appearance_model.forward_micro(t,m.features,m._deformation_table,
        graph.gaussian_node_indices,graph.gaussian_node_weights,m.node_mlp.position_scale)
    return xyz+delta,scales,rotations


def geometry_at(root,scene,key,iteration):
    import torch
    from tools import run_stage8 as run
    from tools import evaluate_rectum_ch_hd95 as geo
    from utils.general_utils import build_rotation
    from utils.exact_split_branchpoint import digest,model_digest
    from gaussian_renderer import render
    run.configure(scene)
    study=root/scene/'runs'/key;out=snapshot(study,iteration)
    dest=root/scene/'geometry_gt_evaluation'/key/str(iteration);dest.mkdir(parents=True,exist_ok=False)
    m,s,p,b,c,st=run.restore(out,iteration);del st
    before=(model_digest(m),digest(m.optimizer.state_dict()),run.sha(out/f'fine_state_{iteration}.pth'))
    manifest=json.loads((dest.parent.parent/'gt_manifest.json').read_text());rows=[];started=time.monotonic()
    rng=np.random.default_rng(99);a=rng.normal(size=(31,3));bb=rng.normal(size=(47,3));pair=((a[:,None]-bb[None])**2).sum(2)
    exact=geo.metric(a,bb);assert np.isclose(exact['CH'],pair.min(1).mean()+pair.min(0).mean(),atol=1e-12)
    assert np.isclose(exact['HD95'],max(np.quantile(np.sqrt(pair.min(1)),.95),np.quantile(np.sqrt(pair.min(0)),.95)))
    with torch.no_grad():
        for idx,(cam,record) in enumerate(zip(s.getTestCameras(),manifest)):
            assert abs(float(cam.time)-record['time_index'])<1e-7
            xyz,log_scale,quaternion=surface_state(m,cam.time)
            if idx in [0,99,199]:torch.testing.assert_close(xyz,render(cam,m,p,b)['transformed_points'],atol=2e-6,rtol=0)
            scale=m.scaling_activation(log_scale).clamp_max(.05*m.spatial_lr_scale)
            opacity=m.get_opacity.squeeze();keep=opacity>0;opacity=opacity[keep]
            weight=(opacity-opacity.min())/(opacity.max()-opacity.min()+1e-8)
            counts=(weight*10).long().clamp_min(1)
            gen=torch.Generator(device='cuda').manual_seed(1729+idx)
            noise=torch.randn((len(xyz),10,3),generator=gen,device='cuda')[keep]
            chosen=torch.arange(10,device='cuda')[None]<counts[:,None]
            rotated=torch.bmm(noise*scale[keep,None],build_rotation(quaternion[keep]).transpose(1,2))
            samples=(rotated+xyz[keep,None])[chosen].cpu().numpy()
            assert geo.sha(record['cache'])==record['cache_sha256']
            row=geo.metric(samples,np.load(record['cache'],allow_pickle=False),workers=2)
            row.update(frame=idx+1,time=float(cam.time),sampled_points=len(samples));rows.append(row)
            with (dest/'per_frame.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if (idx+1)%10==0:run.write(dest/'status.json',dict(state='EVALUATING',frames=idx+1,total=200))
    assert len(rows)==200
    assert before==(model_digest(m),digest(m.optimizer.state_dict()),run.sha(out/f'fine_state_{iteration}.pth'))
    run.write(dest/'summary.json',dict(metrics={k:float(np.mean([r[k] for r in rows])) for k in ['CH','HD95','CH_pred_to_GT','CH_GT_to_pred','HD95_pred_to_GT','HD95_GT_to_pred']},
        frames=200,evaluation_seconds=time.monotonic()-started,model_Adam_checkpoint_unchanged=True,optimizer_updates=0,exact_KDTree_verified=True))
    run.write(dest/'status.json',dict(state='COMPLETE',frames=200))



def geometry(root,scene,key):
    assert scene=='transverse'
    from tools import run_stage8 as run
    means=[]
    for iteration in STEPS:
        geometry_at(root,scene,key,iteration)
        q=json.loads((root/scene/'geometry_gt_evaluation'/key/str(iteration)/'summary.json').read_text())
        means.append(dict(iteration=iteration,**q['metrics']))
    dest=root/scene/'geometry_gt_evaluation'/key
    run.write(dest/'learning_curve.json',means)
    run.write(dest/'summary.json',dict(metrics={k:v for k,v in means[-1].items() if k!='iteration'},frames=200,iterations=STEPS,
        optimizer_updates=0,model_Adam_checkpoint_unchanged=True))


def exports(root,scene,key):
    from tools import run_dynamiccolon_rectum_r6_30k as native
    run.configure(scene)
    out=root/scene/'runs'/key/'J'
    m,s,p,b,c,st=run.restore(out,30000);del st
    assert int(m.appearance_model.micro_iteration)==30000
    native.DATA=run.DATASETS['transverse']
    native.native_exports(root,out,m,s,p,b)


def clean_reference(root):
    """O depth-only replay to Stage8 sidecars; historical files stay untouched."""
    scene='transverse';run.configure(scene)
    study=root/scene/'reference_snapshots/O';dest=root/scene/'reference_depth/O';dest.mkdir(parents=True,exist_ok=False)
    curves=[];checks=[]
    for i in STEPS:
        out=snapshot(study,i);m,s,p,b,c,st=run.restore(out,i);del st
        before=(model_digest(m),digest(m.optimizer.state_dict()),run.sha(out/f'fine_state_{i}.pth'))
        rows=[]
        with torch.no_grad():
            for cam in s.getTestCameras():
                q=render(cam,m,p,b)
                from tools.evaluate_geometry_blending import masks,depth_metrics
                _,valid,*_=masks(cam)
                legacy,_=depth_metrics(q['depth'],cam,q['depth'].squeeze()[valid].max())
                rows.append(dict(frame=str(cam.image_name),legacy_depth_MSE=legacy['DEPTH_FLOAT_VALID_MSE'],**clean_depth(q['depth'],cam)))
        means={k:float(np.mean([v[k] for v in rows])) for k in rows[0] if k!='frame'}
        old=json.loads((root/scene/'runs/O/J/evaluation'/f'per_view_{i}.json').read_text())
        error=max(abs(v['legacy_depth_MSE']-o['depth_MSE']) for v,o in zip(rows,old))
        assert error<2e-6,error
        assert before==(model_digest(m),digest(m.optimizer.state_dict()),run.sha(out/f'fine_state_{i}.pth'))
        curves.append(dict(iteration=i,**means));checks.append(dict(iteration=i,pass_=True,model_Adam_checkpoint_unchanged=True,legacy_max_abs_diff=error,optimizer_updates=0))
        run.write(dest/f'per_view_{i}.json',rows)
        del m,s,p,b,c;gc.collect();torch.cuda.empty_cache()
    run.write(dest/'learning_curve.json',curves)
    run.write(dest/'summary.json',dict(metrics={k:v for k,v in curves[-1].items() if k!='iteration'},TEST_frames=200,checkpoint=30000))
    run.write(dest/'integrity.json',dict(pass_=True,checks=checks,all_old_artifacts_unchanged=True,optimizer_updates=0))


def report_sigmoid2(root):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=root/'sigmoid2';keys=[k for k in 'ABCDEFGHIJKLMNOP' if (dest/'runs'/k/'J/evaluation/summary.json').exists()]
    paths={k:dest/'runs'/k/'J' for k in keys};results={}
    proofs=[json.loads((p.parent/'initial_integrity.json').read_text()) for p in paths.values()]
    for field in ['raw_sha256','node_mlp_sha256','KNN99_sha256','appearance_graph_sha256']:
        assert len({r[field] for r in proofs})==1,field
    camera=[json.loads((p/'integrity.json').read_text())['camera_sha256'] for p in paths.values()]
    assert len(set(camera))==1
    for k,out in paths.items():
        assert json.loads((out/'reload_evaluation_integrity.json').read_text())['pass_']
        q=json.loads((out/'evaluation/summary.json').read_text());r=q['runtime']
        met=dict(q['metrics'])
        if k in 'ABC':met.update(json.loads((root/'historical_float_metrics'/k/'metrics.json').read_text())['metrics'])
        results[k]=dict(met,FPS=q['performance']['FPS'],training_minutes=r['training_seconds']/60,
            peak_VRAM_MiB=r['peak_VRAM_MiB'],appearance_parameters=r['appearance_parameters'],
            total_trainable_parameters=r['total_trainable_parameters'],
            temporal_residual_delta_L1=q['temporal_residual_delta_L1'])
    changes={f'{a}-{b}':{k:results[a][k]-results[b][k] for k in results[a]} for a,b in [('D','A'),('D','B'),('D','C'),('E','D'),('E','B'),('F','E'),('F','A'),('G','F'),('H','G'),('H','A'),('I','G'),('J','I'),('J','A'),('K','J'),('L','K'),('L','A'),('M','J'),('N','M'),('N','A'),('O','M'),('O','J'),('O','A'),('P','O'),('P','M'),('P','A')] if a in results and b in results}
    quality={pair:{k:float(v[k]) for k in ['PSNR','SSIM','LPIPS_A','LPIPS_V']} for pair,v in changes.items()}
    run.write(dest/'comparison.json',dict(results=results,differences=changes,RGB_changes=quality))
    final='P' in keys
    visual=dest/('visuals_final' if final else 'visuals_stage1');visual.mkdir(exist_ok=False)
    labels={'A':'A R6','B':'B R6+Additive','C':'C R7-Core','D':'D R7-V2','E':'E LossUnlock','F':'F LossUnlock+Weight2400','G':'G DynamicsUnlock','H':'H SpecCam+Aniso','I':'I L2-Handoff','J':'J L2-Handoff+Additive025','K':'K Cap035/Handoff10k','L':'L PriorRelease','M':'M Charb/DeepRelease','N':'N WideR6-96','O':'O MicroDisp195','P':'P O+FixedKNN001'}
    labels={k:labels[k] for k in keys}
    viskeys=[k for k in ['A','M','O','P'] if k in keys]
    vislabels={k:labels[k] for k in viskeys}
    rows=json.loads((paths['A']/'evaluation/per_view.json').read_text())
    chosen=set(np.linspace(0,len(rows)-1,6,dtype=int));crop_manifest=[]
    with imageio.get_writer(visual/'GT_comparison.mp4',fps=20,codec='libx264',macro_block_size=1,
                            ffmpeg_params=['-pix_fmt','yuv444p']) as writer:
        for idx,row in enumerate(rows):
            name=row['frame'];gt=Image.open(paths['A']/'evaluation/GT'/(name+'.png')).convert('RGB')
            ims=[gt]+[Image.open(paths[k]/'evaluation/frames'/(name+'.png')).convert('RGB') for k in viskeys]
            w,h=gt.size;panel=Image.new('RGB',(len(ims)*w,h+28));draw=ImageDraw.Draw(panel)
            for j,(im,label) in enumerate(zip(ims,['GT',*vislabels.values()])):
                panel.paste(im,(j*w,28));draw.text((j*w+5,5),label,fill='white')
            writer.append_data(np.asarray(panel.resize((1440,round((h+28)*1440/(len(ims)*w))))))
            if idx in chosen:
                panel.save(visual/f'{name}_full.png')
                gray=np.asarray(gt).mean(2)/255;gy,gx=np.gradient(gray)
                # GT-only proxies; no anatomical labels claimed for fold/bright crops.
                scores={'texture':gx*gx+gy*gy,'fold_edge_proxy':abs(gx)+abs(gy),'specular_brightness_proxy':gray}
                for kind,score in scores.items():
                    boxes=[(x,y,x+96,y+96) for y in range(0,h-95,48) for x in range(0,w-95,48)]
                    box=max(boxes,key=lambda z:float(score[z[1]:z[3],z[0]:z[2]].mean()))
                    cropped=Image.new('RGB',(len(ims)*288,316));draw=ImageDraw.Draw(cropped)
                    for j,(im,label) in enumerate(zip(ims,['GT',*vislabels.values()])):
                        cropped.paste(im.crop(box).resize((288,288)),(j*288,28));draw.text((j*288+4,5),label,fill='white')
                    cropped.save(visual/f'{name}_{kind}.png');crop_manifest.append(dict(frame=name,kind=kind,box=box))
    reader=imageio.get_reader(visual/'GT_comparison.mp4');count=sum(1 for _ in reader);reader.close();assert count==36
    run.write(visual/'crop_selection.json',crop_manifest)
    fig,axes=plt.subplots(1,4,figsize=(17,4))
    for k,out in paths.items():
        curve=json.loads((out/'evaluation/learning_curve.json').read_text())
        for ax,field in zip(axes,['PSNR','SSIM','LPIPS_A','LPIPS_V']):
            ax.plot([r['iteration'] for r in curve],[r[field] for r in curve],label=k,marker='o')
            ax.set_title(field);ax.set_xlabel('Fine iteration');ax.grid(alpha=.2)
    axes[0].legend();fig.tight_layout();fig.savefig(visual/'RGB_learning_curves.png',dpi=160);plt.close(fig)
    fields=['RGB_L1','opacity_mean','opacity_below_001','absolute_deformation_mean',
            'node_motion_max','canonical_scale_clamped_fraction','gain_abs_mean','additive_abs_mean']
    fig,axes=plt.subplots(2,4,figsize=(18,8))
    for k,out in paths.items():
        log=[json.loads(x) for x in (out/'training_metrics.jsonl').read_text().splitlines()]
        for ax,field in zip(axes.flat,fields):
            ax.plot([r['iteration'] for r in log],[r[field] for r in log],label=k)
            ax.set_title(field);ax.set_xlabel('Fine iteration');ax.grid(alpha=.2)
    axes.flat[0].legend();fig.tight_layout();fig.savefig(visual/'training_stability.png',dpi=150);plt.close(fig)
    fields=['PSNR','SSIM','LPIPS_A','LPIPS_V','RGB_L1','RMSE','RMSE_MAE_ratio','image_gradient_L1','FPS','training_minutes','peak_VRAM_MiB']
    selection=json.loads((root/'stage2_selection.json').read_text())
    lines=['# Stage8 O + fixed KNN99=0.01 — C3VD sigmoid2','',
        'A–O reused without retraining. P reuses exact O common fine0, geometry2400 IDs and appearance2000K4. '
        'Geometry and appearance train jointly for30000 Adam updates. G3 after2k, J8 after5k; '
        'appearance retains original2000 K4 nodes. F adds400 TRAIN coarse-gradient geometry nodes.', '',
        'Stage2 appearance: '+selection['appearance_mode']+'. Selection uses final D versus B TEST PSNR as explicitly requested; '
        'Stage2 is exploratory, not an independent confirmation on untouched TEST data.', '',
        '| Run | '+' | '.join(fields)+' |','|---|'+'---:|'*len(fields)]
    for k,r in results.items():lines.append('| '+labels[k]+' | '+' | '.join(f'{r[f]:.6f}' for f in fields)+' |')
    lines+=['','| Contrast | '+' | '.join(fields)+' |','|---|'+'---:|'*len(fields)]
    for k,r in changes.items():lines.append('| '+k+' | '+' | '.join(f'{r[f]:+.6f}' for f in fields)+' |')
    lines+=['','Matched loss contrast: '+selection['pure_loss_contrast']+'. '
        'If fallback R6+Additive is selected, E-D also changes architecture. F-E changes geometry node sampling/count only. '
        'G-F changes the reset/LR/prior bundle; H-G changes camera conditioning, node feature dimension, beta range and scale/rotation clips together. These contrasts do not isolate one mechanism. Root-cause claims remain hypotheses. I-G changes the photometric handoff only; J-I changes additive cap. Both retain G scale/rotation clips .10 and have13922443 trainable parameters. Identical early objective/camera draws do not guarantee bitwise trajectories under native CUDA atomics. M-J bundles deeper priors, Charbonnier, handoff endpoints, cosine floor and XYZ LR endpoint. N-M changes node feature dimension and decoder width together. No isolated prior/Charbonnier causality, proven optimal timing/cap, or additive prediction of gains is established. Single paired seed with native CUDA nondeterminism.',
        '', '| Run | Appearance parameters | Total trainable parameters |','|---|---:|---:|']
    for k,r in results.items():lines.append(f"| {k} | {r['appearance_parameters']} | {r['total_trainable_parameters']} |")
    lines+=['','No estimated-depth accuracy metric. RGBL1 and RMSE use float clamped RGB; RMSE pools squared errors over all36 equally sized TEST frames, '
        'and RMSE/MAE uses the same float errors. PSNR/SSIM/LPIPS retain historical8bit evaluator semantics.',
        'DSSIM=(1-SSIM)/2. E/F/G/H use .8L1+.2DSSIM+2MSE, TV0, colorvar1e-5; D unchanged pureL1, TV.001, colorvar1e-4.',
        'I/J use G weights until15k; linearly hand off L1 .8->.2 and MSE2->12 through30k, with DSSIM.2 unchanged. I beta.15; J beta.25 from initialization. '
        'M/N use exact Charbonnier mean(sqrt(e^2+1e-6)) fromstep1; handoff15k->30k .8->.15 Charb,2->14MSE,DSSIM.2; cosinefloor.12,XYZcoefficient4e-6,depth.25->.005/KNN99.01->.0002 from8k->22k; beta.25. N preservesfine0nodefirst8 andstartsnew8zero,MLP45->96->96->6. '
        'Training minutes exclude graph/checkpoint/evaluation. Peak VRAM is Torch allocated. FPS is end-to-end renderer on an idle compute GPU; '
        'different learned geometry can also affect throughput.',
        'Temporal diagnostics are unwarped successive error changes, not motion-compensated flicker. '
        'Crops are GT-only proxies on six uniform TEST frames, no anatomical labels or prediction-selected crops.',
        '', f'[Video]({visual.name}/GT_comparison.mp4) · [RGB curves]({visual.name}/RGB_learning_curves.png) · '
        f'[Stability curves]({visual.name}/training_stability.png) · [Metrics](comparison.json)']
    micro=json.loads((paths['P']/'evaluation/summary.json').read_text())['micro_displacement']
    lines += ['', 'Run O: exactly M plus a zero-initialized64->3 micro-displacement head (195 parameters), '
        'ramp6k->10k, coordinatewise .005*a*tanh bound, and .001*mean(sum(tanh(raw)^2)) penalty. '
        'Offset is added before rasterization and KNN99; shared trunk/embedding/features receive geometry gradients. '
        'O-M isolates this whole specified package, not an unregularized head alone. '
        'N-M does not establish appearance saturation; sparse-motion phase error remains a hypothesis. '
        'An improvement would not by itself demonstrate anatomically accurate motion.',
        '', f"P30k normalized micro displacement ||delta||/a: mean={micro['mean']:.9g}, p95={micro['p95']:.9g}, max={micro['maximum']:.9g}. "
        'Pooled over all Gaussian IDs and all36 TEST cameras; per-axis bound .005, L2 bound sqrt(3)*.005.']
    lines+=['','Run P differs from O only by constant lambda_knn99=.01 at all30000 updates. Depth remains O .25->.005 over8k->22k; RGB/micro/LR and all architecture unchanged. Surface anchoring and retained29dB quality are hypotheses, not guaranteed. P-O isolates KNNschedule in this paired single-seed block; native CUDA atomics can change trajectories.']
    report='\n'.join(lines)+'\n'
    (dest/('report_final.md' if final else 'report_stage1.md')).write_text(report)
    (root/'sigmoid2_report_links.md').write_text(report.replace(']('+visual.name+'/','](sigmoid2/'+visual.name+'/').replace('](comparison.json)','](sigmoid2/comparison.json)'))
    run.write(dest/('completion_final.json' if final else 'completion_stage1.json'),dict(pass_=True,
        evaluated_runs=keys,steps=30000,video_frames=36,depth_accuracy_metric=False))

def report_transverse(root):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=root/'transverse';results={};sources={};curves={}
    for key in ['O','P1','P2']:
        out=dest/'runs'/key/'J';q=json.loads((out/'evaluation/summary.json').read_text())
        runtime=q['runtime'];met=dict(q['metrics']);sources[key]=q
        if key=='O':
            clean=json.loads((dest/'reference_depth/O/summary.json').read_text())['metrics']
            met.update(clean)
            # O is a symlinked historical study; its geometry summary is reused unchanged.
            original=json.loads((root/'historical_paths.json').read_text())['transverse']
            geo=json.loads((Path(original)/'transverse/geometry_gt_evaluation/O/summary.json').read_text())
            curves[key]=json.loads((dest/'reference_depth/O/learning_curve.json').read_text())
        else:
            geo=json.loads((dest/'geometry_gt_evaluation'/key/'summary.json').read_text())
            curves[key]=json.loads((out/'evaluation/learning_curve.json').read_text())
        met.update(geo['metrics'])
        met['legacy_depth_MSE']=q['metrics']['depth_MSE']
        met.update(FPS=q['performance']['FPS'],training_minutes=runtime['training_seconds']/60,
            peak_VRAM_MiB=runtime['peak_VRAM_MiB'],total_trainable_parameters=runtime['total_trainable_parameters'],
            appearance_parameters=runtime['appearance_parameters'],geometry_nodes=2400)
        results[key]=met
    fields=['PSNR','SSIM','LPIPS_A','LPIPS_V','legacy_depth_MSE','Depth_MSE_raw','Depth_MSE_normalized','CH','HD95','FPS','training_minutes']
    contrasts={f'{a}-{b}':{k:results[a][k]-results[b][k] for k in fields} for a,b in [('P1','O'),('P2','O'),('P2','P1')]}
    historical=json.loads((root/'coverage_reference_results.json').read_text())['completed']
    context={}
    for key in ['F0','U2']:
        q=historical[key];context[key]=dict(q['metrics'],FPS=q['performance']['FPS'],training_minutes=q['train_seconds']/60,geometry_nodes=q['geometry_nodes'])
    run.write(dest/'comparison.json',dict(results=results,context_only=context,differences=contrasts,
        depth_mask='GT finite & >0 & !=1e10 & sourceRGBAalpha>0; exactly F0/U2',legacy_depth_mask='legacy finitepositive including1e10',
        context_note='F0/U2 have different geometry topology/losses/initialization. Context rows, not matched causal controls.'))
    visual=dest/'visuals_final';visual.mkdir(exist_ok=False)
    rows=json.loads((dest/'runs/O/J/evaluation/per_view.json').read_text())
    with imageio.get_writer(visual/'GT_O_P1_P2.mp4',fps=30,codec='libx264',macro_block_size=1,ffmpeg_params=['-pix_fmt','yuv444p']) as writer:
        for row in rows:
            name=row['frame'];gt=Image.open(dest/'runs/O/J/evaluation/GT'/(name+'.png')).convert('RGB')
            ims=[gt]+[Image.open(dest/'runs'/key/'J/evaluation/frames'/(name+'.png')).convert('RGB') for key in ['O','P1','P2']]
            w,h=gt.size;panel=Image.new('RGB',(4*w,h+28));draw=ImageDraw.Draw(panel)
            for j,(im,label) in enumerate(zip(ims,['GT','O reference','P1 FixedPrior Mild','P2 FixedPrior Full'])):
                panel.paste(im,(j*w,28));draw.text((j*w+5,5),label,fill='white')
            writer.append_data(np.asarray(panel.resize((1440,round((h+28)*1440/(4*w))))))
    reader=imageio.get_reader(visual/'GT_O_P1_P2.mp4');assert sum(1 for _ in reader)==200;reader.close()
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for key,curve in curves.items():
        for ax,field in zip(axes,['legacy_depth_MSE','Depth_MSE_raw','Depth_MSE_normalized']):
            ax.plot([v['iteration'] for v in curve],[v[field] for v in curve],label=key,marker='o');ax.set_title(field);ax.grid(alpha=.2)
    axes[0].legend();fig.tight_layout();fig.savefig(visual/'legacy_clean_depth_curves.png',dpi=160);plt.close(fig)
    lines=['# Stage8 DynamicColon transverse — O / P1 / P2','',
        'P1/P2 reuse the exact Stage7 O post-coarse fine0 and2400geometry/2000appearanceK4 node IDs. No coarse replay or old training. '
        'O TriHead architecture,200000Gaussians, all200rootTRAIN/provided200TEST. KNN99=.01 and depth=.25 fixed through30k; '
        'P1 handoff15k->30k to.4Charb+.2DSSIM+6MSE; P2 retains O full handoff to.15Charb+.2DSSIM+14MSE. '
        'All other hyperparameters/LR/micro/TV/colorvar/KL unchanged. Geometry and appearance jointly train30000Adam.', '',
        '| Run | Geometry nodes | '+' | '.join(fields)+' |','|---|---:|'+'---:|'*len(fields)]
    for key,met in [*context.items(),*results.items()]:
        lines.append('| '+key+(' (context)' if key in context else '')+' | '+str(met['geometry_nodes'])+' | '+' | '.join(f'{met[f]:.8g}' if f in met else '—' for f in fields)+' |')
    lines += ['', '| Contrast | '+' | '.join(fields)+' |','|---|'+'---:|'*len(fields)]
    for key,met in contrasts.items():lines.append('| '+key+' | '+' | '.join(f'{met[f]:+.8g}' for f in fields)+' |')
    lines += ['', 'Depth validity is copied exactly from ColonGraph-texture/tools/evaluate_adaptive_j8_transverse.py::depth_valid_mask and '
        'evaluate_adaptive_residual_A20.py::depth_statistics used by coverage-lock F0/U2: finite positive GT, excludes exact1e10 background and sourceRGBAalpha0. '
        'Raw and normalized metrics use this same mask; each prediction/GT is normalized by its own valid-pixel maximum with1e-6 floor and clamp[0,1]. '
        'No threshold>=1e5 or new tuned cutoff. Legacy depth_MSE is retained alongside clean values. '
        'Original O files are immutable; clean O results are read-only sidecars under reference_depth/O.',
        '', 'P1-O changes priors plus RGB handoff; P2-O changes both depth/KNN priors; P2-P1 isolates handoff strength. '
        'F0/U2 are historical context with different topology/loss/init and independently measured FPS, not matched causal controls. '
        'Fixed priors do not guarantee better surface geometry. Single paired block/nativeCUDA variability; fixed30k reporting, no TEST checkpoint selection.',
        '', 'CH is directional squared nearest-neighbor mean sum (scene_unit²); HD95 max directional Euclidean p95 (scene_unit), '
        'all200times at all5epochs, active micro centers, native opacity samples with seed1729+idx, exact KDTree/noICP/scalefit/crop. '
        'Native3lookat camera sequences exported per P1/P2. Training minutes exclude evaluation/checkpoint/setup; FPS idle-GPU native renderer.',
        '', '[Video GT/O/P1/P2](visuals_final/GT_O_P1_P2.mp4) · [Depth curves](visuals_final/legacy_clean_depth_curves.png) · [Metrics](comparison.json)']
    for key in ['P1','P2']:
        lines+=['',key+' lookat: '+' · '.join(f'[{i}](runs/{key}/J/test/ours_30000/lookat_cameras/lookat_{i}_video.mp4)' for i in [1,2,3])]
    (dest/'report.md').write_text('\n'.join(lines)+'\n')
    run.write(dest/'completion_final.json',dict(pass_=True,full_new_runs=['P1','P2'],TEST_frames=200,geometry_frames=200,steps=STEPS,
        O_clean_replay_readonly=True,legacy_and_clean_depth=True,checkpoint_selection='fixed30k'))


def report(root):
    report_sigmoid2(root);report_transverse(root)
    parts=[]
    for scene,name in [('sigmoid2','report_final.md'),('transverse','report.md')]:
        q=(root/scene/name).read_text().replace('](visuals_final/',f']({scene}/visuals_final/').replace('](comparison.json)',f']({scene}/comparison.json)').replace('](runs/',f']({scene}/runs/')
        parts.append(q)
    (root/'report.md').write_text('# Stage8 — fixed KNN99 and modality-dependent depth\n\n'+'\n\n'.join(parts))
