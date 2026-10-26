"""Read-only fixed-checkpoint TEST metrics, with optional DynamicColon GT geometry."""
import json,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from gaussian_renderer import render
from baseline.checkpoint import load
from baseline.common import write
from utils.exact_split_branchpoint import model_digest,digest

def latest(path,iteration):
    return iteration if iteration>=0 else max(int(p.stem.split('_')[-1]) for p in path.glob('fine_state_*.pth'))

def cloud_metric(pred,gt):
    from scipy.spatial import cKDTree
    a=cKDTree(gt).query(pred,workers=2)[0];b=cKDTree(pred).query(gt,workers=2)[0]
    return dict(CH=float(np.mean(a*a)+np.mean(b*b)),HD95=float(max(np.quantile(a,.95),np.quantile(b,.95))),
        HD95_pred_to_GT=float(np.quantile(a,.95)),HD95_GT_to_pred=float(np.quantile(b,.95)))

@torch.no_grad()
def evaluate(args):
    from utils.image_utils import psnr,ssim,lpips_score
    from utils.stage8_depth import depth_statistics
    from utils.general_utils import build_rotation
    path=args.model_path.resolve();iteration=latest(path,args.iteration);data=args.source_path.resolve()
    dest=path/'evaluation'/('iteration_'+str(iteration)+('_probe' if args.max_frames else ''));dest.mkdir(parents=True,exist_ok=False)
    m,s,p,b,c,state=load(path,data,iteration)
    before=(model_digest(m),digest(m.optimizer.state_dict()),digest(m.harmonic_nodes.payload() if hasattr(m,'harmonic_nodes') else None))
    cams=s.getTestCameras();sample=cams[0]
    for _ in range(10):render(sample,m,p,b)
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
    for _ in range(50):render(sample,m,p,b)
    end.record();end.synchronize();fps=50000/start.elapsed_time(end)
    selected=cams[:args.max_frames] if args.max_frames else cams;rows=[]
    for idx,cam in enumerate(tqdm(selected,desc='TEST metrics')):
        q=render(cam,m,p,b);rgb=q['render'].clamp(0,1);gt=cam.original_image[:3].cuda().float()
        a=(rgb*255+.5).to(torch.uint8).float()/255;bgt=(gt*255+.5).to(torch.uint8).float()/255
        row=dict(frame=cam.image_name,PSNR=float(psnr(a[None],bgt[None]).mean()),SSIM=float(ssim(a[None],bgt[None])),
            LPIPS_A=float(lpips_score(a[None],bgt[None],'alex').mean()),LPIPS_V=float(lpips_score(a[None],bgt[None],'vgg').mean()),
            RGB_L1=float((rgb-gt).abs().mean()),RGB_MSE=float((rgb-gt).square().mean()))
        dep=cam.original_depth.cuda().float().squeeze();pred=q['depth'].squeeze();valid=torch.isfinite(dep)&(dep>0)
        if state['is_c3vd']:
            pn=(pred/pred[valid].max().clamp_min(1e-6)).clamp(0,1);gn=(dep/dep[valid].max().clamp_min(1e-6)).clamp(0,1)
            row['Depth_MSE_normalized']=float((pn-gn).square()[valid].mean())
        else:
            name=getattr(cam,'original_image_name',cam.image_name.removeprefix('test__'))
            alpha=torch.tensor(np.array(Image.open(data/'test/img'/(name+'.png')).convert('RGBA'))[...,3],device='cuda')
            row.update(depth_statistics(pred,dep,alpha))
            if args.geometry:
                from plyfile import PlyData
                source=data/'ply'/f'pc_{idx+1:04d}.ply';vertex=PlyData.read(source)['vertex'];gtcloud=np.column_stack([vertex[n] for n in ['x','y','z']])
                xyz=q['transformed_points'];_,logscale,rotation=m.deform_geometry_from_nodes(cam.time);scale=m.scaling_activation(logscale).clamp_max(.05*m.spatial_lr_scale)
                opacity=m.get_opacity.squeeze();keep=opacity>0;op=opacity[keep];count=(((op-op.min())/(op.max()-op.min()+1e-8))*10).long().clamp_min(1)
                noise=torch.randn((len(xyz),10,3),device='cuda',generator=torch.Generator(device='cuda').manual_seed(1729+idx))[keep]
                chosen=torch.arange(10,device='cuda')[None]<count[:,None]
                points=(torch.bmm(noise*scale[keep,None],build_rotation(rotation[keep]).transpose(1,2))+xyz[keep,None])[chosen].cpu().numpy()
                row.update(cloud_metric(points,gtcloud))
        dx=(rgb[:,:,1:]-rgb[:,:,:-1])-(gt[:,:,1:]-gt[:,:,:-1]);dy=(rgb[:,1:,:]-rgb[:,:-1,:])-(gt[:,1:,:]-gt[:,:-1,:])
        row['image_gradient_L1']=.5*float(dx.abs().mean()+dy.abs().mean());rows.append(row)
        with (dest/'per_view.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    assert before==(model_digest(m),digest(m.optimizer.state_dict()),digest(m.harmonic_nodes.payload() if hasattr(m,'harmonic_nodes') else None))
    mean={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='frame'};mean['RMSE']=mean['RGB_MSE']**.5
    write(dest/'summary.json',dict(metrics=mean,FPS=fps,frames=len(rows),total_TEST=len(cams),checkpoint=iteration,partial=bool(args.max_frames),read_only=True))
    print(json.dumps(mean,indent=2));print('Results:',dest)
