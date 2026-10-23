#!/usr/bin/env python
"""Canonical interface to the packaged U7 baseline."""
import argparse,ast,datetime,hashlib,json,os,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))


def write(path,q):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n')


def history_guard(event,args):
    if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
    p=Path(os.fsdecode(args[0])).absolute()
    if ROOT in p.parents:return
    if '/output/experiments/' in str(p) or any(x.startswith('ColonGraph-stage') or x=='ColonGraph-texture' for x in p.parts):
        raise RuntimeError('U7 package tried to access external experiment/source: '+str(p))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def materialize(source,target,iteration):
    target.mkdir(parents=True)
    files=['cfg_args',f'fine_state_{iteration}.pth',f'checkpoint_{iteration}.json']
    if (source/'runtime.json').exists():files.append('runtime.json')
    for name in files:
        shutil.copy2(source/name,target/name)
    shutil.copytree(source/'point_cloud'/f'iteration_{iteration}',target/'point_cloud'/f'iteration_{iteration}')
    return target


def setup(args,out):
    sys.addaudithook(history_guard)
    data=(args.data or ROOT.parent/'datasets/DynamicColon'/('data_'+args.scene)).resolve()
    assert (data/'transforms.json').is_file() and (data/'test/transforms.json').is_file(),data
    expected=json.loads((ROOT/'assets'/args.scene/'dataset_metadata_sha256.json').read_text())
    for rel,h in expected.items():
        assert sha(data/rel)==h, 'Dataset metadata differs from packed '+args.scene+' prefix: '+rel
    os.environ['COLONGRAPH_DATASET']=str(data)
    from tools import u7_engine as engine
    key='R' if args.scene=='rectum' else 'U7'
    engine.DATA=data;engine.ASSET=ROOT/'assets'/args.scene;engine.OUT=out
    engine.SOURCES={key:engine.ASSET/'prefix_10000'}
    engine.configure(key)
    return engine,key,data


def load_run(engine,key,args,out):
    source=args.run.resolve() if args.run else engine.ASSET/'u7_30000'
    iteration=args.iteration
    path=materialize(source,out/'loaded_checkpoint',iteration)
    m,s,p,b,c,st=engine.load(path,iteration,key)
    engine.opt_setup(m,c);m.optimizer.load_state_dict(st['optimizer'])
    assert m.stage11_key in ('U7','R')
    return m,s,p,b,c,st,path


def train(args,out):
    import torch
    engine,key,data=setup(args,out)
    write(out/'matrix_protocol.json',dict(baseline='Stage11 U7',scene=args.scene,source='packed exact10k prefix' if not args.run else str(args.run),fine_end=args.end,GT_PLY_training=False))
    (out/'immutable_native').mkdir()
    dest=out/'model';dest.mkdir()
    if args.run:
        m,s,p,b,c,st,path=load_run(engine,key,args,out)
        assert 'stage11' in st
        c.model_path=str(dest);s.model_path=str(dest);m.gats_joint=True;m.fine_resume_state=st
    else:m,s,p,b,c,st=engine.prepare_model(key,dest)
    begin=int(st['iteration']);end=begin+args.steps if args.command=='smoke' else args.end
    assert begin<end<=30000
    hooks=engine.Hooks(dest,m,s,p,b,c,st,key,end)
    fn=engine.instrument(hooks)
    from utils.timer import Timer
    timer=Timer();timer.start();torch.cuda.reset_peak_memory_stats()
    fn(c,c,m.args,p,[],[],[],None,-1,m,s,'fine',None,end,timer)
    if args.command=='smoke':
        from utils.exact_split_branchpoint import model_digest,digest
        expected=engine.reload_probe(m,s.getTrainCameras()[0],p,b)
        md=model_digest(m);adam=digest(m.optimizer.state_dict())
        # Same-process cold checkpoint reconstruction, including masks and Adam.
        cold,sc,pc,bc,cc,state=engine.load(dest,end,key);engine.opt_setup(cold,cc);cold.optimizer.load_state_dict(state['optimizer'])
        assert model_digest(cold)==md and digest(cold.optimizer.state_dict())==adam
        actual=engine.reload_probe(cold,sc.getTrainCameras()[0],pc,bc)
        errors={k:float((actual[k]-v).abs().max()) for k,v in expected.items()}
        assert max(errors.values())<2e-6,errors
        assert torch.equal(cold.stage11_boundary,m.stage11_boundary)
        assert torch.all(cold._opacity[cold.stage11_boundary]>=cold.stage11_floor[cold.stage11_boundary])
        write(out/'smoke.json',dict(pass_=True,Adam_updates=end-begin,checkpoint=end,model_Adam_reload_exact=True,mask_reload_exact=True,reload_errors=errors,TRAIN_only=True))
    write(out/'status.json',dict(state='COMPLETE',checkpoint=end,updates=end-begin,model=str(dest)))


def validate(args,out):
    import torch,numpy as np
    from utils.exact_split_branchpoint import model_digest,digest
    engine,key,data=setup(args,out)
    m,s,p,b,c,st,path=load_run(engine,key,args,out)
    before=(model_digest(m),digest(m.optimizer.state_dict()))
    assert before[0]==st['model_sha256']
    saved=torch.load(engine.ASSET/'boundary/selection.pth',map_location='cuda',weights_only=True)
    assert torch.equal(saved['mask'],m.stage11_boundary) and torch.equal(saved['floor'],m.stage11_floor)
    q=engine.reload_probe(m,s.getTrainCameras()[0],p,b)
    import copy
    cam=copy.copy(s.getTrainCameras()[1]);cam.time=s.getTrainCameras()[0].time
    other=engine.reload_probe(m,cam,p,b)
    assert torch.equal(q['transformed_points'],other['transformed_points'])
    if args.iteration>=10000:
        assert torch.all(m._opacity[m.stage11_boundary]>=m.stage11_floor[m.stage11_boundary])
        assert torch.count_nonzero(m._micro_offset[m.stage11_boundary])==0
        assert float(m._micro_offset.abs().max())<=.002*float(m.node_mlp.position_scale)+1e-7
    errors=None
    if args.oracle:
        expected=np.load(args.oracle)
        errors={k:float(np.abs(q[k].numpy()-expected[k]).max()) for k in q}
        assert max(errors.values())<2e-6,errors
    np.savez_compressed(out/'render_probe.npz',**{k:v.numpy() for k,v in q.items()})
    assert before==(model_digest(m),digest(m.optimizer.state_dict()))
    write(out/'validation.json',dict(pass_=True,scene=args.scene,checkpoint=args.iteration,Gaussians=len(m._xyz),geometry_nodes=len(m.deformation_graph.node_xyz),appearance_nodes=len(m.appearance_graph.node_xyz),model_sha256=before[0],Adam_sha256=before[1],mask_fraction=float(m.stage11_boundary.float().mean()),camera_independent=True,original_stage11_render_errors=errors,read_only=True,external_experiment_IO_blocked=True))


def evaluate(args,out):
    # GT cache is an explicitly requested post-training stage. It is prepared
    # before the training loader's GT-PLY prohibition is installed.
    import numpy as np,torch
    data=(args.data or ROOT.parent/'datasets/DynamicColon'/('data_'+args.scene)).resolve()
    limit=min(200,args.max_frames)
    if args.geometry:
        from tools.evaluate_rectum_ch_hd95 import read_xyz
        cache=out/'gt_cache';cache.mkdir()
        for idx in range(limit):np.save(cache/f'{idx:04d}.npy',read_xyz(data/'ply'/f'pc_{idx+1:04d}.ply'),allow_pickle=False)
    engine,key,data=setup(args,out)
    m,s,p,b,c,st,path=load_run(engine,key,args,out)
    from gaussian_renderer import render
    from utils.exact_split_branchpoint import model_digest,digest
    from tools.evaluate_texture_experiment import metrics,quantize
    from tools.evaluate_geometry_blending import benchmark
    from utils.stage8_depth import depth_statistics
    from utils.general_utils import build_rotation
    from PIL import Image
    before=(model_digest(m),digest(m.optimizer.state_dict()))
    rows=[]
    with torch.no_grad():
        perf=benchmark(m,s,p,b)
        for idx,cam in enumerate(s.getTestCameras()[:limit]):
            q=render(cam,m,p,b);rgb=q['render'].clamp(0,1);gt=cam.original_image[:3].cuda().float()
            row=metrics(torch.tensor(quantize(rgb),device='cuda').float()/255,torch.tensor(quantize(gt),device='cuda').float()/255,True)
            name=getattr(cam,'original_image_name',cam.image_name.removeprefix('test__'))
            alpha=torch.as_tensor(np.array(Image.open(data/'test/img'/(name+'.png')).convert('RGBA'))[...,3],device='cuda')
            row.update(depth_statistics(q['depth'],cam.original_depth.cuda().float().squeeze(),alpha))
            row.update(frame=cam.image_name,time=float(cam.time),RGB_L1=float((rgb-gt).abs().mean()),RGB_MSE=float((rgb-gt).square().mean()))
            dx=(rgb[:,:,1:]-rgb[:,:,:-1])-(gt[:,:,1:]-gt[:,:,:-1]);dy=(rgb[:,1:,:]-rgb[:,:-1,:])-(gt[:,1:,:]-gt[:,:-1,:])
            row['image_gradient_L1']=.5*float(dx.abs().mean()+dy.abs().mean())
            if args.geometry:
                from tools.evaluate_rectum_ch_hd95 import metric
                xyz=q['transformed_points'];_,logscale,rot=m.deform_geometry_from_nodes(cam.time)
                scale=m.scaling_activation(logscale).clamp_max(.05*m.spatial_lr_scale)
                op=m.get_opacity.squeeze();keep=op>0;v=op[keep]
                counts=(((v-v.min())/(v.max()-v.min()+1e-8))*10).long().clamp_min(1)
                gen=torch.Generator(device='cuda').manual_seed(1729+idx)
                noise=torch.randn((len(xyz),10,3),generator=gen,device='cuda')[keep]
                chosen=torch.arange(10,device='cuda')[None]<counts[:,None]
                samples=(torch.bmm(noise*scale[keep,None],build_rotation(rot[keep]).transpose(1,2))+xyz[keep,None])[chosen].cpu().numpy()
                row.update(metric(samples,np.load(cache/f'{idx:04d}.npy'),workers=2))
            rows.append(row)
            with (out/'per_view.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            write(out/'status.json',dict(state='EVALUATING',frames=len(rows),total=limit))
    assert before==(model_digest(m),digest(m.optimizer.state_dict()))
    mean={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ['frame','time']}
    mean.update(RMSE=float(np.sqrt(mean['RGB_MSE'])))
    mean['RMSE_MAE_ratio']=mean['RMSE']/mean['RGB_L1']
    write(out/'summary.json',dict(metrics=mean,performance=perf,frames=limit,partial=limit!=200,geometry=args.geometry,checkpoint=args.iteration,read_only=True))
    write(out/'status.json',dict(state='COMPLETE',frames=limit))


def render_video(args,out):
    import torch,imageio.v2 as imageio
    from PIL import Image
    from gaussian_renderer import render
    engine,key,data=setup(args,out)
    m,s,p,b,c,st,path=load_run(engine,key,args,out)
    from tools.evaluate_texture_experiment import quantize
    frames=out/'frames';frames.mkdir()
    with imageio.get_writer(out/'reconstruction.mp4',fps=args.fps,macro_block_size=1) as writer,torch.no_grad():
        for cam in s.getTestCameras()[:args.max_frames]:
            byte=quantize(render(cam,m,p,b)['render'].clamp(0,1)).transpose(1,2,0)
            writer.append_data(byte);Image.fromarray(byte).save(frames/(cam.image_name+'.png'))
    write(out/'status.json',dict(state='COMPLETE',video=str(out/'reconstruction.mp4')))


def main():
    ap=argparse.ArgumentParser(description='Packaged Stage11 U7 baseline; two DynamicColon scenes.')
    ap.add_argument('command',choices=['validate','smoke','train','evaluate','render'])
    ap.add_argument('--scene',choices=['transverse','rectum'],required=True)
    ap.add_argument('--data',type=Path,help='Dataset root with rootTRAIN200 and providedtest200')
    ap.add_argument('--out',type=Path,help='New output directory; existing output is never overwritten')
    ap.add_argument('--run',type=Path,help='Optional trained U7 model directory, instead of packed30k')
    ap.add_argument('--iteration',type=int,default=30000)
    ap.add_argument('--end',type=int,default=30000)
    ap.add_argument('--steps',type=int,default=3,help='TRAIN-only smoke updates from packed10k')
    ap.add_argument('--geometry',action='store_true',help='Also evaluateCH/HD95 against post-training GTPLY')
    ap.add_argument('--max-frames',type=int,default=200)
    ap.add_argument('--fps',type=int,default=25)
    ap.add_argument('--oracle',type=Path,help='Optional original deterministic TRAIN render probe for packaging audit')
    args=ap.parse_args()
    assert os.environ.get('CONDA_PREFIX'),'Use ./run.sh or an activated project Conda'
    assert 1<=args.max_frames<=200 and args.steps>0
    out=(args.out or ROOT/'output'/f'{args.scene}_{args.command}_{datetime.datetime.now():%Y%m%d_%H%M%S_%f}').resolve()
    out.mkdir(parents=True,exist_ok=False)
    try:
        if args.command in ('smoke','train'):train(args,out)
        elif args.command=='validate':validate(args,out)
        elif args.command=='evaluate':evaluate(args,out)
        else:render_video(args,out)
    except Exception as e:
        write(out/'status.json',dict(state='FAILED',error=repr(e)));raise
    print('Artifacts:',out)
if __name__=='__main__':main()
