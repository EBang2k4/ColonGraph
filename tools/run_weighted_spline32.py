"""Fresh shared Weighted-FPS prefix, then matched K4/J joint Spline32 to30k.
No retry, no held-out selection. Native core files are unchanged.
"""
import argparse, copy, gc, inspect, json, os, random, shutil, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tools.run_j_baseline30k import write, sha, configure
from tools.run_gats20_joint import seed, preserve_rng, rng_state, native_function, camera_plan
from tools.experiment_common import load_scene
from utils.exact_split_branchpoint import RAW_NAMES, digest, model_digest, restore_knn_reference
from utils.joint_spline32 import install, configure_optimizer
from utils.independent_temporal_rgb import active_basis
SCENE='c1_sigmoid2_t4_v4'
DEFAULT=ROOT/'output/experiments/weighted_spline32_joint'/SCENE/'run_20261013_startup03'
STEPS=[5000,10000,20000,30000]

class PrefixDone(Exception):pass

def update(m,c,**kw):
    for k,v in kw.items():setattr(c,k,v);setattr(m.args,k,v)

def parameters(m):
    return dict(appearance_parameters=m.spline.controls.numel(),trainable_groups={g['name']:sum(p.numel() for p in g['params']) for g in m.optimizer.param_groups},
                optimizer_steps={g['name']:[int(m.optimizer.state[p]['step']) for p in g['params'] if 'step' in m.optimizer.state.get(p,{})] for g in m.optimizer.param_groups})

def save(out,m,s,c,i):
    out.mkdir(parents=True,exist_ok=True);cfg=copy.deepcopy(c);cfg.model_path=str(out)
    (out/'cfg_args').write_text(str(cfg));old=s.model_path;s.model_path=str(out)
    try:s.save(i,'fine')
    finally:s.model_path=old
    state=dict(iteration=i,phase='after_optimizer',exact_raw={k:getattr(m,k).detach().cpu().clone() for k in RAW_NAMES},
        active_sh_degree=m.active_sh_degree,spatial_lr_scale=m.spatial_lr_scale,optimizer=m.optimizer.state_dict(),
        spline=m.spline.state_dict(),model_sha256=model_digest(m),spline_sha256=digest(m.spline.state_dict()),
        train_camera_order=[c.image_name for c in s.getTrainCameras()],**rng_state(),
        densification_stats={k:getattr(m,k).detach().cpu() for k in ['xyz_gradient_accum','denom','max_radii2D']})
    path=out/f'fine_state_{i}.pth';tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)
    write(out/f'checkpoint_{i}.json',dict(sha256=sha(path),model_sha256=state['model_sha256'],spline_sha256=state['spline_sha256'],optimizer_sha256=digest(state['optimizer']),**parameters(m)))

def load(path,i):
    m,s,p,b,c=load_scene(path,i)
    state=torch.load(path/f'fine_state_{i}.pth',map_location='cpu',weights_only=False)
    with torch.no_grad():
        for k,v in state['exact_raw'].items():getattr(m,k).copy_(v.cuda())
    m.active_sh_degree=state['active_sh_degree'];m.spatial_lr_scale=state['spatial_lr_scale']
    restore_knn_reference(m,c.gaussian_knn_reference)
    install(m);m.spline.load_state_dict(state['spline'])
    assert model_digest(m)==state['model_sha256'] and digest(m.spline.state_dict())==state['spline_sha256']
    return m,s,p,b,c,state

class Hooks:
    def __init__(self,study,out,m,s,p,b,c,branch):
        self.study,self.out,self.m,self.s,self.p,self.b,self.c,self.branch=study,out,m,s,p,b,c,branch
        self.plan=camera_plan(s.getTrainCameras());self.i=0;self.started=time.monotonic();self.overhead=0
        self.bases={str(cam.image_name):active_basis(cam.time,device='cuda') for cam in [*s.getTrainCameras(),*s.getTestCameras()]}
        write(out/'camera_plan.json',self.plan)
    def setup(self,m):
        configure_optimizer(m)
        if hasattr(m,'fine_resume_state'):
            for k,v in m.fine_resume_state['densification_stats'].items():setattr(m,k,v.cuda())
        if self.branch=='common':
            r=random.Random(0)
            for _ in range(100):r.randint(0,251)
            random.setstate(r.getstate())
    def begin(self,i):
        self.i=i;self.renders=0;self.backwards=0
        if i==5001:
            initial=dict(model_sha256=model_digest(self.m),spline_sha256=digest(self.m.spline.state_dict()),optimizer_sha256=digest(self.m.optimizer.state_dict()))
            expected=json.loads((self.study/'common_5000/checkpoint_5000.json').read_text())
            assert all(initial[k]==expected[k] for k in initial)
            write(self.out/'matched_initial.json',dict(pass_=True,**initial))
            random.seed(0)
            if self.branch=='J':
                update(self.m,self.c,ggas_variant='J_NO_MOTION_SELECTION',ggas_study_version=2,ggas_candidates=str(self.study/'candidates.pth'),
                    ggas_lr=.001,ggas_lambda=.01,ggas_ramp_end=6000,geometry_study_refresh='none')
                from utils.ggas import initialize
                with preserve_rng():initialize(self.m,self.out)
                (self.out/'cfg_args').write_text(str(self.c))
            if hasattr(self.m,'fine_resume_state'):del self.m.fine_resume_state
    def render(self,cam,m,*args,**kw):
        from gaussian_renderer import render
        self.renders+=1;expected=self.plan[self.i-1]
        assert str(cam.image_name)==str(expected['camera_name']) and float(cam.time)==expected['time']
        m.spline_basis=self.bases[str(cam.image_name)]
        with (self.out/'camera_sequence.jsonl').open('a') as f:f.write(json.dumps(expected)+'\n')
        result=render(cam,m,*args,**kw)
        if self.i<=3 or self.i%100==0:self.render_stats=dict(alpha_mean=float(result['alpha'].detach().mean()),visible_fraction=float((result['radii']>0).float().mean()))
        return result
    def backward(self,m,i,loss,rgb,depth,knn):
        self.backwards+=1
        if i<=3 or i%100==0 or i==5001:
            norms={k:float(getattr(m,k).grad.norm()) if getattr(m,k).grad is not None else 0. for k in ['_xyz','_scaling','_rotation','_opacity']}
            norms['spline32']=float(m.spline.controls.grad.norm())
            norms['node_geometry']=float(sum(p.grad.square().sum() for p in m.node_mlp.parameters() if p.grad is not None).sqrt())
            assert all(np.isfinite(v) for v in norms.values())
            assert m.features.grad is None and m._deformation_table.grad is None
            assert all(p.grad is None for p in m.appearance_model.parameters())
            if i==1:assert norms['_xyz']>0 and norms['spline32']>0 and norms['node_geometry']>0
            row=dict(iteration=i,total=float(loss),RGB_L1=float(rgb),depth=float(depth),KNN99=float(knn),grad_norms=norms,
                elapsed_training_seconds=time.monotonic()-self.started-self.overhead,**getattr(self,'render_stats',{}))
            with (self.out/'training_metrics.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    def after(self,m,s,cameras,i):
        assert self.renders==self.backwards==1;m.spline_basis=None
        if i%100==0:write(self.out/'status.json',dict(state='RUNNING',iteration=i,pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),elapsed_seconds=time.monotonic()-self.started))
        if i==100:
            start=time.monotonic()
            with preserve_rng():save(self.out,m,s,self.c,i)
            self.overhead+=time.monotonic()-start
        if i==2000:
            start=time.monotonic()
            with preserve_rng():
                save(self.study/'setup/warmup',m,s,self.c,i)
                helper=configure(self.study,SCENE);helper.reference()
                update(m,self.c,graph_forward_mode='g3',graph_forward_reference=str(helper.REF))
                m.initialize_graph_forward(str(helper.REF));m.add_graph_residual_optimizer_group()
                (self.out/'cfg_args').write_text(str(self.c))
            self.overhead+=time.monotonic()-start
        if i in STEPS:
            start=time.monotonic()
            with preserve_rng():
                save(self.out,m,s,self.c,i)
                from tools.spline_appearance_common import evaluate
                evaluate(self.out,i,m,s,self.p,self.b,spline=m.spline,save_frames=i==30000)
            self.overhead+=time.monotonic()-start
            write(self.out/'runtime.json',dict(training_seconds=time.monotonic()-self.started-self.overhead,setup_checkpoint_evaluation_seconds=self.overhead,total_seconds=time.monotonic()-self.started,
                peak_VRAM_MiB=torch.cuda.max_memory_allocated()/2**20,**parameters(m)))
        if i==5000 and self.branch=='common':
            m.pre_split_branchpoint_saved_at=5000
            write(self.out/'status.json',dict(state='PREFIX_COMPLETE',iteration=i));raise PrefixDone()
        if i==30000:
            steps=parameters(m)['optimizer_steps']
            assert steps['xyz']==[30000] and steps['spline32']==[30000]
            assert all(x==30000 for x in steps['deformation'])
            if self.branch=='J':assert steps['skinning']==[25000]
            write(self.out/'status.json',dict(state='TRAINED',iteration=i,actual_Adam_updates=30000,geometry_frozen=False))

def configuration(study):
    from arguments import ModelParams,ModelHiddenParams,OptimizationParams,PipelineParams
    parser=argparse.ArgumentParser();groups=[ModelParams(parser),ModelHiddenParams(parser),OptimizationParams(parser),PipelineParams(parser)]
    cfg=parser.parse_args([])
    template=json.loads((ROOT/'output/experiments/j_baseline30k/fresh_20261011'/SCENE/'configs/warmup.json').read_text())
    template['ModelParams']['model_path']=str(study/'common_5000')
    template['ModelHiddenParams'].update(node_sampling='fps_plus_importance',node_importance_extra_nodes=400,
        gaussian_knn_save=str(study/'setup/gaussian_knn99.pth'),fine_stop_iteration=-1,full_state_checkpoints='')
    for section in template.values():
        for k,v in section.items():setattr(cfg,k,v)
    write(study/'configuration.json',template)
    return cfg,groups

def common(study):
    import train
    seed();cfg,groups=configuration(study);train.args=cfg
    native=train.scene_reconstruction
    def wrapped(*args,**kwargs):
        if args[11]=='coarse':return native(*args,**kwargs)
        m,s=args[9:11];install(m)
        assert len(m.deformation_graph.node_xyz)==2400
        cfg.gaussian_knn_reference=str(study/'setup/gaussian_knn99.pth');m.args.gaussian_knn_reference=cfg.gaussian_knn_reference
        sampling=torch.load(study/'common_5000/weighted_fps_sampling.pth',map_location='cpu',weights_only=True)
        from utils.node_sampling import fps_plus_importance_sampling
        recomputed=fps_plus_importance_sampling(sampling['canonical_xyz'].cuda(),2000,400,sampling['importance'].cuda(),sampling['raw_score'].cuda()).cpu()
        checks=dict(coarse100=sampling['coarse_steps']==100,unique2400=sampling['selected_indices'].unique().numel()==2400,
                    selection_exact=torch.equal(recomputed,sampling['selected_indices']),nonconstant_importance=float(sampling['importance'].std())>0)
        write(study/'sampling_audit.json',dict(pass_=all(checks.values()),checks=checks,nodes=2400));assert all(checks.values())
        b=torch.tensor([0.,0.,0.],device='cuda');hooks=Hooks(study,study/'common_5000',m,s,args[3],b,cfg,'common')
        train.scene_reconstruction=native
        try:fn=native_function(hooks)
        finally:train.scene_reconstruction=wrapped
        (study/'native_joint_instrumented.py').write_text(hooks.patched_source)
        try:return fn(*args,**kwargs)
        except PrefixDone:return
    train.scene_reconstruction=wrapped
    lp,hp,op,pp=groups
    train.training(lp.extract(cfg),hp.extract(cfg),op.extract(cfg),pp.extract(cfg),[],[],[],None,-1,'weighted_spline32',None)

def candidates(study):
    helper=configure(study,SCENE)
    source=inspect.getsource(helper.candidates)
    assert source.count('assert n==2000')==1
    source=source.replace('assert n==2000','assert n==2400')
    (study/'weighted_candidates_builder.py').write_text(source)
    namespace=dict(helper.__dict__);exec(compile(source,str(study/'weighted_candidates_builder.py'),'exec'),namespace)
    namespace['candidates']()

def worker(study,branch):
    out=study/branch;out.mkdir(exist_ok=True)
    with (out/'FULL_RUN_STARTED').open('x') as f:f.write(str(os.getpid()))
    seed();m,s,p,b,c,state=load(study/'common_5000',5000)
    c.model_path=str(out);s.model_path=str(out)
    m.fine_resume_state=state
    hooks=Hooks(study,out,m,s,p,b,c,branch);fn=native_function(hooks)
    (out/'cfg_args').write_text(str(c))
    from utils.timer import Timer
    timer=Timer();timer.start();torch.cuda.reset_peak_memory_stats()
    fn(c,c,m.args,p,[],[],[],None,-1,m,s,'fine',None,30000,timer)

def postprocess(study):
    from tools.spline_appearance_common import timing
    results={}
    for branch in ['K4','J']:
        out=study/branch
        # Serialization / Adam integrity at every checkpoint, final benchmark after both GPUs finish.
        audits={}
        for i in [10000,20000,30000]:
            m,s,p,b,c,state=load(out,i);m.training_setup(c);configure_optimizer(m);m.add_graph_residual_optimizer_group()
            if m.skinning is not None:
                from utils.ggas import add_optimizer
                add_optimizer(m)
            m.optimizer.load_state_dict(state['optimizer'])
            expected=json.loads((out/f'checkpoint_{i}.json').read_text())
            assert sha(out/f'fine_state_{i}.pth')==expected['sha256']
            assert digest(m.optimizer.state_dict())==expected['optimizer_sha256']
            audits[str(i)]=dict(model_exact=True,spline_exact=True,Adam_exact=True,file_sha256_exact=True)
            if i==30000:
                perf=timing(m,s,p,b,m.spline);write(out/'performance.json',perf)
            del m,s,p,b,c,state;gc.collect();torch.cuda.empty_cache()
        write(out/'reload_integrity.json',dict(pass_=True,checkpoints=audits,note='Exact serialization audit; independent CUDA rasterization repeatability is not asserted.'))
        summary=json.loads((out/'evaluation/30000/summary.json').read_text())
        runtime=json.loads((out/'runtime.json').read_text());prefix=json.loads((study/'common_5000/runtime.json').read_text())
        runtime['full_trajectory_training_seconds']=prefix['training_seconds']+runtime['training_seconds']
        runtime['shared_prefix_training_seconds']=prefix['training_seconds'];write(out/'runtime.json',runtime)
        rows=[json.loads(x) for path in [study/'common_5000/camera_sequence.jsonl',out/'camera_sequence.jsonl'] for x in path.read_text().splitlines()]
        plan=json.loads((out/'camera_plan.json').read_text());assert len(rows)==30000 and rows==plan
        write(out/'camera_integrity.json',dict(pass_=True,views=30000,sha256=digest(rows)))
        results[branch]=dict(metrics=summary['test'],runtime=runtime,performance=perf,camera_sha256=digest(rows))
    assert results['K4']['camera_sha256']==results['J']['camera_sha256']
    write(study/'comparison.json',results)
    lines=['# Weighted FPS + G3-Surf + joint Spline32, 30k','',
      'Scene c1_sigmoid2_t4_v4; fresh100-coarse, shared joint prefix1..5k, matched K4/J continuations5001..30k. FPS2000 + importance400, G3 after2k, J after5k. Geometry and Spline32 both updated through30k. No held-out tuning.', '',
      '| Branch | PSNR | SSIM | LPIPS-A | LPIPS-V | Depth MSE | Train min (includes shared prefix) | FPS |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for branch,r in results.items():
        q=r['metrics'];lines.append(f"| {branch} | {q['PSNR']:.6f} | {q['SSIM']:.6f} | {q['LPIPS_A']:.6f} | {q['LPIPS_V']:.6f} | {q['DEPTH_FLOAT_VALID_MSE']:.9f} | {r['runtime']['full_trajectory_training_seconds']/60:.2f} | {r['performance']['FPS']:.2f} |")
    lines+=['','Depth is normalized pseudo-depth MSE, not metric anatomical depth. Historical frozen30k-geometry + extra30k Spline32: K4 PSNR28.211585, J28.149784. Those use different optimization budgets/protocols; no isolated causal Weighted-FPS claim against them.',
      'J remains selected blending baseline; this run compares K4/J without automatic promotion based on TEST results. Exact model/Spline/Adam reload and all30000 camera IDs audited.']
    (study/'report.md').write_text('\n'.join(lines)+'\n');write(study/'status.json',dict(state='COMPLETE',full_trajectories=2,iterations=30000))

def prepare(study):
    assert not study.exists();assert shutil.disk_usage(ROOT).free>12*1024**3
    (study/'logs').mkdir(parents=True);(study/'setup').mkdir();(study/'common_5000').mkdir()
    configuration(study)
    protocol=dict(scene=SCENE,branches=['K4','J'],appearance='Independent cubic Spline32',training='joint geometry+appearance fine1..30000, 30000 Adam updates',
        initialization='fresh common100-coarse; native R6 initialized for serialization only, excluded from fine forward and optimizer',
        sampler='FPS2000+top importance400; fresh100coarse total XYZ gradient deviation, visible only, P5/P95 normalization',
        G3_start=2001,J_start=5001,common_prefix='same exact model/Adam/RNG/G3/KNN/Spline through5k, no reused learned source',
        geometry_frozen=False,knn99_weight=.01,spline_lr=.0025,Adam_eps=1e-15,gradient_value_clip=.5,
        spline_controls=32,spline_initialization='postcoarse canonical RGB replicated32; no additive R6/base term',
        timestamps='original idx/288',camera_protocol='native seed0 consume100coarse draws; reset0 at5001 identically',
        checkpoints=STEPS,full_runs=2,no_auto_retry=True,held_out_selection=False,
        historical_reference='28.211585dB used geometry30k THEN frozenSplineadditional30k; joint is a different user-selected protocol')
    write(study/'protocol.json',protocol)
    core=json.loads((ROOT/'output/experiments/j_baseline30k/fresh_20261011/training_implementation_manifest.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in core.items());write(study/'native_implementation_manifest.json',core)
    data=ROOT.parent/'ColonSplat/data/coloncrafter'/SCENE
    write(study/'data_manifest.json',{str(p):sha(p) for p in [data/'pose.txt',data/'points3d.ply',*sorted((data/'color').glob('*.png')),*sorted((data/'depth').glob('*.npy'))]})
    write(study/'code_manifest.json',{str(ROOT/p):sha(ROOT/p) for p in ['tools/run_weighted_spline32.py','utils/joint_spline32.py','utils/independent_temporal_rgb.py','utils/node_sampling.py','tools/run_gats20_joint.py','tools/run_cross_scene_aj.py']})
    write(study/'status.json',dict(state='PREPARED'))

def supervise(study):
    with (study/'PIPELINE_STARTED').open('x') as f:f.write(str(os.getpid()))
    for action in ['common','candidates']:
        write(study/'status.json',dict(state='RUNNING',stage=action,pid=os.getpid()))
        with (study/'logs'/(action+'.log')).open('w') as f:
            subprocess.run([sys.executable,'-u',__file__,action,'--out',str(study)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'),stdout=f,stderr=subprocess.STDOUT,check=True)
    processes=[]
    for gpu,branch in enumerate(['K4','J']):
        out=study/branch;out.mkdir();f=(study/'logs'/(branch+'.log')).open('w')
        p=subprocess.Popen([sys.executable,'-u',__file__,'worker','--out',str(study),'--branch',branch],cwd=ROOT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=f,stderr=subprocess.STDOUT);processes.append((branch,p,f))
    write(study/'processes.json',{k:dict(pid=p.pid,gpu=i) for i,(k,p,f) in enumerate(processes)})
    write(study/'status.json',dict(state='RUNNING',stage='paired_continuations'))
    codes={}
    for k,p,f in processes:codes[k]=p.wait();f.close()
    assert not any(codes.values()),codes
    for p,h in json.loads((study/'native_implementation_manifest.json').read_text()).items():assert sha(ROOT/p)==h
    with (study/'logs/postprocess.log').open('w') as f:
        subprocess.run([sys.executable,'-u',__file__,'postprocess','--out',str(study)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'),stdout=f,stderr=subprocess.STDOUT,check=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','supervise','common','candidates','worker','postprocess']);parser.add_argument('--out',type=Path,default=DEFAULT);parser.add_argument('--branch',choices=['K4','J']);a=parser.parse_args()
    assert os.environ.get('CONDA_PREFIX')=='/home/robot/colongraph/.env','Use project Conda'
    try:
        if a.action=='worker':worker(a.out,a.branch)
        else:globals()[a.action](a.out)
    except Exception as e:
        if a.out.exists():write(a.out/((a.branch+'/status.json') if a.action=='worker' else 'status.json'),dict(state='FAILED',stage=a.action,error=repr(e),pid=os.getpid()))
        raise
