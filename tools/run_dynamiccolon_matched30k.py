"""Fresh DynamicColon RGBD coarse100; matched Spline32 x FPS/K4/J + FPS R6 K4/J.
Only process-local adapters; historical native code and artifacts remain intact.
"""
import argparse, copy, gc, inspect, json, os, random, shutil, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from tools import run_matched_fps_spline32 as matched
from tools import run_weighted_spline32 as base
from tools.run_gats20_joint import restore_rng
from tools.experiment_common import load_scene
from scene.deformation_graph import SparseDeformationGraph
from utils.node_sampling import build_gaussian_node_knn
from utils.exact_split_branchpoint import RAW_NAMES, model_digest, restore_knn_reference
from utils.joint_spline32 import install, configure_optimizer
SCENE = 'data_transverse'
DATA = ROOT.parent / 'datasets/DynamicColon' / SCENE
DEFAULT = Path('/media/robot/data/colongraph_experiments/dynamiccolon_matched30k/data_transverse/run_20261014')
STUDIES = ['old', 'weight', 'r6_old']
STEPS = [5000, 10000, 20000, 30000]
write, sha, digest = base.write, base.sha, base.digest

# A runtime guard covers every Python file-open path, including indirect PLY readers.
def no_gt_ply(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        p = Path(os.fsdecode(args[0])).absolute()
        if p.suffix.lower() == '.ply' and DATA in p.parents:
            raise RuntimeError('GT PLY access forbidden by user: ' + str(p))
sys.addaudithook(no_gt_ply)


def configuration(study):
    from arguments import ModelParams, ModelHiddenParams, OptimizationParams, PipelineParams
    parser = argparse.ArgumentParser()
    groups = [ModelParams(parser), ModelHiddenParams(parser), OptimizationParams(parser), PipelineParams(parser)]
    cfg = parser.parse_args([])
    template = json.loads((ROOT/'output/experiments/j_baseline30k/fresh_20261011/c1_sigmoid2_t4_v4/configs/warmup.json').read_text())
    template['ModelParams'].update(source_path=str(DATA), model_path=str(study/'common_5000'),
                                 dynamic_point_initialization='depth', init_pts=200000)
    template['ModelHiddenParams'].update(node_sampling='fps_plus_importance', node_importance_extra_nodes=400,
        gaussian_knn_save=str(study/'setup/gaussian_knn99.pth'), fine_stop_iteration=-1, full_state_checkpoints='')
    for section in template.values():
        for k, v in section.items(): setattr(cfg, k, v)
    write(study/'configuration.json', template)
    return cfg, groups


def configure(study, scene=SCENE):
    from tools.run_j_baseline30k import configure as original
    helper = original(study, scene)
    helper.DATA = DATA
    return helper


def parameters(m):
    appearance = m.spline.controls.numel() if hasattr(m,'spline') else (
        m.features.numel()+m._deformation_table.numel()+sum(p.numel() for p in m.appearance_model.parameters()))
    return dict(appearance='Spline32' if hasattr(m,'spline') else 'R6', appearance_parameters=appearance,
        trainable_groups={g['name']:sum(p.numel() for p in g['params']) for g in m.optimizer.param_groups},
        optimizer_steps={g['name']:[int(m.optimizer.state[p]['step']) for p in g['params']
            if 'step' in m.optimizer.state.get(p,{})] for g in m.optimizer.param_groups})


def save(out,m,s,c,i):
    out.mkdir(parents=True,exist_ok=True)
    cfg=copy.deepcopy(c);cfg.model_path=str(out)
    from utils.bundle_paths import portable
    for k,v in vars(cfg).items():setattr(cfg,k,portable(v))
    (out/'cfg_args').write_text(str(cfg));old=s.model_path;s.model_path=str(out)
    try:s.save(i,'fine')
    finally:s.model_path=old
    state=dict(iteration=i,phase='after_optimizer',exact_raw={k:getattr(m,k).detach().cpu().clone() for k in RAW_NAMES},
        active_sh_degree=m.active_sh_degree,spatial_lr_scale=m.spatial_lr_scale,optimizer=m.optimizer.state_dict(),
        model_sha256=model_digest(m),train_camera_order=[cam.image_name for cam in s.getTrainCameras()],**base.rng_state(),
        densification_stats={k:getattr(m,k).detach().cpu() for k in ['xyz_gradient_accum','denom','max_radii2D']})
    if hasattr(m,'spline'):state.update(spline=m.spline.state_dict(),spline_sha256=digest(m.spline.state_dict()))
    path=out/f'fine_state_{i}.pth';tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)
    checkpoint=dict(sha256=sha(path),model_sha256=state['model_sha256'],optimizer_sha256=digest(state['optimizer']),**parameters(m))
    if hasattr(m,'spline'):checkpoint['spline_sha256']=state['spline_sha256']
    write(out/f'checkpoint_{i}.json',checkpoint)
    root=next(p for p in [out,*out.parents] if (p/'matrix_protocol.json').exists())
    cache=root/'immutable_native'
    for name in ['deformation.pth','deformation_table.pth','appearance.pth']:
        p=out/'point_cloud'/f'iteration_{i}'/name;target=cache/(sha(p)+'_'+name)
        if target.exists():p.unlink();os.link(target,p)
        else:os.link(p,target)


def load(path,i,appearance=None):
    m,s,p,b,c=load_scene(path,i)
    state=torch.load(path/f'fine_state_{i}.pth',map_location='cpu',weights_only=False)
    with torch.no_grad():
        for k,v in state['exact_raw'].items():getattr(m,k).copy_(v.cuda())
    m.active_sh_degree=state['active_sh_degree'];m.spatial_lr_scale=state['spatial_lr_scale']
    restore_knn_reference(m,c.gaussian_knn_reference)
    if appearance!='R6' and 'spline' in state:
        install(m);m.spline.load_state_dict(state['spline'])
        assert digest(m.spline.state_dict())==state['spline_sha256']
    else:m.gats_joint=True
    assert model_digest(m)==state['model_sha256']
    assert s.gt_plys is None and s.initialization_audit['gt_ply_read'] is False
    assert len(s.getTrainCameras())==160 and len(s.getTestCameras())==40
    return m,s,p,b,c,state


# Reuse native metric functions; camera names and metadata are dataset-specific.
from tools import spline_appearance_common as evaluation

def all_cameras(scene):
    return sorted([*scene.getTrainCameras(),*scene.getTestCameras()],key=lambda c:c.uid)
evaluation.all_cameras=all_cameras
source=inspect.getsource(evaluation.evaluate)
assert source.count('test_views=36,train_views=252')==1
source=source.replace('test_views=36,train_views=252','test_views=len(held),train_views=len(train)')
namespace=dict(evaluation.__dict__);exec(compile(source,__file__+':dynamic_evaluate','exec'),namespace)
evaluation.evaluate=namespace['evaluate']


class Hooks(base.Hooks):
    def __init__(self,*args,initial_state=None,smoke=False,**kw):
        super().__init__(*args,**kw)
        self.initial_state,self.smoke=initial_state,smoke
    def setup(self,m):
        if hasattr(m,'spline'):configure_optimizer(m)
        if hasattr(m,'fine_resume_state'):
            for k,v in m.fine_resume_state['densification_stats'].items():setattr(m,k,v.cuda())
    def begin(self,i):
        self.i=i;self.renders=0;self.backwards=0
        if i==1 and self.initial_state is not None:
            restore_rng(self.initial_state)
            for k,v in self.initial_state['geometry_gradients'].items():getattr(self.m,k).grad=None if v is None else v.cuda().clone()
        if i==5001:
            initial=dict(model_sha256=model_digest(self.m),optimizer_sha256=digest(self.m.optimizer.state_dict()))
            if hasattr(self.m,'spline'):initial['spline_sha256']=digest(self.m.spline.state_dict())
            expected=json.loads((self.study/'common_5000/checkpoint_5000.json').read_text())
            assert all(initial[k]==expected[k] for k in initial)
            write(self.out/'matched_initial.json',dict(pass_=True,**initial))
            random.seed(0)
            if self.branch=='J':
                base.update(self.m,self.c,ggas_variant='J_NO_MOTION_SELECTION',ggas_study_version=2,
                    ggas_candidates=str(self.study/'candidates.pth'),ggas_lr=.001,ggas_lambda=.01,ggas_ramp_end=6000,geometry_study_refresh='none')
                from utils.ggas import initialize
                with base.preserve_rng():initialize(self.m,self.out)
                (self.out/'cfg_args').write_text(str(self.c))
            if hasattr(self.m,'fine_resume_state'):del self.m.fine_resume_state
    def render(self,cam,m,*args,**kw):
        from gaussian_renderer import render
        self.renders+=1;expected=self.plan[self.i-1]
        assert str(cam.image_name)==str(expected['camera_name']) and float(cam.time)==expected['time']
        if hasattr(m,'spline'):m.spline_basis=self.bases[str(cam.image_name)]
        with (self.out/'camera_sequence.jsonl').open('a') as f:f.write(json.dumps(expected)+'\n')
        result=render(cam,m,*args,**kw)
        if self.i<=3 or self.i%100==0:self.render_stats=dict(alpha_mean=float(result['alpha'].detach().mean()),visible_fraction=float((result['radii']>0).float().mean()))
        return result
    def backward(self,m,i,loss,rgb,depth,knn):
        self.backwards+=1
        if i<=3 or i%100==0 or i==5001:
            keys=['_xyz','_scaling','_rotation','_opacity']+([] if hasattr(m,'spline') else ['features','_deformation_table'])
            norms={k:float(getattr(m,k).grad.norm()) if getattr(m,k).grad is not None else 0. for k in keys}
            norms['node_geometry']=float(sum(p.grad.square().sum() for p in m.node_mlp.parameters() if p.grad is not None).sqrt())
            if hasattr(m,'spline'):
                norms['spline32']=float(m.spline.controls.grad.norm())
                assert m.features.grad is None and m._deformation_table.grad is None
                assert all(p.grad is None for p in m.appearance_model.parameters())
            else:norms['R6']=float(sum(p.grad.square().sum() for p in m.appearance_model.parameters() if p.grad is not None).sqrt())
            assert all(np.isfinite(v) for v in norms.values())
            if i==1:assert norms['_xyz']>0 and norms['node_geometry']>0 and norms['spline32' if hasattr(m,'spline') else 'features']>0
            row=dict(iteration=i,total=float(loss),RGB_L1=float(rgb),depth=float(depth),KNN99=float(knn),grad_norms=norms,
                elapsed_training_seconds=time.monotonic()-self.started-self.overhead,**getattr(self,'render_stats',{}))
            with (self.out/'training_metrics.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    def after(self,m,s,cameras,i):
        assert self.renders==self.backwards==1
        if hasattr(m,'spline'):m.spline_basis=None
        if i%100==0:write(self.out/'status.json',dict(state='RUNNING',iteration=i,pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))
        if self.smoke:return
        start=time.monotonic()
        if i==100:
            with base.preserve_rng():save(self.out,m,s,self.c,i)
        if i==2000:
            with base.preserve_rng():
                save(self.study/'setup/warmup',m,s,self.c,i)
                helper=configure(self.study);helper.reference()
                base.update(m,self.c,graph_forward_mode='g3',graph_forward_reference=str(helper.REF))
                m.initialize_graph_forward(str(helper.REF));m.add_graph_residual_optimizer_group()
                (self.out/'cfg_args').write_text(str(self.c))
        if i in STEPS:
            with base.preserve_rng():
                save(self.out,m,s,self.c,i)
                evaluation.evaluate(self.out,i,m,s,self.p,self.b,spline=getattr(m,'spline',None),save_frames=i==30000)
        if i==100 or i==2000 or i in STEPS:self.overhead+=time.monotonic()-start
        if i in STEPS:
            write(self.out/'runtime.json',dict(training_seconds=time.monotonic()-self.started-self.overhead,
                setup_checkpoint_evaluation_seconds=self.overhead,total_seconds=time.monotonic()-self.started,
                peak_VRAM_MiB=torch.cuda.max_memory_allocated()/2**20,**parameters(m)))
        if i==5000 and self.branch=='common':
            m.pre_split_branchpoint_saved_at=5000
            write(self.out/'status.json',dict(state='PREFIX_COMPLETE',iteration=i));raise base.PrefixDone()
        if i==30000:
            steps=parameters(m)['optimizer_steps'];assert steps['xyz']==[30000]
            assert steps['spline32' if hasattr(m,'spline') else 'f_dc']==[30000]
            assert all(x==30000 for x in steps['deformation'])
            if self.branch=='J':assert steps['skinning']==[25000]
            write(self.out/'status.json',dict(state='TRAINED',iteration=i,actual_Adam_updates=30000,geometry_frozen=False))


# Overrides apply only inside this new launcher process.
base.configuration=configuration;base.configure=configure;base.SCENE=SCENE
base.save=save;base.load=load;base.parameters=parameters;base.Hooks=Hooks
matched.SCENE=SCENE;matched.compact_save=save


def sampler_initial(root,sampler,study):
    if sampler!='r6_old':return matched.sampler_initial(root,sampler,study)
    base.seed();m,s,p,b,c,state=load(root/'initial',1,appearance='R6')
    selected=torch.load(root/'initial/weighted_fps_sampling.pth',map_location='cpu',weights_only=True)['selected_indices'][:2000].cuda()
    xyz=m._xyz.detach()[selected];ids,w=build_gaussian_node_knn(m._xyz.detach(),xyz,4)
    m.deformation_graph=SparseDeformationGraph(xyz,ids,w)
    # Fixed appearance K4 uses the original FPS2000 graph, before any G3 repair.
    m.appearance_graph=None
    with base.preserve_rng():m.initialize_appearance_model()
    m.gats_joint=True;c.model_path=str(study/'common_5000');s.model_path=c.model_path
    base.update(m,c,node_sampling='fps',node_importance_extra_nodes=0,gaussian_knn_reference=str(study/'setup/gaussian_knn99.pth'),gaussian_knn_save='')
    expected=json.loads((root/'initial/initial_integrity.json').read_text())
    checks=dict(raw_sha256=digest({k:getattr(m,k) for k in RAW_NAMES}),node_mlp_sha256=digest(m.node_mlp.state_dict()),
        KNN99_sha256=digest(m.closest_point_indices),geometry_gradients_sha256=digest(state['geometry_gradients']))
    assert all(checks[k]==expected[k] for k in checks)
    assert len(m.appearance_graph.node_xyz)==2000 and torch.equal(m.appearance_graph.gaussian_node_indices,ids)
    torch.save(selected.cpu(),study/'selected_node_ids.pth')
    write(study/'initial_integrity.json',dict(pass_=True,**checks,nodes=2000,fine_updates_before_start=0,
        appearance='untrained original fixed-initial K4 R6; seed0 decoder zero output; no Spline adapter'))
    (study/'common_5000/cfg_args').write_text(str(c))
    return m,s,p,b,c,state


def prefix(root,sampler,smoke=False):
    study=root/('preflight/'+sampler if smoke else sampler);out=study/'common_5000'
    if smoke:
        (study/'setup').mkdir(parents=True);out.mkdir()
        os.link(root/'setup/gaussian_knn99.pth',study/'setup/gaussian_knn99.pth')
    else:(out/'PREFIX_STARTED').write_text(str(os.getpid()))
    m,s,p,b,c,state=sampler_initial(root,sampler,study)
    hooks=Hooks(study,out,m,s,p,b,c,'common',initial_state=state,smoke=smoke)
    fn=base.native_function(hooks);(study/'native_joint_instrumented.py').write_text(hooks.patched_source)
    from utils.timer import Timer
    timer=Timer();timer.start();torch.cuda.reset_peak_memory_stats()
    try:fn(c,c,m.args,p,[],[],[],None,-1,m,s,'fine',None,101 if smoke else 30000,timer)
    except base.PrefixDone:assert not smoke
    if smoke:
        rows=[json.loads(x) for x in (out/'camera_sequence.jsonl').read_text().splitlines()];assert rows==hooks.plan[:101]
        with base.preserve_rng():save(out,m,s,c,101)
        expected=(model_digest(m),digest(m.optimizer.state_dict()),digest(m.spline.state_dict()) if hasattr(m,'spline') else None)
        del hooks,fn,m,s,p,b,c,state;gc.collect();torch.cuda.empty_cache()
        m,s,p,b,c,state=load(out,101);m.training_setup(c)
        if hasattr(m,'spline'):configure_optimizer(m)
        m.optimizer.load_state_dict(state['optimizer'])
        assert model_digest(m)==expected[0] and digest(m.optimizer.state_dict())==expected[1]
        if expected[2]:assert digest(m.spline.state_dict())==expected[2]
        m.fine_resume_state=state;hooks=Hooks(study,out,m,s,p,b,c,'SMOKE',smoke=True);fn=base.native_function(hooks)
        timer=Timer();timer.start();fn(c,c,m.args,p,[],[],[],None,-1,m,s,'fine',None,104,timer)
        rows=[json.loads(x) for x in (out/'camera_sequence.jsonl').read_text().splitlines()];assert rows==hooks.plan[:104]
        assert parameters(m)['optimizer_steps']['xyz']==[104]
        write(study/'integrity.json',dict(pass_=True,disposable_updates=101,native_resume_updates=3,camera_replay_exact=True,
            model_exact=True,Adam_exact=True,GT_PLY_forbidden=True,TRAIN=160,TEST=40,note='Primary trajectories restart from shared coarse100/fine0.'))


def candidates(root,sampler):
    if sampler=='r6_old':
        helper=configure(root/sampler);helper.candidates()
    else:matched.candidates(root,sampler)


def prepare(root):
    assert not root.exists()
    assert shutil.disk_usage(root.parent if root.parent.exists() else '/media/robot/data').free>11*1024**3
    root.mkdir(parents=True)
    for folder in ['logs','setup','initial','immutable_native']:(root/folder).mkdir()
    for sampler in STUDIES:
        for folder in ['logs','setup','common_5000','K4','J']:(root/sampler/folder).mkdir(parents=True)
    frames=json.loads((DATA/'transforms.json').read_text())['frames'];assert len(frames)==200
    paths=[DATA/'transforms.json',*[DATA/f[key] for f in frames for key in ['file_path','depth_path']]]
    assert all(p.exists() for p in paths)
    write(root/'data_manifest.json',{str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in paths})
    protocol=dict(scene=SCENE,dataset=str(DATA),branches=['K4_old_Spline32','K4_weight_Spline32','J_old_Spline32','J_weight_Spline32','K4_old_R6','J_old_R6'],
        full_trajectories=6,fresh_initialization='one fresh RGB+depth backprojection from first TRAIN frame r_0001; native coarse100; shared raw/nodeMLP/KNN/RNG/fine0',
        GT_PLY='forbidden file-open guard in every worker; no GT PLY initialization, loss or evaluation',init_pts_cap=200000,
        samplers={'old':'FPS2000','weight':'same FPS2000 backbone + importance400; same coarse100 totalXYZ gradient deviation'},
        appearances={'Spline32':'joint independent per-Gaussian cubic32 from fine1, constantLR.0025','R6':'original fixed-initial appearance K4 FPS2000, native LR schedule'},
        fine_iterations=30000,geometry_trainable_through=30000,appearance_trainable_through=30000,terminal_Adam_included=True,
        G3='built independently from own sampler/appearance warmup2k; original E4 surface topology +16TRAIN trajectory gate',
        J='GGAS-v2 spatial-only learned connectedK8 starts5001; LR.001 KL.01 ramp6000; no v3/v4 loss',
        KNN99_weight=.01,depth_weight=.25,RGB_loss='same native L1+TV.001+colorvariance.0001',
        split='native DynamicColon every5, (idx-1)%5==0 held out',TRAIN=160,TEST=40,resolution=[650,650],times='original idx/200',
        camera='same30000 draws; seed0 consumes100coarse draws from160TRAIN then resets0at5001; every camera/time asserted',
        paired_prefixes='each sampler/appearance has one1..5k prefix shared within K4/J pair; counted per full-trajectory runtime',
        checkpoints=STEPS,loss_and_LR='unchanged previous matched protocol',no_auto_retry=True,held_out_selection=False,
        final_metrics=['PSNR','SSIM','LPIPS_A','LPIPS_V','normalized_depth_MSE','runtime','FPS'],
        depth_note='normalized image depth MSE; physical mm and anatomical stability not certified',conda=os.environ['CONDA_PREFIX'],output_storage=str(root))
    write(root/'matrix_protocol.json',protocol)
    paths=['train.py','scene/gaussian_model.py','scene/dataset_readers.py','scene/endo_loader.py','scene/__init__.py',
        'utils/rgbd_initialization.py','utils/node_sampling.py','utils/joint_spline32.py','utils/ggas.py','utils/ggas_v2.py',
        'tools/run_dynamiccolon_matched30k.py','tools/run_matched_fps_spline32.py','tools/run_weighted_spline32.py','tools/run_gats20_joint.py',
        'tools/run_cross_scene_aj.py','tools/build_graph_forward_reference.py']
    write(root/'code_manifest.json',{str(ROOT/p):sha(ROOT/p) for p in paths})
    configuration(root)
    write(root/'status.json',dict(state='PREPARED',full_trajectories=6))


def run_phase(root,commands,stage):
    write(root/'status.json',dict(state='RUNNING',stage=stage,pid=os.getpid()))
    jobs=[];processes={}
    for gpu,(name,argv) in enumerate(commands):
        f=(root/'logs'/(name+'.log')).open('x')
        cmd=[sys.executable,'-u',__file__,*argv,'--out',str(root)]
        p=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=f,stderr=subprocess.STDOUT)
        jobs.append((name,p,f));processes[name]=dict(pid=p.pid,gpu=gpu,command=cmd)
    write(root/('processes_'+stage+'.json'),processes)
    codes={}
    for name,p,f in jobs:codes[name]=p.wait();f.close()
    write(root/('exit_'+stage+'.json'),codes)
    assert not any(codes.values()),codes


def postprocess(root):
    results={}
    from tools.evaluate_geometry_blending import benchmark
    for sampler in STUDIES:
        for branch in ['K4','J']:
            study=root/sampler;out=study/branch;audits={}
            for i in [10000,20000,30000]:
                m,s,p,b,c,state=load(out,i);m.training_setup(c)
                if hasattr(m,'spline'):configure_optimizer(m)
                m.add_graph_residual_optimizer_group()
                if m.skinning is not None:
                    from utils.ggas import add_optimizer
                    add_optimizer(m)
                m.optimizer.load_state_dict(state['optimizer'])
                expected=json.loads((out/f'checkpoint_{i}.json').read_text())
                assert sha(out/f'fine_state_{i}.pth')==expected['sha256']
                assert digest(m.optimizer.state_dict())==expected['optimizer_sha256']
                audits[str(i)]=dict(model_exact=True,Adam_exact=True,file_sha256_exact=True)
                if i==30000:
                    perf=evaluation.timing(m,s,p,b,m.spline) if hasattr(m,'spline') else benchmark(m,s,p,b)
                    write(out/'performance.json',perf)
                del m,s,p,b,c,state;gc.collect();torch.cuda.empty_cache()
            write(out/'reload_integrity.json',dict(pass_=True,checkpoints=audits))
            runtime=json.loads((out/'runtime.json').read_text());prefix_time=json.loads((study/'common_5000/runtime.json').read_text())
            runtime.update(full_trajectory_training_seconds=runtime['training_seconds']+prefix_time['training_seconds'],shared_prefix_training_seconds=prefix_time['training_seconds'])
            write(out/'runtime.json',runtime)
            rows=[json.loads(x) for f in [study/'common_5000/camera_sequence.jsonl',out/'camera_sequence.jsonl'] for x in f.read_text().splitlines()]
            assert len(rows)==30000 and rows==json.loads((out/'camera_plan.json').read_text())
            write(out/'camera_integrity.json',dict(pass_=True,views=30000,sha256=digest(rows)))
            name=branch+'_'+sampler
            results[name]=dict(metrics=json.loads((out/'evaluation/30000/summary.json').read_text())['test'],runtime=runtime,performance=perf,camera_sha256=digest(rows))
    assert len({r['camera_sha256'] for r in results.values()})==1
    write(root/'comparison.json',results)
    lines=['# DynamicColon transverse — six matched full30k runs','','Fresh RGB/depth coarse100; no GT PLY read. Native split160TRAIN/40TEST. Geometry and appearance active through30k. Same cameras/losses/LRs; new G3 at2k and K4/J fork5k for each sampler/appearance prefix.','',
        '| Branch | PSNR | SSIM | LPIPS-A | LPIPS-V | normalized depth MSE | Train min | FPS |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,r in results.items():
        q=r['metrics'];lines.append(f"| {name} | {q['PSNR']:.6f} | {q['SSIM']:.6f} | {q['LPIPS_A']:.6f} | {q['LPIPS_V']:.6f} | {q['DEPTH_FLOAT_VALID_MSE']:.9f} | {r['runtime']['full_trajectory_training_seconds']/60:.2f} | {r['performance']['FPS']:.2f} |")
    lines+=['','Shared5k compute included in each trajectory runtime. TEST does not select checkpoints or thresholds. Weighted sampling adds400nodes; R6/Spline comparisons include own2k G3 rebuilt from each appearance trajectory. Depth is normalized image depth, not verified physical mm.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    write(root/'status.json',dict(state='COMPLETE',full_trajectories=6,iterations=30000))


def supervise(root):
    with (root/'PIPELINE_STARTED').open('x') as f:f.write(str(os.getpid()))
    run_phase(root,[('fresh_coarse',['initial'])],'fresh_coarse')
    initial=json.loads((root/'initial/initial_integrity.json').read_text())
    assert initial['pass_'] and sha(root/'initial/fine_state_1.pth')==initial['file_sha256']
    for sampler in STUDIES:os.link(root/'setup/gaussian_knn99.pth',root/sampler/'setup/gaussian_knn99.pth')
    for subset in [STUDIES[:2],STUDIES[2:]]:
        run_phase(root,[(s+'_smoke',['smoke','--sampler',s]) for s in subset],'preflight_'+'_'.join(subset))
    assert all(json.loads((root/'preflight'/s/'integrity.json').read_text())['pass_'] for s in STUDIES)
    for subset in [STUDIES[:2],STUDIES[2:]]:
        run_phase(root,[(s+'_prefix',['prefix','--sampler',s]) for s in subset],'prefix_'+'_'.join(subset))
    for subset in [STUDIES[:2],STUDIES[2:]]:
        run_phase(root,[(s+'_candidates',['candidates','--sampler',s]) for s in subset],'candidates_'+'_'.join(subset))
    for branch in ['K4','J']:
        for subset in [STUDIES[:2],STUDIES[2:]]:
            run_phase(root,[(branch+'_'+s,['worker','--sampler',s,'--branch',branch]) for s in subset],branch+'_'+'_'.join(subset))
    for p,h in json.loads((root/'code_manifest.json').read_text()).items():assert sha(p)==h
    for p,v in json.loads((root/'data_manifest.json').read_text()).items():assert sha(p)==v['sha256']
    run_phase(root,[('postprocess',['postprocess'])],'postprocess')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','initial','smoke','prefix','candidates','worker','supervise','postprocess'])
    parser.add_argument('--out',type=Path,default=DEFAULT);parser.add_argument('--sampler',choices=STUDIES);parser.add_argument('--branch',choices=['K4','J']);a=parser.parse_args()
    assert os.environ.get('CONDA_PREFIX')=='/home/robot/colongraph/.env','Use project Conda'
    try:
        if a.action=='initial':matched.fresh_initial(a.out)
        elif a.action in ['prefix','smoke']:prefix(a.out,a.sampler,smoke=a.action=='smoke')
        elif a.action=='candidates':candidates(a.out,a.sampler)
        elif a.action=='worker':base.worker(a.out/a.sampler,a.branch)
        else:globals()[a.action](a.out)
    except Exception as e:
        if a.out.exists():write(a.out/'status.json',dict(state='FAILED',stage=a.action,sampler=a.sampler,branch=a.branch,error=repr(e),pid=os.getpid(),no_automatic_retry=True))
        raise
