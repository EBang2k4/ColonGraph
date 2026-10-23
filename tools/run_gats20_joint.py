"""Two isolated native joint30k GATS experiments. No auto retry or tuning."""
import argparse, copy, gc, hashlib, inspect, json, os, random, shutil, subprocess, sys, time, types
from contextlib import contextmanager
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tools.run_j_baseline30k import write,sha,configure
from tools.experiment_common import load_scene
from utils.exact_split_branchpoint import digest,model_digest,RAW_NAMES,restore_knn_reference
from utils.gats20 import GATS20,initialize_dictionary,install,configure_optimizer,install_lifecycle
SOURCE=ROOT/'output/experiments/j_baseline30k/fresh_20261011/c1_sigmoid2_t4_v4'
SCENE='c1_sigmoid2_t4_v4';VARIANTS=['J_R6_GATS20','J_GATS20_ONLY'];STEPS=[5000,10000,20000,30000]
DEFAULT=ROOT/'output/experiments/gats20_joint/c1_sigmoid2_t4_v4/run_20261012'

def seed():
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

@contextmanager
def preserve_rng():
    state=rng_state()
    try:yield
    finally:restore_rng(state)

def rng_state():
    return dict(torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),numpy_rng=np.random.get_state())

def restore_rng(state):
    torch.set_rng_state(state['torch_rng'].cpu());torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']]);random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])

def camera_plan(cameras):
    order=[c.image_name for c in cameras];r=random.Random(0)
    for _ in range(100):r.randint(0,len(order)-1)
    result=[]
    for i in range(1,30001):
        if i==5001:r.seed(0)
        c=cameras[r.randint(0,len(order)-1)]
        result.append(dict(iteration=i,camera_name=c.image_name,time=float(c.time)))
    return result

def native_function(hooks):
    import train
    source=inspect.getsource(train.scene_reconstruction)
    substitutions=[
      ('    gaussians.training_setup(opt)','    gaussians.training_setup(opt)\n    gats_hooks.setup(gaussians)'),
      ('    for iteration in range(first_iter, final_iter+1):', '    for iteration in range(first_iter, final_iter+1):'),
      ('        if network_gui.conn == None:', '        gats_hooks.begin(iteration)\n        if network_gui.conn == None:'),
      ('        loss.backward()', '        loss.backward()\n        gats_hooks.backward(gaussians, iteration, loss, Ll1, depth_loss, loss_clusters)'),
      ("(stage == 'fine' and hyper.geometry_study_enabled)):","(stage == 'fine' and (hyper.geometry_study_enabled or getattr(gaussians, 'gats_joint', False)))):"),
      ("            if stage == 'fine' and hyper.geometry_study_enabled:\n                from utils.geometry_refresh import after_optimizer", "            gats_hooks.after(gaussians, scene, viewpoint_stack, iteration)\n\n            if stage == 'fine' and hyper.geometry_study_enabled:\n                from utils.geometry_refresh import after_optimizer"),
    ]
    for a,b in substitutions:
        assert source.count(a)==1,(a,source.count(a));source=source.replace(a,b)
    namespace=dict(train.__dict__);namespace.update(gats_hooks=hooks,render=hooks.render)
    exec(compile(source,str(ROOT/'tools/run_gats20_joint.py')+':native_scene_reconstruction','exec'),namespace)
    hooks.patched_source=source
    return namespace['scene_reconstruction']

def fresh(out,variant):
    seed();m,s,p,b,c=load_scene(SOURCE/'setup/warmup',1)
    restore_knn_reference(m,SOURCE/'setup/gaussian_knn99.pth')
    assert len(s.getTrainCameras())==252 and len(s.getTestCameras())==36 and len(m._xyz)==364500
    m.active_sh_degree=0;m.spatial_lr_scale=s.cameras_extent
    changes=dict(model_path=str(out),fine_stop_iteration=-1,full_state_checkpoints='',fine_stage_state_save='',fine_warmup_reference='',fine_warmup_state='',
        gaussian_knn_reference=str(out/'setup/gaussian_knn99.pth'),gaussian_knn_save='',geometry_study_enabled=False,geometry_blend_cache='',
        ggas_variant='none',ggas_sc_edges='',ggas_candidates='',matched_texture_audit=False,connectivity_ablation_audit=False,
        freeze_geometry=False,reset_fine_rng=False,graph_forward_mode='none',graph_forward_reference='',graph_forward_monitor_iterations='',
        graph_forward_motion_reference='',render_process=False,deformation_profile_interval=100)
    for k,v in changes.items():setattr(c,k,v);setattr(m.args,k,v)
    s.model_path=str(out)
    initial=torch.load(out.parent/'dictionary_initial.pth',map_location='cpu',weights_only=True)
    g=m.appearance_graph;assert g is not None and len(g.node_xyz)==2000
    install(m,GATS20(initial,g.gaussian_node_indices,g.gaussian_node_weights),variant);install_lifecycle(m);m.gats_joint=True
    for cam in [*s.getTrainCameras(),*s.getTestCameras()]:m.gats.basis(float(cam.time))
    m.gats_basis=None
    return m,s,p,b,c

def parameters(m):
    groups={g['name']:sum(p.numel() for p in g['params']) for g in m.optimizer.param_groups}
    appearance=m.features.numel()+m.gats.A.numel()+m.gats.D.numel()
    if m.gats_variant==VARIANTS[0]:appearance+=m._deformation_table.numel()+sum(p.numel() for p in m.appearance_model.parameters())
    return dict(optimizer_groups=groups,total_trainable_parameters=sum(groups.values()),appearance_parameters=appearance,
        Adam_state_bytes=sum(v.numel()*v.element_size() for st in m.optimizer.state.values() for v in st.values() if torch.is_tensor(v)))

def save(out,m,s,c,iteration):
    out.mkdir(parents=True,exist_ok=True)
    old=s.model_path;s.model_path=str(out)
    cfg=copy.deepcopy(c);cfg.model_path=str(out);(out/'cfg_args').write_text(str(cfg))
    try:s.save(iteration,'fine')
    finally:s.model_path=old
    # Native serialized models plus exact raw fields, GATS, optimizer and sampler state.
    state=dict(format_version=1,phase='after_optimizer',iteration=iteration,variant=m.gats_variant,
        exact_raw={k:getattr(m,k).detach().cpu().clone() for k in RAW_NAMES},active_sh_degree=m.active_sh_degree,spatial_lr_scale=m.spatial_lr_scale,
        optimizer=m.optimizer.state_dict(),gats=m.gats.state_dict(),model_sha256=model_digest(m),gats_sha256=digest(m.gats.state_dict()),
        train_camera_order=[cam.image_name for cam in s.getTrainCameras()],**rng_state(),
        densification_stats={k:getattr(m,k).detach().cpu() for k in ['xyz_gradient_accum','denom','max_radii2D']})
    target=out/f'fine_state_{iteration}.pth';torch.save(state,target)
    write(out/f'checkpoint_{iteration}.json',dict(file=str(target),sha256=sha(target),model_sha256=state['model_sha256'],gats_sha256=state['gats_sha256'],
        optimizer_sha256=digest(state['optimizer']),iteration=iteration,**parameters(m)))
    return target

def reload_model(out,iteration,study):
    m,s,p,b,c=load_scene(out,iteration)
    state=torch.load(out/f'fine_state_{iteration}.pth',map_location='cpu',weights_only=False)
    for k,v in state['exact_raw'].items():setattr(m,k,torch.nn.Parameter(v.cuda(),requires_grad=True))
    m.active_sh_degree=state['active_sh_degree'];m.spatial_lr_scale=state['spatial_lr_scale']
    restore_knn_reference(m,c.gaussian_knn_reference)
    initial=torch.load(study/'dictionary_initial.pth',map_location='cpu',weights_only=True)
    gs=state['gats'];g=GATS20(initial,gs['indices'].cuda(),gs['weights'].cuda());g.load_state_dict(gs)
    install(m,g,state['variant']);install_lifecycle(m);m.gats_joint=True
    m.training_setup(c);configure_optimizer(m);m.add_graph_residual_optimizer_group()
    if m.skinning is not None:
        from utils.ggas import add_optimizer
        add_optimizer(m)
    m.optimizer.load_state_dict(state['optimizer'])
    for k,v in state['densification_stats'].items():setattr(m,k,v.cuda())
    assert model_digest(m)==state['model_sha256'];assert digest(m.gats.state_dict())==state['gats_sha256']
    return m,s,p,b,c,state

class Hooks:
    def __init__(self,out,m,s,p,b,c,preflight=False):
        self.out,self.m,self.scene,self.pipe,self.bg,self.cfg=out,m,s,p,b,c;self.preflight=preflight
        self.plan=camera_plan(s.getTrainCameras());self.renders=0;self.backwards=0;self.iteration=0;self.started=time.monotonic();self.overhead=0
        self.grad_rows=[];self.actual=[];self.interval_start=time.monotonic();self.peak=0;self.setup_peak=0
        write(out/'camera_plan.json',self.plan)
    def setup(self,m):
        configure_optimizer(m)
        r=random.Random(0)
        for _ in range(100):r.randint(0,251)
        random.setstate(r.getstate())
        write(self.out/'initial_integrity.json',dict(native_model_sha256=model_digest(m),GATS_sha256=digest(m.gats.state_dict()),
            optimizer_sha256=digest(m.optimizer.state_dict()),**parameters(m)))
    def begin(self,i):
        self.iteration=i;self.renders=0;self.backwards=0
    def render(self,cam,m,*args,**kwargs):
        from gaussian_renderer import render
        assert torch.is_grad_enabled(),'Unexpected extra renderer in native training hook'
        self.renders+=1;expected=self.plan[self.iteration-1]
        assert str(cam.image_name)==str(expected['camera_name']) and float(cam.time)==expected['time']
        m.gats_basis=m.gats.basis(float(cam.time))
        self.actual.append(expected)
        with (self.out/'camera_sequence.jsonl').open('a') as f:f.write(json.dumps(expected)+'\n')
        return render(cam,m,*args,**kwargs)
    def backward(self,m,i,loss,rgb,depth,knn):
        self.backwards+=1
        if i<=3 or i%100==0:
            norms={k:float(getattr(m,k).grad.norm()) if getattr(m,k).grad is not None else 0. for k in ['_xyz','_scaling','_rotation','_opacity','features']}
            norms.update(A=float(m.gats.A.grad.norm()),D=float(m.gats.D.grad.norm()),node_geometry=float(sum(p.grad.square().sum() for p in m.node_mlp.parameters() if p.grad is not None).sqrt()))
            row=dict(iteration=i,total=float(loss),RGB_L1=float(rgb),depth=float(depth),KNN99=float(knn),grad_norms=norms,elapsed_training_seconds=time.monotonic()-self.started-self.overhead,
                peak_VRAM_MiB=torch.cuda.max_memory_allocated()/2**20)
            self.grad_rows.append(row)
            with (self.out/'training_metrics.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    def after(self,m,s,cameras,i):
        assert self.renders==self.backwards==1
        m.gats_basis=None
        if i%100==0 or self.preflight:
            write(self.out/'status.json',dict(state='PREFLIGHT' if self.preflight else 'RUNNING',iteration=i,pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),elapsed_seconds=time.monotonic()-self.started))
        if self.preflight:return
        if i in (2000,5000):
            self.peak=max(self.peak,torch.cuda.max_memory_allocated()/2**20)
            torch.cuda.reset_peak_memory_stats()
            t=time.monotonic()
            with preserve_rng():self.boundary(i)
            if i==5000:random.seed(0)
            self.overhead+=time.monotonic()-t
            self.setup_peak=max(self.setup_peak,torch.cuda.max_memory_allocated()/2**20)
            torch.cuda.reset_peak_memory_stats()
        if i in STEPS:
            self.peak=max(self.peak,torch.cuda.max_memory_allocated()/2**20)
            t=time.monotonic()
            with preserve_rng():
                save(self.out,m,s,self.cfg,i)
                from tools.spline_appearance_common import evaluate
                expected=digest(m.gats.state_dict());evaluate(self.out,i,m,s,self.pipe,self.bg,save_frames=i==30000)
                assert expected==digest(m.gats.state_dict())
                if i==30000:
                    from PIL import Image
                    from tools.evaluate_texture_experiment import quantize
                    gtout=self.out/'evaluation/GT';gtout.mkdir(exist_ok=True)
                    for cam in s.getTrainCameras():
                        if str(cam.image_name) in ('144','264'):
                            Image.fromarray(quantize(cam.original_image[:3]).transpose(1,2,0)).save(gtout/(str(cam.image_name)+'.png'))
            self.overhead+=time.monotonic()-t;torch.cuda.reset_peak_memory_stats()
            write(self.out/'runtime.json',dict(training_wall_seconds=time.monotonic()-self.started-self.overhead,setup_checkpoint_evaluation_seconds=self.overhead,
                total_wall_seconds=time.monotonic()-self.started,peak_training_VRAM_MiB=self.peak,peak_graph_setup_VRAM_MiB=self.setup_peak,**parameters(m)))
            write(self.out/'camera_sequence.json',self.actual)
        if i==30000:
            assert len(self.actual)==30000 and self.actual==self.plan
            write(self.out/'status.json',dict(state='TRAINED',iteration=i,actual_Adam_updates=30000,geometry_frozen=False,camera_sha256=digest(self.actual)))
    def boundary(self,i):
        helper=configure(self.out,SCENE)
        if i==2000:
            save(self.out/'setup/warmup',self.m,self.scene,self.cfg,i)
            helper.reference()
            updates=dict(graph_forward_mode='g3',graph_forward_reference=str(helper.REF))
            for k,v in updates.items():setattr(self.cfg,k,v);setattr(self.m.args,k,v)
            self.m.initialize_graph_forward(str(helper.REF));self.m.add_graph_residual_optimizer_group()
        else:
            save(self.out/'common_5000',self.m,self.scene,self.cfg,i)
            helper.candidates()
            updates=dict(ggas_variant='J_NO_MOTION_SELECTION',ggas_study_version=2,ggas_candidates=str(self.out/'candidates.pth'),
                ggas_lr=.001,ggas_lambda=.01,ggas_ramp_end=6000,geometry_regularizer='knn99',geometry_study_refresh='none')
            for k,v in updates.items():setattr(self.cfg,k,v);setattr(self.m.args,k,v)
            from utils.ggas import initialize
            initialize(self.m,self.out);self.m.skinning.set_iteration(5000)
        (self.out/'cfg_args').write_text(str(self.cfg))

def execute(study,variant,preflight=False):
    out=study/variant
    if not preflight:
        assert json.loads((study/'preflight_integrity.json').read_text())['pass_']
        guard=out/'FULL_RUN_STARTED'
        with guard.open('x') as f:f.write(str(os.getpid()))
    m,s,p,b,c=fresh(out,variant);hooks=Hooks(out,m,s,p,b,c,preflight)
    fn=native_function(hooks);(out/'native_joint_instrumented.py').write_text(hooks.patched_source)
    (out/'cfg_args').write_text(str(c))
    from utils.timer import Timer
    timer=Timer();timer.start();torch.cuda.reset_peak_memory_stats()
    fn(c,c,m.args,p,[],[],[],None,-1,m,s,'fine',None,3 if preflight else 30000,timer)
    if preflight:return m,s,p,b,c,hooks


def prepare(study):
    assert not study.exists(),'No overwriting an existing study'
    core=json.loads((SOURCE.parent/'training_implementation_manifest.json').read_text())
    assert all(sha(ROOT/p)==v for p,v in core.items())
    source=json.loads((SOURCE/'source_manifest.json').read_text());assert all(sha(p)==v for p,v in source.items())
    relocation=json.loads((ROOT/'output/experiments/frozen_r6_control'/SCENE/'K4_20261012/source_manifest.json').read_text())
    relocated={p.replace('/colongraph/ColonGraph/','/colongraph/ColonGraph-GSurf/'):v for p,v in relocation.items()}
    assert all(sha(p)==v for p,v in relocated.items())
    study.mkdir(parents=True)
    source.update({str(p):sha(p) for p in (SOURCE/'setup/warmup/point_cloud/iteration_1').iterdir() if p.is_file()})
    source[str(SOURCE/'setup/warmup/cfg_args')]=sha(SOURCE/'setup/warmup/cfg_args')
    write(study/'source_manifest.json',source);write(study/'native_implementation_manifest.json',core)
    write(study/'source_relocation.json',dict(original='/home/robot/colongraph/ColonGraph',validated_relocation='/home/robot/colongraph/ColonGraph-GSurf',files=relocated,verified=True,
        experimental_core=str(ROOT),note='Dangling Git worktree pointer: evidence uses per-file SHA256; no git repository modifications.'))
    # The native split is idx%8==1 TEST; obtain TRAIN times from recorded native sampler order.
    state=torch.load(SOURCE/'setup/warmup/fine_state_2000.pth',map_location='cpu',weights_only=False)
    order=state['train_camera_order'];assert len(order)==252
    r=random.Random(0)
    for _ in range(2100):r.randint(0,251)
    assert r.getstate()==state['python_rng'];del state
    initial=initialize_dictionary([int(x)/288 for x in order]);torch.save(initial,study/'dictionary_initial.pth')
    for variant in VARIANTS:
        out=study/variant;(out/'setup').mkdir(parents=True);(out/'logs').mkdir()
        shutil.copy2(SOURCE/'setup/gaussian_knn99.pth',out/'setup/gaussian_knn99.pth')
    protocol=dict(scene=SCENE,variants=VARIANTS,full_runs=2,labels=30000,Adam_updates=30000,
        initial_native_state=str(SOURCE/'setup/warmup/point_cloud/iteration_1'),initial_state_convention='pre first fine update, same native100-coarse initialization; no fine pretrained geometry',
        phases=['fine1..2000 nativeK4 + GATS', 'own native G3 built after2000; fine2001..5000 G3K4 + GATS','own native spatialK8 J built after5000; fine5001..30000 J + GATS'],
        GATS_start=1,geometry_trainable_through=30000,appearance_graph='validated original fixed_initial K4 independent of geometry K8',
        camera_protocol='native random.Random(0), consume100 coarse draws then5000fine, reset0 at5001 as native J; identical full30k streams',
        camera_order=order,dictionary_seed=1729,dictionary_initialization=initial['initialization'],seed=0,TRAIN=252,TEST=36,resolution=[675,540],timestamps='idx/288',
        losses=json.loads((SOURCE/'configs/warmup.json').read_text())['OptimizationParams'],GATS_LRs=dict(A=.0025,D=.0001),Adam=dict(betas=[.9,.999],eps=1e-15),
        native_terminal_convention='J geometry_study historically performs Adam at30000. Isolated runner preserves this without enabling the old freezing/refresh study manager.',
        densification='original disabledfrom500000; original opacity resets3000,6000,9000,12000 retained',
        one_training_rasterizer_and_backward=True,checkpoints=STEPS,no_TEST_selection=True,no_auto_retry=True,
        branch_B_inactive_serialization_tensors='R6/embedding retained only for native serialization compatibility, excluded from forward and optimizer',
        baseline_comparison='Existing native J historical until every starting state, phase, sampler and terminal convention check is verified; no third control',
        capacity_references='Frozen Spline32 K4 28.2116 / J28.1498 are extra30k-budget references only',
        initial_B_note='B is canonical-only. Native untrained R6 decoder final layer is zero, so A/B happen to agree initially; no trained R6 distillation.',
        conda=os.environ['CONDA_PREFIX'],command='python -u tools/run_gats20_joint.py supervise --out '+str(study))
    write(study/'experiment_protocol.json',protocol)


def supervise(study):
    assert json.loads((study/'preflight_integrity.json').read_text())['pass_']
    write(study/'experiment_code_manifest.json',{str(ROOT/p):sha(ROOT/p) for p in ['utils/gats20.py','tools/run_gats20_joint.py','tools/preflight_gats20.py','tools/preflight_gats20_phases.py','tools/report_gats20_joint.py']})
    procs=[]
    for gpu,variant in enumerate(VARIANTS):
        f=(study/variant/'logs/full_train.log').open('x')
        cmd=[sys.executable,'-u',str(Path(__file__)), 'worker','--out',str(study),'--variant',variant]
        proc=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=f,stderr=subprocess.STDOUT)
        procs.append((variant,proc,f));write(study/variant/'process.json',dict(pid=proc.pid,gpu=gpu,command=cmd))
    while any(p.poll() is None for _,p,_ in procs):
        subprocess.run(['nvidia-smi','--query-gpu=timestamp,index,name,utilization.gpu,memory.used,power.draw','--format=csv,noheader'],stdout=(study/'gpu_monitor.csv').open('a'),check=False)
        time.sleep(30)
    for _,p,f in procs:f.close()
    write(study/'training_completion.json',{v:p.returncode for v,p,_ in procs})
    assert all(p.returncode==0 for _,p,_ in procs),'Run failed; NO automatic retry'
    subprocess.run([sys.executable,'-u',str(ROOT/'tools/report_gats20_joint.py'),'--out',str(study)],check=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','worker','supervise']);a.add_argument('--out',type=Path,default=DEFAULT);a.add_argument('--variant',choices=VARIANTS);a=a.parse_args()
    assert os.environ.get('CONDA_PREFIX')=='/home/robot/colongraph/.env'
    if a.action=='prepare':prepare(a.out)
    elif a.action=='supervise':supervise(a.out)
    else:
        try:execute(a.out,a.variant)
        except Exception as e:
            write(a.out/a.variant/'status.json',dict(state='FAILED_NO_RETRY',error=repr(e),pid=os.getpid()));raise
