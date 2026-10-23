"""Fresh six-run TRAIN200/PROVIDEDTEST200 matrix, full R6 trainability.
Native files and all earlier studies are untouched. Native coarse/G3/J math reused.
Retain own5k + final30k full states, disposable smoke files removed only after audit.
"""
import argparse, copy, gc, json, os, random, shutil, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tools import run_dynamiccolon_matched30k as legacy
from tools import run_weighted_spline32 as base
from tools import run_matched_fps_spline32 as matched
from tools.run_gats20_joint import restore_rng
from tools.experiment_common import load_scene
from utils.exact_split_branchpoint import RAW_NAMES,model_digest,restore_knn_reference
from utils.joint_spline32 import install,configure_optimizer
from utils.independent_temporal_rgb import active_basis
DATA=legacy.DATA;SCENE=legacy.SCENE
DEFAULT=Path('/media/robot/data/colongraph_experiments/dynamiccolon_full200_30k/data_transverse/run_20261014')
STUDIES=['old','weight','r6_old'];write,sha,digest=legacy.write,legacy.sha,legacy.digest

# Patch only this process: all root frames TRAIN; all provided/test frames TEST.
import scene.endo_loader as endo
NativeDataset=endo.ColonSplat_Dataset
class ProvidedSplitDataset(NativeDataset):
    def __init__(self,datadir,downsample=1.0,test_every=5):
        assert Path(datadir).resolve()==DATA.resolve(), 'Root dataset must be loaded for geometry'
        super().__init__(datadir,downsample=downsample,test_every=test_every)
        train=list(self.all_cams);test=self.readCamerasFromTransforms(str(DATA/'test'),'transforms.json',False)
        assert len(train)==len(test)==200
        for cam in train:cam.source_split='TRAIN';cam.source_dataset=str(DATA)
        for idx,cam in enumerate(test):
            cam.original_uid=cam.uid;cam.uid=200+idx
            cam.original_image_name=cam.image_name;cam.image_name='test__'+cam.image_name
            cam.source_split='TEST';cam.source_dataset=str(DATA/'test')
        self.all_cams=train+test;self.train_idxs=list(range(200));self.test_idxs=list(range(200,400));self.video_idxs=list(range(400))
endo.ColonSplat_Dataset=ProvidedSplitDataset


def assert_split(scene):
    train,test=scene.getTrainCameras(),scene.getTestCameras()
    assert len(train)==len(test)==200 and scene.gt_plys is None
    assert all(c.source_split=='TRAIN' and not c.image_name.startswith('test__') for c in train)
    assert all(c.source_split=='TEST' and c.image_name.startswith('test__') for c in test)
    assert {c.image_name for c in train}.isdisjoint(c.image_name for c in test)
    assert [c.uid for c in train]==list(range(200)) and [c.uid for c in test]==list(range(200,400))


def configuration(study):
    cfg,groups=legacy.configuration(study)
    cfg.dynamic_split='root_all200_train__provided_test_all200'
    return cfg,groups


def parameters(m):
    value=legacy.parameters(m)
    value['requires_grad_fields']={k:getattr(m,k).requires_grad for k in RAW_NAMES}
    value['actual_trainable_groups']={g['name']:sum(p.numel() for p in g['params'] if p.requires_grad) for g in m.optimizer.param_groups}
    value['R6_all_appearance_trainable']=None if hasattr(m,'spline') else (
        m.features.requires_grad and m._deformation_table.requires_grad and all(p.requires_grad for p in m.appearance_model.parameters()))
    return value


_native_save=legacy.save

def save(out,m,s,c,i):
    # Original native snapshot/exactraw/Adam/RNG format, immutable tensor hardlinks.
    _native_save(out,m,s,c,i)
    p=out/f'fine_state_{i}.pth';state=torch.load(p,map_location='cpu',weights_only=False)
    state['requires_grad_fields']={k:getattr(m,k).requires_grad for k in RAW_NAMES}
    state['appearance_parameter_requires_grad']={k:v.requires_grad for k,v in m.appearance_model.named_parameters()}
    if hasattr(m,'fresh_coarse_raw_gradients'):state['coarse_raw_gradients']=m.fresh_coarse_raw_gradients
    torch.save(state,p)
    ck=json.loads((out/f'checkpoint_{i}.json').read_text());ck.update(sha256=sha(p),**parameters(m))
    write(out/f'checkpoint_{i}.json',ck)


def load(path,i,appearance=None):
    m,s,p,b,c=load_scene(path,i);assert_split(s)
    state=torch.load(path/f'fine_state_{i}.pth',map_location='cpu',weights_only=False)
    with torch.no_grad():
        for k,v in state['exact_raw'].items():getattr(m,k).copy_(v.cuda())
    m.active_sh_degree=state['active_sh_degree'];m.spatial_lr_scale=state['spatial_lr_scale']
    restore_knn_reference(m,c.gaussian_knn_reference)
    if appearance!='R6' and 'spline' in state:
        install(m);m.spline.load_state_dict(state['spline']);assert digest(m.spline.state_dict())==state['spline_sha256']
    else:
        m.gats_joint=True
        # Critical fix: native torch.load Parameter preserves the Spline-init flag.
        m.features.requires_grad_(True);m._deformation_table.requires_grad_(True)
        m.appearance_model.requires_grad_(True)
        assert m._deformation_table.is_leaf and m._deformation_table.requires_grad
    assert model_digest(m)==state['model_sha256']
    if appearance is None and i!=1:
        assert {k:getattr(m,k).requires_grad for k in RAW_NAMES}==state['requires_grad_fields']
    return m,s,p,b,c,state


# Capture raw coarse gradients BEFORE the Spline adapter freezes base appearance.
_original_install=base.install

def capture_install(m,*args,**kw):
    m.fresh_coarse_raw_gradients={k:None if getattr(m,k).grad is None else getattr(m,k).grad.detach().cpu().clone() for k in RAW_NAMES}
    _original_install(m,*args,**kw)
base.install=capture_install


class Hooks(legacy.Hooks):
    def setup(self,m):
        assert_split(self.s)
        if not hasattr(m,'spline'):
            assert m.features.requires_grad and m._deformation_table.requires_grad
            assert all(p.requires_grad for p in m.appearance_model.parameters())
        super().setup(m)
        if not hasattr(m,'spline'):
            group=next(g for g in m.optimizer.param_groups if g['name']=='deformation_table')
            assert group['params']==[m._deformation_table]
    def begin(self,i):
        super().begin(i)
        if i==1 and self.initial_state is not None and not hasattr(self.m,'spline'):
            raw=self.initial_state['coarse_raw_gradients']
            for k in ['features','_deformation_table']:
                getattr(self.m,k).grad=None if raw[k] is None else raw[k].cuda().clone()
    def render(self,cam,m,*args,**kw):
        assert cam.source_split=='TRAIN' and cam.source_dataset==str(DATA)
        assert torch.is_grad_enabled()
        return super().render(cam,m,*args,**kw)
    def backward(self,m,i,loss,rgb,depth,knn):
        super().backward(m,i,loss,rgb,depth,knn)
        if not hasattr(m,'spline'):
            assert m._deformation_table.requires_grad and m._deformation_table.grad is not None
            assert torch.isfinite(m._deformation_table.grad).all()
            if i==100:assert float(m._deformation_table.grad.norm())>0
    def after(self,m,s,cameras,i):
        assert self.renders==self.backwards==1
        if hasattr(m,'spline'):m.spline_basis=None
        if i%100==0:
            if not hasattr(m,'spline'):
                assert parameters(m)['optimizer_steps']['deformation_table']==[i]
            write(self.out/'status.json',dict(state='RUNNING',iteration=i,pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),TRAIN=200,TEST=200))
        if self.smoke:return
        start=time.monotonic()
        if i==2000:
            with base.preserve_rng():
                save(self.study/'setup/warmup',m,s,self.c,i)
                helper=legacy.configure(self.study);helper.reference()
                base.update(m,self.c,graph_forward_mode='g3',graph_forward_reference=str(helper.REF))
                m.initialize_graph_forward(str(helper.REF));m.add_graph_residual_optimizer_group()
                (self.out/'cfg_args').write_text(str(self.c))
        if i in [5000,30000]:
            with base.preserve_rng():save(self.out,m,s,self.c,i)
        if i in [2000,5000,30000]:self.overhead+=time.monotonic()-start
        if i in [5000,10000,20000,30000]:
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
            if not hasattr(m,'spline'):
                assert steps['deformation_table']==[30000]
                initial=torch.load(self.study.parent/'initial/fine_state_1.pth',map_location='cpu',weights_only=False)
                assert not torch.equal(initial['exact_raw']['_deformation_table'],m._deformation_table.detach().cpu())
                write(self.out/'R6_embedding_integrity.json',dict(pass_=True,requires_grad=True,Adam_updates=30000,value_changed_from_fine0=True))
            write(self.out/'status.json',dict(state='TRAINED',iteration=i,actual_Adam_updates=30000,geometry_frozen=False,evaluation='providedTEST200 pending'))


base.configuration=configuration;base.load=load;base.save=save;base.parameters=parameters;base.Hooks=Hooks
legacy.load=load;legacy.Hooks=Hooks;legacy.save=save
matched.compact_save=save


def initial(root):
    import train
    original_render=train.render
    draws=[]
    def train_render(cam,*args,**kw):
        assert cam.source_split=='TRAIN' and cam.source_dataset==str(DATA)
        if kw.get('stage')=='coarse':draws.append(dict(camera_name=cam.image_name,time=float(cam.time),source=cam.source_dataset))
        return original_render(cam,*args,**kw)
    train.render=train_render
    try:matched.fresh_initial(root)
    finally:train.render=original_render
    assert len(draws)==100
    rng=random.Random(0);expected=['r_'+str(rng.randint(0,199)+1).zfill(4) for _ in range(100)]
    assert [d['camera_name'] for d in draws]==expected
    write(root/'coarse_camera_audit.json',dict(pass_=True,TRAIN=200,TEST_calls=0,draws=draws))


def smoke(root,sampler):
    legacy.prefix(root,sampler,smoke=True)
    p=root/'preflight'/sampler/'common_5000';m,s,pipe,b,c,state=load(p,101)
    flags={k:getattr(m,k).requires_grad for k in RAW_NAMES}
    check=dict(pass_=True,TRAIN=200,TEST=200,camera_names_disjoint=True,raw_requires_grad=flags,
        R6_full_appearance_trainable=None if hasattr(m,'spline') else flags['features'] and flags['_deformation_table'],
        R6_embedding_Adam_updates=None if hasattr(m,'spline') else int(next(v['step'] for k,v in state['optimizer']['state'].items() if k==next(g['params'][0] for g in state['optimizer']['param_groups'] if g['name']=='deformation_table'))),
        model_exact=True,Adam_exact=True,native_resume_verified=3,GT_PLY_forbidden=True,primary_fine_updates=0)
    if not hasattr(m,'spline'):assert check['R6_embedding_Adam_updates']==101
    del m,s,pipe,b,c,state;gc.collect();torch.cuda.empty_cache()
    write(root/'preflight'/sampler/'integrity.json',check)


def prepare(root):
    assert not root.exists()
    assert shutil.disk_usage('/media/robot/data').free>4.5*1024**3,'Need4.5GiB for compact retained fresh states/evaluation'
    root.mkdir(parents=True)
    for f in ['logs','setup','initial','immutable_native']:(root/f).mkdir()
    for sampler in STUDIES:
        for folder in ['logs','setup','common_5000','K4','J']:(root/sampler/folder).mkdir(parents=True)
    manifest={}
    for dataset in [DATA,DATA/'test']:
        frames=json.loads((dataset/'transforms.json').read_text())['frames'];assert len(frames)==200
        for p in [dataset/'transforms.json',*[dataset/f[k] for f in frames for k in ['file_path','depth_path']]]:
            manifest[str(p)]=dict(bytes=p.stat().st_size,sha256=sha(p))
    write(root/'data_manifest.json',manifest)
    protocol=dict(scene=SCENE,branches=['K4_old_Spline32','K4_weight_Spline32','J_old_Spline32','J_weight_Spline32','K4_old_R6','J_old_R6'],
        full_trajectories=6,TRAIN=200,TEST=200,train_dataset=str(DATA),test_dataset=str(DATA/'test'),
        split='ALL200root images TRAIN, ALL200provided/test images TEST; separate camera namespaces',
        initialization='fresh firstTRAIN RGB/depth backprojection200000 points + nativecoarse100; no learned checkpoint reused',
        GT_PLY='forbidden; initial RGB/depth only; Scene.gt_plys=None',coarse_steps=100,coarse_Adam_updates=99,
        samplers={'old':'FPS2000','weight':'sameFPS2000+importance400 from100coarse visible totalXYZ gradient deviation'},
        appearance={'Spline32':'joint32controls perGaussian fine1..30k LR.0025','R6':'FULL original fixed_initial FPS2000K4; canonicalRGB,16DGaussianembedding,nodecontext,decoder alltrainable'},
        R6_fix='Explicit embedding requires_gradTrue on load; nonzero finite TRAINgradient at100; Adamcounts every100; bitwisevaluechange final; flags saved and reload checked',
        G3_start=2001,G3='own2k E4surfacegraph +16TRAINtrajectorygate',J_start=5001,J='GGAS-v2 spatial-only connectedK8, LR.001 KL.01 ramp6000',
        fine_iterations=30000,geometry_trainable_through=30000,appearance_trainable_through=30000,terminal_Adam_included=True,
        KNN99_weight=.01,depth_weight=.25,RGB_loss='nativeL1+TV.001+colorvariance.0001',LR_schedules='unchanged previousprotocol',
        camera='native seed0 consumes100coarse draws from200TRAIN; reset0at5001; exactsame30000cameras/times for6',
        coarse_gradient_carryover='native lastcoarse geometry gradients shared; R6 also restores native coarse canonicalRGB gradient; embedding coarsegradientNone',
        full_checkpoints=[5000,30000],G3_warmup_checkpoint=2000,training_logs_every=100,
        test_evaluation='ALL200providedTEST only AFTER all6 trainings complete; fixed30k, no checkpoint/threshold selection',
        storage='compact newrun only: retain own5k/final30k + own2kwarmup; omitdebug100/10k/20kfullstates; remove disposable smoke model files only after exactreloadaudit',
        no_auto_retry=True,held_out_selection=False,conda=os.environ['CONDA_PREFIX'])
    write(root/'matrix_protocol.json',protocol);configuration(root)
    files=['tools/run_dynamiccolon_full200_30k.py','tools/run_dynamiccolon_matched30k.py','tools/run_matched_fps_spline32.py','tools/run_weighted_spline32.py',
        'tools/run_gats20_joint.py','tools/run_cross_scene_aj.py','tools/build_graph_forward_reference.py','train.py','scene/gaussian_model.py','scene/__init__.py',
        'scene/endo_loader.py','scene/dataset_readers.py','tools/export_dynamiccolon_full200_videos.py','utils/joint_spline32.py','utils/node_sampling.py','utils/ggas.py','utils/ggas_v2.py','utils/rgbd_initialization.py']
    write(root/'code_manifest.json',{str(ROOT/p):sha(ROOT/p) for p in files})
    write(root/'status.json',dict(state='PREPARED',full_trajectories=6,TRAIN=200,TEST=200))


def run_phase(root,commands,stage):
    assert len(commands)<=2
    write(root/'status.json',dict(state='RUNNING',stage=stage,pid=os.getpid(),TRAIN=200,TEST=200))
    jobs=[];processes={}
    for gpu,(name,argv) in enumerate(commands):
        f=(root/'logs'/(name+'.log')).open('x');cmd=[sys.executable,'-u',__file__,*argv,'--out',str(root)]
        p=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=f,stderr=subprocess.STDOUT)
        jobs.append((name,p,f));processes[name]=dict(pid=p.pid,gpu=gpu,command=cmd)
    write(root/('processes_'+stage+'.json'),processes)
    codes={}
    for name,p,f in jobs:codes[name]=p.wait();f.close()
    write(root/('exit_'+stage+'.json'),codes);assert not any(codes.values()),codes


def evaluate_final(out,m,s,p,b):
    from gaussian_renderer import render
    from tools.evaluate_texture_experiment import metrics,quantize
    from tools.evaluate_geometry_blending import masks,depth_metrics
    from utils.image_utils import psnr
    from PIL import Image
    dest=out/'evaluation/30000';dest.mkdir(parents=True,exist_ok=False);(dest/'frames').mkdir()
    expected=model_digest(m);rows={}
    with torch.no_grad():
        for cam in s.getTestCameras():
            assert cam.source_split=='TEST'
            if hasattr(m,'spline'):m.spline_basis=active_basis(cam.time,device='cuda')
            q=render(cam,m,p,b);rgb=q['render'].clamp(0,1);gt=cam.original_image[:3].cuda().float()
            byte=quantize(rgb);gtbyte=quantize(gt)
            v=metrics(torch.as_tensor(byte,device='cuda').float()/255,torch.as_tensor(gtbyte,device='cuda').float()/255,perceptual=True)
            _,valid,*_=masks(cam);d,_=depth_metrics(q['depth'],cam,q['depth'].squeeze()[valid].max())
            v.update(PSNR_float=float(psnr(rgb[None],gt[None]).mean()),DEPTH_FLOAT_VALID_MSE=d['DEPTH_FLOAT_VALID_MSE'],
                DEPTH_GRADIENT_ERROR=d['DEPTH_GRADIENT_ERROR'],alpha_mean=float(q['alpha'].mean()),frame=cam.original_image_name,
                camera_name=cam.image_name,time=float(cam.time),split='PROVIDED_TEST',source=str(DATA/'test'))
            rows[cam.original_image_name]=v
            Image.fromarray(byte.transpose(1,2,0)).save(dest/'frames'/(cam.original_image_name+'.png'))
    assert len(rows)==200 and model_digest(m)==expected
    keys=['PSNR','PSNR_float','SSIM','LPIPS_A','LPIPS_V','DEPTH_FLOAT_VALID_MSE','DEPTH_GRADIENT_ERROR','alpha_mean']
    summary=dict(test={k:float(np.mean([v[k] for v in rows.values()])) for k in keys},TEST_views=200,TRAIN_views=200,test_dataset=str(DATA/'test'),checkpoint=30000)
    write(dest/'summary.json',summary);write(dest/'per_view.json',rows)
    return summary


def postprocess(root):
    from tools.evaluate_geometry_blending import benchmark
    results={}
    for sampler in STUDIES:
        for branch in ['K4','J']:
            study=root/sampler;out=study/branch;m,s,p,b,c,state=load(out,30000)
            m.training_setup(c)
            if hasattr(m,'spline'):configure_optimizer(m)
            m.add_graph_residual_optimizer_group()
            if m.skinning is not None:
                from utils.ggas import add_optimizer
                add_optimizer(m)
            m.optimizer.load_state_dict(state['optimizer'])
            expected=json.loads((out/'checkpoint_30000.json').read_text())
            assert sha(out/'fine_state_30000.pth')==expected['sha256']
            assert digest(m.optimizer.state_dict())==expected['optimizer_sha256']
            assert parameters(m)['requires_grad_fields']==state['requires_grad_fields']
            if not hasattr(m,'spline'):assert parameters(m)['optimizer_steps']['deformation_table']==[30000]
            write(out/'reload_integrity.json',dict(pass_=True,model_exact=True,Adam_exact=True,trainability_flags_exact=True,fileSHA_exact=True))
            summary=evaluate_final(out,m,s,p,b)
            perf=legacy.evaluation.timing(m,s,p,b,m.spline) if hasattr(m,'spline') else benchmark(m,s,p,b)
            write(out/'performance.json',perf)
            runtime=json.loads((out/'runtime.json').read_text());pre=json.loads((study/'common_5000/runtime.json').read_text())
            runtime.update(full_trajectory_training_seconds=runtime['training_seconds']+pre['training_seconds'],shared_prefix_training_seconds=pre['training_seconds'])
            write(out/'runtime.json',runtime)
            rows=[json.loads(x) for f in [study/'common_5000/camera_sequence.jsonl',out/'camera_sequence.jsonl'] for x in f.read_text().splitlines()]
            assert len(rows)==30000 and rows==json.loads((out/'camera_plan.json').read_text())
            assert not any(x['camera_name'].startswith('test__') for x in rows)
            assert len(set(x['camera_name'] for x in rows))==200
            write(out/'camera_integrity.json',dict(pass_=True,draws=30000,unique_TRAIN=200,TEST_calls=0,sha256=digest(rows)))
            results[branch+'_'+sampler]=dict(metrics=summary['test'],runtime=runtime,performance=perf,camera_sha256=digest(rows))
            del m,s,p,b,c,state;gc.collect();torch.cuda.empty_cache()
    assert len({x['camera_sha256'] for x in results.values()})==1
    write(root/'comparison.json',results)
    lines=['# DynamicColon transverse — fresh six full30k, TRAIN200 / providedTEST200','',
        'All200root images train; all200dataset/test cameras evaluate only after all6 trainings. Fresh RGB/depth coarse100; no GTPLY. Original full R6 flags/gradients/Adam/valuechange guarded. Geometry and appearance active throughout30k. OwnG3after2k, pairedK4/Jfork5k.','',
        '| Run | PSNR | SSIM | LPIPS-A | LPIPS-V | normalized depth MSE | Train min | FPS |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in ['K4_old','K4_weight','J_old','J_weight','K4_r6_old','J_r6_old']:
        r=results[name];q=r['metrics'];lines.append(f"| {name} | {q['PSNR']:.6f} | {q['SSIM']:.6f} | {q['LPIPS_A']:.6f} | {q['LPIPS_V']:.6f} | {q['DEPTH_FLOAT_VALID_MSE']:.9f} | {r['runtime']['full_trajectory_training_seconds']/60:.2f} | {r['performance']['FPS']:.2f} |")
    lines+=['','Trainmin includes own shared5kprefix, excludescoarse/setup/checkpoint/evaluation. Full5k/final30k retained, traincurves every100. Test is supplied200camera trajectory, no internal temporalholdout. No TEST selection. Normalized image depth, not physicalmm. Historical160TRAIN/40internal and frozen-embeddingR6 scores remain separate.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    # Durable reconstruction video export: final frames were evaluated on providedtest.
    for name in results:
        branch,sampler=name.split('_',1);target=root/'provided_test_evaluation'/name
        target.mkdir(parents=True);os.symlink(root/sampler/branch/'evaluation/30000/frames',target/'frames')
    write(root/'provided_test_evaluation/comparison.json',{name:dict(metrics=x['metrics']) for name,x in results.items()})
    write(root/'status.json',dict(state='TRAINED_AND_EVALUATED',full_trajectories=6,TRAIN=200,TEST=200,iterations=30000))
    # Video exporter is started below after final COMPLETE flag, no model changes.
    write(root/'status.json',dict(state='COMPLETE',full_trajectories=6,TRAIN=200,TEST=200,iterations=30000))
    subprocess.run([sys.executable,'tools/export_dynamiccolon_full200_videos.py','--out',str(root)],cwd=ROOT,stdout=(root/'logs/video_export.log').open('x'),stderr=subprocess.STDOUT,check=True)


def supervise(root):
    with (root/'PIPELINE_STARTED').open('x') as f:f.write(str(os.getpid()))
    run_phase(root,[('fresh_coarse',['initial'])],'fresh_coarse')
    for sampler in STUDIES:os.link(root/'setup/gaussian_knn99.pth',root/sampler/'setup/gaussian_knn99.pth')
    for subset in [STUDIES[:2],STUDIES[2:]]:
        run_phase(root,[(s+'_smoke',['smoke','--sampler',s]) for s in subset],'preflight_'+'_'.join(subset))
    assert all(json.loads((root/'preflight'/s/'integrity.json').read_text())['pass_'] for s in STUDIES)
    removed=[]
    for sampler in STUDIES:
        folder=root/'preflight'/sampler/'common_5000'
        for p in list(folder.glob('fine_state_*.pth')):
            removed.append(str(p));p.unlink()
        p=folder/'point_cloud';removed.append(str(p));shutil.rmtree(p)
    write(root/'disposable_preflight_cleanup.json',dict(exact_audits_passed=True,removed=removed,historical_artifacts_touched=False))
    for subset in [STUDIES[:2],STUDIES[2:]]:
        run_phase(root,[(s+'_prefix',['prefix','--sampler',s]) for s in subset],'prefix_'+'_'.join(subset))
    for subset in [STUDIES[:2],STUDIES[2:]]:
        run_phase(root,[(s+'_candidates',['candidates','--sampler',s]) for s in subset],'candidates_'+'_'.join(subset))
    for branch in ['K4','J']:
        run_phase(root,[(branch+'_'+s,['worker','--sampler',s,'--branch',branch]) for s in ['old','weight']],branch+'_Spline_continuations')
    run_phase(root,[(b+'_r6_old',['worker','--sampler','r6_old','--branch',b]) for b in ['K4','J']],'R6_continuations')
    for p,h in json.loads((root/'code_manifest.json').read_text()).items():assert sha(p)==h
    for p,v in json.loads((root/'data_manifest.json').read_text()).items():assert sha(p)==v['sha256']
    run_phase(root,[('postprocess',['postprocess'])],'postprocess')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','initial','smoke','prefix','candidates','worker','supervise','postprocess'])
    parser.add_argument('--out',type=Path,default=DEFAULT);parser.add_argument('--sampler',choices=STUDIES);parser.add_argument('--branch',choices=['K4','J']);a=parser.parse_args()
    assert os.environ.get('CONDA_PREFIX')=='/home/robot/colongraph/.env','Use project Conda'
    try:
        if a.action=='prefix':legacy.prefix(a.out,a.sampler)
        elif a.action=='smoke':smoke(a.out,a.sampler)
        elif a.action=='candidates':legacy.candidates(a.out,a.sampler)
        elif a.action=='worker':base.worker(a.out/a.sampler,a.branch)
        else:globals()[a.action](a.out)
    except Exception as e:
        if a.out.exists():write(a.out/'status.json',dict(state='FAILED',stage=a.action,sampler=a.sampler,branch=a.branch,error=repr(e),pid=os.getpid(),no_automatic_retry=True))
        raise
