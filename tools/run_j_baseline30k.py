"""Fresh two-scene J baseline; isolated setup and all geometry updates through 30k."""
import argparse, copy, datetime, hashlib, json, os, random, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCENES = ['c1_sigmoid2_t4_v4', 'c1_sigmoid1_t4_v4']
CHECKPOINTS = [5000, 10000, 15000, 20000, 25000, 30000]

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); tmp.replace(path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''): h.update(b)
    return h.hexdigest()

def configure(out, scene):
    from tools import run_cross_scene_aj as helper
    helper.SCENE = scene; helper.OUT = out; helper.LOGS = out/'logs'
    helper.DATA = ROOT.parent/'ColonSplat/data/coloncrafter'/scene
    helper.SOURCE = out/'common_5000'; helper.KNN = out/'setup/gaussian_knn99.pth'
    helper.REF = out/'setup/references/graph_forward_reference.pth'
    return helper

def prepare(out):
    if out.exists(): raise FileExistsError('New study must have a new directory: '+str(out))
    template = json.loads((ROOT/'output/experiments/ggas_v2/c1_sigmoid2_t4_v4/configs/block00_J_NO_MOTION_SELECTION.json').read_text())
    core = json.loads((ROOT/'output/experiments/ggas_v3/c1_sigmoid2_t4_v4/training_implementation_manifest.json').read_text())
    assert all(sha(ROOT/p)==s for p,s in core.items())
    out.mkdir(parents=True)
    write(out/'training_implementation_manifest.json', core)
    protocol = dict(method='J — GGAS-v2 Spatial-only Learned K8', main_blending_baseline='J',
        selected_by='explicit user instruction', scenes=SCENES, full_runs=2, fresh_initialization=True,
        fine_iterations=30000, coarse_iterations=100, camera_seed=0, experiment_seed=0,
        phases=['fresh coarse100 + fine1..2000 K4', 'new G3 + fine2001..5000 K4', 'new spatial K8 J + fine5001..30000'],
        geometry_and_appearance_train_through=30000, terminal_geometry_freeze='after final Adam30k, no frozen training period',
        gaussian_knn99_weight=.01, skinning_lr=.001, skinning_KL=.01, ramp=[5000,6000],
        appearance='unchanged R6, independent fixed K4', texture=False, new_loss=False,
        optimizer_schedules='unchanged historical schedules; XYZ LR reaches final value at20k and remains active',
        external_K0_motion_guard='disabled for both fresh scenes: no matched fresh K0 full run; no reused historical topology/motion guard',
        G3_trajectory_gate='unchanged original E4 surface graph and 16 TRAIN warmup times',
        comparison_limit='fresh initialization and continued geometry20k..30k differ from historical frozen20k runs; no causal matched A/J claim',
        anatomical_motion_stability='not certified by baseline selection or RGB/depth diagnostics', checkpoints=CHECKPOINTS)
    write(out/'protocol.json', protocol)
    for scene in SCENES:
        dest=out/scene; helper=configure(dest,scene); (dest/'logs').mkdir(parents=True)
        data=helper.DATA
        rgb=sorted((data/'color').glob('*.png')); depth=sorted((data/'depth').glob('*.npy'))
        assert len(rgb)==len(depth)==(288 if 'sigmoid2' in scene else 297)
        assert [p.stem for p in rgb]==[p.stem for p in depth]
        write(dest/'data_manifest.json',{str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in [data/'pose.txt',data/'points3d.ply',*rgb,*depth]})
        cfg=copy.deepcopy(template); cfg['ModelParams'].update(source_path=str(data), model_path=str(dest/'setup/warmup'))
        h=cfg['ModelHiddenParams']; h.update(node_initialization_reference='', gaussian_knn_reference='', gaussian_knn_save=str(helper.KNN),
            graph_forward_mode='none', graph_forward_reference='', graph_forward_motion_reference='', graph_forward_monitor_iterations='',
            fine_warmup_reference='', fine_warmup_state='', fine_stage_state_save='', fine_stop_iteration=2000,
            full_state_checkpoints='2000', ggas_variant='none', ggas_candidates='', ggas_sc_edges='',
            geometry_study_enabled=False, geometry_blend_cache='', geometry_blend_mode='idw',
            matched_texture_audit=False, connectivity_ablation_audit=False, texture_enabled=False, freeze_geometry=False, reset_fine_rng=False)
        write(dest/'configs/warmup.json',cfg)
        cfg=copy.deepcopy(cfg);cfg['ModelParams']['model_path']=str(helper.SOURCE)
        cfg['ModelHiddenParams'].update(gaussian_knn_reference=str(helper.KNN), gaussian_knn_save='', graph_forward_mode='g3',
            graph_forward_reference=str(helper.REF), fine_warmup_reference=str(dest/'setup/warmup/point_cloud/iteration_2000'),
            fine_warmup_state=str(dest/'setup/warmup/fine_state_2000.pth'), fine_stop_iteration=5000, full_state_checkpoints='5000')
        write(dest/'configs/common.json',cfg)
        cfg=copy.deepcopy(cfg);cfg['ModelParams']['model_path']=str(dest/'runs/J_GGAS_V2')
        cfg['ModelHiddenParams'].update(fine_warmup_reference=str(helper.SOURCE/'point_cloud/iteration_5000'),
            fine_warmup_state=str(helper.SOURCE/'fine_state_5000.pth'), fine_stop_iteration=30000,
            full_state_checkpoints=','.join(map(str,CHECKPOINTS)), geometry_study_enabled=True,
            geometry_study_refresh='none', geometry_study_freeze_iteration=30000, geometry_study_camera_seed=0,
            geometry_blend_cache=str(dest/'caches/A_idw.pth'), ggas_candidates=str(dest/'candidates.pth'),
            ggas_variant='J_NO_MOTION_SELECTION', ggas_study_version=2, ggas_lr=.001, ggas_lambda=.01,
            ggas_ramp_end=6000, matched_texture_audit=True, matched_texture_source_iteration=5000)
        assert cfg['OptimizationParams']==template['OptimizationParams']
        write(dest/'configs/J_GGAS_V2.json',cfg)
        write(dest/'status.json',dict(state='PREPARED',scene=scene))
        write(ROOT/'configs/baselines'/('ggas_v2_j_'+scene+'.json'),cfg)
    write(ROOT/'configs/baselines/current_blending.json',dict(method=protocol['method'], main='J', study=str(out), protocol=str(out/'protocol.json'),
        per_scene_configs={s:str(ROOT/'configs/baselines'/('ggas_v2_j_'+s+'.json')) for s in SCENES}, anatomical_motion_certified=False))
    write(out/'launcher_manifest.json',{str(ROOT/'tools/run_j_baseline30k.py'):sha(ROOT/'tools/run_j_baseline30k.py')})
    write(out/'status.json',dict(state='PREPARED',scenes=SCENES))

def train_stage(dest, name, saves):
    cfg=dest/'configs'/(name+'.json'); target=Path(json.loads(cfg.read_text())['ModelParams']['model_path'])
    if target.exists(): raise FileExistsError('Preserve existing stage '+str(target))
    cmd=[sys.executable,'-u','train.py','--configs',str(cfg),'--expname',str(target.relative_to(ROOT/'output')),
        '--save_iterations',*map(str,saves),'--test_iterations','999999']
    write(dest/'status.json',dict(state='RUNNING',stage=name,pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),started_unix=time.time()))
    start=time.monotonic()
    with (dest/'logs'/(name+'.log')).open('w') as f:
        subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    elapsed=time.monotonic()-start
    write(dest/'setup'/('stage_'+name+'.json'),dict(pass_=True,seconds=elapsed,command=cmd))
    return elapsed

def integrity(dest):
    import torch
    from utils.exact_split_branchpoint import digest
    path=dest/'runs/J_GGAS_V2'; source=torch.load(dest/'common_5000/fine_state_5000.pth',map_location='cpu',weights_only=False)
    initial=json.loads((path/'matched_initial.json').read_text())
    final=torch.load(path/'fine_state_30000.pth',map_location='cpu',weights_only=False)
    mid=torch.load(path/'fine_state_20000.pth',map_location='cpu',weights_only=False)
    seq=[json.loads(x) for x in (path/'connectivity_camera_sequence.jsonl').read_text().splitlines()]
    rng=random.Random(0); expected=[source['train_camera_order'][rng.randint(0,len(source['train_camera_order'])-1)] for _ in range(25000)]
    first=torch.load(path/'point_cloud/iteration_5000/skinning.pth',map_location='cpu',weights_only=True)
    skin=torch.load(path/'point_cloud/iteration_30000/skinning.pth',map_location='cpu',weights_only=True)
    mid_skin=torch.load(path/'point_cloud/iteration_20000/skinning.pth',map_location='cpu',weights_only=True)
    groups={g['name']:[float(final['optimizer']['state'][p]['step']) for p in g['params'] if p in final['optimizer']['state']] for g in final['optimizer']['param_groups']}
    midgroups={g['name']:[float(mid['optimizer']['state'][p]['step']) for p in g['params'] if p in mid['optimizer']['state']] for g in mid['optimizer']['param_groups']}
    old=torch.load(dest/'common_5000/point_cloud/iteration_5000/deformation_nodes.pth',map_location='cpu',weights_only=True)
    new=torch.load(path/'point_cloud/iteration_30000/deformation_nodes.pth',map_location='cpu',weights_only=True)
    checks=dict(source_model_exact=initial['non_texture_model_sha256']==source['model_sha256'],
        source_Adam_exact=initial['optimizer_sha256']==digest(source['optimizer']),
        source_RNG_exact=initial['rng_sha256']==digest({k:source[k] for k in ('torch_rng','cuda_rng','python_rng','numpy_rng')}),
        camera_replay_exact=[x['camera_name'] for x in seq]==expected,actual_J_calls=len(seq)==25000, final_iteration=final['iteration']==30000,
        all_raw_finite=all(torch.isfinite(v).all().item() for v in final['exact_raw'].values()),
        geometry_updates_after20k=all(groups[k] and all(b-a==10000 for a,b in zip(midgroups[k],groups[k])) for k in ['xyz','scaling','rotation','opacity','skinning','deformation','graph_residual']),
        appearance_updates_after20k=groups['f_dc'][0]-midgroups['f_dc'][0]==10000,
        skinning_steps=groups['skinning']==[25000.], skinning_finite=bool(torch.isfinite(skin['state_dict']['delta']).all()),
        skinning_changed_after20k=not torch.equal(mid_skin['state_dict']['delta'],skin['state_dict']['delta']),
        candidates_and_prior_fixed=all(torch.equal(first['state_dict'][k],skin['state_dict'][k]) for k in ['indices','initial','target']),
        unbounded_J=skin['format_version']==1, nodes_fixed=torch.equal(old['graph']['node_xyz'],new['graph']['node_xyz']),
        appearance_K4_fixed=digest(old['appearance_graph'])==digest(new['appearance_graph']),
        G3_topology_fixed=all(torch.equal(old['graph_motion_residual'][k],new['graph_motion_residual'][k]) for k in ['edge_indices','edge_weights','canonical_xyz','weight_denominator']),
        xyz_changed_after20k=not torch.equal(mid['exact_raw']['_xyz'],final['exact_raw']['_xyz']))
    result=dict(pass_=all(checks.values()),checks=checks,optimizer_steps=groups,optimizer_steps20k=midgroups,actual_J_calls=len(seq),
        camera_sequence_sha256=digest([x['camera_name'] for x in seq]),model_sha256=final['model_sha256'])
    write(path/'integrity.json',result)
    if not result['pass_']:raise RuntimeError('Integrity failed: '+str(checks))
    for p,h in json.loads((dest/'source_manifest.json').read_text()).items():assert sha(p)==h
    return result

def worker(out,scene):
    dest=out/scene; helper=configure(dest,scene); start=time.monotonic(); timings={}
    try:
        timings['warmup_seconds']=train_stage(dest,'warmup',[1,2000])
        write(dest/'status.json',dict(state='RUNNING',stage='build_new_G3',pid=os.getpid()))
        t=time.monotonic();helper.reference();timings['G3_build_seconds']=time.monotonic()-t
        timings['common_seconds']=train_stage(dest,'common',[5000])
        write(dest/'status.json',dict(state='RUNNING',stage='build_new_K8',pid=os.getpid()))
        t=time.monotonic();helper.candidates();timings['K8_build_seconds']=time.monotonic()-t
        timings['J_continuation_seconds']=train_stage(dest,'J_GGAS_V2',CHECKPOINTS[1:])
        integrity(dest);timings['total_training_setup_wall_seconds']=time.monotonic()-start
        write(dest/'runtime.json',timings);write(dest/'status.json',dict(state='TRAINED',scene=scene,runtime=timings))
    except Exception as e:
        write(dest/'status.json',dict(state='FAILED',scene=scene,error=str(e),pid=os.getpid()));raise

def supervise(out):
    processes=[]
    for gpu,scene in enumerate(SCENES):
        f=(out/scene/'logs/worker.log').open('w')
        p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--out',str(out),'--scene',scene],
            cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=f,stderr=subprocess.STDOUT)
        processes.append((scene,p,f))
    write(out/'processes.json',{s:dict(pid=p.pid,gpu=i) for i,(s,p,f) in enumerate(processes)})
    while any(p.poll() is None for s,p,f in processes):
        write(out/'status.json',dict(state='RUNNING',scenes={s:json.loads((out/s/'status.json').read_text()) for s,p,f in processes}))
        time.sleep(10)
    for s,p,f in processes:f.close()
    if any(p.returncode for s,p,f in processes):
        write(out/'status.json',dict(state='FAILED',returncodes={s:p.returncode for s,p,f in processes}));raise RuntimeError('Training worker failed')
    for p,h in json.loads((out/'training_implementation_manifest.json').read_text()).items():assert sha(ROOT/p)==h
    write(out/'status.json',dict(state='TRAINED',completed=2,evaluation='pending'))
    subprocess.run([sys.executable,'-u',str(ROOT/'tools/evaluate_j_baseline30k.py'),'--out',str(out)],cwd=ROOT,check=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','worker']);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--scene',choices=SCENES);a=parser.parse_args()
    assert os.environ.get('CONDA_PREFIX')=='/home/robot/colongraph/.env','Use project Conda'
    if a.action=='prepare':prepare(a.out)
    elif a.action=='run':supervise(a.out)
    else:worker(a.out,a.scene)
