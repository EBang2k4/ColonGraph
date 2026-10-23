"""Exactly two matched A/J continuations on the second local colon sequence.

Shared 2k initialization and G3 2k->5k prefix are setup, not extra full runs.
No loss/architecture changes, tuning, or motion-baseline full training.
"""
import argparse, copy, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
ROOT=Path(__file__).resolve().parents[1]
SCENE='c1_sigmoid1_t4_v4'
OUT=ROOT/'output/experiments/cross_scene_aj'/SCENE
LOGS=ROOT/'logs/cross_scene_aj'/SCENE
DATA=ROOT.parent/'ColonSplat/data/coloncrafter'/SCENE
SOURCE=OUT/'common_5000'
KNN=OUT/'setup/gaussian_knn99.pth'
REF=OUT/'setup/references/graph_forward_reference.pth'
CHECKPOINTS=[5000,10000,15000,20000,25000,30000]

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def prepare():
    import numpy as np
    from PIL import Image
    if (OUT/'protocol.json').exists():raise FileExistsError('Preserve prepared study')
    rgb=sorted((DATA/'color').glob('*.png'));depth=sorted((DATA/'depth').glob('*.npy'))
    assert len(rgb)==len(depth)==297
    assert [p.stem for p in rgb]==[p.stem for p in depth]
    poses=np.loadtxt(DATA/'pose.txt',delimiter=',');assert poses.shape==(297,16) and np.isfinite(poses).all()
    for r,d in zip(rgb,depth):
        a=np.load(d);assert a.shape[-2:]==Image.open(r).size[::-1] and np.isfinite(a).all()
    OUT.mkdir(parents=True,exist_ok=True);LOGS.mkdir(parents=True,exist_ok=True)
    template=json.loads((ROOT/'output/experiments/ggas_v2/c1_sigmoid2_t4_v4/configs/block00_A_BASE.json').read_text())
    cfg=copy.deepcopy(template);cfg['ModelParams'].update(source_path=str(DATA),model_path=str(OUT/'setup/warmup'))
    h=cfg['ModelHiddenParams'];h.update(node_initialization_reference='',gaussian_knn_reference='',gaussian_knn_save=str(KNN),
        graph_forward_mode='none',graph_forward_reference='',graph_forward_motion_reference='',graph_forward_monitor_iterations='',
        fine_warmup_reference='',fine_warmup_state='',fine_stage_state_save='',fine_stop_iteration=2000,
        full_state_checkpoints='2000',ggas_variant='none',ggas_candidates='',ggas_sc_edges='',geometry_study_enabled=False,
        geometry_blend_cache='',geometry_blend_mode='idw',matched_texture_audit=False,connectivity_ablation_audit=False,
        texture_enabled=False,freeze_geometry=False,reset_fine_rng=False)
    write(OUT/'configs/warmup.json',cfg)
    cfg=copy.deepcopy(cfg);cfg['ModelParams']['model_path']=str(SOURCE)
    cfg['ModelHiddenParams'].update(gaussian_knn_reference=str(KNN),gaussian_knn_save='',graph_forward_mode='g3',graph_forward_reference=str(REF),
        fine_warmup_reference=str(OUT/'setup/warmup/point_cloud/iteration_2000'),fine_warmup_state=str(OUT/'setup/warmup/fine_state_2000.pth'),
        fine_stop_iteration=5000,full_state_checkpoints='5000')
    write(OUT/'configs/common.json',cfg)
    rows=[]
    for name,variant in [('A_BASE','A_BASE'),('J_GGAS_V2','J_NO_MOTION_SELECTION')]:
        cfg=copy.deepcopy(template);cfg['ModelParams'].update(source_path=str(DATA),model_path=str(OUT/'runs'/('block00_'+name)))
        cfg['ModelHiddenParams'].update(node_initialization_reference='',gaussian_knn_reference=str(KNN),gaussian_knn_save='',
            graph_forward_reference=str(REF),graph_forward_motion_reference='',graph_forward_monitor_iterations='',
            fine_warmup_reference=str(SOURCE/'point_cloud/iteration_5000'),fine_warmup_state=str(SOURCE/'fine_state_5000.pth'),
            geometry_blend_cache=str(OUT/'caches/A_idw.pth'),ggas_candidates=str(OUT/'candidates.pth'),ggas_variant=variant,
            ggas_study_version=2,ggas_lr=.001,ggas_lambda=.01,ggas_ramp_end=6000,ggas_sc_edges='',
            geometry_study_enabled=True,geometry_study_refresh='none',geometry_study_freeze_iteration=20000,geometry_study_camera_seed=0,
            fine_stop_iteration=30000,full_state_checkpoints=','.join(map(str,CHECKPOINTS)),fine_stage_state_save='',
            texture_enabled=False,freeze_geometry=False,reset_fine_rng=False,matched_texture_audit=True,matched_texture_source_iteration=5000)
        assert cfg['OptimizationParams']==template['OptimizationParams'] and cfg['OptimizationParams']['knn_weight']==.01
        write(OUT/'configs'/('block00_'+name+'.json'),cfg)
        rows.append(dict(name='block00_'+name,variant=name,config=str(OUT/'configs'/('block00_'+name+'.json')),
            path=cfg['ModelParams']['model_path'],status='PENDING',block=0,seed=0))
    write(OUT/'matrix.json',rows)
    write(OUT/'data_manifest.json',{str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in [DATA/'pose.txt',DATA/'points3d.ply',*rgb,*depth]})
    write(OUT/'protocol.json',dict(scene=SCENE,variants=['A_BASE','J_GGAS_V2'],full_runs=2,blocks=1,camera_seed=0,
        shared_setup='100 coarse + fine1..2000 initialization, then shared G3 fine2001..5000; all full30k schedules retained',
        continuation=[5001,30000],Adam_calls_per_branch=25000,geometry_freeze_after_Adam=20000,R6_until=30000,
        KNN99_weight=.01,G3='same E4 surface topology / repairedK4 / 16TRAIN warmup trajectory gate',R6='independent fixed appearanceK4',
        J='same spatial-only connected-growth K8, warm-start mass cap0.20, ramp5000..6000, unbounded logits, Adam0.001 and KL0.01',
        no_architecture_loss_changes=True,no_TEST_selection=True,checkpoints=CHECKPOINTS,
        motion_monitor_difference='No second-scene K0 full baseline exists. External 2x-K0 displacement abort monitor disabled identically for A/J and shared G3 prefix; no substitute threshold or extra full run. G3 trajectory gating itself unchanged. This is a runtime-guard difference from historical scene2, not a loss/architecture/LR difference.',
        stability='geometry-aware same-ID tangent deformation/renderer-depth support; thresholds locked before outcomes; primary20k',
        generalization='Method transfer via independent training on another sequence, not zero-shot model transfer; one block, no cross-seed significance.',
        promotion='No automatic main promotion; A/main, J/research, K not promoted, L not implemented'))
    core=json.loads((ROOT/'output/experiments/ggas_v3/c1_sigmoid2_t4_v4/training_implementation_manifest.json').read_text())
    assert all(sha(ROOT/p)==s for p,s in core.items());write(OUT/'training_implementation_manifest.json',core)

def command(config,saves):
    cfg=OUT/'configs'/(config+'.json');dest=Path(json.loads(cfg.read_text())['ModelParams']['model_path'])
    return [sys.executable,'-u','train.py','--configs',str(cfg),'--expname',str(dest.relative_to(ROOT/'output')),
        '--save_iterations',*map(str,saves),'--test_iterations','999999']

def stage(name,config,saves):
    marker=OUT/'setup'/('stage_'+name+'.json')
    if marker.exists():assert json.loads(marker.read_text())['pass_'];return
    dest=Path(json.loads((OUT/'configs'/(config+'.json')).read_text())['ModelParams']['model_path'])
    if dest.exists():raise FileExistsError('Preserve incomplete training: '+str(dest))
    write(OUT/'status.json',dict(state='RUNNING',stage=name,pid=os.getpid(),conda=os.environ['CONDA_PREFIX']))
    start=time.monotonic()
    with (LOGS/(name+'.log')).open('w') as f:subprocess.run(command(config,saves),cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    write(marker,dict(pass_=True,seconds=time.monotonic()-start,config=config))

def reference():
    import torch
    from tools.experiment_common import load_scene
    from tools.audit_sparse_run import read_config
    from tools.build_node_connectivity_graphs import build_e4,co_influence_candidates,audit_graph
    from tools import build_graph_forward_reference as builder
    warm=OUT/'setup/warmup';cfg=read_config(warm/'cfg_args');cfg['gaussian_knn_reference']=str(KNN)
    (warm/'cfg_args').write_text(str(argparse.Namespace(**cfg)))
    m,_,_,_,_=load_scene(warm,2000);g=m.deformation_graph
    graph=build_e4(m._xyz.detach().cpu(),g.node_xyz.detach().cpu(),g.gaussian_node_indices.detach().cpu(),g.gaussian_node_weights.detach().cpu(),torch.load(KNN,map_location='cpu',weights_only=True))
    keys,strength,count=co_influence_candidates(g.gaussian_node_indices.cpu(),g.gaussian_node_weights.cpu(),len(g.node_xyz))
    audit=audit_graph(graph,keys,strength,count);assert audit['connected_components']==1 and audit['isolated_nodes']==0
    REF.parent.mkdir(parents=True,exist_ok=True)
    surface=REF.parent/'surface_graph.pth';torch.save({k:v.cpu() if torch.is_tensor(v) else v for k,v in graph.state_dict_payload().items()},surface)
    write(REF.parent/'surface_audit.json',audit);del m;torch.cuda.empty_cache()
    # Original builder computes exactly the original G3 topology/trajectory gate.
    # Skip only unavailable K0-derived runtime monitor references; no full K0 run.
    original=builder.motion_reference;builder.motion_reference=lambda *args,**kwargs:{}
    oldargv=sys.argv
    try:
        sys.argv=['builder','--common',str(warm),'--surface-graph',str(surface),'--k0',str(warm),'--output',str(REF.parent)];builder.main()
    finally:builder.motion_reference=original;sys.argv=oldargv

def candidates():
    import torch
    from tools.experiment_common import load_scene
    from utils.exact_split_branchpoint import digest
    from utils.graph_forward import _dense_adjacency,k4_topology_tensors
    from utils.geometry_blending import graph_spacing,validate_cache
    from utils.ggas_v2 import connected_growth,warm_start_prior
    model,_,_,_,_=load_scene(SOURCE,5000);g=model.deformation_graph
    ref=torch.load(REF,map_location='cuda',weights_only=True);n=len(g.node_xyz)
    ids=g.gaussian_node_indices;w=g.gaussian_node_weights;x=model._xyz.detach();nodes=g.node_xyz
    assert n==2000 and torch.equal(ids,ref['graph_k4_indices']) and torch.equal(w,ref['graph_k4_weights'])
    assert k4_topology_tensors(ids,ref['surface_edge_indices'],n)['connected'].all()
    adj=_dense_adjacency(ref['surface_edge_indices'],n);two=adj|(adj.float()@adj.float()>0);two.fill_diagonal_(True)
    h,_=graph_spacing(nodes,ref['surface_edge_indices']);anchor=ids.gather(1,w.argmax(1)[:,None])[:,0]
    indices=[];prior=[]
    for lo in range(0,len(x),512):
        dist=torch.cdist(x[lo:lo+512],nodes,compute_mode='donot_use_mm_for_euclid_dist')
        score=dist.square()/(2*h[anchor[lo:lo+512],None].square())
        q=connected_growth(ids[lo:lo+512],adj,two,score)
        indices.append(q.cpu());prior.append(warm_start_prior(score,q,w[lo:lo+512]).cpu())
    ii=torch.cat(indices);pp=torch.cat(prior)
    assert torch.equal(ii[:,:4],ids.cpu()) and (ii.sort(1).values.diff(dim=1)>0).all()
    assert torch.isfinite(pp).all() and (pp>0).all() and torch.allclose(pp.sum(1),torch.ones(len(x)),atol=2e-6,rtol=0)
    reach=adj[ii.cuda()[:,:,None],ii.cuda()[:,None,:]]|torch.eye(8,dtype=torch.bool,device='cuda')[None]
    for k in range(8):reach|=reach[:,:,k:k+1]&reach[:,k:k+1,:]
    assert reach.all()
    torch.save(dict(format_version=1,spatial_graph_indices=ii,spatial_graph_prior=pp,source_xyz=x.cpu(),node_xyz=nodes.cpu(),
        original_indices=ids.cpu(),original_weights=w.cpu(),original_indices_sha256=digest(ids),original_weights_sha256=digest(w),source_xyz_sha256=digest(x)),OUT/'candidates.pth')
    invariants=dict(nodes=digest(nodes),indices=digest(ids),appearance=digest(model.appearance_graph.state_dict()),
        g3_graph=digest({k:v for k,v in model.graph_motion_residual.state_dict().items() if not k.startswith('decoder.')}))
    cache=dict(format_version=1,mode='idw',beta=2.5,weights=w.cpu(),weights_sha256=digest(w),baseline_weights_sha256=digest(w),invariants=invariants,
        node_spacing=h.cpu(),node_radius=(2.5*h).cpu(),fallback=torch.zeros(len(x),dtype=torch.bool),provenance=dict(scene=SCENE,source=str(SOURCE)))
    validate_cache(model,cache);(OUT/'caches').mkdir(exist_ok=True);torch.save(cache,OUT/'caches/A_idw.pth')
    write(OUT/'candidate_audit.json',dict(pass_=True,Gaussians=len(x),nodes=n,connected=True,fallbacks=0,
        exact_original4=True,spatial_only=True,mass_cap=.20,new_mass_mean=float(pp[:,4:].sum(1).mean()),candidate_sha256=sha(OUT/'candidates.pth')))
    inputs=[KNN,REF,OUT/'candidates.pth',OUT/'caches/A_idw.pth',SOURCE/'fine_state_5000.pth',*sorted((SOURCE/'point_cloud/iteration_5000').iterdir())]
    write(OUT/'source_manifest.json',{str(p):sha(p) for p in inputs if p.is_file()})

def integrity(path):
    from tools import run_geometry_full30k as base
    base.SOURCE=SOURCE
    r=base.integrity(path,30000,20000)
    import torch
    from utils.exact_split_branchpoint import digest
    if (path/'skinning_initial.json').exists():
        first=torch.load(path/'point_cloud/iteration_5000/skinning.pth',map_location='cpu',weights_only=True)
        mid=torch.load(path/'point_cloud/iteration_20000/skinning.pth',map_location='cpu',weights_only=True)
        last=torch.load(path/'point_cloud/iteration_30000/skinning.pth',map_location='cpu',weights_only=True)
        assert digest(mid)==digest(last) and torch.equal(first['state_dict']['target'],last['state_dict']['target'])
        assert last['format_version']==1
    return r

def full():
    if not (OUT/'protocol.json').exists():prepare()
    stage('warmup','warmup',[1,2000])
    if not REF.exists():reference()
    stage('common','common',[5000])
    if not (OUT/'candidates.pth').exists():candidates()
    rows=json.loads((OUT/'matrix.json').read_text());active=[]
    for gpu,row in enumerate(rows):
        if row['status']=='COMPLETED':continue
        dest=Path(row['path'])
        if dest.exists():raise FileExistsError('No automatic training retry or overwrite: '+str(dest))
        f=(LOGS/(row['name']+'.log')).open('w')
        p=subprocess.Popen(command(row['name'],CHECKPOINTS[1:]),cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=f,stderr=subprocess.STDOUT)
        row.update(status='RUNNING',gpu=gpu,pid=p.pid,started_unix=time.time());active.append((row,p,f,time.monotonic()))
    write(OUT/'matrix.json',rows)
    while active:
        for row,p,f,start in active[:]:
            code=p.poll()
            if code is None:continue
            f.close();active.remove((row,p,f,start));row.update(wall_seconds=time.monotonic()-start,returncode=code)
            if code:row['status']='FAILED'
            else:integrity(Path(row['path']));row['status']='COMPLETED'
        write(OUT/'matrix.json',rows);write(OUT/'status.json',dict(state='RUNNING',stage='paired_training',completed=sum(r['status']=='COMPLETED' for r in rows),total=2,active=[r['name'] for r,p,f,s in active],pid=os.getpid()))
        if active:time.sleep(5)
    if any(r['status']!='COMPLETED' for r in rows):raise RuntimeError('Preserved failed paired run; no automatic retry')
    for path,s in json.loads((OUT/'source_manifest.json').read_text()).items():assert sha(Path(path))==s
    for path,s in json.loads((OUT/'training_implementation_manifest.json').read_text()).items():assert sha(ROOT/path)==s
    write(OUT/'status.json',dict(state='TRAINED',stage='evaluation_pending',completed=2,total=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','full']);a=p.parse_args()
    assert os.environ.get('CONDA_PREFIX')=='/home/robot/colongraph/.env','Use actual project Conda'
    try:prepare() if a.action=='prepare' else full()
    except Exception as e:write(OUT/'status.json',dict(state='FAILED',error=str(e),pid=os.getpid()));raise

if __name__=='__main__':main()
