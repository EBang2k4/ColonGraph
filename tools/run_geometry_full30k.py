"""Durable sequential GPU queue: 3 paired blocks x 6 full 5k->30k continuations."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import copy
import json
import os
import random
import shutil
import subprocess
import time
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
SCENE='c1_sigmoid2_t4_v4'
OUT=ROOT/'output/experiments/geometry_full30k'/SCENE
LOGS=ROOT/'logs/geometry_full30k'/SCENE
SOURCE=ROOT/'output/zero_mean_texture'/SCENE/'common_5000'
PRIOR=ROOT/'output/experiments/geometry_blending'/SCENE
SEEDS=[0,101,202]
VARIANTS={'A_BASE':('none','idw',.01),'B_WENDLAND':('none','wendland_c2',.01),
    'C_WEIGHT_REFRESH':('weights_only','idw',.01),'D_K4_REFRESH':('neighbors_and_weights','idw',.01),
    'E_NO_KNN':('none','idw',0.),'F_K4_REFRESH_NO_KNN':('neighbors_and_weights','idw',0.)}
CHECKPOINTS=[5000,10000,15000,20000,25000,30000]


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def prepare():
    from tools.evaluate_texture_experiment import file_hash
    prior=json.loads((PRIOR/'source_manifest.json').read_text())
    for name,row in prior.items():
        if ('common_5000' in name or 'references/' in name or 'gaussian_knn99' in name) and file_hash(name)!=row['sha256']:
            raise ValueError('Immutable input changed: '+name)
    baseline=json.loads((ROOT/'output/zero_mean_texture'/SCENE/'configs/A.json').read_text())
    (OUT/'caches').mkdir(parents=True,exist_ok=True)
    for n in ('A_idw','C_beta2_5'):
        destination=OUT/'caches'/(n+'.pth')
        if not destination.exists():shutil.copy2(PRIOR/'caches'/(n+'.pth'),destination)
        if file_hash(destination)!=file_hash(PRIOR/'caches'/(n+'.pth')):raise ValueError('Cache changed')
    matrix=[]
    for block,seed in enumerate(SEEDS):
        for variant,(refresh,kernel,knn) in VARIANTS.items():
            name=f'block{block:02d}_{variant}'
            cfg=copy.deepcopy(baseline)
            cfg['ModelParams']['model_path']=str(OUT/'runs'/name)
            cfg['ModelHiddenParams'].update(geometry_study_enabled=True,geometry_study_refresh=refresh,
                geometry_study_refresh_interval=1000,geometry_study_freeze_iteration=20000,geometry_study_camera_seed=seed,
                geometry_blend_mode=kernel,geometry_blend_beta=2.5,
                geometry_blend_cache=str(OUT/'caches'/('C_beta2_5.pth' if kernel=='wendland_c2' else 'A_idw.pth')),
                freeze_geometry=False,texture_enabled=False,reset_fine_rng=False,matched_texture_audit=True,
                matched_texture_source_iteration=5000,fine_stop_iteration=30000,
                full_state_checkpoints=','.join(map(str,CHECKPOINTS)),fine_stage_state_save='')
            cfg['OptimizationParams']['knn_weight']=knn
            original=copy.deepcopy(cfg['OptimizationParams']);original['knn_weight']=.01
            if original!=baseline['OptimizationParams']:raise ValueError('Unauthorized optimizer change')
            write(OUT/'configs'/(name+'.json'),cfg)
            matrix.append(dict(name=name,block=block,seed=seed,variant=variant,refresh=refresh,kernel=kernel,knn_weight=knn,
                config=str(OUT/'configs'/(name+'.json')),path=str(OUT/'runs'/name),status='PENDING'))
    # Tests exercise the exact freeze implementation at 5010 on every variant.
    for variant in VARIANTS:
        cfg=json.loads((OUT/'configs'/('block00_'+variant+'.json')).read_text())
        cfg['ModelParams']['model_path']=str(OUT/'preflight'/variant)
        cfg['ModelHiddenParams'].update(geometry_study_freeze_iteration=5010,fine_stop_iteration=5020,full_state_checkpoints='5010,5020')
        write(OUT/'configs'/('preflight_'+variant+'.json'),cfg)
    write(OUT/'matrix.json',matrix)
    write(OUT/'protocol.json',dict(source=str(SOURCE),source_iteration=5000,stop_iteration=30000,
        optimizer_calls=25000,geometry_training=[5001,20000],appearance_training=[5001,30000],
        checkpoints=CHECKPOINTS,primary_geometry=20000,primary_final=30000,blocks=3,variants=VARIANTS,camera_seeds=SEEDS,
        refresh_iterations=list(range(5000,20000,1000)),refresh_phase='source before first step, thereafter after Adam before next iteration; no 20k refresh',
        diagnostic_train_views=[0,144,264],visual_selection='GT-only high-gradient views 129,177,121 from previous predeclared GT ranking, plus descriptive best/worst views after evaluation',
        camera_rng='Restore common full source first; reseed only Python camera RNG within each block. Torch/CUDA/NumPy RNG remains source.',
        schedule_difference_from_historical_joint_baseline='All six variants now freeze geometry after 20k per this user task. Historical baseline trained geometry through 30k. Opt-in final Adam call ensures exactly 25000 fine calls; original entrypoint skipped final iteration.',
        fold='Historical official source absent locally; GT-P80 explicitly named new region-error diagnostic, not Fold',
        nondeterminism='Original graph index_add and rasterizer backward atomics retained; no bitwise trajectory claim',
        no_cherry_pick=True))


def command(name,preflight=False):
    cfg=OUT/'configs'/('preflight_'+name+'.json' if preflight else name+'.json')
    dest=json.loads(cfg.read_text())['ModelParams']['model_path']
    saves=[5020] if preflight else CHECKPOINTS[1:]
    return [sys.executable,'-u','train.py','--configs',str(cfg),'--expname',str(Path(dest).relative_to(ROOT/'output')),
        '--save_iterations',*map(str,saves),'--test_iterations',*map(str,saves)]


def integrity(path,stop,freeze):
    import torch
    from utils.exact_split_branchpoint import digest
    source=torch.load(SOURCE/'fine_state_5000.pth',map_location='cpu',weights_only=False)
    initial=json.loads((path/'matched_initial.json').read_text())
    final=torch.load(path/f'fine_state_{stop}.pth',map_location='cpu',weights_only=False)
    frozen=torch.load(path/f'fine_state_{freeze}.pth',map_location='cpu',weights_only=False)
    seq=[json.loads(line) for line in (path/'connectivity_camera_sequence.jsonl').read_text().splitlines()]
    seed=json.loads((path/'geometry_study_initial.json').read_text())['camera_seed']
    rng=random.Random(seed);expected=[source['train_camera_order'][rng.randint(0,len(source['train_camera_order'])-1)] for _ in range(stop-5000)]
    frozen_rows=[json.loads(line) for line in (path/'frozen_geometry_audit.jsonl').read_text().splitlines()]
    checks=dict(source_model_exact=initial['non_texture_model_sha256']==source['model_sha256'],
        source_Adam_exact=initial['optimizer_sha256']==digest(source['optimizer']),
        source_RNG_exact=initial['rng_sha256']==digest({k:source[k] for k in ('torch_rng','cuda_rng','python_rng','numpy_rng')}),
        camera_order_exact=initial['camera_order']==source['train_camera_order'],
        camera_replay_exact=[s['camera_name'] for s in seq]==expected,actual_calls=len(seq)==stop-5000,
        final_iteration=final['iteration']==stop,geometry_frozen_raw_exact=all(torch.equal(frozen['exact_raw'][k],final['exact_raw'][k]) for k in ('_xyz','_scaling','_rotation','_opacity')),
        frozen_periodic_all_pass=bool(frozen_rows) and all(all(r[k] for k in ('geometry_exact','frozen_gradients_absent','frozen_adam_unchanged')) for r in frozen_rows),
        appearance_continues=not torch.equal(frozen['exact_raw']['features'],final['exact_raw']['features']),
        final_all_raw_finite=all(torch.isfinite(v).all().item() for v in final['exact_raw'].values()))
    old_graph=torch.load(SOURCE/'point_cloud/iteration_5000/deformation_nodes.pth',map_location='cpu',weights_only=True)
    new_graph=torch.load(path/f'point_cloud/iteration_{stop}/deformation_nodes.pth',map_location='cpu',weights_only=True)
    mid_graph=torch.load(path/f'point_cloud/iteration_{freeze}/deformation_nodes.pth',map_location='cpu',weights_only=True)
    checks.update(appearance_graph_exact=digest(old_graph['appearance_graph'])==digest(new_graph['appearance_graph']),
        nodes_fixed=torch.equal(old_graph['graph']['node_xyz'],new_graph['graph']['node_xyz']),
        G3_edges_fixed=all(torch.equal(old_graph['graph_motion_residual'][k],new_graph['graph_motion_residual'][k]) for k in ('edge_indices','edge_weights','canonical_xyz','weight_denominator')),
        frozen_native_geometry_exact=all(digest(mid_graph[k])==digest(new_graph[k]) for k in ('graph','mlp','graph_motion_residual')))
    cache=torch.load(path/f'point_cloud/iteration_{stop}/geometry_blending.pth',map_location='cpu',weights_only=True)
    checks['cached_weights_exact']=digest(new_graph['graph']['gaussian_node_weights'])==cache['weights_sha256']
    result=dict(pass_=all(checks.values()),checks=checks,seed=seed,actual_calls=len(seq),
        source_iteration=5000,last_iteration=stop,freeze_iteration=freeze,model_sha256=final['model_sha256'],
        camera_sequence_sha256=digest([r['camera_name'] for r in seq]),last_refresh=final['geometry_study_meta']['last_refresh'])
    write(path/'integrity.json',result)
    if not result['pass_']:raise RuntimeError('Run integrity failed '+str(path)+' '+str(checks))
    return result


def provenance():
    from tools.evaluate_texture_experiment import file_hash
    inputs={p:row for p,row in json.loads((PRIOR/'source_manifest.json').read_text()).items()
        if ('common_5000' in p or 'references/' in p or 'gaussian_knn99' in p)}
    for p in (OUT/'caches').glob('*.pth'):inputs[str(p)]=dict(sha256=file_hash(p),bytes=p.stat().st_size)
    write(OUT/'source_manifest.json',inputs)
    names=['train.py','arguments/__init__.py','scene/gaussian_model.py','utils/geometry_refresh.py','utils/geometry_blending.py',
        'utils/texture_training.py','utils/graph_forward.py','scene/deformation.py','gaussian_renderer/__init__.py',
        'tools/run_geometry_full30k.py','tests/test_geometry_refresh.py']
    write(OUT/'training_implementation_manifest.json',{n:file_hash(ROOT/n) for n in names})
    for n in names:
        dest=OUT/'provenance/training_code'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
    for args,n in [(['git','rev-parse','HEAD'],'commit.txt'),(['git','diff','--binary'],'after.patch'),(['nvidia-smi'],'GPU.txt'),
            ([sys.executable,'-m','pip','freeze'],'environment.txt')]:
        r=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,check=True);(OUT/'provenance'/n).write_text(r.stdout)
    import torch
    write(OUT/'provenance/runtime.json',dict(conda=os.environ.get('CONDA_PREFIX'),torch=torch.__version__,cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),deterministic=torch.are_deterministic_algorithms_enabled(),
        concurrency=2,one_training_process_per_gpu=True))


def launch_queue(matrix,preflight=False):
    pending=list(matrix);active={};done=[];fatal=None
    try:
        while pending or active:
            for gpu in (0,1):
                if gpu in active or not pending or fatal is not None:continue
                row=pending.pop(0);name=row['name'];dest=OUT/'preflight'/name if preflight else Path(row['path'])
                stop,freeze=(5020,5010) if preflight else (30000,20000)
                if (dest/f'fine_state_{stop}.pth').exists():
                    integrity(dest,stop,freeze);row['status']='COMPLETED';done.append(row);continue
                if dest.exists():raise FileExistsError('Partial run preserved; explicit recovery required: '+str(dest))
                logfile=LOGS/('preflight_'+name+'.log' if preflight else name+'.log');stream=logfile.open('w')
                p=subprocess.Popen(command(name,preflight),cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=stream,stderr=subprocess.STDOUT)
                row.update(status='RUNNING',gpu=gpu,pid=p.pid,started_utc=datetime.now(timezone.utc).isoformat())
                active[gpu]=(row,p,stream,time.monotonic())
            for gpu,(row,p,stream,started) in list(active.items()):
                code=p.poll()
                if code is None:continue
                stream.close();del active[gpu]
                row.update(returncode=code,wall_seconds=time.monotonic()-started,finished_utc=datetime.now(timezone.utc).isoformat())
                if code:
                    row['status']='FAILED';done.append(row);fatal='Run failed: '+row['name'];continue
                integrity(OUT/'preflight'/row['name'] if preflight else Path(row['path']),5020 if preflight else 30000,5010 if preflight else 20000)
                row['status']='COMPLETED';done.append(row)
            write(OUT/('preflight_matrix.json' if preflight else 'matrix.json'),done+[v[0] for v in active.values()]+pending)
            write(OUT/'status.json',dict(state='RUNNING',stage='preflight' if preflight else 'training',completed=len(done),total=len(matrix),
                active=[dict(name=v[0]['name'],gpu=gpu,pid=v[1].pid) for gpu,v in active.items()],updated_utc=datetime.now(timezone.utc).isoformat()))
            if fatal is not None and not active:raise RuntimeError(fatal)
            time.sleep(5)
    finally:
        for row,p,stream,_ in active.values():
            # Preserve a healthy branch if its peer failed; no GPU work is killed.
            stream.close()
    return done


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','preflight','full']);action=parser.parse_args().action
    if not (OUT/'protocol.json').exists():prepare()
    if action=='prepare':return
    try:
        if not (OUT/'preflight.json').exists():
            rows=[dict(name=n) for n in VARIANTS];launch_queue(rows,True)
            write(OUT/'preflight.json',dict(pass_=True,variants=list(VARIANTS),freeze_rehearsal_iteration=5010))
        if action=='preflight':return
        provenance();rows=json.loads((OUT/'matrix.json').read_text());launch_queue(rows)
        write(OUT/'status.json',dict(state='TRAINED',stage='evaluation',updated_utc=datetime.now(timezone.utc).isoformat()))
        with (LOGS/'evaluation.log').open('w') as stream:
            subprocess.run([sys.executable,'-u','tools/evaluate_geometry_full30k.py'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        write(OUT/'status.json',dict(state='COMPLETED',stage='report',updated_utc=datetime.now(timezone.utc).isoformat()))
    except Exception as e:
        write(OUT/'status.json',dict(state='FAILED',error=str(e),updated_utc=datetime.now(timezone.utc).isoformat()));raise


if __name__=='__main__':main()
