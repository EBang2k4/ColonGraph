import contextlib,copy,hashlib,json,random
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def rng_state():
    return dict(torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),numpy_rng=np.random.get_state())

@contextlib.contextmanager
def preserve_rng():
    q=rng_state()
    try:yield
    finally:
        torch.set_rng_state(q['torch_rng']);torch.cuda.set_rng_state_all(q['cuda_rng']);random.setstate(q['python_rng']);np.random.set_state(q['numpy_rng'])

def configuration(data,out):
    import argparse
    from arguments import ModelParams,ModelHiddenParams,OptimizationParams,PipelineParams
    parser=argparse.ArgumentParser();groups=[ModelParams(parser),ModelHiddenParams(parser),OptimizationParams(parser),PipelineParams(parser)]
    c=parser.parse_args([])
    for key,value in json.loads((ROOT/'configs/m2.json').read_text()).items():setattr(c,key,value)
    c.source_path=str(data);c.model_path=str(out);c.dynamic_point_initialization='depth'
    c.node_initialization_reference='';c.gaussian_knn_reference='';c.gaussian_knn_save=''
    return c,groups

def optimizer(m,c):
    m.training_setup(c);m.add_graph_residual_optimizer_group()
    from utils.ggas import add_optimizer
    add_optimizer(m)
    if hasattr(m,'harmonic_nodes'):
        from utils.stage12 import register_optimizer
        register_optimizer(m)

def migrate_optimizer(m,c,old):
    """Preserve existing Adam moments when adding the two appearance heads."""
    optimizer(m,c);fresh=m.optimizer.state_dict();groups={g['name']:g for g in old['param_groups']}
    for g in fresh['param_groups']:
        if g['name'] not in groups:continue
        prev=groups[g['name']]
        for a,b in zip(prev['params'],g['params']):
            if a in old['state']:fresh['state'][b]=copy.deepcopy(old['state'][a])
        for k,v in prev.items():
            if k!='params':g[k]=v
    m.optimizer.load_state_dict(fresh)

def schedule(m,c,k,is_c3vd,late):
    m.update_learning_rate(k,stage='fine')
    if m.skinning is not None:m.skinning.set_iteration(k)
    if hasattr(m.appearance_model,'micro_iteration'):m.appearance_model.micro_iteration.fill_(k)
    if hasattr(m,'harmonic_nodes'):m.harmonic_nodes.iteration.fill_(k)
    if is_c3vd:
        from utils.stage3 import update_training
        update_training(m,c,k,'fine')
        from utils.stage8 import dynamics
        c.depth_weight=dynamics(k,'P')[1];c.knn_weight=.01
        if hasattr(m,'harmonic_nodes'):
            next(g for g in m.optimizer.param_groups if g['name']=='harmonic_nodes')['lr']=.001*dynamics(k,'P')[0]
    elif late:
        from utils.stage9 import cosine
        factor=cosine(k)
        fixed=dict(f_dc=c.feature_lr,deformation_table=c.feature_lr,opacity=c.opacity_lr,scaling=c.scaling_lr,rotation=c.rotation_lr,skinning=c.ggas_lr,harmonic_nodes=.001)
        for g in m.optimizer.param_groups:g['lr']=fixed.get(g['name'],g['lr'])*factor

def protect_inputs(data,out):
    """Training cannot read GT PLY or any old experiment artifact."""
    import os,sys
    data=Path(data).resolve();out=Path(out).resolve()
    def guard(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).absolute()
        if p==out or out in p.parents:return
        if p.suffix.lower() in {'.pth','.pt','.ckpt'}:
            raise RuntimeError('Fresh training forbids input model checkpoints: '+str(p))
        if '/output/experiments/' in str(p) or any(part.startswith('ColonGraph-stage') or part in ('ColonGraph-texture','ColonGraph_M2_checkpoint_archive') for part in p.parts):
            raise RuntimeError('Old experiment dependency: '+str(p))
        if (data/'ply') in p.parents and p.suffix.lower()=='.ply':raise RuntimeError('GT PLY is evaluation-only: '+str(p))
    sys.addaudithook(guard)
