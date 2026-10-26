"""Fresh coarse -> K4 -> G3 -> J8 -> frozen TRAIN selector -> M2 training."""
import copy,json,random,time
import numpy as np
import torch
from tqdm import tqdm
from gaussian_renderer import render
from scene import Scene
from scene.gaussian_model import GaussianModel
from scene.deformation_graph import SparseDeformationGraph
from baseline.common import configuration,optimizer,migrate_optimizer,preserve_rng,schedule,seed,write,protect_inputs
from baseline.data import configure as configure_data
from baseline.graphs import build_g3,build_j8,build_m2
from baseline.checkpoint import save
from utils.loss_utils import TV_loss,ssim
from utils.stage10 import depth_loss
from utils.stage7 import micro_penalty,R6AdditiveMicro
from utils.stage14 import energy
from utils.exact_split_branchpoint import model_digest,digest

def upgrade_appearance(m,c):
    previous=m.appearance_model.state_dict();c.appearance_mode=m.args.appearance_mode='r6_additive_micro'
    c.r6_additive_cap=m.args.r6_additive_cap=.25
    with preserve_rng():app=R6AdditiveMicro(m.args,2000).cuda()
    result=app.load_state_dict(previous,strict=False)
    assert set(result.missing_keys)=={'additive_head.weight','additive_head.bias','micro_head.weight','micro_head.bias','micro_iteration'}
    m.appearance_model=app

def objective(m,q,cam,c,k,coarse=False,is_c3vd=False,late=False):
    rgb=q['render'][None];gt=cam.original_image[:3].cuda().float()[None]
    pred=q['depth'].reshape(1,1,*q['depth'].shape[-2:]);dep=cam.original_depth.cuda().float().reshape_as(pred)
    valid=torch.isfinite(dep)&(dep>0)&torch.isfinite(pred)
    l1=(rgb-gt).abs().mean()
    if not coarse and is_c3vd:
        from utils.photometric_unlock import rgb_loss
        photo,_=rgb_loss(rgb,gt,l1,'charbonnier_psnr_ssim_handoff',iteration=k,profile='P')
    elif not coarse and late:photo=.8*((rgb-gt).square()+1e-6).sqrt().mean()+.15*(1-ssim(rgb,gt))/2+3*(rgb-gt).square().mean()
    else:photo=l1
    depth=depth_loss(pred,dep,valid,c.depth_weight,.30 if late and not is_c3vd and m.metric_depth_enabled else 0.)[0]
    if not coarse and late:
        knn=energy('M2',q['transformed_points'],m.stage14_ref,node=q['g2g_node'],micro=q['micro_offset'],base=m._xyz)
    else:
        x=q['transformed_points'];knn=.01*(x-x[m.closest_point_indices].mean(1)).square().sum(-1).mean()
    color=c.col_smooth_weight*(q['transformed_color']-q['transformed_color'].mean(0)).square().mean()
    total=photo+depth+knn+color+c.tv_weight*TV_loss(rgb)+c.lambda_res*q['dcol'].square().mean()
    if not coarse and m.skinning is not None:total=total+.01*m.skinning.kl()
    if not coarse and c.appearance_mode=='r6_additive_micro':total=total+micro_penalty(q['micro_raw'])
    if not coarse and hasattr(m,'harmonic_nodes'):total=total+m.harmonic_nodes.regularization()
    return total,dict(photo=photo,L1=l1,depth=depth,geometry=knn)

def gradients_step(m,loss,perform_step=True,reset_opacity=False):
    assert torch.isfinite(loss),float(loss.detach())
    loss.backward()
    for group in m.optimizer.param_groups:
        for p in group['params']:
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(),group['name']
                torch.nn.utils.clip_grad_value_(p,.5)
    if reset_opacity:m.reset_opacity()
    if perform_step:m.optimizer.step();m.optimizer.zero_grad(set_to_none=True)

@torch.no_grad()
def metric_gate(m,s,p,b,out,is_c3vd):
    if is_c3vd:
        m.metric_depth_enabled=False;write(out/'setup/scale_gate.json',dict(enabled=False,reason='estimated C3VD depth modality',TEST_used=False));return
    predictions=[];targets=[]
    for cam in s.getTrainCameras():
        q=render(cam,m,p,b);d=cam.original_depth.cuda().float().squeeze();pred=q['depth'].squeeze();valid=torch.isfinite(d)&(d>0)&torch.isfinite(pred)
        predictions.append(pred[valid].cpu().numpy());targets.append(d[valid].cpu().numpy())
    ratio=float(np.median(np.concatenate(predictions))/np.median(np.concatenate(targets)))
    m.metric_depth_enabled=.5<=ratio<=2.
    write(out/'setup/scale_gate.json',dict(enabled=m.metric_depth_enabled,ratio=ratio,range=[.5,2.],source='fresh post-coarse model; all TRAIN views',TEST_used=False))

def collect_selector(m,s,p,b,out):
    """A weight-frozen, arithmetic-mean pass over all TRAIN observations."""
    from baseline.selector import select
    from utils.adaptive_coverage_lock import SelectionStatistics
    before=model_digest(m);adam=digest(m.optimizer.state_dict());statistics=SelectionStatistics(len(m._xyz),'cuda')
    cams=sorted(s.getTrainCameras(),key=lambda x:(float(x.time),x.image_name));previous=m.graph_motion_residual.deterministic_aggregation
    m.graph_motion_residual.deterministic_aggregation=True
    with preserve_rng():
        for cam in tqdm(cams,desc='10k frozen TRAIN selector'):
            q=render(cam,m,p,b);rgb=(q['render']-cam.original_image[:3].cuda().float()).abs().mean()
            d=cam.original_depth.cuda().float().squeeze();pred=q['depth'].squeeze();valid=torch.isfinite(d)&(d>0)&torch.isfinite(pred)
            dl=depth_loss(pred[None,None],d[None,None],valid[None,None],.25,0.)[0]
            g=torch.autograd.grad(rgb,q['transformed_points'],retain_graph=True)[0].detach()
            v=torch.autograd.grad(dl,q['transformed_points'])[0].detach() if dl.requires_grad else torch.zeros_like(g)
            visible=q['visibility_filter']&torch.isfinite(g).all(1)&torch.isfinite(v).all(1)
            statistics.rgb[visible]+=g[visible].norm(dim=1);statistics.depth[visible]+=v[visible].norm(dim=1)
            statistics.vector[visible]+=g[visible]+v[visible];statistics.count[visible]+=1;statistics.steps+=1
            m.optimizer.zero_grad(set_to_none=True)
            del q,g,v,rgb,dl
        counts=statistics.count.clamp_min(1);statistics.rgb/=counts;statistics.depth/=counts;statistics.vector/=counts[:,None]
        times=np.unique([float(cam.time) for cam in cams]);times=times[np.rint(np.linspace(0,len(times)-1,16)).astype(int)]
        proposals,audit=select(m,statistics,times)
    m.graph_motion_residual.deterministic_aggregation=previous
    assert model_digest(m)==before and digest(m.optimizer.state_dict())==adam
    assert 16<=len(proposals)<=150,'Insufficient feasible harmonic support; do not relax selector thresholds'
    torch.save(dict(proposals=proposals),out/'setup/harmonic_proposals.pth')
    write(out/'setup/harmonic_selector.json',dict(audit,model_Adam_unchanged=True,source='fresh current10k model',TRAIN_views=len(cams)))
    return proposals

@torch.no_grad()
def boundary_mask(m,s,p,b):
    count=torch.zeros(len(m._xyz),device='cuda',dtype=torch.int32)
    for cam in s.getTrainCameras():count+=(render(cam,m,p,b)['radii']>0).int()
    n=len(count);quota=int(np.ceil(.15*n));ids=np.arange(n);low=np.lexsort((ids,count.cpu().numpy()))[:quota]
    radius=(m._xyz-m._xyz.mean(0)).norm(dim=1).cpu().numpy();outer=np.lexsort((ids,-radius))[:quota]
    mask=torch.zeros(n,device='cuda',dtype=torch.bool);mask[low]=True;mask[outer]=True
    return mask

def insert_adaptive(m,s,p,b,c,out,initial_opacity,is_c3vd):
    from utils.stage12 import HarmonicNodes,register_optimizer
    # Selector and M2 references precede the Dynamic opacity restoration/head migration.
    proposals=collect_selector(m,s,p,b,out)
    mask=None if is_c3vd else boundary_mask(m,s,p,b)
    build_m2(m,mask,out)
    if not is_c3vd:
        old=copy.deepcopy(m.optimizer.state_dict());upgrade_appearance(m,c);migrate_optimizer(m,c,old)
        m.appearance_model.micro_iteration.fill_(10000)
        m.stage11_boundary=mask;m.stage11_floor=torch.minimum(initial_opacity,torch.zeros_like(initial_opacity))
        with torch.no_grad():m._opacity.copy_(torch.maximum(m._opacity,initial_opacity))
        c.position_lr_max_steps=30000;c.position_lr_final=1.6e-6;c.tv_weight=.0005;c.col_smooth_weight=.0001
        # Refresh the scheduler while preserving every existing Adam state.
        old=copy.deepcopy(m.optimizer.state_dict());migrate_optimizer(m,c,old)
    previous=m.graph_motion_residual.deterministic_aggregation
    m.graph_motion_residual.deterministic_aggregation=True
    with torch.no_grad():before=render(s.getTrainCameras()[0],m,p,b)
    m.harmonic_nodes=HarmonicNodes(len(m._xyz),'cuda',True,len(proposals));m.harmonic_nodes.insert(proposals);register_optimizer(m)
    with torch.no_grad():after=render(s.getTrainCameras()[0],m,p,b)
    m.graph_motion_residual.deterministic_aggregation=previous
    errors={k:float((before[k]-after[k]).abs().max()) for k in ['render','depth','alpha','transformed_points']}
    assert max(errors.values())<2e-6,errors
    write(out/'setup/harmonic_zero_shock.json',dict(pass_=True,errors=errors,admitted=len(proposals),handshake_steps=500))

def train(args):
    out=args.model_path.resolve();data=args.source_path.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'setup').mkdir();protect_inputs(data,out);seed(args.seed);is_c3vd=configure_data(data)
    c,groups=configuration(data,out);c.experiment_seed=args.seed;c.appearance_seed=args.seed
    sname='sigmoid2' if is_c3vd else 'rectum' if 'rectum' in data.name else 'transverse'
    write(out/'protocol.json',dict(baseline='M2_nearest16',fresh=True,external_checkpoints=False,GT_PLY_training=False,scene=sname,smoke=args.smoke,
        coarse_steps=100,fine_steps=12 if args.smoke else args.iterations,stages=dict(G3=2000,J8=5000,harmonic_and_M2=10000)))
    lp,hp,op,pp=groups;m=GaussianModel(c.sh_degree,hp.extract(c));s=Scene(lp.extract(c),m,shuffle=False);p=pp.extract(c);b=torch.tensor([0.,0.,0.],device='cuda')
    assert not s.gt_plys
    write(out/'split.json',dict(TRAIN=len(s.getTrainCameras()),TEST=len(s.getTestCameras()),TEST_rendered_for_training=False))
    torch.save(m.closest_point_indices.cpu(),out/'setup/knn99.pth')
    m.training_setup(c)
    from utils.node_sampling import CoarseGradientImportance,fps_plus_importance_sampling,build_gaussian_node_knn
    importance=CoarseGradientImportance(m._xyz) if is_c3vd else None
    started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for i in tqdm(range(1,101),desc='Coarse RGB/depth'):
        m.update_learning_rate(i,stage='coarse');cam=random.choice(s.getTrainCameras());q=render(cam,m,p,b,stage='coarse')
        loss,_=objective(m,q,cam,c,i,coarse=True)
        loss.backward()
        if importance is not None:importance.update(m._xyz.grad,q['visibility_filter'])
        for group in m.optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:torch.nn.utils.clip_grad_value_(param,.5)
        # Preserve native ColonSplat's last-coarse gradient handoff.
        if i<100:m.optimizer.step();m.optimizer.zero_grad(set_to_none=True)
    m.initialize_deformation_nodes();m.initialize_appearance_model()
    if is_c3vd:
        stats=importance.finalize();selected=fps_plus_importance_sampling(m._xyz.detach(),2000,400,stats['importance'],stats['raw_score'])
        nodes=m._xyz.detach()[selected];ids,w=build_gaussian_node_knn(m._xyz.detach(),nodes,4);m.deformation_graph=SparseDeformationGraph(nodes,ids,w)
        upgrade_appearance(m,c);c.stage3_dynamics=True;c.stage5_profile='P';c.loss_rgb_mode='charbonnier_psnr_ssim_handoff'
        c.tv_weight=0.;c.col_smooth_weight=1e-5;c.position_lr_max_steps=30000;c.position_lr_final=4e-6
    initial_opacity=m._opacity.detach().clone();optimizer(m,c)
    with preserve_rng():metric_gate(m,s,p,b,out,is_c3vd)
    app_hash=digest(m.appearance_graph.state_dict());late=False
    total=12 if args.smoke else args.iterations
    marks={2:2000,4:5000,8:10000,12:30000} if args.smoke else {}
    coarse_seconds=time.monotonic()-started
    for i in tqdm(range(1,total+1),desc='Fine joint M2'):
        k=marks.get(i,{3:3000,5:5001,6:6000,7:9000,9:10001,10:10500,11:12000}.get(i,i)) if args.smoke else i
        schedule(m,c,k,is_c3vd,late)
        if k%500==0:m.oneupSHdegree()
        cam=random.choice(s.getTrainCameras());q=render(cam,m,p,b)
        loss,parts=objective(m,q,cam,c,k,is_c3vd=is_c3vd,late=late)
        gradients_step(m,loss,reset_opacity=not is_c3vd and k in (3000,6000,9000))
        if late and sname=='transverse':
            with torch.no_grad():m._opacity[m.stage11_boundary]=torch.maximum(m._opacity[m.stage11_boundary],m.stage11_floor[m.stage11_boundary])
        with (out/'camera_sequence.jsonl').open('a') as f:f.write(json.dumps(dict(iteration=i,schedule_iteration=k,camera=cam.image_name))+'\n')
        if i%100==0 or args.smoke:
            row=dict(iteration=i,schedule_iteration=k,loss=float(loss.detach()),**{a:float(v.detach()) for a,v in parts.items()})
            with (out/'training_metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            write(out/'status.json',dict(state='TRAINING',**row))
        del q,loss,parts
        if hasattr(m,'_stage14_node'):m._stage14_node=None
        if k in (2000,5000,10000):
            with preserve_rng():
                if k==2000:build_g3(m,s.getTrainCameras(),out)
                elif k==5000:build_j8(m,out,args.seed)
                else:insert_adaptive(m,s,p,b,c,out,initial_opacity,is_c3vd);late=True
            if k==5000:random.seed(args.seed)
        if i==total or (not args.smoke and i in args.save_iterations):save(out,m,s,c,i,is_c3vd)
    assert digest(m.appearance_graph.state_dict())==app_hash
    write(out/'runtime.json',dict(training_wall_seconds=time.monotonic()-started,coarse_seconds=coarse_seconds,peak_VRAM_MiB=torch.cuda.max_memory_allocated()/2**20,
        fine_updates=total,Gaussians=len(m._xyz),geometry_nodes=len(m.deformation_graph.node_xyz)+len(m.harmonic_nodes.parents),appearance_nodes=2000,from_scratch=True))
    write(out/'status.json',dict(state='SMOKE_COMPLETE' if args.smoke else 'COMPLETE',iteration=total,all_stages_executed=True,external_checkpoints=False))
