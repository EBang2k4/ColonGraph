"""Stage8 P/P1/P2: constantKNN99 and modality-specific depth, unchanged O model, fixed final30k, no old retraining.
Run using the project Conda; each subprocess owns one GPU and one experiment.
Native training, G3 builder, K8 candidates and checkpoint functions are reused.
"""
import argparse
import copy
import gc
import inspect
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from tools import run_dynamiccolon_rectum_r6_30k as shared
from tools import run_dynamiccolon_full200_30k as full
from tools import run_dynamiccolon_matched30k as legacy
from tools import run_weighted_spline32 as base
from tools.run_gats20_joint import seed, preserve_rng, rng_state, restore_rng, native_function, camera_plan
from tools.experiment_common import load_scene
from utils.exact_split_branchpoint import RAW_NAMES, model_digest, digest, restore_knn_reference
from gaussian_renderer import render
from utils.r6_hf import initialize as initialize_hf
write, sha = shared.write, shared.sha
DEFAULT = ROOT / 'output/experiments/stage8_fixedknn_modality/run_20261023'
DATASETS = {'transverse': ROOT.parent/'datasets/DynamicColon/data_transverse',
            'sigmoid2': ROOT.parent/'ColonSplat_data/data/coloncrafter/c1_sigmoid2_t4_v4'}
MODES = {k:('r6_additive_micro',0.) for k in ('P','P1','P2')}
S2 = ROOT/'output/experiments/stage5_confirmation_refinement/run_20261022'
HIST = ROOT/'output/experiments/r7_core_fixed2000/run_20261021'
PILOT_STEPS = 200
SCENE = 'sigmoid2'


def expected_gaussians():return 200000 if SCENE=='transverse' else 364500


def expected_params():
    return dict(total=7671638,appearance=3823177,pilot=6066835) if SCENE=='transverse' else dict(total=13922638,appearance=6948677,pilot=11001835)


def assert_split(s):
    train, test = s.getTrainCameras(), s.getTestCameras()
    assert len(train) == (200 if SCENE == 'transverse' else 252)
    assert len(test) == (200 if SCENE == 'transverse' else 36)
    assert {c.image_name for c in train}.isdisjoint(c.image_name for c in test)
    if SCENE == 'transverse':
        assert s.gt_plys is None and not s.initialization_audit['gt_ply_read']
        assert all(c.source_split == 'TRAIN' for c in train)
        assert all(c.source_split == 'TEST' for c in test)


def configure(scene,root=None):
    global SCENE
    SCENE = scene
    assert scene in ('sigmoid2','transverse')
    # DynamicColon guard never points at the preprocessed C3VD input point cloud.
    full.DATA = legacy.DATA = shared.DATA = DATASETS['transverse']
    full.assert_split = assert_split


def configuration(root):
    from arguments import ModelParams, ModelHiddenParams, OptimizationParams, PipelineParams
    parser = argparse.ArgumentParser()
    groups = [ModelParams(parser), ModelHiddenParams(parser), OptimizationParams(parser), PipelineParams(parser)]
    c = parser.parse_args([])
    template = json.loads((ROOT/'output/experiments/j_baseline30k/fresh_20261011/c1_sigmoid2_t4_v4/configs/warmup.json').read_text())
    for section in template.values():
        for k, v in section.items(): setattr(c, k, v)
    c.source_path = str(DATASETS[SCENE]); c.model_path = str(root/'initial')
    c.node_sampling = 'fps'; c.node_importance_extra_nodes = 0
    c.adaptive_node_mode='none'; c.graph_forward_adaptive=False; c.surface_graph_refine=False
    c.num_nodes=2000; c.node_appearance_graph_mode='fixed_initial'
    c.gaussian_knn_save = str(root/'setup/gaussian_knn99.pth')
    c.gaussian_knn_reference = ''; c.fine_stop_iteration = -1; c.full_state_checkpoints = ''
    c.appearance_mode = 'r6'; c.hf_eta = .15; c.hf_lr = 1e-4
    if SCENE == 'transverse': c.dynamic_point_initialization = 'depth'; c.init_pts = 200000
    assert c.knn_weight == .01 and c.tv_weight == .001 and c.col_smooth_weight == .0001
    assert c.color_model == 'node_residual_mlp' and c.color_residual_scale == .5
    write(root/'configuration.json', vars(c))
    return c, groups


def parameters(m):
    q = full.parameters(m)
    hf = 0 if m.hf_appearance is None else sum(p.numel() for p in m.hf_appearance.parameters())
    q.update(HF_parameters=hf, appearance_mode=getattr(m.args, 'appearance_mode', 'r6'))
    q['appearance_parameters'] += hf
    q['total_trainable_parameters'] = sum(q['actual_trainable_groups'].values())
    q['appearance'] = getattr(m.args,'appearance_mode','r6')
    return q


def save(out, m, s, c, i):
    full.save(out, m, s, c, i)
    manifest = json.loads((out/f'checkpoint_{i}.json').read_text())
    manifest.update(parameters(m)); write(out/f'checkpoint_{i}.json', manifest)


def graph_helper(study):
    from tools.run_j_baseline30k import configure as setup
    helper = setup(study, DATASETS[SCENE].name); helper.DATA = DATASETS[SCENE]
    return helper


def candidates(study):
    helper=graph_helper(study)
    source=inspect.getsource(helper.candidates)
    assert source.count('assert n==2000')==1
    source=source.replace('assert n==2000', 'assert n=='+str(2400))
    scope=dict(helper.__dict__);exec(compile(source,__file__+':K8','exec'),scope)
    scope['candidates']()



def fresh_branch(root,key,pilot=False):
    study=root/('pilot' if pilot else 'runs')/key
    for name in ['setup','logs','common_5000','J']:(study/name).mkdir(parents=True,exist_ok=True)
    os.link(root/'setup/gaussian_knn99.pth',study/'setup/gaussian_knn99.pth')
    seed();m,s,p,b,c,st=full.load(root/'initial',1,appearance='R6')
    expected=json.loads((root/'initial/initial_integrity.json').read_text())
    checks=dict(raw_sha256=digest({k:getattr(m,k) for k in RAW_NAMES}),
        node_mlp_sha256=digest(m.node_mlp.state_dict()),KNN99_sha256=digest(m.closest_point_indices),
        geometry_gradients_sha256=digest(st['geometry_gradients']))
    assert all(checks[k]==expected[k] for k in checks)
    ids=torch.load(root/'initial/fps_source_ids.pth',map_location='cuda',weights_only=True)
    assert torch.equal(m.deformation_graph.node_xyz,m._xyz.detach()[ids])
    assert torch.equal(m.appearance_graph.node_xyz,m.deformation_graph.node_xyz)
    original_r6={k:v.detach().clone() for k,v in m.appearance_model.state_dict().items()}
    mode,tv=MODES[key]
    geometry_ids=ids
    if key in MODES:
        from utils.node_sampling import build_gaussian_node_knn
        from scene.deformation_graph import SparseDeformationGraph
        sampling=torch.load(root/'setup/importance_sampling.pth',map_location='cuda',weights_only=True)
        assert torch.equal(sampling['canonical_xyz'],m._xyz.detach())
        geometry_ids=sampling['selected_indices']
        assert torch.equal(geometry_ids[:2000],ids)
        centers=m._xyz.detach()[geometry_ids]
        indices,weights=build_gaussian_node_knn(m._xyz.detach(),centers,4)
        m.deformation_graph=SparseDeformationGraph(centers,indices,weights)
        assert torch.equal(m.appearance_graph.node_xyz,m._xyz.detach()[ids])
    base.update(m,c,appearance_mode=mode,tv_weight=tv,node_sampling='fps',node_importance_extra_nodes=0,
        gaussian_knn_reference=str(study/'setup/gaussian_knn99.pth'),gaussian_knn_save='',
        loss_rgb_mode='charbonnier_psnr_ssim_handoff', col_smooth_weight=.00001,
        r6_additive_cap=.25, stage5_profile=key,
        stage3_dynamics=True, position_lr_max_steps=30000,position_lr_final=4e-6,
        node_appearance_dim=16 if key=='N' else 8,
        node_scale_residual_ratio=.1, node_rotation_residual_ratio=.1)
    base.update(m,c,node_sampling='fps_plus_importance',node_importance_extra_nodes=400)
    locks=root.parent/'locked_configurations.json'
    config=json.loads(locks.read_text())[key]
    fields=['appearance_mode','node_appearance_dim','r6_additive_cap','loss_rgb_mode','stage5_profile','stage3_dynamics','position_lr_max_steps','position_lr_final','node_scale_residual_ratio','node_rotation_residual_ratio']
    assert all(getattr(c,k)==config[k] for k in fields)
    write(study/'locked_configuration_check.json',dict(pass_=True,fields=fields,sha256=sha(locks)))
    with preserve_rng():m.initialize_appearance_model()
    if mode=='r6_additive_micro':
        state=m.appearance_model.state_dict()
        assert all(torch.equal(v,state[k]) for k,v in original_r6.items())
    if mode=='r6_additive_wide96':
        m.appearance_model.bind_fine0_features(original_r6['node_appearance'])
        assert torch.equal(m.appearance_model.node_appearance[:,:8],original_r6['node_appearance'])
        assert torch.count_nonzero(m.appearance_model.node_appearance[:,8:])==0
    if mode=='r7_v2':
        m.appearance_model.bind_nodes(ids,geometry_ids,m.appearance_graph.node_xyz,m.deformation_graph.node_xyz,
                                      m.node_mlp.position_scale)
    if mode=='r6_speccam':
        m.appearance_model.camera_scale.copy_(m.node_mlp.position_scale.detach())
    fixed=json.loads((root/'initial/fixed_node_audit.json').read_text())
    old=json.loads((root/'runs/O/initial_integrity.json').read_text())
    assert digest(geometry_ids)==digest(sampling['selected_indices'])==old['selected_ids_sha256']
    assert digest(m.appearance_graph.state_dict())==fixed['appearance_graph_sha256']==old['appearance_graph_sha256']
    from utils.r7_core import R6Additive
    with preserve_rng():reference=R6Additive(c,2000).cuda()
    reference.load_state_dict({**reference.state_dict(),**original_r6},strict=True)
    assert digest({k:v for k,v in m.appearance_model.state_dict().items() if not k.startswith('micro_')})==digest(reference.state_dict())
    del reference
    m.gats_joint=True
    c.model_path=str(study/'J');s.model_path=c.model_path
    torch.save(geometry_ids.cpu(),study/'selected_node_ids.pth')
    write(study/'initial_integrity.json',dict(pass_=True,**checks,selected_ids_sha256=digest(geometry_ids),
        appearance_graph_sha256=digest(m.appearance_graph.state_dict()),
        common_R6_sha256=digest(original_r6),appearance_sha256=digest(m.appearance_model.state_dict()),
        appearance_mode=mode,r6_additive_cap=c.r6_additive_cap,loss_rgb_mode=c.loss_rgb_mode,
        initial_node_features_first8_sha256=digest(m.appearance_model.node_appearance[:,:8]),
        fine0_node_features_sha256=digest(original_r6['node_appearance']),
        micro_head_zero=all(torch.count_nonzero(v)==0 for v in m.appearance_model.micro_head.parameters()),
        final_expected_parameters=expected_params()['total'],geometry_nodes=len(geometry_ids),appearance_nodes=2000,K=4,fine_updates_before_start=0))
    (study/'J/cfg_args').write_text(str(c))
    return study,m,s,p,b,c,st



class Hooks:
    def __init__(self, study,m,s,p,b,c,initial=None,pilot=False):
        self.study,self.out,self.m,self.s,self.p,self.b,self.c=study,study/'J',m,s,p,b,c
        self.initial,self.pilot=initial,pilot
        self.plan=camera_plan(s.getTrainCameras());write(self.out/'camera_plan.json',self.plan)
        self.started=time.monotonic();self.overhead=0.;self.i=0
        self.fixed_support=digest(m.appearance_graph.state_dict())
        self.fixed_centers=digest(m.deformation_graph.node_xyz)
        self.prior_checks=0;self.knn_min=float('inf');self.knn_max=0.;self.depth_min=float('inf');self.depth_max=0.
    def setup(self,m):
        def forbidden_reset():
            raise AssertionError('Stage7 fine opacity reset must never execute')
        m.reset_opacity=forbidden_reset
        assert self.c.stage3_dynamics and self.c.position_lr_max_steps==30000
        assert sum(v.numel() for v in m.appearance_model.micro_head.parameters())==195
        assert parameters(m)['appearance_parameters']==expected_params()['appearance']
        assert m.appearance_model.additive_cap==self.c.r6_additive_cap==.25
        assert self.c.loss_rgb_mode=='charbonnier_psnr_ssim_handoff'
        assert self.c.node_appearance_dim==(16 if self.study.name=='N' else 8)
        assert self.c.stage5_profile==self.study.name
        assert self.c.position_lr_final==4e-6
        assert self.c.node_scale_residual_ratio==self.c.node_rotation_residual_ratio==.1
        assert all(getattr(m,k).requires_grad for k in RAW_NAMES)
        assert all(p.requires_grad for p in m.node_mlp.parameters())
        assert all(p.requires_grad for p in m.appearance_model.parameters())
        assert len(m.deformation_graph.node_xyz)==(2400) and len(m.appearance_graph.node_xyz)==2000
        if hasattr(m,'fine_resume_state'):
            for k,v in m.fine_resume_state['densification_stats'].items():setattr(m,k,v.cuda())
    def begin(self,i):
        self.i=i;self.renders=0;self.backwards=0
        self.m.appearance_model.micro_iteration.fill_(i)
        if i in [1,2001,5001,30000] or i%1000==0:
            assert digest(self.m.appearance_graph.state_dict())==self.fixed_support
            assert digest(self.m.deformation_graph.node_xyz)==self.fixed_centers
            assert self.m.appearance_graph.gaussian_node_indices.shape==(expected_gaussians(),4)
            assert self.m.surface_children is None and not self.m.args.graph_forward_adaptive
            if i>5001:assert self.m.skinning.indices.shape==(expected_gaussians(),8)
        if i==1:
            assert self.c.node_scale_residual_ratio==self.c.node_rotation_residual_ratio==.1
            assert self.initial is not None
            restore_rng(self.initial)
            for k,v in self.initial['coarse_raw_gradients'].items():getattr(self.m,k).grad=None if v is None else v.cuda().clone()
        if i==5001:
            random.seed(0)
            base.update(self.m,self.c,ggas_variant='J_NO_MOTION_SELECTION',ggas_study_version=2,
                ggas_candidates=str(self.study/'candidates.pth'),ggas_lr=.001,ggas_lambda=.01,
                ggas_ramp_end=6000,geometry_study_refresh='none')
            from utils.ggas import initialize
            with preserve_rng():initialize(self.m,self.out)
            (self.out/'cfg_args').write_text(str(self.c))
    def render(self,cam,m,*args,**kw):
        self.renders+=1
        expected=self.plan[self.i-1]
        assert str(cam.image_name)==str(expected['camera_name']) and float(cam.time)==expected['time']
        assert torch.is_grad_enabled()
        from utils.stage8 import dynamics
        eta,depth,knn=dynamics(self.i,self.c.stage5_profile)
        assert self.c.knn_weight==knn==.01
        assert self.c.depth_weight==depth
        self.prior_checks+=1;self.knn_min=min(self.knn_min,knn);self.knn_max=max(self.knn_max,knn)
        self.depth_min=min(self.depth_min,depth);self.depth_max=max(self.depth_max,depth)
        q=render(cam,m,*args,**kw)
        with (self.out/'camera_sequence.jsonl').open('a') as f:f.write(json.dumps(expected)+'\n')
        if self.i<=3 or self.i%100==0:
            with torch.no_grad():
                node_motion=m.node_positions_at_time(cam.time)-m.deformation_graph.node_xyz
                self.stats=dict(alpha_mean=float(q['alpha'].mean()),visible_fraction=float((q['radii']>0).float().mean()),
                    node_motion_mean=float(node_motion.norm(dim=-1).mean()),node_motion_max=float(node_motion.norm(dim=-1).max()),
                    gaussian_RGB_below0=float((q['transformed_color']<0).float().mean()),
                    gaussian_RGB_above1=float((q['transformed_color']>1).float().mean()),
                    render_RGB_below0=float((q['render']<0).float().mean()),render_RGB_above1=float((q['render']>1).float().mean()),
                    absolute_deformation_mean=float((q['transformed_points']-m._xyz).norm(dim=-1).mean()),
                    opacity_mean=float(m.get_opacity.mean()),opacity_below_001=float((m.get_opacity<.01).float().mean()),
                    canonical_scale_clamped_fraction=float((m.get_scaling>.05*m.spatial_lr_scale).float().mean()))
                self.stats.update(appearance_statistics(m,cam.time,cam))
                mag=q['micro_offset'].norm(dim=-1)/m.node_mlp.position_scale
                self.stats.update(micro_alpha=float(((m.appearance_model.micro_iteration-6000)/4000.).clamp(0,1)),
                    micro_normalized_mean=float(mag.mean()),micro_normalized_p95=float(torch.quantile(mag,.95)),
                    micro_normalized_max=float(mag.max()),micro_raw_abs_max=float(q['micro_raw'].abs().max()))
                assert mag.max()<=np.sqrt(3)*.005+1e-7
                if self.i<=6000:assert torch.count_nonzero(q['micro_offset'])==0
                if m.hf_appearance is not None:
                    r=q['hf_residual'];a=m.hf_appearance.coefficients
                    self.stats.update(HF_abs_mean=float(r.abs().mean()),HF_abs_max=float(r.abs().max()),
                        HF_saturation_fraction=float((r.abs()>.99*.15).float().mean()),
                        HF_coefficient_RMS=float(a.square().mean().sqrt()),HF_coefficient_abs_max=float(a.abs().max()))
                assert all(torch.isfinite(q[k]).all() for k in ['render','depth','alpha','transformed_points','transformed_color'])
        return q
    def backward(self,m,i,loss,rgb,depth,knn):
        self.backwards+=1
        if i<=3 or i%100==0:
            from utils.photometric_unlock import handoff_weights
            expected=handoff_weights(i,self.c.stage5_profile)
            observed=tuple(m.rgb_loss_audit[k] for k in ['lambda_Charb','lambda_DSSIM','lambda_MSE'])
            assert observed==expected
            norms={k:float(getattr(m,k).grad.norm()) if getattr(m,k).grad is not None else 0. for k in RAW_NAMES}
            for name,mod in [('R6',m.appearance_model),('node_geometry',m.node_mlp),('HF',m.hf_appearance)]:
                if mod is not None:norms[name]=float(sum(p.grad.square().sum() for p in mod.parameters() if p.grad is not None).sqrt())
            assert np.isfinite(list(norms.values())).all() and torch.isfinite(loss)
            if i==1:
                assert norms['_xyz']>0 and norms['node_geometry']>0 and norms['features']>0
                if m.hf_appearance is not None:assert norms['HF']>0
            if i==100:
                assert norms['_deformation_table']>0 and norms['R6']>0
                if self.study.name=='N':
                    assert m.appearance_model.node_appearance.grad[:,8:].norm()>0
                    write(self.out/'wide_extra8_gradient.json',dict(iteration=i,gradient_norm=float(m.appearance_model.node_appearance.grad[:,8:].norm()),pass_=True))
            if i in [100,5100]:
                app_grad={k:float(v.grad.norm()) if v.grad is not None else 0.
                          for k,v in m.appearance_model.named_parameters()}
                assert all(np.isfinite(v) and (v==0 if k.startswith('micro_head.') else v>0) for k,v in app_grad.items()),app_grad
                write(self.out/f'appearance_gradient_{i}.json',app_grad)
            micro_grad={k:float(v.grad.norm()) if v.grad is not None else 0. for k,v in m.appearance_model.micro_head.named_parameters()}
            assert all(np.isfinite(v) for v in micro_grad.values())
            if i<=6000:assert all(v==0 for v in micro_grad.values())
            if i in (6100,10000):
                assert all(v>0 for v in micro_grad.values())
                write(self.out/f'micro_gradient_{i}.json',dict(pass_=True,gradients=micro_grad))
            row=dict(iteration=i,micro_loss=float(m.micro_loss),micro_grad_norms=micro_grad,loss=float(loss),RGB_L1=float(rgb),training_depth_loss=float(depth),KNN99=float(knn),
                loss_components=getattr(m,"rgb_loss_audit",{}),grad_norms=norms,
                learning_rates={g['name']:g['lr'] for g in m.optimizer.param_groups},
                lambda_depth=self.c.depth_weight,lambda_knn=self.c.knn_weight,lambda_kl=self.m.args.ggas_lambda,peak_VRAM_MiB=torch.cuda.max_memory_allocated()/2**20,
                elapsed_training_seconds=time.monotonic()-self.started-self.overhead,**self.stats)
            with (self.out/'training_metrics.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    def after(self,m,s,cameras,i):
        assert self.renders==self.backwards==1
        if i%100==0:write(self.out/'status.json',dict(state='PILOT' if self.pilot else 'RUNNING',iteration=i,pid=os.getpid()))
        start=time.monotonic()
        if i==2000:
            with preserve_rng():
                save(self.study/'setup/warmup',m,s,self.c,i)
                helper=graph_helper(self.study);helper.reference()
                base.update(m,self.c,graph_forward_mode='g3',graph_forward_reference=str(helper.REF))
                m.initialize_graph_forward(str(helper.REF));m.add_graph_residual_optimizer_group()
        if i==5000:
            with preserve_rng():
                save(self.study/'common_5000',m,s,self.c,i);candidates(self.study)
        if i in [3000,6000,9000,12000,30000]:
            assert parameters(m)['optimizer_steps']['opacity']==[i], 'Opacity reset/missing Adam update'
        if i in [5000,10000,20000,30000] and m.skinning is not None:
            assert parameters(m)['total_trainable_parameters']==expected_params()['total']
        if i in [10000,20000,30000]:
            with preserve_rng():save(self.out,m,s,self.c,i)
        if i in [2000,5000,10000,20000,30000]:self.overhead+=time.monotonic()-start
        if i in [5000,10000,20000,30000]:
            write(self.out/'runtime.json',dict(training_seconds=time.monotonic()-self.started-self.overhead,
                setup_checkpoint_seconds=self.overhead,peak_VRAM_MiB=torch.cuda.max_memory_allocated()/2**20,**parameters(m)))
        if i==30000:
            assert len(m._xyz)==expected_gaussians()
            q=parameters(m)['optimizer_steps']
            for k in ['xyz','f_dc','deformation_table','opacity']:assert q[k]==[30000]
            assert q['skinning']==[25000] and all(v==30000 for v in q['deformation'])
            if m.hf_appearance is not None:assert q['r6_hf']==[30000]
            rows=[json.loads(x) for x in (self.out/'camera_sequence.jsonl').read_text().splitlines()]
            assert rows==self.plan
            assert self.prior_checks==30000 and self.knn_min==self.knn_max==.01
            write(self.out/'integrity.json',dict(pass_=True,prior_checks=self.prior_checks,lambda_knn_min=self.knn_min,lambda_knn_max=self.knn_max,lambda_depth_min=self.depth_min,lambda_depth_max=self.depth_max,actual_fine_Adam_updates=30000,camera_sha256=digest(rows),
                geometry_and_appearance_trainable=True,appearance_mode=self.c.appearance_mode,
                fixed_appearance_support_sha256=self.fixed_support,fixed_geometry_centers_sha256=self.fixed_centers,
                geometry_nodes=len(m.deformation_graph.node_xyz),appearance_nodes=2000,appearance_K=4,geometry_K=8,
                skinning_Adam_updates=25000,G3_Adam_updates=28000))
            write(self.out/'status.json',dict(state='TRAINED',iteration=30000))


def forward_checks(study,m,s,p,b,c):
    from utils.r7_core import R6Additive
    original=m.appearance_model
    cfg=copy.deepcopy(c);cfg.node_appearance_dim=8;cfg.r6_additive_cap=.25
    reference=R6Additive(cfg,2000).cuda()
    reference.load_state_dict({k:v for k,v in original.state_dict().items() if not k.startswith('micro_')},strict=True)
    original_ratios=(m.args.node_scale_residual_ratio,m.args.node_rotation_residual_ratio)
    rows=[]
    with torch.no_grad():
        for cam in [s.getTrainCameras()[0],s.getTrainCameras()[-1],s.getTestCameras()[0]]:
            m.appearance_model=reference;oldmode=m.args.appearance_mode;m.args.appearance_mode='r6_additive'
            m.args.node_scale_residual_ratio=m.args.node_rotation_residual_ratio=.1
            a=render(cam,m,p,b)
            m.appearance_model=original;m.args.appearance_mode=oldmode
            m.args.node_scale_residual_ratio,m.args.node_rotation_residual_ratio=original_ratios
            z=render(cam,m,p,b)
            delta={k:float((a[k]-z[k]).abs().max()) for k in ['render','alpha','depth','transformed_color']}
            assert max(delta.values())==0,delta
            assert torch.equal(z['transformed_color'],m.features)
            rows.append(dict(camera=str(cam.image_name),max_abs_error=delta))
    write(study/'renderer_zero_init_check.json',dict(pass_=True,rows=rows,
        canonical_color_identity=True,original_J_F_canonical_equivalence=True,no_extra_RGB_clamp=True,
        fine0_and_all_training_clip=.1,additive_cap=c.r6_additive_cap))



def micro_probe(study,m,s,p,b,c):
    """Check native render/KNN gradients after activation, without training/tuning."""
    app=m.appearance_model;before=model_digest(m);adam=digest(m.optimizer.state_dict())
    old=app.micro_iteration.detach().clone();cam=s.getTrainCameras()[7];rows=[]
    try:
        for iteration in (6000,6001,10000):
            app.micro_iteration.fill_(iteration)
            q=render(cam,m,p,b)
            rgb=(q['render']-cam.original_image[:3].cuda()).square().mean()
            xyz=q['transformed_points'];idx=m.closest_point_indices
            knn=(xyz-xyz[idx].mean(1)).square().sum(-1).mean()
            gradients={}
            for name,loss in [('RGB',rgb),('KNN99',knn)]:
                grads=torch.autograd.grad(loss,tuple(app.micro_head.parameters()),retain_graph=True)
                gradients[name]=float(sum(g.square().sum() for g in grads).sqrt())
                assert np.isfinite(gradients[name])
                assert (gradients[name]==0) if iteration==6000 else (gradients[name]>0)
            assert torch.count_nonzero(q['micro_offset'])==0
            rows.append(dict(iteration=iteration,gradient_norms=gradients,zero_init_offset=True))
            del q,rgb,xyz,knn,grads
        # Known nonzero offset checks exact composition before native renderer/KNN.
        with torch.no_grad():
            saved=app.micro_head.bias.clone();app.micro_head.bias.copy_(torch.tensor([.01,-.02,.03],device='cuda'))
            q=render(cam,m,p,b);prior=m.deform_geometry_from_nodes(cam.time)[0]
            expected=.005*m.node_mlp.position_scale*app.micro_head.bias.tanh().expand_as(prior)
            torch.testing.assert_close(q['micro_offset'],expected,rtol=0,atol=0)
            torch.testing.assert_close(q['transformed_points'],prior+expected,rtol=0,atol=0)
            state={k:v.detach().clone() for k,v in app.state_dict().items()}
            from utils.r7_core import create
            cold=create(c,2000).cuda();cold.load_state_dict(state,strict=True)
            assert int(cold.micro_iteration)==10000
            args=(cam.time,m.features,m._deformation_table,m.appearance_graph.gaussian_node_indices,
                  m.appearance_graph.gaussian_node_weights,m.node_mlp.position_scale)
            assert torch.equal(app.forward_micro(*args)[1],cold.forward_micro(*args)[1])
            app.micro_head.bias.copy_(saved)
            del cold,state,q,prior,expected
    finally:
        app.micro_iteration.copy_(old)
    assert model_digest(m)==before and digest(m.optimizer.state_dict())==adam
    write(study/'micro_activation_probe.json',dict(pass_=True,TRAIN_only=True,optimizer_updates=0,
        rows=rows,native_render_centers_include_offset=True,KNN99_uses_same_transformed_points=True,
        checkpoint_buffer_active_ramp_preserved=True,model_and_Adam_unchanged=True,
        half_extent=float(m.node_mlp.position_scale),half_extent_source='saved NodeMLP.position_scale'))


def restore(out,i):
    m,s,p,b,c,st=full.load(out,i)
    m.training_setup(c);m.add_graph_residual_optimizer_group()
    if m.skinning is not None:
        from utils.ggas import add_optimizer
        add_optimizer(m)
    m.optimizer.load_state_dict(st['optimizer'])
    if getattr(c,'loss_rgb_mode','')=='charbonnier_psnr_ssim_handoff':
        assert m.appearance_model.additive_cap==c.r6_additive_cap
        assert m.args.node_scale_residual_ratio==m.args.node_rotation_residual_ratio==.1
    assert model_digest(m)==st['model_sha256']
    assert digest(m.optimizer.state_dict())==digest(st['optimizer'])
    assert sha(out/f'fine_state_{i}.pth')==json.loads((out/f'checkpoint_{i}.json').read_text())['sha256']
    return m,s,p,b,c,st


def reload_check_render(m,cam,p,b):
    """Only the zero-update reload probe uses deterministic graph reduction.

    Native CUDA index_add varies by ulps after G3 activates; at this world
    scale that can alter per-pixel sort order. Leave training flags untouched.
    """
    residual=m.graph_motion_residual
    previous=None if residual is None else residual.deterministic_aggregation
    try:
        if residual is not None:residual.deterministic_aggregation=True
        with torch.no_grad():
            return {k:v.detach().cpu().clone() for k,v in render(cam,m,p,b).items()
                    if k in ['render','alpha','depth']}
    finally:
        if residual is not None:residual.deterministic_aggregation=previous


def run(root,key,pilot=False):
    study,m,s,p,b,c,st=fresh_branch(root,key,pilot)
    with (study/'STARTED').open('x') as f:f.write(str(os.getpid()))
    if pilot:
        with preserve_rng():forward_checks(study,m,s,p,b,c)
    m.gats_joint=True
    hooks=Hooks(study,m,s,p,b,c,initial=st,pilot=pilot)
    fn=native_function(hooks);(study/'instrumented_training.py').write_text(hooks.patched_source)
    from utils.timer import Timer
    timer=Timer();timer.start();torch.cuda.reset_peak_memory_stats()
    fn(c,c,m.args,p,[],[],[],None,-1,m,s,'fine',None,PILOT_STEPS if pilot else 30000,timer)
    if pilot:
        save(study/'J',m,s,c,PILOT_STEPS)
        expected=reload_check_render(m,s.getTrainCameras()[7],p,b)
        del m,s,p,b,c,st,hooks,fn;gc.collect();torch.cuda.empty_cache()
        m,s,p,b,c,st=restore(study/'J',PILOT_STEPS)
        actual=reload_check_render(m,s.getTrainCameras()[7],p,b)
        errors={k:float((actual[k]-v).abs().max()) for k,v in expected.items()}
        assert max(errors.values())<2e-6
        with preserve_rng():micro_probe(study,m,s,p,b,c)
        m.gats_joint=True;m.fine_resume_state=st
        hooks=Hooks(study,m,s,p,b,c,pilot=True);fn=native_function(hooks);timer=Timer();timer.start()
        fn(c,c,m.args,p,[],[],[],None,-1,m,s,'fine',None,PILOT_STEPS+3,timer)
        assert parameters(m)['optimizer_steps']['xyz']==[PILOT_STEPS+3]
        if m.hf_appearance is not None:assert parameters(m)['optimizer_steps']['r6_hf']==[PILOT_STEPS+3]
        actual=[json.loads(x) for x in (study/'J/camera_sequence.jsonl').read_text().splitlines()]
        assert actual==hooks.plan[:PILOT_STEPS+3]
        write(study/'pilot_integrity.json',dict(pass_=True,steps=PILOT_STEPS,resume_steps=3,render_reload_max_abs=errors,
            model_Adam_reload_exact=True,camera_replay_exact=True,geometry_appearance_gradients_active=True,
            fine_opacity_reset_guard=True,stage3_dynamics=True,
            G3_and_J8_activation_tested=False,appearance_K4_unchanged=True,**parameters(m)))


@torch.no_grad()
def appearance_statistics(m,t,cam=None):
    graph=m.appearance_graph;app=m.appearance_model
    args=(t,m.features,m._deformation_table,graph.gaussian_node_indices,graph.gaussian_node_weights)
    if m.args.appearance_mode=='r7_v2':
        xyz,_,_=m.deform_geometry_from_nodes(t)
        low,high,beta=app.components(t,m._deformation_table,graph.gaussian_node_indices,
                                    graph.gaussian_node_weights,m._query_node_motion(t),xyz-m._xyz)
        gain=low+high
    elif m.args.appearance_mode=='r7_core':
        low,high,beta=app.components(t,m._deformation_table,graph.gaussian_node_indices,
                                    graph.gaussian_node_weights,m._query_node_motion(t))
        gain=low+high
    elif m.args.appearance_mode=='r6_speccam':
        from utils.stage3 import camera_cues
        xyz,_,_=m.deform_geometry_from_nodes(t)
        cue=camera_cues(xyz,cam.world_view_transform,app.camera_scale)
        gain,beta=app.components(*args,camera_input=cue);low=gain;high=torch.zeros_like(gain)
    elif m.args.appearance_mode in ('r6_additive','r6_additive_wide96','r6_additive_micro'):
        gain,beta=app.components(*args);low=gain;high=torch.zeros_like(gain)
    else:
        _,gain=app(m._xyz,t,m.features,m._deformation_table,graph.gaussian_node_indices,graph.gaussian_node_weights)
        low=gain;high=beta=torch.zeros_like(gain)
    result={}
    for name,value in [('gain',gain),('gain_low',low),('gain_high',high),('additive',beta)]:
        result.update({name+'_mean':float(value.mean()),name+'_abs_mean':float(value.abs().mean()),
                       name+'_abs_max':float(value.abs().max()),name+'_RMS':float(value.square().mean().sqrt())})
    cap=getattr(app,'additive_cap',.30 if m.args.appearance_mode=='r6_speccam' else (.35 if m.args.appearance_mode=='r7_core' else .15))
    result['additive_saturation_fraction']=float((beta.abs()>.99*cap).float().mean())
    return result


def free_gpus():
    uuids=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
    mapping={u.strip():int(i) for i,u in (line.split(',') for line in uuids.splitlines())}
    busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,process_name','--format=csv,noheader,nounits'],text=True)
    display={'gnome-remote-desktop-daemon','Xorg','gnome-shell'}
    occupied=set()
    for line in busy.splitlines():
        uuid,name=[x.strip() for x in line.split(',',1)]
        if uuid in mapping and Path(name).name not in display:occupied.add(mapping[uuid])
    return [i for i in [1,0] if i not in occupied]


def phase(root,commands,name,parallel=True):
    pending=list(commands);jobs=[];codes={}
    write(root/'status.json',dict(state='RUNNING',phase=name,full_runs=3))
    while pending or jobs:
        for job in list(jobs):
            tag,proc,log,gpu=job
            if proc.poll() is not None:
                codes[tag]=proc.returncode;log.close();jobs.remove(job)
        if any(v!=0 for v in codes.values()):
            # Let already-running independent checks finish; never restart a failed run.
            for tag,proc,log,gpu in jobs:codes[tag]=proc.wait();log.close()
            write(root/'logs'/f'{name}_exit.json',codes)
            raise RuntimeError(str(codes))
        slots=[g for g in free_gpus() if g not in [j[3] for j in jobs]]
        if not parallel and jobs:slots=[]
        for gpu in slots[:len(pending)]:
            tag,argv=pending.pop(0)
            log=(root/'logs'/f'{name}_{tag}.log').open('x')
            cmd=[sys.executable,'-u',__file__,*argv,'--out',str(root)]
            proc=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),
                OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT)
            jobs.append((tag,proc,log,gpu))
            write(root/'logs'/f'{name}_{tag}_process.json',dict(pid=proc.pid,gpu=gpu,command=cmd))
            if not parallel:break
        if pending or jobs:time.sleep(15)
    write(root/'logs'/f'{name}_exit.json',codes)



def prepare(root):
    from tools import prepare_stage8 as prep
    prep.prepare(root)



def pilot_gate(root,keys):
    checks={}
    for key in keys:
        study=root/SCENE/'pilot'/key
        proof=json.loads((study/'pilot_integrity.json').read_text());assert proof['pass_'] and proof['steps']==200
        assert json.loads((study/'micro_activation_probe.json').read_text())['pass_']
        assert proof['appearance_parameters']==expected_params()['appearance']
        assert proof['total_trainable_parameters']==expected_params()['pilot']
        rows=[json.loads(x) for x in (study/'J/training_metrics.jsonl').read_text().splitlines()]
        row=rows[-1]
        checks[key]=dict(finite=True,RGB_not_divergent=row['RGB_L1']<.2,opacity_not_collapsed=row['opacity_mean']>.01,
                         additive_not_saturated=row['additive_saturation_fraction']<.01)
        if key in MODES:
            lc=row['loss_components']
            assert all(lc[k+'_image_gradient_norm']>0 for k in ['Charb','DSSIM','MSE'])
            assert abs(lc['weighted_RGB']-sum(lc[k+'_weighted_value'] for k in ['Charb','DSSIM','MSE']))<1e-6
        assert all(checks[key].values()),checks
    write(root/('pilot_gate_'+''.join(keys)+'.json'),dict(pass_=True,checks=checks,TEST_tuning=False))


def supervise(root):
    from tools import prepare_stage8 as prep
    with (root/'PIPELINE_STARTED').open('x') as f:f.write(str(os.getpid()))
    prep.verify(root,final=False)
    jobs=[('P','sigmoid2'),('P1','transverse'),('P2','transverse')]
    phase(root,[(key,['pilot','--variant',key,'--scene',scene]) for key,scene in jobs],'stage8_pilots')
    for key,scene in jobs:
        configure(scene);pilot_gate(root,[key])
    phase(root,[(key,['train','--variant',key,'--scene',scene]) for key,scene in jobs],'stage8_full30k')
    phase(root,[(key,['evaluate','--variant',key,'--scene',scene]) for key,scene in jobs],'stage8_readonly_eval',parallel=False)
    # Old transverse O receives clean depth replay only; preserve all original artifacts.
    phase(root,[('O',['clean-reference','--scene','transverse','--variant','P1'])],'historical_O_clean_depth',parallel=False)
    phase(root,[(key,['geometry','--scene','transverse','--variant',key]) for key in ['P1','P2']],'CH_HD95_all5epochs',parallel=False)
    phase(root,[(key,['exports','--scene','transverse','--variant',key]) for key in ['P1','P2']],'native_TEST_lookat',parallel=False)
    phase(root,[('final',['report'])],'stage8_final_report',parallel=False)
    prep.verify(root,final=True)
    write(root/'status.json',dict(state='COMPLETE',new_full_runs=3,iterations_each=30000,old_runs_retrained=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','pilot','train','evaluate','report','supervise','geometry','exports','clean-reference'])
    parser.add_argument('--out',type=Path,default=DEFAULT);parser.add_argument('--variant',choices=MODES,default='P')
    parser.add_argument('--scene',choices=['sigmoid2','transverse'],default='sigmoid2')
    args=parser.parse_args()
    assert os.environ.get('CONDA_PREFIX')=='/home/robot/colongraph/.env','Use project Conda'
    configure(args.scene)
    try:
        if args.action in ['prepare','supervise']:globals()[args.action](args.out)
        elif args.action in ['pilot','train']:run(args.out/args.scene,args.variant,pilot=args.action=='pilot')
        else:
            from tools import evaluate_stage8 as evaluator
            if args.action in ['evaluate','geometry','exports']:getattr(evaluator,args.action)(args.out,args.scene,args.variant)
            elif args.action=='clean-reference':evaluator.clean_reference(args.out)
            else:getattr(evaluator,args.action)(args.out)
    except Exception as exc:
        write(args.out/('failure_'+args.action+'_'+args.variant+'.json'),dict(error=repr(exc),pid=os.getpid(),no_auto_retry=True))
        if args.action=='supervise':write(args.out/'status.json',dict(state='FAILED',error=repr(exc),no_auto_retry=True))
        raise
