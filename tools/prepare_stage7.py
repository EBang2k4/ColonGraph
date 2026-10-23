"""Freeze Stage7 source, exact commonfine0/IDs and historical A-L references."""
import hashlib,json,os,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
HIST=ROOT/'output/experiments/stage6_charb_wide96/run_20261022'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def prepare(root):
    assert not root.exists()
    assert shutil.disk_usage(ROOT).free>30*1024**3
    for sub in ['logs','c3vd/setup','c3vd/runs','c3vd/pilot','c3vd/immutable_native']:(root/sub).mkdir(parents=True)
    shutil.copytree(HIST/'c3vd/initial',root/'c3vd/initial')
    for name in ['gaussian_knn99.pth','importance_sampling.pth']:os.link(HIST/'c3vd/setup'/name,root/'c3vd/setup'/name)
    for k in 'ABCDEFGHIJKLMN':(root/'c3vd/runs'/k).symlink_to(HIST/'c3vd/runs'/k,target_is_directory=True)
    shutil.copy2(HIST/'c3vd/data_manifest.json',root/'c3vd/data_manifest.json')
    shutil.copy2(ROOT/'output/experiments/stage4_l2_handoff/run_20261022/stage2_selection.json',root/'stage2_selection.json')
    for k in 'ABC':
        d=root/'historical_float_metrics'/k;d.mkdir(parents=True)
        shutil.copy2(HIST/'historical_float_metrics'/k/'metrics.json',d/'metrics.json')
    common=dict(scene='c1_sigmoid2_t4_v4',TRAIN=252,TEST=36,Gaussians=364500,fine_updates=30000,
        geometry_nodes=2400,geometry_source='exact FPS2000+importance400 IDs fromF/G/I/J/K/L',
        appearance_nodes=2000,appearance_support='original fixed FPS2000 IDs and K4 IDW weights',
        common_fine0=str(HIST/'c3vd/initial'),coarse_replay=False,old_runs_retrained=False,
        beta=.25,gain=.50,geometry_scale_rotation_clip=.1,no_extra_RGB_clamp=True,
        RGB='lambda_Charb*mean(sqrt(e^2+1e-6))+.2DSSIM+lambda_MSE*MSE',DSSIM='(1-SSIM)/2',
        handoff='15k->30k, Charb.8->.15,MSE2->14,DSSIM.2 unchanged; Charb activefromfine1',
        cosine='Gaussianattributes+skinning baseLR *eta1->.12 from15k->30k; no recursive decay',
        XYZ='native exponential1.6e-4->4e-6 coefficient across30k, multiplied by spatial_lr_scale',
        MLP='unchanged native1e-3->1e-4 across30k, geometry/appearance/G3',
        priors='8k->22k depth.25->.005,KNN99.01->.0002; endpoints held constant',TV=0,colorvar=1e-5,KL=.01,
        fine_opacity_resets=False,G3_after=2000,J8_after=5000,geometry_appearance_joint=True,
        checkpoints=[2000,5000,10000,20000,30000],final_checkpoint=30000,pilots='200+3 actualAdam, freshfine0primary',
        full_parallel='one idle A4000 GPU for O',
        metric_protocol='native8bit PSNR/SSIM/LPIPS-A,V, floatclamped MAE/pooledRMSE/ratio/gradientL1; idleGPUFPS/fine_minutes/peakTorchMiB',
        C3VD_depth='estimatedprior only, not anatomical accuracy',
        causal_limit='O-M isolates the specified micro head+ramp+L2 package, including its gradients into shared appearance trunk. N-M does not prove appearance saturation. Sparse-motion phase error remains a hypothesis; one paired block with CUDA variability.',
        micro=dict(epsilon_x=.005,alpha_start=6000,alpha_end=10000,L2=.001,head=[64,3],parameters=195,half_extent='saved NodeMLP.position_scale',norm_bound='sqrt(3)*.005*a (coordinatewise tanh)',before_renderer_and_KNN99=True),
        new_runs=['O'],historical_reused=list('ABCDEFGHIJKLMN'))
    write(root/'protocol.json',common);write(root/'c3vd/matrix_protocol.json',common)
    old=json.loads((HIST/'locked_configurations.json').read_text())['M']
    locked=dict(old,appearance_mode='r6_additive_micro',stage5_profile='O',
        final_expected_parameters=13922638,appearance_parameters=6948677,
        micro_head=[64,3],micro_epsilon=.005,micro_ramp=[6000,10000],micro_lambda=.001,
        node_initialization='exact sharedfine0 R6Additive state plus zero micro head',
        micro_iteration_buffer='persisted at every checkpoint')
    locks={'O':locked}
    write(root/'locked_configurations.json',locks);write(root/'locked_configurations_sha256.json',dict(sha256=sha(root/'locked_configurations.json')))
    files=[p for folder in ['arguments','gaussian_renderer','scene','utils','tools','tests'] for p in (ROOT/folder).rglob('*.py')]+[ROOT/'train.py']
    write(root/'code_manifest.json',{str(p):sha(p) for p in files})
    history=[p for k in 'ABCDEFGHIJKLMN' for p in (HIST/'c3vd/runs'/k/'J').glob('fine_state_*.pth')]
    history += [p for k in 'ABCDEFGHIJKLMN' for p in (HIST/'c3vd/runs'/k).rglob('*') if p.is_file() and p.suffix in ('.json','.pth') and ('point_cloud' in p.parts or p.name.startswith('checkpoint_'))]
    history+=[HIST/'c3vd/initial/fine_state_1.pth',HIST/'c3vd/setup/gaussian_knn99.pth',HIST/'c3vd/setup/importance_sampling.pth']
    write(root/'historical_manifest.json',{str(p):sha(p) for p in history})
    write(root/'status.json',dict(state='PREPARED',new_runs=['O'],full_iterations=30000))
    (root/'report.md').write_text('# Stage7 O prepared\n\nNo new final metrics yet.\n')


def verify(root,final=False):
    for p,h in json.loads((root/'code_manifest.json').read_text()).items():assert sha(p)==h,p
    assert sha(root/'locked_configurations.json')==json.loads((root/'locked_configurations_sha256.json').read_text())['sha256']
    if final:
        for p,h in json.loads((root/'historical_manifest.json').read_text()).items():assert sha(p)==h,p
        for p,h in json.loads((root/'c3vd/data_manifest.json').read_text()).items():assert sha(p)==h['sha256'],p
        write(root/'final_integrity.json',dict(pass_=True,code_data_history_configuration_locks_unchanged=True))
