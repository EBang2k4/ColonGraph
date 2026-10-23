"""Freeze exactly three Stage8 runs and preserve all historical fine0/results."""
import ast,json,os,shutil,subprocess,sys
from pathlib import Path
from tools.prepare_stage7 import sha,write
ROOT=Path(__file__).resolve().parents[1]
OLD={'sigmoid2':ROOT/'output/experiments/stage7_microdisp195/run_20261022',
     'transverse':ROOT/'output/experiments/stage7_O_dynamiccolon_transverse/run_20261022'}
COVERAGE=ROOT/'output/experiments/J8_R6_transverse_coverage_lock_pair_seed0/run_20261022_top132'


def old_scene(scene):return OLD[scene]/('c3vd' if scene=='sigmoid2' else scene)


def prepare(root):
    assert not root.exists()
    assert shutil.disk_usage(ROOT).free>25*1024**3
    root.mkdir(parents=True);(root/'logs').mkdir()
    configs={};history=[]
    for scene,keys in [('sigmoid2',['P']),('transverse',['P1','P2'])]:
        src=old_scene(scene);dest=root/scene
        for name in ['setup','runs','pilot','immutable_native','logs']:(dest/name).mkdir(parents=True)
        shutil.copytree(src/'initial',dest/'initial')
        for name in ['gaussian_knn99.pth','importance_sampling.pth']:os.link(src/'setup'/name,dest/'setup'/name)
        shutil.copy2(src/'data_manifest.json',dest/'data_manifest.json')
        for key in (list('ABCDEFGHIJKLMNO') if scene=='sigmoid2' else ['O']):
            (dest/'runs'/key).symlink_to(src/'runs'/key,target_is_directory=True)
            study=src/'runs'/key
            history += list((study/'J').glob('fine_state_*.pth'))
            history += list((study/'setup/warmup').glob('fine_state_*.pth'))
            history += list((study/'common_5000').glob('fine_state_*.pth'))
            history += [study/'J/evaluation/summary.json',study/'initial_integrity.json']
        history += list((src/'initial').rglob('*.pth'))+[src/'setup/gaussian_knn99.pth',src/'setup/importance_sampling.pth',OLD[scene]/'locked_configurations.json']
        locks=json.loads((OLD[scene]/'locked_configurations.json').read_text())['O']
        for key in keys:
            configs[key]=dict(locks,stage5_profile=key,scene=scene,knn_schedule='constant.01 from1->30k',
                depth_schedule='linear.25->.005 from8k->22k' if scene=='sigmoid2' else 'constant.25 from1->30k',
                photometric_schedule='15k->30k Charb.8->.4,MSE2->6,DSSIM.2' if key=='P1' else 'exactO15k->30k Charb.8->.15,MSE2->14,DSSIM.2')
            assert all(configs[key][k]==v for k,v in locks.items() if k!='stage5_profile')
        protocol=dict(scene=scene,new_full_runs=keys,coarse_replayed=False,fine0=str(src/'initial'),
            geometry_nodes='exact O FPS2000+importance400 IDs',appearance_nodes='exact O2000 K4 IDs/weights',
            joint_fine_Adam_updates=30000,TRAIN=252 if scene=='sigmoid2' else 200,TEST=36 if scene=='sigmoid2' else 200,
            Gaussians=364500 if scene=='sigmoid2' else 200000,checkpoint_steps=[2000,5000,10000,20000,30000],
            no_TEST_tuning=True,no_TEST_checkpoint_selection=True,GT_PLY='never in training/initialization',
            knn_weight='.01 constant',depth='estimatedprior anneal8k->22k' if scene=='sigmoid2' else '.25constant syntheticdepth',
            unchanged='O TriHead model/micro/gain/addcap/noresets/LR/TV/colorvar/KL',
            pilot='200+3 actualAdam structural validation only; primary starts sharedfine0; no design changes')
        write(dest/'matrix_protocol.json',protocol)
        write(dest/'fine0_manifest.json',{str(p):sha(p) for p in [*sorted((dest/'initial').rglob('*.pth')),
            dest/'setup/gaussian_knn99.pth',dest/'setup/importance_sampling.pth']})
    # Scene construction writes dataset_initialization.json. Keep even that
    # metadata write out of historical O by loading read-only snapshot adapters.
    old_study=old_scene('transverse')/'runs/O'
    scratch=root/'transverse/reference_snapshots/O'
    for rel,steps in [('setup/warmup',[2000]),('common_5000',[5000]),('J',[10000,20000,30000])]:
        src=old_study/rel;dest=scratch/rel;dest.mkdir(parents=True)
        shutil.copy2(src/'cfg_args',dest/'cfg_args')
        (dest/'point_cloud').symlink_to(src/'point_cloud',target_is_directory=True)
        for step in steps:
            for name in [f'fine_state_{step}.pth',f'checkpoint_{step}.json']:(dest/name).symlink_to(src/name)
        if (src/'dataset_initialization.json').exists():history.append(src/'dataset_initialization.json')
    # Reuse completed GT cache; never load dataset PLY in a trainer process.
    geo=root/'transverse/geometry_gt_evaluation';geo.mkdir()
    for name in ['gt_manifest.json','protocol.json']:shutil.copy2(old_scene('transverse')/'geometry_gt_evaluation'/name,geo/name)
    (geo/'gt_cache').symlink_to(old_scene('transverse')/'geometry_gt_evaluation/gt_cache',target_is_directory=True)
    for key in 'ABC':
        d=root/'historical_float_metrics'/key;d.mkdir(parents=True)
        shutil.copy2(OLD['sigmoid2']/'historical_float_metrics'/key/'metrics.json',d/'metrics.json')
    shutil.copy2(OLD['sigmoid2']/'stage2_selection.json',root/'stage2_selection.json')
    write(root/'historical_paths.json',{k:str(v) for k,v in OLD.items()})
    write(root/'locked_configurations.json',configs)
    write(root/'locked_configurations_sha256.json',dict(sha256=sha(root/'locked_configurations.json')))
    write(root/'protocol.json',dict(new_full_runs=3,variants=['P_sigmoid2','P1_transverse','P2_transverse'],
        old_runs_retrained=False,coarse_replayed=False,primary_initialization='each exact originalO fine0',
        uniform_KNN99=.01,depth='C3VDanneal.25->.0058k->22k;Dynamicfixed.25',
        comparison='P-O KNNonly;P1-O prior+handoffbundle;P2-O priorbundle;P2-P1 handoffonly',
        depth_metrics='legacy finitepositive plus exact F0/U2 clean sentinel/alpha mask, both reported',
        methodology='physical surface anchoring and preserved~29dB are hypotheses; single paired block; no TESTcheckpointselection',
        execution='two idleGPUs, third run queued; projectConda.env, no unrelated jobs stopped'))
    depth_sources={str(ROOT.parent/'ColonGraph-texture/tools'/f):sha(ROOT.parent/'ColonGraph-texture/tools'/f)
        for f in ['evaluate_adaptive_j8_transverse.py','evaluate_adaptive_residual_A20.py','evaluate_coverage_lock_pair.py']}
    depth_sources[str(ROOT.parent/'ColonGraph-texture/utils/adaptive_deformation_graph.py')]=sha(ROOT.parent/'ColonGraph-texture/utils/adaptive_deformation_graph.py')
    write(root/'depth_reference_manifest.json',depth_sources)
    core=['train.py','scene/gaussian_model.py','gaussian_renderer/__init__.py','utils/stage7.py','utils/stage6.py','utils/r7_core.py','utils/photometric_unlock.py','utils/stage3.py']
    core_files={str(ROOT.parent/'ColonGraph-stage7'/f):sha(ROOT.parent/'ColonGraph-stage7'/f) for f in core}
    assert all(sha(ROOT/f)==core_files[str(ROOT.parent/'ColonGraph-stage7'/f)] for f in core)
    write(root/'model_reference_manifest.json',core_files)
    history += [COVERAGE/'results.json',COVERAGE/'report.md',COVERAGE/'postprocess_repair/matched_inference.json']
    write(root/'historical_manifest.json',{str(p):sha(p) for p in history})
    shutil.copy2(COVERAGE/'results.json',root/'coverage_reference_results.json')
    shutil.copy2(COVERAGE/'postprocess_repair/matched_inference.json',root/'coverage_reference_matched_inference.json')
    files=[p for folder in ['arguments','gaussian_renderer','scene','utils','tools','tests'] for p in (ROOT/folder).rglob('*.py')]+[ROOT/'train.py']
    write(root/'code_manifest.json',{str(p):sha(p) for p in files})
    write(root/'status.json',dict(state='PREPARED',new_full_runs=3))
    (root/'report.md').write_text('# Stage8 prepared\n\nNo new final metrics yet.\n')


def verify(root,final=False):
    for manifest in ['code_manifest.json','model_reference_manifest.json','depth_reference_manifest.json']:
        for p,h in json.loads((root/manifest).read_text()).items():assert sha(p)==h,p
    assert sha(root/'locked_configurations.json')==json.loads((root/'locked_configurations_sha256.json').read_text())['sha256']
    if final:
        for manifest in ['historical_manifest.json','sigmoid2/fine0_manifest.json','transverse/fine0_manifest.json']:
            for p,h in json.loads((root/manifest).read_text()).items():assert sha(p)==h,p
        for scene in ['sigmoid2','transverse']:
            for p,h in json.loads((root/scene/'data_manifest.json').read_text()).items():assert sha(p)==h['sha256'],p
        for scene,keys in [('sigmoid2',['P']),('transverse',['P1','P2'])]:
            for key in keys:
                proof=json.loads((root/scene/'runs'/key/'J/integrity.json').read_text())
                assert proof['prior_checks']==30000 and proof['lambda_knn_min']==proof['lambda_knn_max']==.01
                if scene=='transverse':
                    assert proof['lambda_depth_min']==proof['lambda_depth_max']==.25
                    assert json.loads((root/scene/'runs'/key/'J/native_export_integrity.json').read_text())['pass_']
        subprocess.run([sys.executable,str(ROOT/'tools/verify_stage5_gt.py'),str(root/'transverse/geometry_gt_evaluation/gt_manifest.json')],check=True)
        write(root/'final_integrity.json',dict(pass_=True,source_data_history_fine0_config_locks_unchanged=True,
            exactly_three_new_full_runs=True,KNN001_all30000_updates_all_datasets=True,DynamicDepth025_constant=True,
            O_clean_depth_replay_readonly=True))
