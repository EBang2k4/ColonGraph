"""Four fresh matched trajectories: FPS/importance sampling x K4/J, joint Spline32.

One fresh coarse initialization, two sampler-specific 1..5k prefixes, four
5001..30000 continuations. No historical learned checkpoint or automatic retry.
"""
import argparse
import copy
import gc
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from tools import run_weighted_spline32 as base
from tools.run_gats20_joint import restore_rng
from scene.deformation_graph import SparseDeformationGraph
from utils.node_sampling import build_gaussian_node_knn, fps_plus_importance_sampling

SCENE = base.SCENE
DEFAULT = ROOT / 'output/experiments/matched_fps_spline32' / SCENE / 'run_20261013'
SAMPLERS = ['old', 'weight']
NODES = {'old': 2000, 'weight': 2400}
write, sha, digest = base.write, base.sha, base.digest
original_save = base.save


def compact_save(out, m, s, c, i):
    """Hardlink byte-identical inactive legacy tensors within this new study."""
    original_save(out, m, s, c, i)
    root = next(p for p in [out, *out.parents] if (p / 'matrix_protocol.json').exists())
    cache = root / 'immutable_native'
    cache.mkdir(exist_ok=True)
    for name in ['deformation.pth', 'deformation_table.pth', 'appearance.pth']:
        p = out / 'point_cloud' / ('iteration_' + str(i)) / name
        identity = sha(p)
        target = cache / (identity + '_' + name)
        if target.exists():
            assert sha(target) == identity
            p.unlink()
            os.link(target, p)
        else:
            os.link(p, target)


base.save = compact_save


def fresh_initial(root):
    """Run native coarse100, intercept before the first fine update."""
    import train
    base.seed()
    cfg, groups = base.configuration(root)
    cfg.model_path = str(root / 'initial')
    train.args = cfg
    native = train.scene_reconstruction

    def wrapped(*args, **kwargs):
        if args[11] == 'coarse':
            return native(*args, **kwargs)
        m, s = args[9:11]
        cfg.gaussian_knn_reference = str(root / 'setup/gaussian_knn99.pth')
        m.args.gaussian_knn_reference = cfg.gaussian_knn_reference
        base.install(m)
        assert len(m.deformation_graph.node_xyz) == 2400
        sampling = torch.load(root / 'initial/weighted_fps_sampling.pth', map_location='cpu', weights_only=True)
        recomputed = fps_plus_importance_sampling(sampling['canonical_xyz'].cuda(), 2000, 400,
                                                  sampling['importance'].cuda(), sampling['raw_score'].cuda()).cpu()
        assert sampling['coarse_steps'] == 100 and torch.equal(recomputed, sampling['selected_indices'])
        assert recomputed.unique().numel() == 2400
        write(root / 'sampling_audit.json', dict(pass_=True, coarse_steps=100, standard_nodes=2000,
                                                extra_importance_nodes=400, selection_exact=True))
        # Native Scene uses truthiness for iteration: label1 is explicitly BEFORE
        # fine1, with zero fine updates; it is not a learned fine checkpoint.
        compact_save(root / 'initial', m, s, cfg, 1)
        state = torch.load(root / 'initial/fine_state_1.pth', map_location='cpu', weights_only=False)
        state.update(iteration=0, phase='post_coarse_before_fine1', fine_Adam_updates=0,
                     geometry_gradients={k: None if getattr(m, k).grad is None else getattr(m, k).grad.detach().cpu().clone()
                                         for k in ['_xyz', '_scaling', '_rotation', '_opacity']})
        torch.save(state, root / 'initial/fine_state_1.pth')
        checkpoint = json.loads((root / 'initial/checkpoint_1.json').read_text())
        checkpoint.update(sha256=sha(root / 'initial/fine_state_1.pth'), fine_updates=0, phase=state['phase'])
        write(root / 'initial/checkpoint_1.json', checkpoint)
        raw = digest(state['exact_raw'])
        write(root / 'initial/initial_integrity.json', dict(pass_=True, fine_updates=0,
              phase=state['phase'], file_sha256=sha(root / 'initial/fine_state_1.pth'),
              raw_sha256=raw, spline_sha256=state['spline_sha256'], node_mlp_sha256=digest(m.node_mlp.state_dict()),
              normalization_sha256=digest({k:v for k,v in m.node_mlp.state_dict().items() if 'position_' in k}),
              KNN99_sha256=digest(m.closest_point_indices), RNG_sha256=digest(base.rng_state()),
              geometry_gradients_sha256=digest(state['geometry_gradients']),
              serialized_inactive_R6='same weighted2400 graph in all models; no forward/optimizer role'))
        m.pre_split_branchpoint_saved_at = 0

    train.scene_reconstruction = wrapped
    lp, hp, op, pp = groups
    try:
        train.training(lp.extract(cfg), hp.extract(cfg), op.extract(cfg), pp.extract(cfg), [], [], [], None,
                       -1, 'matched_fps_spline32', None)
    finally:
        train.scene_reconstruction = native


def sampler_initial(root, sampler, study):
    base.seed()
    m, s, p, b, c, state = base.load(root / 'initial', 1)
    assert state['iteration'] == 0 and state['fine_Adam_updates'] == 0
    sampling = torch.load(root / 'initial/weighted_fps_sampling.pth', map_location='cpu', weights_only=True)
    selected = sampling['selected_indices'][:NODES[sampler]].cuda()
    if sampler == 'old':
        from scene.deformation_graph import farthest_point_sampling
        # The first 2000 weighted-backbone IDs must be the native FPS IDs.
        standard = farthest_point_sampling(m._xyz.detach(), 2000)
        assert torch.equal(standard, selected)
        xyz = m._xyz.detach()[selected]
        ids, weights = build_gaussian_node_knn(m._xyz.detach(), xyz, 4)
        m.deformation_graph = SparseDeformationGraph(xyz, ids, weights)
    assert len(m.deformation_graph.node_xyz) == NODES[sampler]
    c.model_path = str(study / 'common_5000')
    s.model_path = c.model_path
    base.update(m, c, node_sampling='fps' if sampler == 'old' else 'fps_plus_importance',
                node_importance_extra_nodes=0 if sampler == 'old' else 400,
                gaussian_knn_reference=str(study / 'setup/gaussian_knn99.pth'), gaussian_knn_save='')
    checks = dict(raw_sha256=digest({k:getattr(m,k) for k in base.RAW_NAMES}),
                  spline_sha256=digest(m.spline.state_dict()), node_mlp_sha256=digest(m.node_mlp.state_dict()),
                  KNN99_sha256=digest(m.closest_point_indices),
                  geometry_gradients_sha256=digest(state['geometry_gradients']))
    expected = json.loads((root / 'initial/initial_integrity.json').read_text())
    assert all(checks[k] == expected[k] for k in checks)
    write(study / 'initial_integrity.json', dict(pass_=True, **checks, nodes=NODES[sampler],
          FPS_backbone_sha256=digest(sampling['selected_indices'][:2000]),
          selected_ids_sha256=digest(selected), fine_updates_before_start=0))
    torch.save(selected.cpu(), study / 'selected_node_ids.pth')
    (study / 'common_5000/cfg_args').write_text(str(c))
    return m, s, p, b, c, state


class PrefixHooks(base.Hooks):
    def __init__(self, *args, initial_state, smoke=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_state, self.smoke = initial_state, smoke

    def begin(self, i):
        super().begin(i)
        if i == 1:
            # Restore AFTER native LPIPS/optimizer setup, including native coarse
            # geometry gradients retained at its last (non-Adam) iteration100.
            restore_rng(self.initial_state)
            for k, v in self.initial_state['geometry_gradients'].items():
                getattr(self.m, k).grad = None if v is None else v.cuda().clone()

    def after(self, m, s, cameras, i):
        if self.smoke:
            assert self.renders == self.backwards == 1
            m.spline_basis = None
            assert base.parameters(m)['optimizer_steps']['xyz'] == [i]
            assert base.parameters(m)['optimizer_steps']['spline32'] == [i]
            return
        super().after(m, s, cameras, i)


def prefix(root, sampler, smoke=False):
    study = root / ('preflight/' + sampler if smoke else sampler)
    out = study / 'common_5000'
    if smoke:
        (study / 'setup').mkdir(parents=True)
        out.mkdir()
        os.link(root / 'setup/gaussian_knn99.pth', study / 'setup/gaussian_knn99.pth')
    else:
        with (out / 'PREFIX_STARTED').open('x') as f:
            f.write(str(os.getpid()))
    m, s, p, b, c, state = sampler_initial(root, sampler, study)
    hooks = PrefixHooks(study, out, m, s, p, b, c, 'common', initial_state=state, smoke=smoke)
    fn = base.native_function(hooks)
    (study / 'native_joint_instrumented.py').write_text(hooks.patched_source)
    from utils.timer import Timer
    timer = Timer(); timer.start()
    torch.cuda.reset_peak_memory_stats()
    try:
        fn(c, c, m.args, p, [], [], [], None, -1, m, s, 'fine', None, 101 if smoke else 30000, timer)
    except base.PrefixDone:
        assert not smoke
    if smoke:
        rows = [json.loads(x) for x in (out / 'camera_sequence.jsonl').read_text().splitlines()]
        assert rows == hooks.plan[:101]
        assert m.last_deformation_forward_ms is not None and m.last_deformation_forward_ms > 0
        with base.preserve_rng():
            compact_save(out, m, s, c, 101)
        # Discard model and verify cold exact serialization and Adam before any primary fine prefix.
        expected_model = base.model_digest(m)
        expected_spline = digest(m.spline.state_dict())
        expected_Adam = digest(m.optimizer.state_dict())
        del hooks, fn, m, s, p, b, c, state
        gc.collect(); torch.cuda.empty_cache()
        m, s, p, b, c, state = base.load(out, 101)
        m.training_setup(c); base.configure_optimizer(m)
        m.optimizer.load_state_dict(state['optimizer'])
        assert base.model_digest(m) == expected_model and digest(m.spline.state_dict()) == expected_spline
        assert digest(m.optimizer.state_dict()) == expected_Adam
        # Native continuation must reconstruct the optimizer and restore the
        # camera RNG, not merely load tensors successfully.
        m.fine_resume_state = state
        class ResumeAudit(base.Hooks):
            def after(self, m, s, cameras, i):
                assert self.renders == self.backwards == 1
                m.spline_basis = None
                assert base.parameters(m)['optimizer_steps']['xyz'] == [i]
                assert base.parameters(m)['optimizer_steps']['spline32'] == [i]
        hooks = ResumeAudit(study, out, m, s, p, b, c, 'SMOKE')
        fn = base.native_function(hooks)
        timer = Timer(); timer.start()
        fn(c, c, m.args, p, [], [], [], None, -1, m, s, 'fine', None, 104, timer)
        rows = [json.loads(x) for x in (out / 'camera_sequence.jsonl').read_text().splitlines()]
        assert rows == hooks.plan[:104]
        write(study / 'integrity.json', dict(pass_=True, disposable_updates=101, camera_replay_exact=True,
              native_profiling100_pass=True, model_exact=True, spline_exact=True, Adam_exact=True,
              disposable_native_resume_updates=3, next_camera_and_Adam_steps_exact=True,
              note='Disposable smoke only; all primary prefixes restart from shared untrained fine0.'))


def candidates(root, sampler):
    study = root / sampler
    helper = base.configure(study, SCENE)
    source = inspect.getsource(helper.candidates)
    assert source.count('assert n==2000') == 1
    source = source.replace('assert n==2000', 'assert n==' + str(NODES[sampler]))
    (study / 'candidates_builder.py').write_text(source)
    namespace = dict(helper.__dict__)
    exec(compile(source, str(study / 'candidates_builder.py'), 'exec'), namespace)
    namespace['candidates']()


def prepare(root):
    assert not root.exists()
    assert shutil.disk_usage(ROOT).free > 14 * 1024**3, 'Need14GiB free for retained four-run checkpoints'
    root.mkdir(parents=True)
    protocol = dict(scene=SCENE, branches=['K4_old', 'K4_weight', 'J_old', 'J_weight'], full_trajectories=4,
      appearance='joint independent per-Gaussian cubic Spline32 from fine1', controls=32, appearance_LR=.0025,
      initialization='ONE fresh native coarse100; exact shared raw Gaussian/nodeMLP/Spline/KNN99/RNG state; zero fine updates',
      samplers={'old':'standard FPS2000', 'weight':'same FPS2000 backbone + importance400, total2400'},
      importance='native visible totalXYZ gradient deviation from same100coarse; count5, P5/P95, raw-score ties',
      G3='each sampler builds original E4/G3 from own warmup2k, original16TRAIN trajectory gate',
      blending={'K4':'original G3 repairedK4 IDW', 'J':'GGAS-v2 spatial-only learned connectedK8 from own5k'},
      J_start=5001, skinning_LR=.001, skinning_KL=.01, skinning_ramp_end=6000,
      training='fine1..30000: geometry+appearance active throughout, terminalAdam included, 30000updates',
      KNN99_weight=.01, depth_weight=.25, RGB_loss='native RGBL1 +TV.001 +colorvariance.0001',
      camera='same native seed0 consume100coarse draws; reset0 at5001; assert every camera/time',
      checkpoints=[5000,10000,20000,30000], shared_compute='sampler-specific1..5k prefixes reused within K4/J pair; counted in each trajectory runtime',
      inactive_R6='same serialized2400 R6 graph retained; no forward, no gradients, no optimizer; Spline count identical across samplers',
      coarse_gradient_carryover='native lastcoarse100 geometry gradients restored identically beforefine1',
      final_metrics=['PSNR','SSIM','LPIPS_A','LPIPS_V','normalized_pseudo_depth_MSE','runtime','FPS'],
      old_means='original sampler/blending under SAME new jointSpline protocol, not oldR6 appearance',
      no_auto_retry=True, held_out_selection=False, historical_learned_checkpoint_reuse=False)
    write(root / 'matrix_protocol.json', protocol)
    for folder in ['logs', 'setup', 'initial', 'immutable_native']:
        (root / folder).mkdir()
    # Reuse only frozen protocol/code checks, NEVER the learned initialization.
    core = json.loads((ROOT / 'output/experiments/j_baseline30k/fresh_20261011/training_implementation_manifest.json').read_text())
    assert all(sha(ROOT / p) == h for p, h in core.items())
    write(root / 'native_implementation_manifest.json', core)
    data = ROOT.parent / 'ColonSplat/data/coloncrafter' / SCENE
    write(root / 'data_manifest.json', {str(p):sha(p) for p in [data / 'pose.txt', data / 'points3d.ply',
          *sorted((data / 'color').glob('*.png')), *sorted((data / 'depth').glob('*.npy'))]})
    files = ['tools/run_matched_fps_spline32.py', 'tools/run_weighted_spline32.py', 'tools/run_gats20_joint.py',
             'tools/run_cross_scene_aj.py', 'utils/joint_spline32.py', 'utils/independent_temporal_rgb.py', 'utils/node_sampling.py']
    write(root / 'code_manifest.json', {str(ROOT / p):sha(ROOT / p) for p in files})
    for sampler in SAMPLERS:
        study = root / sampler
        for folder in ['logs', 'setup', 'common_5000', 'K4', 'J']:
            (study / folder).mkdir(parents=True, exist_ok=True)
    write(root / 'status.json', dict(state='PREPARED', full_trajectories=4))


def run_phase(root, commands, stage):
    """One job per GPU; no automatic retry; retain commands/exit codes."""
    jobs = []
    write(root / 'status.json', dict(state='RUNNING', stage=stage))
    for gpu, (name, argv) in enumerate(commands):
        f = (root / 'logs' / (name + '.log')).open('x')
        cmd = [sys.executable, '-u', __file__, *argv, '--out', str(root)]
        p = subprocess.Popen(cmd, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=f, stderr=subprocess.STDOUT)
        jobs.append((name, p, f))
    write(root / ('processes_' + stage + '.json'), {name:dict(pid=p.pid, gpu=gpu, command=commands[gpu][1])
                                                  for gpu, (name,p,f) in enumerate(jobs)})
    codes = {}
    for name, p, f in jobs:
        codes[name] = p.wait(); f.close()
    write(root / ('exit_' + stage + '.json'), codes)
    assert not any(codes.values()), codes


def postprocess(root):
    # Both GPUs idle: benchmark one model at a time, retain native summary and exact reload guards.
    results = {}
    for sampler in SAMPLERS:
        base.postprocess(root / sampler)
        pair = json.loads((root / sampler / 'comparison.json').read_text())
        for branch in ['K4', 'J']:
            results[branch + '_' + sampler] = pair[branch]
    assert len({r['camera_sha256'] for r in results.values()}) == 1
    for sampler in SAMPLERS:
        initial = json.loads((root / sampler / 'initial_integrity.json').read_text())
        assert initial['pass_']
    write(root / 'comparison.json', results)
    lines = ['# Matched G3-Surf + joint Spline32, four30k trajectories', '',
             'Fresh exact common coarse100; FPS2000 versus FPS2000+importance400; sampler-specific G3 at2k, K4/J fork at5k. Geometry and Spline32 active to30k. Same30000 cameras, losses and learning-rate schedules.', '',
             '| Branch | PSNR | SSIM | LPIPS-A | LPIPS-V | Depth MSE | Train min | FPS |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in ['K4_old','K4_weight','J_old','J_weight']:
        r = results[name]; q = r['metrics']
        lines.append(f"| {name} | {q['PSNR']:.6f} | {q['SSIM']:.6f} | {q['LPIPS_A']:.6f} | {q['LPIPS_V']:.6f} | {q['DEPTH_FLOAT_VALID_MSE']:.9f} | {r['runtime']['full_trajectory_training_seconds']/60:.2f} | {r['performance']['FPS']:.2f} |")
    deltas = {}
    for a, b in [('K4_old','K4_weight'),('J_old','J_weight'),('K4_old','J_old'),('K4_weight','J_weight')]:
        deltas[b+' minus '+a] = {k:results[b]['metrics'][k]-results[a]['metrics'][k] for k in
                               ['PSNR','SSIM','LPIPS_A','LPIPS_V','DEPTH_FLOAT_VALID_MSE']}
    write(root / 'matched_deltas.json', deltas)
    # Curves use every recorded checkpoint/gradient sample, without selecting a
    # checkpoint or thresholds from the held-out scores.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    checkpoint_curves = {}
    paired_views = {}
    for sampler in SAMPLERS:
        for branch in ['K4','J']:
            name = branch + '_' + sampler
            paths = [root / sampler / 'common_5000/training_metrics.jsonl', root / sampler / branch / 'training_metrics.jsonl']
            rows = [json.loads(x) for p in paths for x in p.read_text().splitlines()]
            for ax, key in zip(axes[0], ['RGB_L1','depth']):
                ax.plot([r['iteration'] for r in rows], [r[key] for r in rows], label=name, linewidth=1)
                ax.set_title('Native training '+key); ax.set_xlabel('Iteration')
            checkpoints = []
            for i in base.STEPS:
                folder = root / sampler / ('common_5000' if i==5000 else branch) / 'evaluation' / str(i)
                summary = json.loads((folder / 'summary.json').read_text())
                checkpoints.append(dict(iteration=i, **summary['test']))
            checkpoint_curves[name] = checkpoints
            for ax, key in zip(axes[1], ['PSNR','DEPTH_FLOAT_VALID_MSE']):
                ax.plot([r['iteration'] for r in checkpoints], [r[key] for r in checkpoints], 'o-', label=name)
                ax.set_title('Held-out '+key); ax.set_xlabel('Iteration')
    for ax in axes.flat:
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(root / 'learning_curves.png', dpi=160); plt.close(fig)
    write(root / 'checkpoint_curves.json', checkpoint_curves)
    for before, after in [('K4_old','K4_weight'),('J_old','J_weight'),('K4_old','J_old'),('K4_weight','J_weight')]:
        views = {}
        for name in [before, after]:
            branch, sampler = name.split('_')
            views[name] = json.loads((root / sampler / branch / 'evaluation/30000/per_view.json').read_text())
        assert views[before].keys() == views[after].keys()
        paired_views[after+' minus '+before] = {
            k:dict(split=views[before][k]['split'],
                   **{metric:views[after][k][metric]-views[before][k][metric]
                      for metric in ['PSNR','SSIM','LPIPS_A','LPIPS_V','DEPTH_FLOAT_VALID_MSE']
                      if metric in views[before][k] and metric in views[after][k]})
            for k in views[before]}
    write(root / 'paired_per_view_deltas.json', paired_views)
    lines += ['', 'Depth is normalized pseudo-depth MSE. One matched seed/camera block; no TEST threshold or checkpoint selection. Weight comparison includes400extra nodes and changed interpolation/G3, not a fixed-node-count sampler test. Shared5k compute counted in each full trajectory runtime. Historical R6 and frozenSpline30k+30k references have different appearance/protocol budgets.',
              'Exact checkpoint model/Spline/Adam/camera guards passed. Independent CUDA raster repeatability and anatomical stability are not asserted. J and Spline32 remain user-selected baselines; no automatic method promotion.']
    (root / 'report.md').write_text('\n'.join(lines)+'\n')
    write(root / 'status.json', dict(state='COMPLETE', full_trajectories=4, iterations=30000))


def supervise(root):
    with (root / 'PIPELINE_STARTED').open('x') as f:
        f.write(str(os.getpid()))
    initial = root / 'initial/initial_integrity.json'
    if not initial.exists():
        run_phase(root, [('fresh_coarse', ['initial'])], 'fresh_coarse')
    expected = json.loads(initial.read_text())
    assert expected['pass_'] and sha(root / 'initial/fine_state_1.pth') == expected['file_sha256']
    for sampler in SAMPLERS:
        target = root / sampler / 'setup/gaussian_knn99.pth'
        if not target.exists():
            os.link(root / 'setup/gaussian_knn99.pth', target)
        assert os.path.samefile(target, root / 'setup/gaussian_knn99.pth')
    if not all((root / 'preflight' / s / 'integrity.json').exists() for s in SAMPLERS):
        run_phase(root, [(s+'_smoke',['smoke','--sampler',s]) for s in SAMPLERS], 'preflight')
    assert all(json.loads((root / 'preflight' / s / 'integrity.json').read_text())['pass_'] for s in SAMPLERS)
    run_phase(root, [(s+'_prefix',['prefix','--sampler',s]) for s in SAMPLERS], 'prefixes')
    run_phase(root, [(s+'_candidates',['candidates','--sampler',s]) for s in SAMPLERS], 'candidates')
    for branch in ['K4','J']:
        run_phase(root, [(branch+'_'+s,['worker','--sampler',s,'--branch',branch]) for s in SAMPLERS], branch+'_continuations')
    for p, h in json.loads((root / 'native_implementation_manifest.json').read_text()).items():
        assert sha(ROOT / p) == h
    for p, h in json.loads((root / 'code_manifest.json').read_text()).items():
        assert sha(p) == h
    run_phase(root, [('postprocess',['postprocess'])], 'postprocess')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare','initial','smoke','prefix','candidates','worker','supervise','postprocess'])
    parser.add_argument('--out', type=Path, default=DEFAULT)
    parser.add_argument('--sampler', choices=SAMPLERS)
    parser.add_argument('--branch', choices=['K4','J'])
    a = parser.parse_args()
    assert os.environ.get('CONDA_PREFIX') == '/home/robot/colongraph/.env', 'Use project Conda'
    try:
        if a.action == 'initial': fresh_initial(a.out)
        elif a.action in ['smoke','prefix']: prefix(a.out, a.sampler, smoke=a.action=='smoke')
        elif a.action == 'candidates': candidates(a.out, a.sampler)
        elif a.action == 'worker': base.worker(a.out / a.sampler, a.branch)
        else: globals()[a.action](a.out)
    except Exception as e:
        if a.out.exists():
            write(a.out / 'status.json', dict(state='FAILED', stage=a.action, sampler=a.sampler, branch=a.branch,
                                             error=repr(e), pid=os.getpid(), no_automatic_retry=True))
        raise
