"""Four matched FULL-R6 TRAIN200/providedTEST200 runs, with native lookat exports.

Reuse the verified transverse native training/serialization protocol. Only new
launcher-process adapters select the rectum dataset and geometry node sampling.
R6 appearance stays on the same fixed FPS2000 K4 graph for every branch.
"""
import argparse
import contextlib
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from tools import run_dynamiccolon_full200_30k as full
from tools import run_dynamiccolon_matched30k as legacy
from tools import run_matched_fps_spline32 as matched
from tools import run_weighted_spline32 as base
from scene.deformation_graph import SparseDeformationGraph, farthest_point_sampling
from utils.node_sampling import build_gaussian_node_knn
from utils.exact_split_branchpoint import RAW_NAMES, model_digest

SCENE = 'data_rectum'
DATA = ROOT.parent / 'datasets/DynamicColon' / SCENE
DEFAULT = ROOT / 'output/experiments/dynamiccolon_rectum_r6_full200_30k' / SCENE / 'run_20261014'
SAMPLERS = ['old', 'weight']
NODES = {'old': 2000, 'weight': 2400}
BRANCHES = [('K4_old', 'old', 'K4'), ('J_old', 'old', 'J'),
            ('K4_weight', 'weight', 'K4'), ('J_weight', 'weight', 'J')]
write, sha, digest = full.write, full.sha, full.digest

# The existing split callback and GT-PLY guard resolve these module globals.
full.DATA = legacy.DATA = DATA
full.SCENE = legacy.SCENE = matched.SCENE = base.SCENE = SCENE


def configure(study, scene=SCENE):
    from tools.run_j_baseline30k import configure as original
    helper = original(study, SCENE)
    helper.DATA = DATA
    return helper


legacy.configure = base.configure = configure


def sampler_initial(root, sampler, study):
    base.seed()
    m, s, p, b, c, state = full.load(root / 'initial', 1, appearance='R6')
    assert state['iteration'] == 0 and state['fine_Adam_updates'] == 0
    sampling = torch.load(root / 'initial/weighted_fps_sampling.pth', map_location='cpu', weights_only=True)
    selected = sampling['selected_indices'][:NODES[sampler]].cuda()
    standard = sampling['selected_indices'][:2000].cuda()
    assert torch.equal(farthest_point_sampling(m._xyz.detach(), 2000), standard)
    # Initialize exactly the historical FULL-R6 FPS2000 appearance branch.
    appearance_xyz = m._xyz.detach()[standard]
    appearance_ids, appearance_w = build_gaussian_node_knn(m._xyz.detach(), appearance_xyz, 4)
    m.deformation_graph = SparseDeformationGraph(appearance_xyz, appearance_ids, appearance_w)
    m.appearance_graph = None
    with base.preserve_rng():
        m.initialize_appearance_model()
    # Only geometry receives the extra importance nodes.
    geometry_xyz = m._xyz.detach()[selected]
    geometry_ids, geometry_w = build_gaussian_node_knn(m._xyz.detach(), geometry_xyz, 4)
    m.deformation_graph = SparseDeformationGraph(geometry_xyz, geometry_ids, geometry_w)
    m.gats_joint = True
    m.features.requires_grad_(True)
    m._deformation_table.requires_grad_(True)
    m.appearance_model.requires_grad_(True)
    c.model_path = str(study / 'common_5000')
    s.model_path = c.model_path
    base.update(m, c, node_sampling='fps' if sampler == 'old' else 'fps_plus_importance',
                node_importance_extra_nodes=0 if sampler == 'old' else 400,
                gaussian_knn_reference=str(study / 'setup/gaussian_knn99.pth'), gaussian_knn_save='')
    expected = json.loads((root / 'initial/initial_integrity.json').read_text())
    checks = dict(raw_sha256=digest({k: getattr(m, k) for k in RAW_NAMES}),
                  node_mlp_sha256=digest(m.node_mlp.state_dict()), KNN99_sha256=digest(m.closest_point_indices),
                  geometry_gradients_sha256=digest(state['geometry_gradients']))
    assert all(checks[k] == expected[k] for k in checks)
    assert not hasattr(m, 'spline') and len(m.deformation_graph.node_xyz) == NODES[sampler]
    assert len(m.appearance_graph.node_xyz) == 2000
    assert torch.equal(m.appearance_graph.gaussian_node_indices, appearance_ids)
    appearance_hash = digest(dict(graph=m.appearance_graph.state_dict(), decoder=m.appearance_model.state_dict()))
    torch.save(selected.cpu(), study / 'selected_node_ids.pth')
    write(study / 'initial_integrity.json', dict(pass_=True, **checks, nodes=NODES[sampler],
          FPS_backbone_sha256=digest(standard), selected_ids_sha256=digest(selected),
          appearance_graph_and_decoder_sha256=appearance_hash, appearance_nodes=2000,
          appearance='FULL R6 fixed FPS2000 K4; geometry sampler is independent', fine_updates_before_start=0))
    (study / 'common_5000/cfg_args').write_text(str(c))
    return m, s, p, b, c, state


class Hooks(full.Hooks):
    def setup(self, m):
        assert not hasattr(m, 'spline')
        assert len(m.appearance_graph.node_xyz) == 2000
        assert len(m.deformation_graph.node_xyz) == NODES[self.study.name]
        super().setup(m)


legacy.sampler_initial = sampler_initial
legacy.Hooks = base.Hooks = Hooks


def prepare(root):
    assert not root.exists(), 'Preserve existing runs; choose a fresh output directory.'
    assert shutil.disk_usage(ROOT).free > 12 * 1024**3
    root.mkdir(parents=True)
    for name in ['logs', 'setup', 'initial', 'immutable_native']:
        (root / name).mkdir()
    for sampler in SAMPLERS:
        for name in ['logs', 'setup', 'common_5000', 'K4', 'J']:
            (root / sampler / name).mkdir(parents=True)
    datasets = [DATA, DATA / 'test', *sorted((DATA / 'lookat_cams').glob('lookat_*'))]
    assert len(datasets) == 5
    manifest = {}
    counts = {}
    for dataset in datasets:
        frames = json.loads((dataset / 'transforms.json').read_text())['frames']
        assert len(frames) == 200
        counts[str(dataset)] = len(frames)
        paths = [dataset / 'transforms.json', *[dataset / f[k] for f in frames for k in ['file_path', 'depth_path']]]
        for path in paths:
            assert path.is_file()
            manifest[str(path)] = dict(bytes=path.stat().st_size, sha256=sha(path))
    write(root / 'data_manifest.json', manifest)
    protocol = dict(scene=SCENE, branches=[x[0] + '_FULL_R6' for x in BRANCHES], full_trajectories=4,
        TRAIN=200, TEST=200, lookat_sequences=3, frames_each_lookat=200,
        train_dataset=str(DATA), test_dataset=str(DATA / 'test'),
        initialization='fresh firstTRAIN RGB/depth backprojection200000 + coarse100; no learned initialization',
        GT_PLY='forbidden for initialization/training/evaluation; predicted PCD export allowed',
        coarse_iterations=100, coarse_Adam_updates=99, fine_iterations=30000, actual_fine_Adam_updates=30000,
        samplers={'old': 'standard FPS2000', 'weight': 'same FPS2000 backbone + importance400, geometry2400'},
        importance='same fresh100coarse TRAIN-only visible totalXYZ gradient deviation; P5/P95 normalization',
        R6='canonicalRGB + Gaussian embedding16D + node context8D + decoder; all trainable, eta0.5 tanh',
        appearance_graph='IDENTICAL fixed initial FPS2000 K4 for all4; WeightedFPS only changes geometry',
        G3='own sampler/R6 warmup2k -> own E4surfacegraph + original16TRAIN trajectory gate; active2001',
        blending={'K4': 'original G3 repaired K4 IDW', 'J': 'GGAS-v2 spatial-only connected learnedK8 from5001'},
        J_LR=.001, J_KL=.01, J_ramp_end=6000, KNN99_weight=.01, depth_weight=.25,
        geometry_trainable_through=30000, appearance_trainable_through=30000,
        losses_and_LRs='unchanged selected FULL-R6 transverse protocol; RGB L1/TV.001/colorvariance.0001/lambda_res0',
        camera='seed0 consumes100coarse root200 draws; reset0at5001; exact same30k camera/time plan all4',
        paired_prefix='fresh sampler-specific1..5k, exact shared K4/J split at5k; no prior scene checkpoint reuse',
        retention='full2k warmup,5k branchpoint,30k terminal; native/raw/Adam/RNG/KNN/trainability',
        R6_guard='embedding enabled, finite nonzero gradient100, Adam30000, changed fromfine0, coldreload exact',
        test_selection=False, evaluate_only_after_all4_trainings=True,
        outputs='matchedPSNR/SSIM/LPIPS-A,V/floatvaliddepthMSE/runtime/FPS + original-format TEST/customtrajectory/lookat1,2,3',
        lookat='native qualitative PNG/video only; no lookat threshold or metrics selection',
        depth_units='normalized image diagnostic, physicalmm not verified',
        conda=os.environ['CONDA_PREFIX'], no_auto_retry=True, data_frame_counts=counts)
    write(root / 'matrix_protocol.json', protocol)
    full.configuration(root)
    source = json.loads((ROOT / 'output/experiments/dynamiccolon_full200_30k/data_transverse/run_20261014/code_manifest.json').read_text())
    assert all(sha(p) == h for p, h in source.items()), 'Historical verified implementation changed.'
    files = ['tools/run_dynamiccolon_rectum_r6_30k.py', 'render.py', 'metrics.py',
             'render_custom_test_trajectory.py', 'render_custom_lookat_cameras.py']
    source.update({str(ROOT / p): sha(ROOT / p) for p in files})
    write(root / 'code_manifest.json', source)
    write(root / 'status.json', dict(state='PREPARED', full_trajectories=4, TRAIN=200, TEST=200, lookat=3))
    (root / 'launch_report.md').write_text(
        '# DynamicColon rectum — four fresh FULL-R6 30k runs\n\n'
        'K4 old, J old, K4 WeightedFPS, J WeightedFPS. Fresh RGB/depth/coarse100; all200root TRAIN, '
        'provided200TEST only after all4 trainings. No dataset GT PLY. Conda project.env. Geometry/R6 active '
        'through30k; terminalAdam included. Same fixedFPS2000 K4 R6 appearance graph all4; importance400 '
        'added only to geometry. Exact paired5k branchpoints and common30k camera stream. See matrix_protocol.json.\n\n'
        'Fresh coarse and two disposable101+3resume preflights precede primary training. Each sampler builds '
        'its own G3 after2k and J candidates at5k. No historical learned checkpoint reuse or automatic retry. '
        'Full R6 embedding trainability/gradient/Adam/value-change and coldserialization guards enabled.\n\n'
        'Two GPUs; old/weight prefixes in parallel, K4 pair then J pair. Final metrics/runtime/FPS, '
        'GT/reconstruction videos, native results.json and customtrajectory, and all3lookat PNG/video sequences '
        'for every branch are durably queued. GT Chamfer/HD95 excluded because they require forbidden sourcePLY. '
        'Outputs are on project disk; previous artifacts remain intact.\n')


def run_phase(root, commands, stage):
    assert len(commands) <= 2
    write(root / 'status.json', dict(state='RUNNING', stage=stage, full_trajectories=4,
                                   TRAIN=200, TEST=200, lookat=3, pid=os.getpid()))
    jobs, info = [], {}
    for gpu, (name, argv) in enumerate(commands):
        log = (root / 'logs' / (name + '.log')).open('x')
        cmd = [sys.executable, '-u', __file__, *argv, '--out', str(root)]
        proc = subprocess.Popen(cmd, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                                stdout=log, stderr=subprocess.STDOUT)
        jobs.append((name, proc, log))
        info[name] = dict(pid=proc.pid, gpu=gpu, command=cmd)
    write(root / ('processes_' + stage + '.json'), info)
    codes = {}
    for name, proc, log in jobs:
        codes[name] = proc.wait()
        log.close()
    write(root / ('exit_' + stage + '.json'), codes)
    assert not any(codes.values()), codes


def smoke(root, sampler):
    full.smoke(root, sampler)
    proof = json.loads((root / 'preflight' / sampler / 'initial_integrity.json').read_text())
    assert proof['nodes'] == NODES[sampler] and proof['appearance_nodes'] == 2000


def candidates(root, sampler):
    matched.candidates(root, sampler)


def restore_final(out):
    m, s, p, b, c, state = full.load(out, 30000)
    assert not hasattr(m, 'spline')
    m.training_setup(c)
    m.add_graph_residual_optimizer_group()
    if m.skinning is not None:
        from utils.ggas import add_optimizer
        add_optimizer(m)
    m.optimizer.load_state_dict(state['optimizer'])
    expected = json.loads((out / 'checkpoint_30000.json').read_text())
    assert sha(out / 'fine_state_30000.pth') == expected['sha256']
    assert digest(m.optimizer.state_dict()) == expected['optimizer_sha256']
    assert full.parameters(m)['requires_grad_fields'] == state['requires_grad_fields']
    assert full.parameters(m)['optimizer_steps']['deformation_table'] == [30000]
    return m, s, p, b, c, state


def raw_cameras(folder, white_background=False):
    # ProvidedSplitDataset is deliberately root-only. Read arbitrary evaluation
    # trajectories with the unchanged native camera method, never a training split.
    folder = Path(folder)
    assert folder == DATA / 'test' or folder.parent == DATA / 'lookat_cams'
    reader = full.NativeDataset.__new__(full.NativeDataset)
    cameras = reader.readCamerasFromTransforms(str(folder), 'transforms.json', white_background)
    assert len(cameras) == 200
    return cameras


def native_exports(root, out, m, s, p, b):
    import render as original_render
    import metrics as original_metrics
    import render_custom_test_trajectory as trajectory
    import render_custom_lookat_cameras as lookat
    import imageio.v2 as imageio
    trajectory.render_custom_trajectory = raw_cameras
    lookat.load_lookat_cameras = raw_cameras
    assert not (out / 'test').exists()
    before = dict(model=model_digest(m), Adam=digest(m.optimizer.state_dict()), file=sha(out / 'fine_state_30000.pth'))
    with torch.no_grad():
        original_render.render_set(str(out), 'test', 30000, s.getTestCameras(), m, p, b)
        original_metrics.evaluate([str(out)])
        trajectory.render_set(str(out), 'test', 30000, s.getTestCameras(), m, p, b, str(DATA))
        lookat.render_set(str(out), 'test', 30000, s.getTestCameras(), m, p, b, str(DATA))
    assert before == dict(model=model_digest(m), Adam=digest(m.optimizer.state_dict()), file=sha(out / 'fine_state_30000.pth'))
    dest = out / 'test/ours_30000'
    checks = {}
    for path in sorted(dest.rglob('*.mp4')):
        reader = imageio.get_reader(path)
        metadata = reader.get_meta_data()
        frames = sum(1 for _ in reader)
        reader.close()
        assert frames == 200
        checks[str(path.relative_to(out))] = dict(frames=frames, fps=metadata['fps'], resolution=metadata['size'])
    assert len(checks) == 9
    for name in ['renders', 'renders_no_dcol', 'gt', 'depth', 'gt_depth',
                 'custom_trajectory/custom_trajectory_renders',
                 *['lookat_cameras/lookat_%d_renders' % i for i in [1, 2, 3]]]:
        assert len(list((dest / name).glob('*.png'))) == 200
    assert len(list((dest / 'pcd').glob('*.ply'))) == 200
    write(out / 'native_export_integrity.json', dict(pass_=True, checks=checks, model_unchanged=True,
        Adam_unchanged=True, checkpoint_unchanged=True, optimizer_updates=0, dataset_GT_PLY_reads=0))


def comparison_videos(root):
    from PIL import Image, ImageDraw, ImageFont
    import imageio.v2 as imageio
    dest = root / 'provided_test_evaluation/videos'
    dest.mkdir(parents=True)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 20)
    frames = json.loads((DATA / 'test/transforms.json').read_text())['frames']
    args = dict(fps=20, codec='libx264', pixelformat='yuv420p', macro_block_size=2,
                output_params=['-crf', '18', '-preset', 'fast', '-threads', '2', '-movflags', '+faststart'])
    with contextlib.ExitStack() as stack:
        writers = {key: stack.enter_context(imageio.get_writer(dest / (key + '_GT_reconstruction.mp4'), **args))
                   for key, _, _ in BRANCHES}
        overview = stack.enter_context(imageio.get_writer(dest / 'GT_all4_R6_comparison.mp4', **args))
        for idx, frame in enumerate(frames):
            name = Path(frame['file_path']).stem
            rgba = Image.open(DATA / 'test' / frame['file_path']).convert('RGBA')
            gt = Image.alpha_composite(Image.new('RGBA', rgba.size, (0, 0, 0, 255)), rgba).convert('RGB')
            panels = [gt]
            for key, sampler, branch in BRANCHES:
                pred = Image.open(root / sampler / branch / 'evaluation/30000/frames' / (name + '.png')).convert('RGB')
                pair = Image.new('RGB', (1300, 704), (18, 18, 24))
                draw = ImageDraw.Draw(pair)
                draw.text((12, 8), 'GT TEST | ' + name, font=font, fill='white')
                draw.text((662, 8), key + ' + FULL R6 | 30k', font=font, fill='white')
                pair.paste(gt, (0, 54)); pair.paste(pred, (650, 54))
                writers[key].append_data(np.asarray(pair)); panels.append(pred)
            canvas = Image.new('RGB', (1440, 1120), (18, 18, 24))
            for i, (image, label) in enumerate(zip(panels, ['GT provided TEST'] + [x[0] + ' R6' for x in BRANCHES])):
                x, y = (i % 3) * 480, (i // 3) * 560
                ImageDraw.Draw(canvas).text((x + 12, y + 14), label + ' | ' + name, font=font, fill='white')
                canvas.paste(image.resize((480, 480)), (x, y + 80))
            overview.append_data(np.asarray(canvas))
            if idx == 99: canvas.save(dest / 'GT_all4_preview.png')
    checks = {}
    for path in sorted(dest.glob('*.mp4')):
        reader = imageio.get_reader(path); count = sum(1 for _ in reader); reader.close()
        assert count == 200
        checks[path.name] = dict(frames=count, playback_fps=20)
    assert len(checks) == 5
    write(dest / 'video_integrity.json', dict(pass_=True, checks=checks, display_fps_only=True, optimizer_updates=0))


def postprocess(root):
    from tools.evaluate_geometry_blending import benchmark
    results = {}
    appearance_graph_hashes = []
    for key, sampler, branch in BRANCHES:
        out = root / sampler / branch
        m, s, p, b, c, state = restore_final(out)
        write(out / 'reload_integrity.json', dict(pass_=True, model_exact=True, Adam_exact=True,
              trainability_flags_exact=True, fileSHA_exact=True, embedding_Adam_updates=30000))
        appearance_graph_hashes.append(digest(m.appearance_graph.state_dict()))
        summary = full.evaluate_final(out, m, s, p, b)
        performance = benchmark(m, s, p, b)
        write(out / 'performance.json', performance)
        runtime = json.loads((out / 'runtime.json').read_text())
        prefix = json.loads((root / sampler / 'common_5000/runtime.json').read_text())
        runtime.update(full_trajectory_training_seconds=runtime['training_seconds'] + prefix['training_seconds'],
                       shared_prefix_training_seconds=prefix['training_seconds'])
        write(out / 'runtime.json', runtime)
        rows = [json.loads(x) for f in [root / sampler / 'common_5000/camera_sequence.jsonl', out / 'camera_sequence.jsonl']
                for x in f.read_text().splitlines()]
        assert len(rows) == 30000 and rows == json.loads((out / 'camera_plan.json').read_text())
        assert len({r['camera_name'] for r in rows}) == 200 and not any(r['camera_name'].startswith('test__') for r in rows)
        write(out / 'camera_integrity.json', dict(pass_=True, draws=30000, unique_TRAIN=200, TEST_calls=0, sha256=digest(rows)))
        results[key] = dict(metrics=summary['test'], runtime=runtime, performance=performance, camera_sha256=digest(rows))
        write(root / 'comparison_partial.json', results)
        native_exports(root, out, m, s, p, b)
        write(out / 'status.json', dict(state='COMPLETE', iteration=30000, TRAIN=200, TEST=200, lookat=3))
        del m, s, p, b, c, state
        gc.collect(); torch.cuda.empty_cache()
    assert len(set(appearance_graph_hashes)) == 1
    assert len({r['camera_sha256'] for r in results.values()}) == 1
    write(root / 'comparison.json', results)
    comparison_videos(root)
    lines = ['# DynamicColon rectum — four FULL-R6 30k runs', '',
        'Fresh RGB/depth/coarse100; TRAIN200/providedTEST200; no dataset GT PLY. Geometry and full R6 active '
        'through30k. Geometry FPS2000 vs FPS2000+importance400; fixed FPS2000 K4 appearance identical all4. '
        'G3 from own2k warmup; K4/J paired at5k; J spatialK8 learned from5001. No TEST selection.', '',
        '| Run | PSNR | SSIM | LPIPS-A | LPIPS-V | float-valid depth MSE | Fine train min | FPS |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for key, sampler, branch in BRANCHES:
        r = results[key]; q = r['metrics']
        lines.append(f"| {key} + FULL R6 | {q['PSNR']:.6f} | {q['SSIM']:.6f} | {q['LPIPS_A']:.6f} | {q['LPIPS_V']:.6f} | {q['DEPTH_FLOAT_VALID_MSE']:.9f} | {r['runtime']['full_trajectory_training_seconds']/60:.2f} | {r['performance']['FPS']:.2f} |")
    lines += ['', 'Training minutes include each sampler5kprefix, exclude coarse/setup/checkpoint/evaluation. '
        'Matched RGB metrics use clamp/round uint8; depth is separately max-normalized float-valid image MSE, '
        'not metricmm. Native results.json uses RGB/depth PNG; customtrajectory uses unclampedfloat RGB. '
        'Keep conventions separate. TEST/customtrajectory use the same provided200 cameras.', '',
        '[GT + all4 comparison](provided_test_evaluation/videos/GT_all4_R6_comparison.mp4)', '']
    for key, sampler, branch in BRANCHES:
        lines += [f'[{key}: GT/reconstruction](provided_test_evaluation/videos/{key}_GT_reconstruction.mp4)',
                  f'[{key}: native metrics]({sampler}/{branch}/results.json)',
                  ' · '.join(f'[{key} lookat{i}]({sampler}/{branch}/test/ours_30000/lookat_cameras/lookat_{i}_video.mp4)' for i in [1, 2, 3]), '']
    lines += ['Native lookat exports are qualitative only,3sequences ×200frames perbranch. All36native videos '
              'and5comparison videos fullydecoded200frames. Native playback30fps; comparison20fps; display '
              'choices differ from measuredrendererFPS. GT Chamfer/HD95 excluded because user forbids sourcePLY.']
    (root / 'report.md').write_text('\n'.join(lines) + '\n')
    write(root / 'status.json', dict(state='COMPLETE', full_trajectories=4, iterations=30000,
          TRAIN=200, TEST=200, lookat_sequences_per_branch=3, videos=41))


def supervise(root):
    with (root / 'PIPELINE_STARTED').open('x') as f: f.write(str(os.getpid()))
    run_phase(root, [('fresh_coarse', ['initial'])], 'fresh_coarse')
    for sampler in SAMPLERS:
        os.link(root / 'setup/gaussian_knn99.pth', root / sampler / 'setup/gaussian_knn99.pth')
    run_phase(root, [(s + '_smoke', ['smoke', '--sampler', s]) for s in SAMPLERS], 'preflight')
    proofs = [json.loads((root / 'preflight' / s / 'initial_integrity.json').read_text()) for s in SAMPLERS]
    assert len({r['appearance_graph_and_decoder_sha256'] for r in proofs}) == 1
    assert all(json.loads((root / 'preflight' / s / 'integrity.json').read_text())['pass_'] for s in SAMPLERS)
    write(root / 'launch_verification.json', dict(pass_=True, preflight101_plus3resume=True,
          identical_R6_initial_appearance=True, GT_PLY_forbidden=True, TRAIN=200, TEST=200))
    # Remove only disposable preflight model files created for this new study.
    for sampler in SAMPLERS:
        folder = root / 'preflight' / sampler / 'common_5000'
        for path in folder.glob('fine_state_*.pth'): path.unlink()
        shutil.rmtree(folder / 'point_cloud')
    run_phase(root, [(s + '_prefix', ['prefix', '--sampler', s]) for s in SAMPLERS], 'paired_prefixes')
    proofs = [json.loads((root / s / 'initial_integrity.json').read_text()) for s in SAMPLERS]
    assert len({r['appearance_graph_and_decoder_sha256'] for r in proofs}) == 1
    run_phase(root, [(s + '_candidates', ['candidates', '--sampler', s]) for s in SAMPLERS], 'spatial_K8_candidates')
    for branch in ['K4', 'J']:
        run_phase(root, [(branch + '_' + s, ['worker', '--sampler', s, '--branch', branch]) for s in SAMPLERS], branch + '_continuations')
    for path, h in json.loads((root / 'code_manifest.json').read_text()).items(): assert sha(path) == h
    for path, v in json.loads((root / 'data_manifest.json').read_text()).items(): assert sha(path) == v['sha256']
    run_phase(root, [('postprocess', ['postprocess'])], 'metrics_and_native_TEST_lookat_exports')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'initial', 'smoke', 'prefix', 'candidates', 'worker', 'supervise', 'postprocess'])
    parser.add_argument('--out', type=Path, default=DEFAULT)
    parser.add_argument('--sampler', choices=SAMPLERS)
    parser.add_argument('--branch', choices=['K4', 'J'])
    args = parser.parse_args()
    assert os.environ.get('CONDA_PREFIX') == '/home/robot/colongraph/.env', 'Use project Conda'
    try:
        if args.action == 'initial': full.initial(args.out)
        elif args.action == 'prefix': legacy.prefix(args.out, args.sampler)
        elif args.action == 'smoke': smoke(args.out, args.sampler)
        elif args.action == 'candidates': candidates(args.out, args.sampler)
        elif args.action == 'worker': base.worker(args.out / args.sampler, args.branch)
        else: globals()[args.action](args.out)
    except Exception as exc:
        if args.out.exists():
            write(args.out / 'status.json', dict(state='FAILED', stage=args.action, sampler=args.sampler,
                  branch=args.branch, error=repr(exc), pid=os.getpid(), no_automatic_retry=True))
        raise
