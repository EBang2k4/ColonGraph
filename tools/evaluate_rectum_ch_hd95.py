"""Post-training-only GT geometry evaluation for four completed rectum runs.

Native ColonSplat opacity-dependent Gaussian sampling and CH/HD95 definitions,
with exact KD-tree nearest neighbors instead of unavailable PyTorch3D. Model
geometry comes from the active G3/K4/J path, not the inactive legacy HexPlane.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUN = ROOT / 'output/experiments/dynamiccolon_rectum_r6_full200_30k/data_rectum/run_20261014'
DATA = ROOT.parent / 'datasets/DynamicColon/data_rectum'
OUT = RUN / 'geometry_gt_evaluation'
BRANCHES = [('K4_old', 'old', 'K4'), ('J_old', 'old', 'J'),
            ('K4_weight', 'weight', 'K4'), ('J_weight', 'weight', 'J')]


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''): h.update(b)
    return h.hexdigest()


def read_xyz(path):
    # Fast, checked reader for the dataset's ASCII XYZ PLY. No deduplication.
    with Path(path).open() as f:
        header = []
        while True:
            line = f.readline()
            assert line, 'Missing PLY end_header'
            header.append(line.strip())
            if line.strip() == 'end_header': break
        assert header[0] == 'ply' and header[1] == 'format ascii 1.0'
        assert header[3:6] == ['property float x', 'property float y', 'property float z']
        count = int(header[2].split()[-1])
        xyz = np.loadtxt(f, dtype=np.float32)
    assert xyz.shape == (count, 3) and np.isfinite(xyz).all()
    return xyz


def cache_one(i):
    path = DATA / 'ply' / ('pc_%04d.ply' % i)
    xyz = read_xyz(path)
    dest = OUT / 'gt_cache' / ('pc_%04d.npy' % i)
    np.save(dest, xyz, allow_pickle=False)
    return dict(frame=i, source=str(path), source_sha256=sha(path), cache=str(dest),
                cache_sha256=sha(dest), points=len(xyz), time_index=(i - 1) / 200)


def prepare():
    assert json.loads((RUN / 'status.json').read_text())['state'] == 'COMPLETE'
    assert not OUT.exists(), 'Preserve existing geometry evaluations.'
    (OUT / 'gt_cache').mkdir(parents=True)
    (OUT / 'logs').mkdir()
    write(OUT / 'status.json', dict(state='PREPARING_GT_CACHE', optimizer_updates=0))
    for folder in [DATA, DATA / 'test']:
        frames = json.loads((folder / 'transforms.json').read_text())['frames']
        assert [Path(f['file_path']).stem for f in frames] == ['r_%04d' % i for i in range(1, 201)]
    # Verify the fast reader against the native PLY parser once.
    from plyfile import PlyData
    vertex = PlyData.read(DATA / 'ply/pc_0001.ply')['vertex']
    native = np.column_stack([vertex[k] for k in ['x', 'y', 'z']]).astype(np.float32)
    assert np.array_equal(native, read_xyz(DATA / 'ply/pc_0001.ply'))
    with ProcessPoolExecutor(max_workers=4) as pool:
        entries = list(pool.map(cache_one, range(1, 201)))
    write(OUT / 'gt_manifest.json', entries)
    checkpoints = {}
    for key, sampler, branch in BRANCHES:
        out = RUN / sampler / branch
        files = [out / 'fine_state_30000.pth', out / 'cfg_args',
                 *sorted((out / 'point_cloud/iteration_30000').iterdir())]
        checkpoints[key] = {str(p): sha(p) for p in files if p.is_file()}
    write(OUT / 'checkpoint_manifest.json', checkpoints)
    write(OUT / 'protocol.json', dict(scene='data_rectum', branches=[b[0] for b in BRANCHES],
        checkpoint=30000, times=200, frame_mapping='r_i at idx/200 -> root ply/pc_i.ply; complete world GT geometry',
        GT_authorization='User explicitly requested CH/HD95 after training; GT used ONLY for this post-training evaluation',
        training_initialization='Unchanged: RGB/depth coarse100, no GT PLY',
        CH='mean_P min_Q ||p-q||^2 + mean_Q min_P ||q-p||^2; original PyTorch3D default norm2 convention',
        HD95='max(quantile95_P min_Q ||p-q||, quantile95_Q min_P ||q-p||); linear quantile interpolation',
        units={'CH': 'dataset_scene_unit^2', 'HD95': 'dataset_scene_unit', 'physical_mm_verified': False},
        geometry='active model.deform_geometry_from_nodes: G3/K4 or G3/J spatial K8, including scale/rotation deformation',
        sampling='original opacity>0; count=max(1,floor(10*(opacity-min)/(max-min+1e-8))); Gaussian normal draw * renderer-activated/clamped scale rotated by quaternion',
        sample_noise='fixed per-frame seed1729+idx; common [GaussianID,10,3] normals, first count_i samples perGaussian for every branch',
        backend='scipy cKDTree exact eps0; no approximate neighbors, no downsampling, no deduplication',
        precision='geometry and GT float32; exact NN distances/reduction float64',
        alignment='native world coordinates; no ICP, no scale normalization, no view/visibility cropping',
        metric_aggregation='arithmetic mean of per-time metrics over ALL200; directional terms/per-view retained',
        optimizer_updates=0, GT_used_for_threshold_or_checkpoint_selection=False,
        original_sampler='ColonSplat/test_geometry.py convert_3dgs_to_pc; vectorized draws with paired noise rather than unseeded per-Gaussian loops'))
    write(OUT / 'status.json', dict(state='PREPARED', optimizer_updates=0))


def metric(pred, gt, workers=4):
    dg = cKDTree(gt).query(pred, k=1, eps=0, workers=workers)[0]
    dp = cKDTree(pred).query(gt, k=1, eps=0, workers=workers)[0]
    a, b = float(np.mean(dg**2)), float(np.mean(dp**2))
    p, q = float(np.quantile(dg, .95)), float(np.quantile(dp, .95))
    return dict(CH=a+b, HD95=max(p, q), CH_pred_to_GT=a, CH_GT_to_pred=b,
                HD95_pred_to_GT=p, HD95_GT_to_pred=q)


def worker(key):
    import torch
    from plyfile import PlyData
    from tools import run_dynamiccolon_rectum_r6_30k as trained
    from utils.general_utils import build_rotation
    from utils.exact_split_branchpoint import model_digest
    torch.set_num_threads(2)
    _, sampler, branch = next(b for b in BRANCHES if b[0] == key)
    out = RUN / sampler / branch
    dest = OUT / key
    dest.mkdir()
    write(dest / 'status.json', dict(state='LOADING_FIXED_30K', optimizer_updates=0))
    m, scene, pipe, bg, cfg, state = trained.restore_final(out)
    before_model = model_digest(m)
    before_adam = trained.digest(m.optimizer.state_dict())
    del state; gc.collect()
    assert not hasattr(m, 'spline')
    cameras = scene.getTestCameras()
    assert len(cameras) == 200
    # Independent exact-NN check against brute-force squared Euclidean distance.
    rng = np.random.default_rng(99)
    a, b = rng.normal(size=(31, 3)).astype(np.float32), rng.normal(size=(47, 3)).astype(np.float32)
    da = np.min(((a.astype(np.float64)[:, None] - b.astype(np.float64)[None])**2).sum(2), axis=1)
    db = np.min(((b.astype(np.float64)[:, None] - a.astype(np.float64)[None])**2).sum(2), axis=1)
    check = metric(a, b)
    assert np.isclose(check['CH'], da.mean() + db.mean(), atol=1e-12)
    assert np.isclose(check['HD95'], max(np.quantile(np.sqrt(da), .95), np.quantile(np.sqrt(db), .95)), atol=1e-12)
    manifests = json.loads((OUT / 'gt_manifest.json').read_text())
    rows = []
    started = time.monotonic()
    with torch.no_grad():
        for idx, (cam, gt_record) in enumerate(zip(cameras, manifests)):
            assert cam.original_image_name == 'r_%04d' % (idx + 1)
            xyz, log_scale, quaternion = m.deform_geometry_from_nodes(cam.time)
            scales = m.scaling_activation(log_scale).clamp_max(.05 * m.spatial_lr_scale)
            opacity = m.get_opacity.squeeze()
            keep = opacity > 0
            assert keep.any()
            opacity = opacity[keep]
            weights = (opacity - opacity.min()) / (opacity.max() - opacity.min() + 1e-8)
            counts = (weights * 10).long().clamp_min(1)
            assert counts.min() >= 1 and counts.max() <= 10
            gen = torch.Generator(device='cuda').manual_seed(1729 + idx)
            noise = torch.randn((len(xyz), 10, 3), generator=gen, device='cuda')[keep]
            chosen = torch.arange(10, device='cuda')[None] < counts[:, None]
            rotated = torch.bmm(noise * scales[keep, None], build_rotation(quaternion[keep]).transpose(1, 2))
            samples = (rotated + xyz[keep, None])[chosen].cpu().numpy()
            assert np.isfinite(samples).all()
            if idx in [0, 99, 199]:
                # The original visual exports are deformed centers. Verify exact
                # active geometry before applying the native Gaussian sampler.
                vertex = PlyData.read(out / 'test/ours_30000/pcd' / ('frame_%05d.ply' % idx))['vertex']
                exported = np.column_stack([vertex[k] for k in ['x', 'y', 'z']]).astype(np.float32)
                assert np.allclose(xyz.cpu().numpy(), exported, atol=2e-6, rtol=0)
            gt_path = Path(gt_record['cache'])
            assert sha(gt_path) == gt_record['cache_sha256']
            gt = np.load(gt_path, allow_pickle=False)
            row = metric(samples, gt)
            row.update(frame=cam.original_image_name, time=float(cam.time), GT_PLY=gt_record['source'],
                       GT_points=len(gt), sampled_points=len(samples), Gaussians=len(xyz),
                       sampled_Gaussians=int(keep.sum()), seed=1729+idx)
            rows.append(row)
            with (dest / 'per_frame.jsonl').open('a') as f: f.write(json.dumps(row) + '\n')
            if (idx+1) % 10 == 0 or idx == 0:
                write(dest / 'status.json', dict(state='EVALUATING', frame=idx+1, total=200,
                    elapsed_seconds=time.monotonic()-started, optimizer_updates=0))
                print(key, idx+1, '/200', 'CH', row['CH'], 'HD95', row['HD95'], 'samples', len(samples), flush=True)
    assert model_digest(m) == before_model and trained.digest(m.optimizer.state_dict()) == before_adam
    for p, h in json.loads((OUT / 'checkpoint_manifest.json').read_text())[key].items(): assert sha(p) == h
    means = {k: float(np.mean([r[k] for r in rows])) for k in ['CH', 'HD95', 'CH_pred_to_GT', 'CH_GT_to_pred', 'HD95_pred_to_GT', 'HD95_GT_to_pred']}
    result = dict(metrics=means, frames=200, checkpoint=30000, evaluation_seconds=time.monotonic()-started,
        sampled_points_mean=float(np.mean([r['sampled_points'] for r in rows])),
        integrity=dict(model_unchanged=True, Adam_unchanged=True, checkpoints_unchanged=True, optimizer_updates=0,
                       active_geometry_matches_native_export=True, exact_KDTree_verified=True),
        units=dict(CH='scene_unit^2', HD95='scene_unit', physical_mm_verified=False))
    write(dest / 'summary.json', result)
    assert not (out / 'test/ours_30000/distance_metrics.json').exists()
    write(out / 'test/ours_30000/distance_metrics.json', dict(chamfer_distance=means['CH'], hd95_distance=means['HD95'],
        evaluated_frames=200, protocol=str(OUT/'protocol.json'), native_metric_definitions=True,
        active_G3_geometry=True, backend='exact_scipy_KDTree', GT_PLY_evaluation_only=True))
    write(dest / 'status.json', dict(state='COMPLETE', frames=200, optimizer_updates=0))


def supervise():
    with (OUT / 'EVALUATION_STARTED').open('x') as f: f.write(str(os.getpid()))
    for subset in [BRANCHES[:2], BRANCHES[2:]]:
        write(OUT / 'status.json', dict(state='RUNNING', branches=[b[0] for b in subset], optimizer_updates=0))
        jobs = []
        for gpu, (key, _, _) in enumerate(subset):
            f = (OUT / 'logs' / (key + '.log')).open('x')
            p = subprocess.Popen([sys.executable, '-u', __file__, 'worker', '--branch', key], cwd=ROOT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1'), stdout=f, stderr=subprocess.STDOUT)
            jobs.append((key, p, f))
        codes = {}
        for key, p, f in jobs: codes[key]=p.wait(); f.close()
        write(OUT / ('exit_' + '_'.join(codes) + '.json'), codes)
        assert not any(codes.values()), codes
    results = {key: json.loads((OUT/key/'summary.json').read_text()) for key, _, _ in BRANCHES}
    for record in json.loads((OUT/'gt_manifest.json').read_text()): assert sha(record['source']) == record['source_sha256']
    write(OUT/'comparison.json', results)
    lines = ['# Rectum — post-training GT geometry CH / HD95', '',
        'Fixed30k checkpoints, all200 time-matched world GTPLYs, no ICP/scale fitting/visibility crop. '
        'Original ColonSplat opacity-adaptive Gaussian sampling (1..10 perGaussian) with paired seeded noise. '
        'Active G3/K4/J geometry. Exact nearest neighbors, full GT/sample clouds.', '',
        '| Run + FULL R6 | CH (scene unit²) ↓ | HD95 (scene unit) ↓ | Frames |', '|---|---:|---:|---:|']
    for key, _, _ in BRANCHES:
        q=results[key]['metrics']; lines.append(f"| {key} | {q['CH']:.8f} | {q['HD95']:.8f} | 200 |")
    lines += ['', 'CH = sum of bidirectional mean squared nearest distances. HD95 = maximum of '
        'bidirectional95th-percentile Euclidean nearest distances. Average per-time metrics over200frames. '
        'Physicalmm conversion is unverified. Sampling uses identical GaussianID-indexed normals seed1729+idx '
        'across branches; native formula/distribution preserved, not bitwise native unseeded loop RNG.', '',
        'GT PLY is authorized ONLY by the later request to add CH/HD95 after training. Training/coarse '
        'initialization remain RGB/depth with noGTPLY. Model/Adam/checkpoint hashes unchanged, zero optimizerupdates. '
        'Native exported center clouds were verified atframes1/100/200. GTframes/caches hashes retained. '
        'Legacy test_geometry.py directly calls inactive HexPlane in node models; this evaluation uses active '
        'deform_geometry_from_nodes to measure the trained G3/K4/J surface representation.', '',
        '[Protocol](protocol.json) · [Per-branch results](comparison.json)']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    p=RUN/'report.md'
    with p.open('a') as f: f.write('\n\n## Added post-training CH / HD95\n\nUser subsequently authorized GTPLY for geometry evaluation. Training/noGTPLY initialization remains unchanged. Earlier GT geometry exclusion is superseded for this post-training addition.\n\n'+ '\n'.join(lines[4:10])+'\n\n[Full GT geometry report](geometry_gt_evaluation/report.md).\n')
    write(OUT/'status.json', dict(state='COMPLETE', branches=4, frames_per_branch=200, optimizer_updates=0))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare','worker','supervise'])
    parser.add_argument('--branch', choices=[b[0] for b in BRANCHES])
    args=parser.parse_args()
    assert os.environ.get('CONDA_PREFIX') == '/home/robot/colongraph/.env', 'Use project Conda'
    try:
        if args.action=='worker': worker(args.branch)
        else: globals()[args.action]()
    except Exception as exc:
        if OUT.exists(): write(OUT/'status.json', dict(state='FAILED', action=args.action, branch=args.branch, error=repr(exc), optimizer_updates=0))
        raise
