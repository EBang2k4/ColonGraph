"""Standalone fixed G3-Surf launcher. Default is a read-only dry run."""
import argparse
import copy
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/baselines/g3_surf.json'
SCENE = 'c1_sigmoid2_t4_v4'


def validate_config(config):
    h = config['ModelHiddenParams']
    o = config['OptimizationParams']
    required = dict(deformation_mode='node', num_nodes=2000, node_k=4,
                    node_position_mode='direct', node_position_normalization='bbox',
                    node_sampling='fps', node_importance_extra_nodes=0,
                    graph_forward_mode='g3', graph_forward_adaptive=False,
                    graph_forward_deterministic=False,
                    adaptive_node_mode='none', surface_child_mode='none',
                    color_model='node_residual_mlp', node_appearance_dim=8,
                    node_appearance_graph_mode='fixed_initial',
                    color_residual_scale=0.5, color_residual_tanh=True,
                    use_do=False, geometry_regularizer='knn99',
                    arap_weight=0.0, local_isometry_weight=0.0,
                    node_edge_weight=0.0, canonical_laplacian_weight=0.0)
    for key, value in required.items():
        if h[key] != value:
            raise ValueError(f'Not the packaged G3-Surf baseline: {key}')
    for key, value in dict(iterations=30000, coarse_iterations=100,
                           depth_weight=0.25, knn_weight=0.01,
                           lambda_res=0.0, lambda_dssim=0, lambda_lpips=0).items():
        if o[key] != value:
            raise ValueError(f'Baseline loss/schedule changed: {key}')
    for key in ('densify_from_iter', 'densify_from_iter_fine',
                'pruning_from_iter', 'pruning_from_iter_fine'):
        if o[key] <= 30000:
            raise ValueError('Gaussian densification/pruning must stay disabled')


def make_config(artifact_root, data_path, output):
    config = copy.deepcopy(json.loads(CONFIG.read_text()))
    validate_config(config)
    h = config['ModelHiddenParams']
    for key in ('node_initialization_reference', 'gaussian_knn_reference',
                'graph_forward_reference', 'graph_forward_motion_reference',
                'fine_warmup_reference', 'fine_warmup_state'):
        h[key] = str(artifact_root / h[key])
    h['fine_stage_state_save'] = str(output / 'fine_state_30000.pth')
    config['ModelParams']['source_path'] = str(data_path)
    config['ModelParams']['model_path'] = str(output)
    return config


def preflight(config):
    h = config['ModelHiddenParams']
    expected = [Path(config['ModelParams']['source_path']),
                Path(h['gaussian_knn_reference']),
                Path(h['graph_forward_reference']),
                Path(h['graph_forward_motion_reference']),
                Path(h['fine_warmup_state']),
                Path(h['node_initialization_reference']) / 'point_cloud.ply']
    checkpoint = Path(h['fine_warmup_reference'])
    expected += [checkpoint / name for name in
                 ('point_cloud.ply', 'deformation.pth', 'deformation_table.pth',
                  'deformation_nodes.pth', 'appearance.pth')]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError('External baseline artifacts are required:\n' + '\n'.join(missing))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('train', 'evaluate', 'audit'))
    p.add_argument('--artifact-root', type=Path, default=ROOT,
                   help='Root containing the original output/reference tree')
    p.add_argument('--data', type=Path)
    p.add_argument('--output', type=Path,
                   default=ROOT / f'output/g3_surf_baseline/{SCENE}')
    p.add_argument('--execute', action='store_true', help='Opt in to running commands')
    a = p.parse_args()
    artifact_root = a.artifact_root.resolve()
    data = a.data.resolve() if a.data else artifact_root / f'data/coloncrafter/{SCENE}'
    output = a.output.resolve()
    config = make_config(artifact_root, data, output)
    h = config['ModelHiddenParams']
    if a.action == 'train':
        cmd = [sys.executable, '-u', 'train.py', '--configs', str(output / 'launch_config.json'),
               '--expname', f'g3_surf_baseline/{SCENE}', '--port', '6212',
               '--save_iterations', '5000', '10000', '20000', '30000']
        commands = [cmd]
    elif a.action == 'evaluate':
        commands = [[sys.executable, 'render.py', '-m', str(output), '-s', str(data),
                     '--iteration', str(step), '--skip_train', '--skip_video', '--eval']
                    for step in (5000, 10000, 20000, 30000)]
        commands += [[sys.executable, 'metrics.py', '-m', str(output)],
                     [sys.executable, 'tools/benchmark_adaptive_render.py', str(output),
                      '--iteration', '30000', '--repeats', '5']]
    else:
        common = Path(h['fine_warmup_reference']).parents[1]
        commands = [[sys.executable, 'tools/audit_graph_forward_checkpoint.py', str(output),
                     '--iteration', '30000', '--reference', h['graph_forward_reference'],
                     '--common', str(common)]]
    print(json.dumps({'action': a.action, 'execute': a.execute, 'output': str(output),
                      'source_iteration': 2000, 'fine_updates': 28000}, indent=2))
    for cmd in commands:
        print(shlex.join(cmd), flush=True)
    if not a.execute:
        print('DRY RUN: no training, rendering, checkpoint or optimizer changes.')
        return
    if a.action == 'train':
        if output.exists():
            raise FileExistsError(f'Refusing to overwrite an existing run: {output}')
        preflight(config)
        output.mkdir(parents=True)
        (output / 'launch_config.json').write_text(json.dumps(config, indent=2) + '\n')
    elif not (output / 'cfg_args').is_file():
        raise FileNotFoundError(f'No trained baseline at {output}')
    for cmd in commands:
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
