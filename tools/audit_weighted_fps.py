"""Paired initialization audit and spatial/influence reports for weighted FPS."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import csv
import json
import numpy as np
import torch
from plyfile import PlyData
from tools.audit_sparse_run import read_config
from utils.node_sampling import (CoarseGradientImportance, farthest_point_sampling,
                                 weighted_farthest_point_sampling, build_gaussian_node_knn)


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def equal_state(a, b):
    return a.keys() == b.keys() and all(torch.equal(v, b[k]) for k, v in a.items())


def summary(x):
    x = x.detach().double().cpu()
    return dict(mean=x.mean().item(), std=x.std(unbiased=False).item(),
                min=x.min().item(), p05=x.quantile(.05).item(), median=x.median().item(),
                p95=x.quantile(.95).item(), max=x.max().item())


def run(output, baseline, initial_only=False):
    cfg, bcfg = read_config(output/'cfg_args'), read_config(baseline/'cfg_args')
    bcfg.setdefault('node_sampling', 'fps')
    bcfg.setdefault('node_sampling_alpha', 1.)
    bcfg.setdefault('node_sampling_importance_reference', '')
    bcfg.setdefault('node_sampling_gradient_source', 'total')
    cfg.setdefault('node_sampling_importance_reference', '')
    cfg.setdefault('node_sampling_gradient_source', 'total')
    differences = {k: [bcfg.get(k), cfg.get(k)] for k in bcfg.keys() | cfg.keys()
                   if bcfg.get(k) != cfg.get(k)}
    allowed = {'model_path', 'expname', 'node_sampling', 'node_sampling_alpha',
               'node_sampling_importance_reference', 'node_sampling_gradient_source'}
    if cfg['iterations'] != 30000:
        allowed |= {'iterations', 'save_iterations'}
    assert not set(differences)-allowed, differences
    alpha = float(cfg['node_sampling_alpha'])
    assert cfg['node_sampling'] == 'weighted_fps' and alpha >= 0
    assert cfg['num_nodes'] == 2000 and cfg['node_k'] == 4
    assert cfg['coarse_iterations'] == 100 and cfg['color_model'] == 'node_residual_mlp'
    assert cfg['knn_weight'] == .01 and cfg['depth_weight'] == .25
    a, b = output/'point_cloud/iteration_1', baseline/'point_cloud/iteration_1'
    pa, pb = PlyData.read(a/'point_cloud.ply')['vertex'].data, PlyData.read(b/'point_cloud.ply')['vertex'].data
    assert np.array_equal(pa, pb), 'Non-sampling canonical attributes differ'
    for file in ('deformation.pth', 'deformation_table.pth', 'appearance.pth'):
        sa, sb = load(a/file), load(b/file)
        if file == 'appearance.pth':
            assert sa['config'] == sb['config']
            assert equal_state(sa['state_dict'], sb['state_dict'])
        elif isinstance(sa, dict):
            assert equal_state(sa, sb)
        else:
            assert torch.equal(sa, sb)
    an, bn = load(a/'deformation_nodes.pth'), load(b/'deformation_nodes.pth')
    assert equal_state(an['mlp'], bn['mlp']), 'Initial motion MLP or bbox buffers differ'
    s = load(output/'weighted_fps_sampling.pth')
    assert s['coarse_steps'] == 100 and s['first_node_index'] == 0 and s['alpha'] == alpha
    gradient_source = s.get('gradient_source', 'total')
    assert gradient_source == cfg['node_sampling_gradient_source']
    alignment = s['reference_alignment']
    assert alignment['passed']
    assert alignment['mismatched_fraction'] <= alignment['max_mismatched_fraction_allowed']
    assert alignment['max_bbox_relative_difference'] <= alignment['max_bbox_relative_difference_allowed']
    assert alignment['bbox_relative_mean_difference'] <= alignment['max_bbox_relative_mean_difference_allowed']
    xyz, importance = s['canonical_xyz'], s['importance']
    assert np.array_equal(xyz.numpy(), np.stack([pa[k] for k in ('x','y','z')], axis=1))
    assert s['grad_count'].min() >= 0 and s['grad_count'].max() <= 100
    stats = CoarseGradientImportance(xyz, gradient_source=gradient_source)
    for key in ('grad_sum','grad_sq_sum','grad_count'):
        setattr(stats, key, s[key])
    recomputed = stats.finalize()
    for key in ('mean_gradient','deviation','confidence','raw_score','percentile_limits','importance'):
        torch.testing.assert_close(s[key], recomputed[key], rtol=1e-6, atol=1e-12)
    assert torch.isfinite(importance).all() and importance.min() >= 0 and importance.max() <= 1
    assert importance.max() > importance.min(), 'Degenerate importance: weighted run is ineffective'
    if cfg['node_sampling_importance_reference']:
        source = load(Path(cfg['node_sampling_importance_reference']))
        for key in ('grad_sum', 'grad_sq_sum', 'grad_count', 'importance',
                    'observed_post_coarse_xyz', 'canonical_xyz'):
            assert torch.equal(s[key], source[key]), f'Importance reference changed: {key}'
        assert s['importance_reference_path'] == cfg['node_sampling_importance_reference']
        assert s['current_coarse_reference_alignment']['passed']

    # Reproduce each FPS sequence on identical XYZ and use exact K4 associations.
    device_xyz = xyz.cuda()
    weighted = weighted_farthest_point_sampling(device_xyz, 2000, importance.cuda(), alpha).cpu()
    standard = farthest_point_sampling(device_xyz, 2000).cpu()
    assert torch.equal(weighted, s['selected_indices'])
    assert len(weighted.unique()) == 2000 and weighted[0] == standard[0] == 0
    assert torch.equal(xyz[standard], bn['graph']['node_xyz'])
    assert torch.equal(xyz[weighted], an['graph']['node_xyz'])
    assert not torch.equal(weighted, standard)
    idx, w = build_gaussian_node_knn(device_xyz, xyz[weighted].cuda(), 4)
    assert torch.equal(idx.cpu(), an['graph']['gaussian_node_indices'])
    torch.testing.assert_close(w.cpu(), an['graph']['gaussian_node_weights'], rtol=0, atol=0)
    del device_xyz, idx, w
    torch.cuda.empty_cache()

    distribution = {}
    output_rows = []
    for name, selected, graph in [('standard', standard, bn['graph']), ('weighted', weighted, an['graph'])]:
        indices, weights = graph['gaussian_node_indices'], graph['gaussian_node_weights']
        nodes = graph['node_xyz']
        assert torch.isfinite(weights).all() and (weights >= 0).all()
        torch.testing.assert_close(weights.sum(1), torch.ones(len(xyz)))
        nearest = (xyz - nodes[indices[:,0]]).norm(dim=1)
        count = torch.bincount(indices.flatten(), minlength=2000)
        mass = torch.zeros(2000).scatter_add_(0, indices.flatten(), weights.flatten())
        mass_sq = torch.zeros(2000).scatter_add_(0, indices.flatten(), weights.square().flatten())
        effective_support = mass.square() / mass_sq.clamp_min(1e-20)
        effective_nodes_per_gaussian = 1 / weights.square().sum(1)
        selected_importance = importance[selected]
        distribution[name] = {
            'selected_importance': summary(selected_importance),
            'nearest_node_distance': summary(nearest),
            'importance_weighted_distance': (nearest*importance).sum().item()/importance.sum().item(),
            'high_importance_nearest_distance': summary(nearest[importance >= .9]),
            'low_importance_nearest_distance': summary(nearest[importance <= .1]),
            'node_association_count': summary(count), 'node_weight_mass': summary(mass),
            'node_effective_gaussian_support': summary(effective_support),
            'effective_nodes_per_gaussian': summary(effective_nodes_per_gaussian),
            'node_mass_cv': mass.std(unbiased=False).item()/mass.mean().item(),
            'unused_nodes': int((count==0).sum()),
            'bbox_min': nodes.amin(0).tolist(), 'bbox_max': nodes.amax(0).tolist(),
        }
        for j in range(2000):
            output_rows.append(dict(mode=name, node=j, gaussian_index=selected[j].item(),
                                    x=nodes[j,0].item(), y=nodes[j,1].item(), z=nodes[j,2].item(),
                                    importance=selected_importance[j].item(),
                                    associations=count[j].item(), weight_mass=mass[j].item(),
                                    effective_support=effective_support[j].item()))
    with (output/'node_distribution.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for row, (name, selected) in enumerate([('standard', standard), ('weighted', weighted)]):
        for col, (x, y) in enumerate([(0,1), (0,2), (1,2)]):
            ax = axes[row,col]
            ax.scatter(xyz[::80,x], xyz[::80,y], s=.3, c='lightgray', rasterized=True)
            points = ax.scatter(xyz[selected,x], xyz[selected,y], c=importance[selected],
                                s=3, cmap='viridis', vmin=0, vmax=1, rasterized=True)
            ax.set(xlabel='XYZ'[x], ylabel='XYZ'[y], title=name)
            ax.set_aspect('equal')
    fig.colorbar(points, ax=axes, label='Coarse gradient importance')
    fig.savefig(output/'node_spatial_distribution.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for name in ('standard','weighted'):
        rows = [r for r in output_rows if r['mode']==name]
        axes[0].hist([r['weight_mass'] for r in rows], bins=40, alpha=.5, label=name)
        axes[1].plot(sorted(r['weight_mass'] for r in rows), label=name)
    axes[0].set(xlabel='Sum of K4 weights per node', ylabel='Nodes')
    axes[1].set(xlabel='Node rank', ylabel='Sum of K4 weights')
    for ax in axes: ax.legend()
    fig.savefig(output/'node_influence_distribution.png', dpi=160)
    plt.close(fig)

    report = dict(config_differences=differences, initialization_except_graph_exact=True,
                  alpha=alpha, importance_reference=cfg['node_sampling_importance_reference'],
                  gradient_source=gradient_source,
                  gradient_moments_recomputed=True, first_node_index=0, selected_nodes=2000,
                  selected_indices_reproduced=True, standard_fps_reproduces_baseline=True,
                  k4_weights_reproduced=True, fixed_graph=True,
                  observation_rule=s['observation_rule'], percentile_population=s['percentile_population'],
                  observed_count=summary(s['grad_count']), importance=summary(importance),
                  coarse_reference_max_xyz_difference=s['reference_max_xyz_difference'],
                  coarse_reference_alignment=alignment,
                  shared_selected_nodes=len(set(weighted.tolist()) & set(standard.tolist())),
                  distribution=distribution,
                  influence_definition='Fixed graph K4 weights; not pixel visibility/contribution',
                  historical_knn_note='Same cuVS KNN-99 algorithm and seed; historical neighbor table was not persisted, so cross-run bitwise KNN-99 identity is not claimed.')
    if not initial_only:
        final = load(output/f"point_cloud/iteration_{cfg['iterations']}/deformation_nodes.pth")
        assert equal_state(an['graph'], final['graph'])
        assert final['config']['node_sampling'] == 'weighted_fps'
        assert all(torch.equal(an['mlp'][k],final['mlp'][k]) for k in ('position_center','position_scale'))
        report['graph_fixed_through_fine'] = True
    report['all_checks_passed'] = True
    (output/'weighted_fps_audit.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--baseline', type=Path, default=Path('output/joint_b1_bbox_r6/c1_sigmoid2_t4_v4'))
    parser.add_argument('--initial-only', action='store_true')
    args = parser.parse_args()
    run(args.output, args.baseline, args.initial_only)
