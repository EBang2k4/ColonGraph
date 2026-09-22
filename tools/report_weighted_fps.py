"""Write the standard-vs-weighted FPS comparison after the pipeline completes."""
import argparse
import csv
import json
from pathlib import Path


def run(output, baseline):
    read = lambda p: json.loads(p.read_text())
    a = read(baseline/'results.json')['ours_30000']
    b = read(output/'results.json')['ours_30000']
    audit = read(output/'weighted_fps_audit.json')
    alpha = audit['alpha']
    joint = read(output/'joint_r6_audit.json')
    assert audit['all_checks_passed'] and joint['all_checks_passed']
    report = dict(alpha=alpha, gradient_source=audit.get('gradient_source', 'total'),
                  baseline=str(baseline), weighted=str(output), standard_metrics=a,
                  weighted_metrics=b, weighted_minus_standard={k:b[k]-a[k] for k in a},
                  distribution=audit['distribution'], audits_passed=True,
                  timing=read(output/'training_summary.json'),
                  profile=read(output/'standard_vs_weighted_profile.json'))
    with (output/'joint_r6_diagnostics.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    report['residual_statistics'] = {
        'mean_abs_delta_sample_mean': sum(float(r['mean_abs_delta']) for r in rows)/len(rows),
        'max_abs_delta': max(float(r['max_abs_delta']) for r in rows),
        'saturation_sample_mean': sum(float(r['saturation_ratio']) for r in rows)/len(rows),
    }
    (output/'weighted_fps_comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    lines = ['# Standard FPS vs Weighted FPS — joint B1-bbox + R6', '',
             f"Same scene, 100 coarse + 30k fine, 2,000 nodes, K4, alpha={alpha:g}, "
             f"gradient source={audit.get('gradient_source', 'total')}.", '',
             '| Metric | Standard FPS | Weighted FPS | Difference |', '|---|---:|---:|---:|']
    for key in ('PSNR','SSIM','LPIPS','DEPTH_MSE'):
        lines.append(f'| {key} | {a[key]:.8f} | {b[key]:.8f} | {b[key]-a[key]:+.8f} |')
    lines += ['', '| Sampling statistic | Standard | Weighted |', '|---|---:|---:|']
    d = audit['distribution']
    for label, key in [('Selected importance mean','selected_importance'),
                       ('Mean nearest-node distance','nearest_node_distance'),
                       ('High-importance nearest-node distance','high_importance_nearest_distance'),
                       ('Mean node association count','node_association_count'),
                       ('Mean node effective Gaussian support','node_effective_gaussian_support')]:
        lines.append(f"| {label} | {d['standard'][key]['mean']:.6f} | {d['weighted'][key]['mean']:.6f} |")
    lines += ['', f"Shared selected Gaussian IDs: {audit['shared_selected_nodes']}/2000.",
              f"All initialization attributes except graph match exactly; graph fixed through fine. Audits passed.",
              '', 'Importance uses total-loss XYZ gradients before clipping, counted only when radii > 0. '
              'P5/P95 includes all Gaussians, with unseen scores zero. '
              'Same historical post-coarse reference is used for paired initialization. '
              'cuVS KNN-99 algorithm/seed are unchanged; old neighbor indices were not persisted, '
              'so bitwise cross-run KNN-99 equality cannot be established.', '',
              'Distribution figures: node_spatial_distribution.png and node_influence_distribution.png. '
              'Influence is graph weight mass, not per-pixel rasterization contribution.']
    (output/'weighted_fps_comparison.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('output', type=Path)
    p.add_argument('--baseline', type=Path, default=Path('output/joint_b1_bbox_r6/c1_sigmoid2_t4_v4'))
    args = p.parse_args()
    run(args.output, args.baseline)
