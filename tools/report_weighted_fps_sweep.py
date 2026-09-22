"""Summarize the matched alpha sweep against standard FPS and alpha=1."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def run(root, scene):
    baseline = root / 'output/joint_b1_bbox_r6' / scene
    runs = {
        0.25: root / 'output/joint_b1_bbox_r6_weighted_fps_a025' / scene,
        0.5: root / 'output/joint_b1_bbox_r6_weighted_fps_a050' / scene,
        1.0: root / 'output/joint_b1_bbox_r6_weighted_fps' / scene,
    }
    standard = read(baseline/'results.json')['ours_30000']
    alpha_one_audit = read(runs[1.0]/'weighted_fps_audit.json')
    rows = [{
        'alpha': 0.0, 'mode': 'standard_fps', **standard,
        'selected_importance_mean': alpha_one_audit['distribution']['standard']['selected_importance']['mean'],
        'nearest_node_distance_mean': alpha_one_audit['distribution']['standard']['nearest_node_distance']['mean'],
        'high_importance_nearest_distance_mean': alpha_one_audit['distribution']['standard']['high_importance_nearest_distance']['mean'],
        'low_importance_nearest_distance_mean': alpha_one_audit['distribution']['standard']['low_importance_nearest_distance']['mean'],
        'node_mass_cv': alpha_one_audit['distribution']['standard']['node_mass_cv'],
        'shared_nodes_with_standard': 2000,
    }]
    for alpha, path in runs.items():
        metrics = read(path/'results.json')['ours_30000']
        audit = read(path/'weighted_fps_audit.json')
        assert audit['all_checks_passed'] and float(audit['alpha']) == alpha
        d = audit['distribution']['weighted']
        rows.append({
            'alpha': alpha, 'mode': 'weighted_fps', **metrics,
            'selected_importance_mean': d['selected_importance']['mean'],
            'nearest_node_distance_mean': d['nearest_node_distance']['mean'],
            'high_importance_nearest_distance_mean': d['high_importance_nearest_distance']['mean'],
            'low_importance_nearest_distance_mean': d['low_importance_nearest_distance']['mean'],
            'node_mass_cv': d['node_mass_cv'],
            'shared_nodes_with_standard': audit['shared_selected_nodes'],
        })
    for row in rows:
        row['metric_delta_vs_standard'] = {
            key: row[key] - standard[key] for key in ('PSNR', 'SSIM', 'LPIPS', 'DEPTH_MSE')
        }
    report = {
        'scene': scene,
        'controlled_variable': 'node_sampling_alpha',
        'shared_coarse_importance_for_alpha_025_050': True,
        'adaptive_nodes': False,
        'rows': rows,
    }
    destination = root/'output/weighted_fps_alpha_sweep'/scene
    destination.mkdir(parents=True, exist_ok=True)
    (destination/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    lines = [
        '# Weighted FPS alpha sweep — joint B1-bbox + R6', '',
        'Same scene/config/seed and a shared audited coarse-gradient importance map; adaptive nodes are disabled.', '',
        '| alpha | PSNR | SSIM | LPIPS | Depth MSE | selected importance | high-importance distance | low-importance distance | shared nodes |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(
            f"| {row['alpha']:.2f} | {row['PSNR']:.6f} | {row['SSIM']:.6f} | "
            f"{row['LPIPS']:.6f} | {row['DEPTH_MSE']:.8f} | "
            f"{row['selected_importance_mean']:.6f} | "
            f"{row['high_importance_nearest_distance_mean']:.6f} | "
            f"{row['low_importance_nearest_distance_mean']:.6f} | "
            f"{row['shared_nodes_with_standard']} |"
        )
    (destination/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--scene', default='c1_sigmoid2_t4_v4')
    args = parser.parse_args()
    run(args.root, args.scene)
