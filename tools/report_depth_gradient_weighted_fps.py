"""Compare standard, total-loss-gradient, and depth-gradient FPS at alpha=.5."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def run(root, scene):
    paths = {
        'standard_fps': root/'output/joint_b1_bbox_r6'/scene,
        'total_gradient_a050': root/'output/joint_b1_bbox_r6_weighted_fps_a050'/scene,
        'depth_gradient_a050': root/'output/joint_b1_bbox_r6_depth_weighted_fps_a050'/scene,
    }
    standard = read(paths['standard_fps']/'results.json')['ours_30000']
    rows = []
    total_audit = read(paths['total_gradient_a050']/'weighted_fps_audit.json')
    for name, path in paths.items():
        metrics = read(path/'results.json')['ours_30000']
        if name == 'standard_fps':
            audit = total_audit
            d = audit['distribution']['standard']
            source, shared = 'none', 2000
        else:
            audit = read(path/'weighted_fps_audit.json')
            assert audit['all_checks_passed'] and float(audit['alpha']) == .5
            d = audit['distribution']['weighted']
            source = audit.get('gradient_source', 'total')
            shared = audit['shared_selected_nodes']
        rows.append({
            'name': name, 'gradient_source': source, **metrics,
            'delta_vs_standard': {key: metrics[key]-standard[key]
                                  for key in ('PSNR','SSIM','LPIPS','DEPTH_MSE')},
            'selected_importance_mean': d['selected_importance']['mean'],
            'nearest_node_distance_mean': d['nearest_node_distance']['mean'],
            'high_importance_nearest_distance_mean': d['high_importance_nearest_distance']['mean'],
            'low_importance_nearest_distance_mean': d['low_importance_nearest_distance']['mean'],
            'node_mass_cv': d['node_mass_cv'],
            'shared_nodes_with_standard': shared,
        })
    report = {'scene': scene, 'alpha': .5, 'adaptive_nodes': False, 'rows': rows}
    destination = root/'output/depth_gradient_weighted_fps_comparison'/scene
    destination.mkdir(parents=True, exist_ok=True)
    (destination/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    lines = [
        '# Depth-gradient Weighted FPS — joint B1-bbox + R6', '',
        'Fixed 2,000 nodes, K=4, alpha=0.5. No adaptive nodes, ARAP, or local-frame deformation.', '',
        '| sampler | PSNR | SSIM | LPIPS | Depth MSE | selected importance | high-importance distance | node mass CV | shared nodes |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['PSNR']:.6f} | {row['SSIM']:.6f} | "
            f"{row['LPIPS']:.6f} | {row['DEPTH_MSE']:.8f} | "
            f"{row['selected_importance_mean']:.6f} | "
            f"{row['high_importance_nearest_distance_mean']:.6f} | "
            f"{row['node_mass_cv']:.6f} | {row['shared_nodes_with_standard']} |"
        )
    (destination/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--scene', default='c1_sigmoid2_t4_v4')
    args = parser.parse_args()
    run(args.root, args.scene)
