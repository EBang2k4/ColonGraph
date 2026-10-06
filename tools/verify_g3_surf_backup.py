"""Read-only state/replay check for an archived G3-Surf checkpoint.

Run from the original artifact root so historical cfg_args relative paths keep
their original meaning. Only this packaged code is imported; no optimizer,
training, file writes or camera/RNG resets are performed.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from gaussian_renderer import render
from tools.experiment_common import load_scene
from utils.exact_split_branchpoint import model_digest


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path, help='Model directory containing cfg_args')
    args = parser.parse_args()
    a, scene, pipe, bg, config = load_scene(args.checkpoint.resolve(), 30000)
    b, scene_b, pipe_b, bg_b, _ = load_scene(args.checkpoint.resolve(), 30000)
    if config.graph_forward_mode != 'g3' or config.adaptive_node_mode != 'none':
        raise ValueError('This is not the fixed G3-Surf archive')
    if len(a.deformation_graph.node_xyz) != 2000 or a.appearance_model.eta != .5:
        raise ValueError('Node count or appearance bound changed')
    before_a, before_b = model_digest(a), model_digest(b)
    if before_a != before_b:
        raise RuntimeError('Serialized state did not reload bit-exactly')
    names = ('render', 'depth', 'alpha', 'transformed_points', 'transformed_color')
    out_a = render(scene.getTestCameras()[0], a, pipe, bg, stage='fine')
    out_b = render(scene_b.getTestCameras()[0], b, pipe_b, bg_b, stage='fine')
    differences = {name: float((out_a[name] - out_b[name]).abs().max()) for name in names}
    print(json.dumps({'replay_max_abs_differences': differences}), flush=True)
    if differences['render'] > 1e-3:
        raise RuntimeError('Rendered reload exceeded the archived 1e-3 tolerance')
    # Preserve the archived graph index_add/CUDA rasterizer behavior. Neither
    # unordered graph aggregation nor rendering is promised bitwise exact.
    torch.testing.assert_close(out_a['transformed_points'], out_b['transformed_points'],
                               rtol=1e-6, atol=1e-3)
    torch.testing.assert_close(out_a['transformed_color'], out_b['transformed_color'],
                               rtol=0, atol=1e-6)
    if before_a != model_digest(a) or before_b != model_digest(b):
        raise RuntimeError('Read-only replay mutated model state')
    print(json.dumps(dict(pass_=True, state_sha256=before_a,
                          tensor_reload_exact=True, replay_state_unchanged=True,
                          geometry_forward_close=True,
                          color_forward_close=True,
                          replay_max_abs_differences=differences,
                          optimizer_steps=0, source_files_written=0), indent=2))


if __name__ == '__main__':
    main()
