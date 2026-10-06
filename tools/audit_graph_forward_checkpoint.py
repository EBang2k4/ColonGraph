#!/usr/bin/env python3
"""Checkpoint/freeze/reload audit for a trained G1/G2/G3 branch."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from gaussian_renderer import render
from tools.experiment_common import load_scene


def digest(tensor):
    return hashlib.sha256(
        tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def module_state_equal(left, right):
    if left is None or right is None:
        return left is right
    left_state, right_state = left.state_dict(), right.state_dict()
    return (left_state.keys() == right_state.keys() and
            all(torch.equal(left_state[key], right_state[key])
                for key in left_state))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--common", type=Path, required=True)
    args = parser.parse_args()
    model, scene, pipe, background, config = load_scene(args.path, args.iteration)
    reference = torch.load(args.reference, map_location="cuda", weights_only=True)
    common, _, _, _, _ = load_scene(args.common, 2000)
    mode = config.graph_forward_mode
    expected_weights = (reference["surface_edge_weights"] if mode == "g2" else
                        reference["trajectory_gated_edge_weights"])
    checks = {
        "mode_valid": mode in ("g1", "g2", "g3", "g4"),
        "nodes_2000": len(model.deformation_graph.node_xyz) == 2000,
        "geometry_k4_repaired_exact": torch.equal(
            model.deformation_graph.gaussian_node_indices,
            reference["graph_k4_indices"]),
        "geometry_weights_repaired_exact": torch.equal(
            model.deformation_graph.gaussian_node_weights,
            reference["graph_k4_weights"]),
        "appearance_k4_original_exact": torch.equal(
            model.appearance_graph.gaussian_node_indices,
            reference["original_k4_indices"]),
        "appearance_weights_original_exact": torch.equal(
            model.appearance_graph.gaussian_node_weights,
            reference["original_k4_weights"]),
        "canonical_nodes_common_exact": torch.equal(
            model.deformation_graph.node_xyz,
            common.deformation_graph.node_xyz),
        "knn99_reference_exact": digest(torch.load(
            config.gaussian_knn_reference, map_location="cpu", weights_only=True)) ==
            reference["audit"]["source_hashes"]["gaussian_knn99"],
        "arap_off": config.arap_weight == 0 and model.node_arap_graph is None,
        "node_edge_off": config.node_edge_weight == 0 and model.node_edge_graph is None,
        "isometry_off": config.local_isometry_weight == 0 and model.gaussian_isometry_graph is None,
        "canonical_laplacian_off": config.canonical_laplacian_weight == 0,
        "adaptive_nodes_off": config.adaptive_node_mode == "none",
    }
    if mode == "g1":
        checks["g1_no_residual_decoder"] = model.graph_motion_residual is None
    else:
        checks.update({
            "residual_decoder_present": model.graph_motion_residual is not None,
            "residual_mode_exact": model.graph_motion_residual.mode == mode,
            "residual_topology_exact": torch.equal(
                model.graph_motion_residual.edge_indices,
                reference["surface_edge_indices"]),
            "residual_weights_exact": torch.equal(
                model.graph_motion_residual.edge_weights, expected_weights),
        })
        if mode == "g4":
            checks["guard_alpha_exact"] = model.graph_motion_residual.cancellation_alpha == .10
            checks["guard_no_new_parameters"] = sum(
                p.numel() for p in model.graph_motion_residual.parameters()) == 4803
    camera = scene.getTestCameras()[0]
    image_a = render(camera, model, pipe, background, stage="fine")["render"]
    model_b, scene_b, pipe_b, background_b, _ = load_scene(args.path, args.iteration)
    image_b = render(scene_b.getTestCameras()[0], model_b, pipe_b,
                     background_b, stage="fine")["render"]
    reload_error = float((image_a-image_b).abs().max())
    # CUDA raster accumulation is not guaranteed bitwise deterministic.  Audit
    # the serialized trainable state bit-exactly and use a strict numerical
    # tolerance only for the replayed image.
    checks["reload_gaussian_tensors_exact"] = all(torch.equal(left, right) for left, right in (
        (model._xyz, model_b._xyz), (model.features, model_b.features),
        (model._opacity, model_b._opacity), (model._scaling, model_b._scaling),
        (model._rotation, model_b._rotation)))
    checks["reload_node_mlp_exact"] = module_state_equal(model.node_mlp, model_b.node_mlp)
    checks["reload_appearance_exact"] = module_state_equal(
        model.appearance_model, model_b.appearance_model)
    checks["reload_graph_residual_exact"] = module_state_equal(
        model.graph_motion_residual, model_b.graph_motion_residual)
    # The CUDA Gaussian rasterizer uses unordered atomic accumulation; replay
    # can vary below one 8-bit intensity bin even when every serialized tensor
    # is bit-exact.  Keep state equality as the exact checkpoint criterion and
    # use a strict sub-bin tolerance for the rendered image.
    checks["reload_render_close_1e-3"] = reload_error <= 1e-3
    diagnostics = model.graph_forward_diagnostics(reference["trajectory_times"])
    report = {
        "pass": all(checks.values()),
        "checks": checks,
        "mode": mode,
        "iteration": args.iteration,
        "reload_max_abs_error": reload_error,
        "diagnostics": diagnostics,
        "changed_k4": int(reference["audit"]["changed_gaussians"]),
        "changed_k4_ratio": float(reference["audit"]["changed_ratio"]),
        "checkpoint_bytes": sum(path.stat().st_size for path in
                                (args.path/"point_cloud"/f"iteration_{args.iteration}").glob("*")),
    }
    (args.path/"graph_forward_checkpoint_audit.json").write_text(
        json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit("Graph-forward checkpoint audit failed")


if __name__ == "__main__":
    main()
