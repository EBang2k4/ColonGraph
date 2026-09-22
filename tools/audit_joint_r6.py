"""Strict structural, numerical, gradient, and reload audit for joint B1-bbox+R6."""
import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

from audit_sparse_run import read_config
from experiment_common import load_scene
from gaussian_renderer import render


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "output/baseline1_bbox_nodes2000/c1_sigmoid2_t4_v4"


def ply_arrays(path):
    vertex = PlyData.read(path)["vertex"]
    return {name: np.asarray(vertex[name]).copy() for name in vertex.data.dtype.names}


def state_equal(left, right):
    return left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)


def select_view(scene, uid=144):
    views = scene.getTrainCameras() + scene.getTestCameras()
    matches = [view for view in views if int(view.uid) == uid]
    if len(matches) != 1:
        raise AssertionError(f"Expected exactly one camera uid={uid}, got {len(matches)}")
    return matches[0]


def reload_render(path, iteration, uid=144):
    model, scene, pipe, background, _ = load_scene(path, iteration)
    view = select_view(scene, uid)
    with torch.no_grad():
        package = render(view, model, pipe, background)
    values = {key: package[key].detach().cpu() for key in
              ("render", "depth", "alpha", "transformed_points", "dcol")}
    del package, view, scene, model
    gc.collect()
    torch.cuda.empty_cache()
    return values


def run(experiment, iteration=None):
    output = ROOT / "output" / experiment
    cfg = read_config(output / "cfg_args")
    if iteration is None:
        iteration = int(cfg["iterations"])
    baseline_cfg = read_config(BASELINE / "cfg_args")

    expected = {
        "deformation_mode": "node", "num_nodes": 2000, "node_k": 4,
        "node_position_mode": "direct", "node_position_normalization": "bbox",
        "knn_weight": .01, "depth_weight": .25, "coarse_iterations": 100,
        "densify_from_iter": 500000, "pruning_from_iter": 500000,
        "densify_from_iter_fine": 500000, "pruning_from_iter_fine": 500000,
        "color_model": "node_residual_mlp", "node_appearance_dim": 8,
        "color_residual_scale": .3, "color_residual_tanh": True,
        "lambda_res": 0.0, "appearance_seed": 0,
    }
    for key, value in expected.items():
        assert cfg.get(key) == value, (key, cfg.get(key), value)

    # Every non-appearance setting that controls B1 geometry/training must match.
    matched = [
        "coarse_iterations", "position_lr_init", "position_lr_final",
        "feature_lr", "opacity_lr", "scaling_lr", "rotation_lr",
        "position_lr_delay_mult", "position_lr_max_steps",
        "deformation_lr_init", "deformation_lr_final", "deformation_lr_delay_mult",
        "grid_lr_init", "grid_lr_final", "depth_weight", "knn_weight",
        "dcol_weight", "col_smooth_weight", "tv_weight", "num_nodes", "node_k",
        "node_mlp_width", "node_mlp_depth", "node_position_mode",
        "node_position_normalization", "node_initialization_reference",
        "densify_from_iter", "pruning_from_iter", "densify_from_iter_fine",
        "pruning_from_iter_fine", "densification_interval", "pruning_interval",
        "densification_interval_fine", "pruning_interval_fine", "densify_until_iter",
        "eval", "depth_mode", "no_dx", "no_ds", "no_dr", "no_dcol", "use_do",
    ]
    assert all(cfg[key] == baseline_cfg[key] for key in matched)

    first_path = output / "point_cloud/iteration_1"
    final_path = output / f"point_cloud/iteration_{iteration}"
    assert first_path.is_dir() and final_path.is_dir()
    first_nodes = torch.load(first_path / "deformation_nodes.pth", map_location="cpu", weights_only=True)
    final_nodes = torch.load(final_path / "deformation_nodes.pth", map_location="cpu", weights_only=True)
    first_app = torch.load(first_path / "appearance.pth", map_location="cpu", weights_only=True)
    final_app = torch.load(final_path / "appearance.pth", map_location="cpu", weights_only=True)

    assert state_equal(first_nodes["graph"], final_nodes["graph"]), "K4 graph changed"
    for key in ("position_center", "position_scale"):
        assert torch.equal(first_nodes["mlp"][key], final_nodes["mlp"][key])
    trainable_mlp_keys = [key for key in first_nodes["mlp"] if key.startswith("net.")]
    assert any(not torch.equal(first_nodes["mlp"][key], final_nodes["mlp"][key])
               for key in trainable_mlp_keys), "Node motion MLP did not train"

    first_legacy = torch.load(first_path / "deformation.pth", map_location="cpu", weights_only=True)
    final_legacy = torch.load(final_path / "deformation.pth", map_location="cpu", weights_only=True)
    assert state_equal(first_legacy, final_legacy), "Replaced HexPlane branch changed"

    app_cfg = final_app["config"]
    for key in ("color_model", "node_appearance_dim", "color_residual_scale",
                "color_residual_tanh", "appearance_seed"):
        assert app_cfg[key] == expected[key]
    initial_state, final_state = first_app["state_dict"], final_app["state_dict"]
    assert tuple(initial_state["node_appearance"].shape) == (2000, 8)
    assert torch.count_nonzero(initial_state["residual_mlp.4.weight"]) == 0
    assert torch.count_nonzero(initial_state["residual_mlp.4.bias"]) == 0
    assert any(not torch.equal(initial_state[key], final_state[key]) for key in initial_state)
    assert all(torch.isfinite(value).all() for value in final_state.values())

    first_ply, final_ply = ply_arrays(first_path / "point_cloud.ply"), ply_arrays(final_path / "point_cloud.ply")
    assert first_ply.keys() == final_ply.keys()
    assert all(np.isfinite(value).all() for value in final_ply.values())
    assert np.any(first_ply["x"] != final_ply["x"]), "Joint canonical XYZ did not train"

    with (output / "joint_r6_diagnostics.csv").open(newline="") as stream:
        rows = [{key: float(value) for key, value in row.items()}
                for row in csv.DictReader(stream)]
    assert rows and int(rows[0]["iteration"]) == 1 and int(rows[-1]["iteration"]) == iteration
    numeric_keys = set(rows[0]) - {"iteration"}
    assert all(np.isfinite(row[key]) for row in rows for key in numeric_keys)
    assert rows[0]["mean_abs_delta"] == 0 and rows[0]["max_abs_delta"] == 0
    assert max(row["max_abs_delta"] for row in rows) <= .300001
    later = rows[1:]
    assert later and max(row["appearance_grad_norm"] for row in later) > 0
    assert max(row["node_appearance_grad_norm"] for row in later) > 0
    assert max(row["embedding_grad_norm"] for row in later) > 0
    assert max(row["node_motion_grad_norm"] for row in later) > 0
    assert min(row["transformed_xyz_finite"] for row in rows) == 1
    assert min(row["depth_finite_ratio"] for row in rows) == 1

    summary = json.loads((output / "training_summary.json").read_text())
    assert summary["color_model"] == "node_residual_mlp"
    assert summary["coarse_iterations"] == 100 and summary["fine_iterations"] == iteration
    assert np.isfinite(summary["peak_vram_mb"]) and summary["peak_vram_mb"] > 0

    render_a = reload_render(output, iteration)
    render_b = reload_render(output, iteration)
    reload_max_abs = {}
    for key in render_a:
        difference = (render_a[key] - render_b[key]).abs().max().item()
        reload_max_abs[key] = difference
        torch.testing.assert_close(render_a[key], render_b[key], rtol=0, atol=0)
    assert render_a["dcol"].abs().max().item() <= .300001

    report = {
        "experiment": experiment,
        "iteration": iteration,
        "protocol_exact": expected,
        "matched_b1_bbox_fields": matched,
        "graph_fixed": True,
        "bbox_normalization_fixed": True,
        "node_motion_mlp_updated": True,
        "canonical_geometry_updated_and_finite": True,
        "legacy_hexplane_bitwise_unchanged": True,
        "r6_zero_initialized_at_fine_iteration_1": True,
        "r6_parameters_updated_and_finite": True,
        "gradient_checks_passed": True,
        "depth_and_geometry_numerically_stable": True,
        "delta_bound_observed": max(row["max_abs_delta"] for row in rows),
        "diagnostic_rows": len(rows),
        "last_diagnostic": rows[-1],
        "training_summary": summary,
        "checkpoint_reload_max_abs_difference": reload_max_abs,
        "checkpoint_reload_exact": True,
        "all_checks_passed": True,
    }
    path = output / "joint_r6_audit.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    parser.add_argument("--iteration", type=int)
    args = parser.parse_args()
    run(args.experiment, args.iteration)
