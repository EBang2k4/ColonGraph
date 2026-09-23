"""Audit the controlled 2,000 standard FPS + extra importance-node graph."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.audit_sparse_run import read_config
from utils.node_sampling import (build_gaussian_node_knn, farthest_point_sampling,
                                 fps_plus_importance_sampling)


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def equal_state(left, right):
    return left.keys() == right.keys() and all(torch.equal(value, right[key])
                                               for key, value in left.items())


def summary(value):
    value = value.detach().double().cpu()
    return {
        "mean": value.mean().item(), "std": value.std(unbiased=False).item(),
        "min": value.min().item(), "p05": value.quantile(.05).item(),
        "median": value.median().item(), "p95": value.quantile(.95).item(),
        "max": value.max().item(),
    }


def graph_summary(xyz, importance, graph):
    indices = graph["gaussian_node_indices"]
    weights = graph["gaussian_node_weights"]
    nodes = graph["node_xyz"]
    count = torch.bincount(indices.flatten(), minlength=len(nodes))
    mass = torch.zeros(len(nodes)).scatter_add_(0, indices.flatten(), weights.flatten())
    nearest = (xyz - nodes[indices[:, 0]]).norm(dim=1)
    return {
        "nodes": len(nodes),
        "nearest_node_distance": summary(nearest),
        "high_importance_nearest_distance": summary(nearest[importance >= .9]),
        "low_importance_nearest_distance": summary(nearest[importance <= .1]),
        "node_association_count": summary(count),
        "node_weight_mass": summary(mass),
        "node_mass_cv": mass.std(unbiased=False).item() / mass.mean().item(),
        "unused_nodes": int((count == 0).sum()),
    }


def run(output, baseline, initial_only=False):
    cfg, base_cfg = read_config(output / "cfg_args"), read_config(baseline / "cfg_args")
    defaults = {
        "node_sampling": "fps", "node_sampling_alpha": 1.,
        "node_sampling_importance_reference": "",
        "node_sampling_gradient_source": "total", "node_importance_extra_nodes": 0,
    }
    for key, value in defaults.items():
        cfg.setdefault(key, value)
        base_cfg.setdefault(key, value)
    differences = {key: [base_cfg.get(key), cfg.get(key)]
                   for key in base_cfg.keys() | cfg.keys()
                   if base_cfg.get(key) != cfg.get(key)}
    allowed = {"model_path", "expname", "node_sampling",
               "node_sampling_importance_reference", "node_sampling_gradient_source",
               "node_importance_extra_nodes"}
    if cfg["iterations"] != 30000:
        allowed |= {"iterations", "save_iterations"}
    assert not set(differences) - allowed, differences
    assert cfg["node_sampling"] == "fps_plus_importance"
    assert cfg["num_nodes"] == 2000 and cfg["node_importance_extra_nodes"] == 400
    assert cfg["node_k"] == 4 and cfg["node_sampling_gradient_source"] == "total"
    assert cfg["coarse_iterations"] == 100 and cfg["color_model"] == "node_residual_mlp"

    first = output / "point_cloud/iteration_1"
    base_first = baseline / "point_cloud/iteration_1"
    vertices = PlyData.read(first / "point_cloud.ply")["vertex"].data
    base_vertices = PlyData.read(base_first / "point_cloud.ply")["vertex"].data
    assert np.array_equal(vertices, base_vertices), "Canonical iteration-1 attributes changed"

    nodes = load(first / "deformation_nodes.pth")
    base_nodes = load(base_first / "deformation_nodes.pth")
    assert equal_state(nodes["mlp"], base_nodes["mlp"]), "Motion MLP initialization changed"
    app = load(first / "appearance.pth")
    base_app = load(base_first / "appearance.pth")
    assert app["config"] == base_app["config"]
    for key in app["state_dict"]:
        if key == "node_appearance":
            assert app["state_dict"][key].shape == (2400, 8)
            assert torch.equal(app["state_dict"][key][:2000], base_app["state_dict"][key])
        else:
            assert torch.equal(app["state_dict"][key], base_app["state_dict"][key])

    state = load(output / "weighted_fps_sampling.pth")
    reference = load(Path(cfg["node_sampling_importance_reference"]))
    for key in ("grad_sum", "grad_sq_sum", "grad_count", "importance", "raw_score",
                "observed_post_coarse_xyz", "canonical_xyz"):
        assert torch.equal(state[key], reference[key]), f"Importance reference changed: {key}"
    assert state.get("gradient_source", "total") == "total" and state["coarse_steps"] == 100
    assert state["standard_nodes"] == 2000 and state["importance_extra_nodes"] == 400
    assert state["effective_num_nodes"] == 2400
    assert state["reference_alignment"]["passed"]
    assert state["current_coarse_reference_alignment"]["passed"]

    xyz = state["canonical_xyz"]
    device_xyz = xyz.cuda()
    standard = farthest_point_sampling(device_xyz, 2000).cpu()
    combined = fps_plus_importance_sampling(
        device_xyz, 2000, 400, state["importance"].cuda(), state["raw_score"].cuda()).cpu()
    assert torch.equal(combined, state["selected_indices"])
    assert torch.equal(combined[:2000], standard)
    assert len(combined) == len(combined.unique()) == 2400
    assert torch.equal(xyz[standard], base_nodes["graph"]["node_xyz"])
    assert torch.equal(xyz[combined], nodes["graph"]["node_xyz"])
    indices, weights = build_gaussian_node_knn(device_xyz, xyz[combined].cuda(), 4)
    assert torch.equal(indices.cpu(), nodes["graph"]["gaussian_node_indices"])
    torch.testing.assert_close(weights.cpu(), nodes["graph"]["gaussian_node_weights"], rtol=0, atol=0)
    del device_xyz, indices, weights
    torch.cuda.empty_cache()

    extra = combined[2000:]
    distribution = {
        "standard": graph_summary(xyz, state["importance"], base_nodes["graph"]),
        "standard_plus_importance": graph_summary(xyz, state["importance"], nodes["graph"]),
    }
    report = {
        "config_differences": differences,
        "base_standard_nodes_retained_exactly": True,
        "selected_indices_reproduced": True,
        "k4_weights_reproduced": True,
        "gradient_source": "total",
        "standard_nodes": 2000,
        "importance_extra_nodes": 400,
        "effective_num_nodes": 2400,
        "extra_selected_importance": summary(state["importance"][extra]),
        "extra_selected_raw_score": summary(state["raw_score"][extra]),
        "distribution": distribution,
        "graph_fixed": True,
    }
    if not initial_only:
        final_nodes = load(output / f"point_cloud/iteration_{cfg['iterations']}" / "deformation_nodes.pth")
        assert equal_state(nodes["graph"], final_nodes["graph"])
        assert final_nodes["config"]["node_sampling"] == "fps_plus_importance"
        report["graph_fixed_through_fine"] = True
    report["all_checks_passed"] = True
    (output / "fps_plus_importance_audit.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline", type=Path,
                        default=Path("output/joint_b1_bbox_r6/c1_sigmoid2_t4_v4"))
    parser.add_argument("--initial-only", action="store_true")
    args = parser.parse_args()
    run(args.output, args.baseline, args.initial_only)
