#!/usr/bin/env python3
"""Evaluate fixed backbone stretch, motion, topology, and checkpoint reload."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from gaussian_renderer import render
from tools.experiment_common import load_scene
from utils.node_arap import CanonicalNodeARAPGraph


def quantiles(values):
    values = values.flatten()
    return {
        "median": float(values.median()),
        "p90": float(torch.quantile(values, 0.90)),
        "p95": float(torch.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--evaluation-graph", type=Path,
                        default=Path("output/arap_backbone_eval_k8.pth"))
    parser.add_argument("--reload-audit", action="store_true")
    args = parser.parse_args()
    payload = torch.load(args.evaluation_graph, map_location="cpu", weights_only=True)
    evaluation_graph = CanonicalNodeARAPGraph.from_state_dict_payload(
        payload, device="cuda")
    model, scene, pipe, background, config = load_scene(
        args.model_path, args.iteration)
    backbone = model.deformation_graph.node_xyz[:2000]
    checks = {
        "backbone_count_at_least_2000": len(model.deformation_graph.node_xyz) >= 2000,
        "backbone_xyz_evaluation_graph_exact": torch.equal(
            backbone.detach().cpu(), payload["canonical_xyz"]),
        "appearance_graph_fixed_2000": (
            model.appearance_graph is not None and
            len(model.appearance_graph.node_xyz) == 2000 and
            len(model.appearance_model.node_appearance) == 2000),
        "arap_graph_saved_for_arap_branch": (
            getattr(config, "geometry_regularizer", "knn99") != "arap" or
            model.node_arap_graph is not None),
    }
    cameras = scene.getTrainCameras()
    with torch.no_grad():
        trajectory = torch.stack([
            model.node_positions_at_time(camera.time)[:2000]
            for camera in cameras
        ])
    stretch = evaluation_graph.stretch_statistics(trajectory)
    displacement = (trajectory - evaluation_graph.canonical_xyz[None]).norm(dim=-1)
    motion = quantiles(displacement)

    reload_error = None
    if args.reload_audit:
        camera = scene.getTestCameras()[0]
        with torch.no_grad():
            image_a = render(camera, model, pipe, background, stage="fine")["render"]
        model_b, scene_b, pipe_b, background_b, _ = load_scene(
            args.model_path, args.iteration)
        with torch.no_grad():
            image_b = render(scene_b.getTestCameras()[0], model_b, pipe_b,
                             background_b, stage="fine")["render"]
        reload_error = float((image_a - image_b).abs().max())
        checks["reload_forward_exact"] = reload_error == 0.0

    report = {
        "pass": all(checks.values()),
        "checks": checks,
        "iteration": args.iteration,
        "regularizer": getattr(config, "geometry_regularizer", "knn99"),
        "experiment_seed": int(getattr(config, "experiment_seed", 0)),
        "arap_weight": float(getattr(config, "arap_weight", 0)),
        "geometry_nodes": len(model.deformation_graph.node_xyz),
        "appearance_nodes": len(model.appearance_graph.node_xyz),
        "timestamps": len(cameras),
        "backbone_stretch": stretch,
        "backbone_motion_magnitude": motion,
        "reload_max_abs_render_error": reload_error,
    }
    output = args.model_path / "arap_evaluation.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
