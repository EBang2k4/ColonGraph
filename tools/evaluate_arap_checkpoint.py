#!/usr/bin/env python3
"""Evaluate fixed backbone stretch, motion, topology, and checkpoint reload."""

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from gaussian_renderer import render
from tools.experiment_common import load_scene
from utils.node_arap import CanonicalNodeARAPGraph


def quantiles(values):
    values = torch.as_tensor(values).float().flatten()
    return {
        "median": float(values.median()),
        "p90": float(torch.quantile(values, 0.90)),
        "p95": float(torch.quantile(values, 0.95)),
        "max": float(values.max()),
    }


@torch.no_grad()
def local_isometry_statistics(model, cameras, pair_count=16, window=4, seed=2718):
    graph = model.gaussian_isometry_graph
    if graph is None:
        return None
    t1_indices = torch.linspace(0, len(cameras)-1, pair_count).round().long()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    losses, relative_samples, pairs = [], [], []
    for t1_index_tensor in t1_indices:
        t1_index = int(t1_index_tensor)
        valid = [index for index in range(max(0, t1_index-window),
                                          min(len(cameras), t1_index+window+1))
                 if index != t1_index]
        t2_index = valid[int(torch.randint(len(valid), (1,), generator=generator))]
        xyz_t1 = model.deform_geometry_from_nodes(cameras[t1_index].time)[0]
        xyz_t2 = model.deform_geometry_from_nodes(cameras[t2_index].time)[0]
        loss, diagnostics = graph(xyz_t1, xyz_t2, return_diagnostics=True)
        losses.append(float(loss))
        relative = diagnostics.relative_length_change.cpu()
        stride = max(1, math.ceil(relative.numel() / 100000))
        relative_samples.append(relative[::stride])
        pairs.append({"t1_index": t1_index, "t2_index": t2_index,
                      "raw_loss": float(loss)})
    relative = torch.cat(relative_samples)
    return {"pair_count": pair_count, "window": window,
            "raw_loss": quantiles(losses),
            "absolute_relative_edge_change": quantiles(relative),
            "pairs": pairs}


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
        "appearance_graph_2000": (
            len(model.appearance_model.node_appearance) == 2000 and
            ((model.appearance_graph is not None and
              len(model.appearance_graph.node_xyz) == 2000) or
             (model.appearance_graph is None and
              len(model.deformation_graph.node_xyz) == 2000))),
        "arap_graph_saved_for_arap_branch": (
            getattr(config, "geometry_regularizer", "knn99") not in
            ("arap", "knn99_arap", "arap_iso") or
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
    isometry = local_isometry_statistics(
        model, cameras,
        window=int(getattr(config, "local_isometry_temporal_window", 4)),
        seed=int(getattr(config, "local_isometry_seed", 2718)))

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
        if model.gaussian_isometry_graph is not None:
            left = model.gaussian_isometry_graph.state_dict_payload()
            right = model_b.gaussian_isometry_graph.state_dict_payload()
            checks["isometry_graph_reload_exact"] = all(
                torch.equal(left[key].cpu(), right[key].cpu())
                if torch.is_tensor(left[key]) else left[key] == right[key]
                for key in left)

    appearance_nodes = (len(model.appearance_graph.node_xyz)
                        if model.appearance_graph is not None
                        else len(model.deformation_graph.node_xyz))
    report = {
        "pass": all(checks.values()),
        "checks": checks,
        "iteration": args.iteration,
        "regularizer": getattr(config, "geometry_regularizer", "knn99"),
        "experiment_seed": int(getattr(config, "experiment_seed", 0)),
        "arap_weight": float(getattr(config, "arap_weight", 0)),
        "local_isometry_weight": float(getattr(config, "local_isometry_weight", 0)),
        "geometry_nodes": len(model.deformation_graph.node_xyz),
        # Older matched K0 checkpoints use the shared geometry graph rather
        # than serializing a separate fixed appearance graph.
        "appearance_nodes": appearance_nodes,
        "timestamps": len(cameras),
        "backbone_stretch": stretch,
        "backbone_motion_magnitude": motion,
        "reload_max_abs_render_error": reload_error,
        "gaussian_isometry_graph": (
            None if model.gaussian_isometry_graph is None else
            model.gaussian_isometry_graph.graph_statistics()),
        "local_gaussian_isometry": isometry,
    }
    output = args.model_path / "arap_evaluation.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
