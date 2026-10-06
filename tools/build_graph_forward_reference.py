#!/usr/bin/env python3
"""Build the fixed G1-G3 K4 repair and G3 trajectory-gating reference."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from tools.experiment_common import load_scene
from utils.graph_forward import repair_gaussian_k4_minimal


def digest(tensor):
    return hashlib.sha256(
        tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


@torch.no_grad()
def motion_reference(path, iterations, canonical_xyz, times):
    result = {}
    for iteration in iterations:
        model, _, _, _, _ = load_scene(path, iteration)
        if not torch.equal(model.deformation_graph.node_xyz, canonical_xyz):
            raise ValueError(f"K0 iteration {iteration} canonical nodes differ from common 2k")
        rows = []
        for time in times.to(model._xyz.device):
            rows.append(model.node_positions_at_time(time)-canonical_xyz)
        magnitude = torch.stack(rows).norm(dim=-1).flatten()
        result[str(iteration)] = float(torch.quantile(magnitude, .95))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--common", type=Path, required=True)
    parser.add_argument("--surface-graph", type=Path, required=True)
    parser.add_argument("--k0", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    model, scene, _, _, config = load_scene(args.common, 2000)
    surface = torch.load(args.surface_graph, map_location=model._xyz.device,
                         weights_only=True)
    node_xyz = model.deformation_graph.node_xyz
    if not torch.equal(surface["canonical_xyz"], node_xyz):
        raise ValueError("Saved E4 topology does not match common 2k nodes")
    original_indices = model.deformation_graph.gaussian_node_indices
    original_weights = model.deformation_graph.gaussian_node_weights
    repaired_indices, repaired_weights, changed, fallback, audit = (
        repair_gaussian_k4_minimal(
            model._xyz, node_xyz, original_indices, original_weights,
            surface["edge_indices"]))

    cameras = sorted(scene.getTrainCameras(),
                     key=lambda camera: float(torch.as_tensor(camera.time).reshape(-1)[0]))
    selected = torch.linspace(0, len(cameras)-1, 16).round().long().tolist()
    times = torch.stack([
        torch.as_tensor(cameras[index].time, device=model._xyz.device,
                        dtype=model._xyz.dtype).reshape(-1)[0]
        for index in selected])
    trajectory = torch.stack([model.node_positions_at_time(time) for time in times])
    displacement = trajectory-node_xyz[None]
    source, target = surface["edge_indices"]
    difference = displacement[:, source]-displacement[:, target]
    distance = difference.square().sum(-1).mean(0).sqrt()
    sigma = distance.median().clamp_min(1e-8)
    similarity = torch.exp(-distance.square()/(2*sigma.square()))
    gated_weights = (surface["edge_weights"]*similarity).clamp_min(1e-12)
    monitor_iterations = [5000, 10000]
    reference_motion = motion_reference(
        args.k0, monitor_iterations, node_xyz, times)

    changed_xyz = model._xyz[changed]
    audit.update({
        "surface_topology": surface["topology"],
        "surface_edges": int(surface["edge_indices"].shape[1]),
        "trajectory_timestamp_indices": selected,
        "trajectory_distance": {
            "median": float(distance.median()),
            "p90": float(torch.quantile(distance, .9)),
            "p95": float(torch.quantile(distance, .95)),
            "max": float(distance.max()),
        },
        "trajectory_sigma": float(sigma),
        "trajectory_similarity": {
            "median": float(similarity.median()),
            "p10": float(torch.quantile(similarity, .1)),
            "p05": float(torch.quantile(similarity, .05)),
            "min": float(similarity.min()),
        },
        "changed_gaussian_xyz": (None if not len(changed_xyz) else {
            "min": changed_xyz.amin(0).tolist(),
            "max": changed_xyz.amax(0).tolist(),
            "mean": changed_xyz.mean(0).tolist(),
        }),
        "motion_reference_p95": reference_motion,
        "source_hashes": {
            "canonical_xyz": digest(model._xyz),
            "canonical_node_xyz": digest(node_xyz),
            "original_k4_indices": digest(original_indices),
            "original_k4_weights": digest(original_weights),
            "appearance_k4_indices": digest(
                model.appearance_graph.gaussian_node_indices),
            "appearance_k4_weights": digest(
                model.appearance_graph.gaussian_node_weights),
            "gaussian_knn99": digest(torch.load(
                config.gaussian_knn_reference, map_location="cpu", weights_only=True)),
        },
    })
    payload = {
        "format_version": 1,
        "canonical_node_xyz": node_xyz.detach().cpu(),
        "original_k4_indices": original_indices.detach().cpu(),
        "original_k4_weights": original_weights.detach().cpu(),
        "graph_k4_indices": repaired_indices.detach().cpu(),
        "graph_k4_weights": repaired_weights.detach().cpu(),
        "changed_mask": changed.detach().cpu(),
        "fallback_mask": fallback.detach().cpu(),
        "surface_edge_indices": surface["edge_indices"].detach().cpu(),
        "surface_edge_weights": surface["edge_weights"].detach().cpu(),
        "trajectory_gated_edge_weights": gated_weights.detach().cpu(),
        "trajectory_distance": distance.detach().cpu(),
        "trajectory_similarity": similarity.detach().cpu(),
        "trajectory_times": times.detach().cpu(),
        "motion_reference_p95": reference_motion,
        "audit": audit,
    }
    torch.save(payload, args.output/"graph_forward_reference.pth")
    (args.output/"audit.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False)+"\n")
    print(json.dumps(audit, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
