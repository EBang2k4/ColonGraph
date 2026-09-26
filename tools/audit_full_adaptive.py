#!/usr/bin/env python3
"""Final topology/checkpoint/locality audit for adaptive C/D runs."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from gaussian_renderer import render
from tools.experiment_common import load_scene


SCHEDULE = [5000, 7500, 10000, 12500, 15000, 17500, 20000]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--branch", choices=("c", "d"), required=True)
    args = parser.parse_args()
    adaptive = args.model_path / "adaptive_nodes"
    events = [json.loads(line) for line in
              (adaptive / "adaptive_events.jsonl").read_text().splitlines()]
    state = torch.load(adaptive / "adaptive_state_final.pth", map_location="cpu",
                       weights_only=False)
    snapshot_path = (args.model_path / "point_cloud" /
                     f"iteration_{args.iteration}")
    node_snapshot = torch.load(snapshot_path / "deformation_nodes.pth",
                               map_location="cpu", weights_only=True)
    appearance = torch.load(snapshot_path / "appearance.pth", map_location="cpu",
                            weights_only=True)
    graph = node_snapshot["graph"]
    checks = {
        "schedule_exact": [event["iteration"] for event in events] == SCHEDULE,
        "node_budget": 1600 <= len(state["node_xyz"]) <= 2400,
        "graph_xyz_exact": torch.equal(state["node_xyz"], graph["node_xyz"]),
        "graph_indices_exact": torch.equal(
            state["gaussian_node_indices"], graph["gaussian_node_indices"]),
        "graph_weights_exact": torch.equal(
            state["gaussian_node_weights"], graph["gaussian_node_weights"]),
        "appearance_exact": torch.equal(
            state["node_appearance"], appearance["state_dict"]["node_appearance"]),
        "metadata_lengths": all(len(state[name]) == len(state["node_xyz"])
                                for name in ("node_ids", "parent_ids", "birth_iterations",
                                             "prune_counters")),
        "optimizer_saved": bool(state["optimizer"]["param_groups"]),
        "no_gaussian_topology_change": len(state["gaussian_score"]) == 364500,
        "all_local_audits": all(
            event.get("outside_affected_physical_k4_exact", False) and
            event.get("outside_affected_weights_exact", False) for event in events),
        "all_shocks_bounded": all(
            event.get("shock_max", float("inf")) <=
            event.get("shock_limit_world", -float("inf")) for event in events),
        "monotonic_node_ids": len(torch.unique(state["node_ids"])) == len(state["node_ids"]),
        "trajectory_within_budget": all(
            1600 <= event["nodes_after"] <= 2400 for event in events),
        "final_matches_trajectory": len(state["node_xyz"]) == events[-1]["nodes_after"],
    }
    if args.branch == "c":
        checks["split_only"] = all(event.get("prunes", 0) == 0 for event in events)
        checks["grew_from_2000"] = len(state["node_xyz"]) > 2000
    else:
        checks["split_and_prune_exercised"] = (
            sum(event.get("splits", 0) for event in events) > 0 and
            sum(event.get("prunes", 0) for event in events) > 0)

    model, scene, pipe, background, _ = load_scene(args.model_path, args.iteration)
    camera = scene.getTestCameras()[0]
    with torch.no_grad():
        image_a = render(camera, model, pipe, background, stage="fine")["render"]
    model_b, scene_b, pipe_b, background_b, _ = load_scene(args.model_path, args.iteration)
    with torch.no_grad():
        image_b = render(scene_b.getTestCameras()[0], model_b, pipe_b,
                         background_b, stage="fine")["render"]
    reload_error = float((image_a - image_b).abs().max())
    checks["reload_forward_exact"] = reload_error == 0.0
    report = {
        "pass": all(checks.values()),
        "branch": args.branch,
        "checks": checks,
        "final_nodes": len(state["node_xyz"]),
        "node_count_trajectory": [
            {"iteration": event["iteration"], "before": event["nodes_before"],
             "after": event["nodes_after"], "splits": event.get("splits", 0),
             "prunes": event.get("prunes", 0)} for event in events],
        "mean_shock_over_events": sum(event["shock_mean"] for event in events) / len(events),
        "max_shock_over_events": max(event["shock_max"] for event in events),
        "mean_changed_k4_percent": sum(event["changed_k4_percent"] for event in events) / len(events),
        "total_splits_suppressed_by_budget": sum(
            event.get("splits_suppressed_by_budget", 0) for event in events),
        "reload_max_abs_render_error": reload_error,
    }
    output = adaptive / f"{args.branch}_full_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
