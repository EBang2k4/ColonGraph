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
LATE_SCHEDULE = SCHEDULE + [22500, 25000]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--branch", choices=("c", "d", "d2", "d3"), required=True)
    parser.add_argument("--mechanism-diagnostic", action="store_true")
    parser.add_argument("--fixed-appearance-graph", action="store_true")
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
        "schedule_exact": [event["iteration"] for event in events] == (
            LATE_SCHEDULE if args.branch == "d3" else SCHEDULE),
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
    if args.fixed_appearance_graph:
        appearance_graph = node_snapshot.get("appearance_graph")
        source = torch.load(
            Path("output/baseline1_nodes2000") / args.model_path.name /
            "point_cloud/iteration_1/deformation_nodes.pth",
            map_location="cpu", weights_only=True)["graph"]
        checks.update({
            "fixed_appearance_graph_present": appearance_graph is not None,
            "fixed_appearance_nodes_2000": (appearance_graph is not None and
                                             len(appearance_graph["node_xyz"]) == 2000),
            "appearance_features_2000": len(
                appearance["state_dict"]["node_appearance"]) == 2000,
            "appearance_graph_xyz_source_exact": (appearance_graph is not None and
                torch.equal(appearance_graph["node_xyz"], source["node_xyz"])),
            "appearance_graph_indices_source_exact": (appearance_graph is not None and
                torch.equal(appearance_graph["gaussian_node_indices"],
                            source["gaussian_node_indices"])),
            "appearance_graph_weights_source_exact": (appearance_graph is not None and
                torch.equal(appearance_graph["gaussian_node_weights"],
                            source["gaussian_node_weights"])),
            "geometry_graph_grew_independently": len(graph["node_xyz"]) > 2000,
            "appearance_graph_mode_saved": node_snapshot["config"].get(
                "node_appearance_graph_mode") == "fixed_initial",
        })
    if args.branch == "c":
        checks["split_only"] = all(event.get("prunes", 0) == 0 for event in events)
        checks["grew_from_2000"] = len(state["node_xyz"]) > 2000
    elif args.branch == "d":
        checks["split_and_prune_exercised"] = (
            sum(event.get("splits", 0) for event in events) > 0 and
            sum(event.get("prunes", 0) for event in events) > 0)
    else:
        diagnostics = [diagnostic for event in events
                       for diagnostic in event.get("prune_candidate_diagnostics", [])]
        accepted = [diagnostic for diagnostic in diagnostics if diagnostic["accepted"]]
        checks.update({
            "split_exercised": sum(event.get("splits", 0) for event in events) > 0,
            "conservative_diagnostics_present": len(diagnostics) > 0,
            "backbone_2000_preserved": torch.equal(
                torch.sort(state["node_ids"][state["node_ids"] < 2000]).values,
                torch.arange(2000)),
            "child_only_candidates": all(
                diagnostic["node_id"] >= 2000 and diagnostic["parent_id"] >= 0
                for diagnostic in diagnostics),
            "accepted_redundant_and_safe": all(
                diagnostic.get("redundancy_pass", False) and
                diagnostic["safe_prune_error"] <= diagnostic["safe_prune_error_limit"]
                for diagnostic in accepted),
            "prune_count_matches_diagnostics": sum(
                event.get("prunes", 0) for event in events) == len(accepted),
        })
        if args.branch == "d3":
            checks.update({
                "split_only_through_20k": all(
                    event.get("prunes", 0) == 0 for event in events
                    if event["iteration"] <= 20000),
                "prune_only_after_22_5k": all(
                    event.get("splits", 0) == 0 for event in events
                    if event["iteration"] >= 22500),
                "late_prune_events_exact": [
                    event["iteration"] for event in events
                    if event["iteration"] >= 22500] == [22500, 25000],
            })

    if args.mechanism_diagnostic:
        pairs = state.get("split_pairs", [])
        motion = state.get("motion_diagnostics", [])
        available = [item for item in motion if item.get("available", False)]
        observed = {(item["child_id"], item["offset"]) for item in available}
        expected = {(pair["child_id"], offset) for pair in pairs
                    for offset in (500, 1000, 2500)}
        checks.update({
            "split_pairs_recorded": len(pairs) == sum(
                event.get("splits", 0) for event in events),
            "positive_parent_child_separation": bool(pairs) and all(
                pair["d_pos"] > 0 for pair in pairs),
            "appearance_copied_exactly_at_split": bool(pairs) and all(
                pair.get("d_appearance_at_split") == 0 for pair in pairs),
            "motion_offsets_complete": observed == expected,
            "motion_is_finite": bool(available) and all(
                torch.isfinite(torch.tensor(item["d_motion"])).all().item()
                for item in available),
            "appearance_offsets_complete": bool(available) and all(
                "d_appearance" in item for item in available),
            "appearance_is_finite": bool(available) and all(
                torch.isfinite(torch.tensor(item["d_appearance"])).item()
                for item in available),
        })

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
        "total_prune_candidates": sum(event.get("prune_candidates", 0) for event in events),
        "total_prunes": sum(event.get("prunes", 0) for event in events),
        "rejected_low_redundancy": sum(
            event.get("rejected_low_redundancy", 0) for event in events),
        "rejected_high_prune_error": sum(
            event.get("rejected_high_prune_error", 0) for event in events),
        "reload_max_abs_render_error": reload_error,
        "split_pairs": len(state.get("split_pairs", [])),
        "motion_diagnostic_records": len(state.get("motion_diagnostics", [])),
    }
    output = adaptive / f"{args.branch}_full_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
