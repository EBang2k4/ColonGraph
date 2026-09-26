#!/usr/bin/env python3
"""Gate S0/S1 adaptive graph outputs before longer experiments."""
import argparse
import json
from pathlib import Path

import torch


def distribution(value):
    value = value.detach().float().flatten()
    q = torch.quantile(value, torch.tensor([.05, .5, .95]))
    return {"min": float(value.min()), "p05": float(q[0]), "median": float(q[1]),
            "p95": float(q[2]), "max": float(value.max()), "mean": float(value.mean())}


def exact_graph(state, snapshot):
    graph = snapshot["graph"]
    return {
        "node_xyz": torch.equal(state["node_xyz"], graph["node_xyz"]),
        "indices": torch.equal(state["gaussian_node_indices"], graph["gaussian_node_indices"]),
        "weights": torch.equal(state["gaussian_node_weights"], graph["gaussian_node_weights"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--stage", choices=("s0", "s1"), required=True)
    parser.add_argument("--iteration", type=int, default=5200)
    args = parser.parse_args()
    adaptive = args.model_path / "adaptive_nodes"
    final = torch.load(adaptive / "adaptive_state_final.pth", map_location="cpu",
                       weights_only=False)
    latest = torch.load(adaptive / "adaptive_state_latest.pth", map_location="cpu",
                        weights_only=False)
    final_snapshot = torch.load(
        args.model_path / "point_cloud" / f"iteration_{args.iteration}" / "deformation_nodes.pth",
        map_location="cpu", weights_only=True)
    latest_snapshot = torch.load(
        args.model_path / "point_cloud" / "iteration_5000" / "deformation_nodes.pth",
        map_location="cpu", weights_only=True)
    appearance = torch.load(
        args.model_path / "point_cloud" / f"iteration_{args.iteration}" / "appearance.pth",
        map_location="cpu", weights_only=True)
    events = [json.loads(line) for line in (adaptive / "adaptive_events.jsonl").read_text().splitlines()]
    event = events[-1]
    expected_nodes = 2000 if args.stage == "s0" else 2001
    expected_splits = 0 if args.stage == "s0" else 1
    checks = {
        "one_event": len(events) == 1,
        "expected_splits": event["splits"] == expected_splits,
        "expected_node_count": len(final["node_xyz"]) == expected_nodes,
        "node_budget": 1600 <= len(final["node_xyz"]) <= 2400,
        "final_graph_snapshot_exact": all(exact_graph(final, final_snapshot).values()),
        "event_graph_snapshot_exact": all(exact_graph(latest, latest_snapshot).values()),
        "appearance_snapshot_exact": torch.equal(
            final["node_appearance"], appearance["state_dict"]["node_appearance"]),
        "optimizer_saved": bool(final["optimizer"]["param_groups"]),
        "metadata_lengths": all(len(final[name]) == len(final["node_xyz"])
                                for name in ("node_ids", "parent_ids", "birth_iterations",
                                             "prune_counters")),
    }
    if args.stage == "s1":
        checks.update({
            "local_k4_audit": bool(event["outside_affected_physical_k4_exact"] and
                                   event["outside_affected_weights_exact"]),
            "changed_subset": 0 < event["changed_k4_percent"] < 100,
            "finite_small_mean_shock": 0 <= event["shock_mean"] < 1e-3,
            "finite_bounded_max_shock": (0 <= event["shock_max"] <=
                                         event["shock_limit_world"]),
            "metric_shock_psnr": abs(event["metrics_after"]["PSNR"] -
                                     event["metrics_before"]["PSNR"]) < .05,
        })
    graph = final_snapshot["graph"]
    weights = graph["gaussian_node_weights"]
    indices = graph["gaussian_node_indices"]
    influence = torch.zeros(len(graph["node_xyz"]))
    coverage = torch.zeros(len(graph["node_xyz"]))
    for k in range(4):
        influence.scatter_add_(0, indices[:, k], weights[:, k])
        coverage.scatter_add_(0, indices[:, k], (weights[:, k] > .05).float())
    report = {
        "stage": args.stage,
        "pass": all(checks.values()),
        "checks": checks,
        "event": event,
        "final_node_count": len(final["node_xyz"]),
        "score_distribution": distribution(final["gaussian_score"]),
        "observation_distribution": distribution(final["gaussian_observations"]),
        "influence_distribution": distribution(influence),
        "coverage_distribution": distribution(coverage),
        "final_graph_exact_fields": exact_graph(final, final_snapshot),
        "event_graph_exact_fields": exact_graph(latest, latest_snapshot),
    }
    output = adaptive / f"{args.stage}_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
