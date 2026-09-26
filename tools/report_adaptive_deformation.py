#!/usr/bin/env python3
"""Build the matched A/B/C/D adaptive deformation report."""
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCENE = "c1_sigmoid2_t4_v4"
RUNS = {
    "A fixed FPS-2000": "adaptive_deformation_a_fps2000_fixed",
    "B fixed 2000+400": "adaptive_deformation_b_fps2000_importance400_fixed",
    "C adaptive split": "adaptive_deformation_c_split_only",
    "D adaptive split+prune": "adaptive_deformation_d_split_prune",
    "D2 conservative prune": "adaptive_deformation_d2_conservative_prune",
}


def read(path):
    return json.loads(path.read_text())


def collect(label, tag):
    path = ROOT / "output" / tag / SCENE
    metrics = read(path / "results.json")["ours_30000"]
    summary = read(path / "training_summary.json")
    benchmark = read(path / "adaptive_render_benchmark.json")
    snapshot = torch.load(
        path / "point_cloud" / "iteration_30000" / "deformation_nodes.pth",
        map_location="cpu", weights_only=True)
    row = {
        "label": label, "path": str(path), **metrics,
        "final_nodes": len(snapshot["graph"]["node_xyz"]),
        "training_time_s": summary["total_training_duration_s"],
        "peak_training_vram_mb": summary["peak_vram_mb"],
        **benchmark,
    }
    adaptive = path / "adaptive_nodes"
    if adaptive.exists() and (adaptive / "adaptive_events.jsonl").exists():
        events = [json.loads(line) for line in
                  (adaptive / "adaptive_events.jsonl").read_text().splitlines()]
        row.update({
            "trajectory": [{key: event.get(key) for key in
                            ("iteration", "nodes_before", "nodes_after", "splits", "prunes")}
                           for event in events],
            "total_splits": sum(event.get("splits", 0) for event in events),
            "total_prunes": sum(event.get("prunes", 0) for event in events),
            "shock_mean": sum(event["shock_mean"] for event in events) / len(events),
            "shock_max": max(event["shock_max"] for event in events),
            "changed_k4_percent_mean": sum(
                event["changed_k4_percent"] for event in events) / len(events),
            "splits_suppressed_by_budget": sum(
                event.get("splits_suppressed_by_budget", 0) for event in events),
            "prune_candidates": sum(event.get("prune_candidates", 0) for event in events),
            "rejected_low_redundancy": sum(
                event.get("rejected_low_redundancy", 0) for event in events),
            "rejected_high_prune_error": sum(
                event.get("rejected_high_prune_error", 0) for event in events),
        })
    return row


def main():
    rows = [collect(label, tag) for label, tag in RUNS.items()]
    destination = ROOT / "output" / "adaptive_deformation_comparison" / SCENE
    destination.mkdir(parents=True, exist_ok=True)
    report = {"scene": SCENE, "runs": rows}
    (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Adaptive deformation graph: matched A/B/C/D", "",
             "| Branch | Nodes | PSNR | SSIM | LPIPS | Depth MSE | Train min | FPS | VRAM MB |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['final_nodes']} | {row['PSNR']:.6f} | "
            f"{row['SSIM']:.6f} | {row['LPIPS']:.6f} | {row['DEPTH_MSE']:.8f} | "
            f"{row['training_time_s']/60:.2f} | {row['render_fps']:.2f} | "
            f"{row['peak_training_vram_mb']:.1f} |")
    lines += ["", "## Adaptive event summary", ""]
    for row in rows:
        if "trajectory" not in row:
            continue
        lines.append(
            f"- {row['label']}: splits={row['total_splits']}, prunes={row['total_prunes']}, "
            f"mean/max shock={row['shock_mean']:.6g}/{row['shock_max']:.6g}, "
            f"mean changed K4={row['changed_k4_percent_mean']:.4f}%, "
            f"budget-suppressed={row['splits_suppressed_by_budget']}, "
            f"prune candidates={row['prune_candidates']}, "
            f"rejected redundancy/error={row['rejected_low_redundancy']}/"
            f"{row['rejected_high_prune_error']}.")
    (destination / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
