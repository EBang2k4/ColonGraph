#!/usr/bin/env python3
"""Summarize and visualize the parent-child mechanism of adaptive splits."""
import argparse
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from plyfile import PlyData


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {key: None for key in ("count", "min", "p05", "median", "p95", "max", "mean")}
    return {
        "count": int(len(values)), "min": float(values.min()),
        "p05": float(np.quantile(values, .05)), "median": float(np.median(values)),
        "p95": float(np.quantile(values, .95)), "max": float(values.max()),
        "mean": float(values.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--baseline-model-path", type=Path)
    args = parser.parse_args()
    adaptive = args.model_path / "adaptive_nodes"
    state = torch.load(adaptive / "adaptive_state_final.pth", map_location="cpu",
                       weights_only=False)
    pairs = state.get("split_pairs", [])
    motion = [item for item in state.get("motion_diagnostics", [])
              if item.get("available", False)]
    by_offset = {}
    for offset in (500, 1000, 2500):
        records = [item for item in motion if item["offset"] == offset]
        values = [item["d_motion_mean"] for item in records]
        by_offset[str(offset)] = {
            "d_motion_mean": distribution(values),
            "d_appearance": distribution(
                [item["d_appearance"] for item in records]),
            "nonzero_fraction_gt_1e-6": (None if not values else
                float(np.mean(np.asarray(values) > 1e-6))),
            "appearance_nonzero_fraction_gt_1e-8": (None if not records else
                float(np.mean(np.asarray([item["d_appearance"] for item in records]) > 1e-8))),
            "pairs": len(records),
        }
    report = {
        "pairs": len(pairs),
        "d_pos": distribution([pair["d_pos"] for pair in pairs]),
        "d_appearance_at_split": distribution(
            [pair["d_appearance_at_split"] for pair in pairs]),
        "motion_by_offset": by_offset,
        "interpretation": (
            "A non-zero, persistent parent-child motion difference shows that the "
            "new interpolation sample is not a forward duplicate of its parent. Because "
            "the node MLP is shared, this is spatially distinct output from the shared "
            "field, not an independent per-node motion parameter."),
    }
    current_cfg = (args.model_path / "cfg_args").read_text()
    current_eta = re.search(r"color_residual_scale=([^,)]*)", current_cfg)
    report["configuration"] = {
        "split_only_color_residual_scale": (
            None if current_eta is None else float(current_eta.group(1))),
    }
    current_summary_path = args.model_path / "training_summary.json"
    if current_summary_path.exists():
        current_summary = json.loads(current_summary_path.read_text())
        timing = {
            "split_only_total_s": current_summary["total_training_duration_s"],
            "split_only_fine_s": current_summary["fine_duration_s"],
            "split_only_peak_vram_mb": current_summary["peak_vram_mb"],
        }
        if args.baseline_model_path:
            baseline_summary = json.loads(
                (args.baseline_model_path / "training_summary.json").read_text())
            timing.update({
                "baseline_path": str(args.baseline_model_path),
                "baseline_total_s": baseline_summary["total_training_duration_s"],
                "baseline_fine_s": baseline_summary["fine_duration_s"],
                "baseline_peak_vram_mb": baseline_summary["peak_vram_mb"],
                "total_time_ratio": (current_summary["total_training_duration_s"] /
                                     baseline_summary["total_training_duration_s"]),
                "total_time_overhead_percent": 100 * (
                    current_summary["total_training_duration_s"] /
                    baseline_summary["total_training_duration_s"] - 1),
            })
            baseline_cfg = (args.baseline_model_path / "cfg_args").read_text()
            baseline_eta = re.search(r"color_residual_scale=([^,)]*)", baseline_cfg)
            report["configuration"].update({
                "baseline_color_residual_scale": (
                    None if baseline_eta is None else float(baseline_eta.group(1))),
                "timing_comparison_note": (
                    "The historical Standard-FPS baseline uses its recorded eta; "
                    "timing is informative but not a matched eta=0.5 quality ablation."),
            })
        report["training_time"] = timing
    (adaptive / "split_mechanism_summary.json").write_text(
        json.dumps(report, indent=2) + "\n")

    ply = PlyData.read(str(args.model_path / "point_cloud" /
                           f"iteration_{args.iteration}" / "point_cloud.ply"))
    vertex = ply["vertex"].data
    xyz = np.column_stack((vertex["x"], vertex["y"], vertex["z"]))
    centered = xyz - xyz.mean(0)
    _, _, basis = np.linalg.svd(centered[::max(1, len(centered)//50000)],
                               full_matrices=False)
    gaussian_2d = centered @ basis[:2].T
    nodes = state["node_xyz"].numpy()
    node_2d = (nodes - xyz.mean(0)) @ basis[:2].T
    node_ids = state["node_ids"].numpy()
    id_to_row = {int(node_id): row for row, node_id in enumerate(node_ids)}
    score = state["gaussian_score"].numpy()
    difficult = score >= np.quantile(score, .995)
    child_rows = np.where(node_ids >= 2000)[0]
    births = state["birth_iterations"].numpy()[child_rows]

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    stride = max(1, len(xyz) // 60000)
    ax.scatter(gaussian_2d[::stride, 0], gaussian_2d[::stride, 1], s=.2,
               c="#c8c8c8", alpha=.15, label="canonical Gaussians")
    hard_stride = max(1, int(difficult.sum()) // 10000)
    hard_xy = gaussian_2d[difficult][::hard_stride]
    ax.scatter(hard_xy[:, 0], hard_xy[:, 1], s=2, c="#d62728", alpha=.25,
               label="top 0.5% geometry difficulty")
    ax.scatter(node_2d[:2000, 0], node_2d[:2000, 1], s=3, c="#555555", alpha=.35,
               label="FPS backbone")
    scatter = ax.scatter(node_2d[child_rows, 0], node_2d[child_rows, 1], s=14,
                         c=births, cmap="viridis", label="split child")
    for pair in pairs:
        parent = id_to_row.get(pair["parent_id"])
        child = id_to_row.get(pair["child_id"])
        if parent is not None and child is not None:
            ax.plot(node_2d[[parent, child], 0], node_2d[[parent, child], 1],
                    color="#1f77b4", linewidth=.35, alpha=.35)
    fig.colorbar(scatter, ax=ax, label="birth iteration")
    ax.set(title="Adaptive split locations in canonical PCA projection",
           xlabel="PC1", ylabel="PC2", aspect="equal")
    ax.legend(loc="best", markerscale=3)
    fig.savefig(adaptive / "split_locations.png", dpi=180)
    plt.close(fig)

    motion_2500 = sorted(
        [item for item in motion if item["offset"] == 2500],
        key=lambda item: item["d_motion_mean"], reverse=True)[:4]
    if motion_2500:
        fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
        for ax, item in zip(axes.flat, motion_2500):
            parent, child = id_to_row[item["parent_id"]], id_to_row[item["child_id"]]
            center = .5 * (node_2d[parent] + node_2d[child])
            neighbor_distance = np.linalg.norm(node_2d - center, axis=1)
            radius = max(float(np.partition(neighbor_distance, min(30, len(nodes)-1))[min(30, len(nodes)-1)]),
                         10 * np.linalg.norm(node_2d[parent] - node_2d[child]))
            local_g = np.linalg.norm(gaussian_2d - center, axis=1) <= radius
            local_n = neighbor_distance <= radius
            local_hard = local_g & difficult
            local_stride = max(1, int(local_g.sum()) // 10000)
            points = gaussian_2d[local_g][::local_stride]
            ax.scatter(points[:, 0], points[:, 1], s=.5, c="#bbbbbb", alpha=.25)
            ax.scatter(gaussian_2d[local_hard, 0], gaussian_2d[local_hard, 1],
                       s=4, c="#d62728", alpha=.5)
            ax.scatter(node_2d[local_n, 0], node_2d[local_n, 1], s=8,
                       c="#666666", alpha=.5)
            ax.plot(node_2d[[parent, child], 0], node_2d[[parent, child], 1],
                    color="#0066cc", linewidth=2)
            ax.scatter(*node_2d[parent], s=50, marker="o", c="#ff7f0e", label="parent")
            ax.scatter(*node_2d[child], s=65, marker="*", c="#2ca02c", label="child")
            ax.set(title=(f"child {item['child_id']} @ {item['birth_iteration']}\n"
                          f"d_pos={item['d_pos']:.4g}, motion@2500={item['d_motion_mean']:.4g}"),
                   aspect="equal")
        for ax in axes.flat[len(motion_2500):]:
            ax.axis("off")
        axes.flat[0].legend(loc="best")
        fig.savefig(adaptive / "split_regions_top_motion.png", dpi=180)
        plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
