#!/usr/bin/env python3
"""Audit matched fixed A/B branches against the common KNN-99 protocol."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from tools.audit_sparse_run import read_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--branch", choices=("a", "b"), required=True)
    args = parser.parse_args()
    cfg = read_config(args.model_path / "cfg_args")
    snapshot = torch.load(
        args.model_path / "point_cloud" / "iteration_30000" / "deformation_nodes.pth",
        map_location="cpu", weights_only=True)
    count = len(snapshot["graph"]["node_xyz"])
    expected = 2000 if args.branch == "a" else 2400
    checks = {
        "fine_30k": cfg["iterations"] == 30000,
        "coarse_100": cfg["coarse_iterations"] == 100,
        "adaptive_off": cfg.get("adaptive_node_mode", "none") == "none",
        "matched_knn_reference": cfg.get("gaussian_knn_reference", "") ==
            "output/adaptive_deformation_matched_knn99.pth",
        "node_count": count == expected,
        "k4": snapshot["graph"]["gaussian_node_indices"].shape[1] == 4,
        "b1_bbox": cfg["node_position_normalization"] == "bbox",
        "r6": cfg["color_model"] == "node_residual_mlp" and
            cfg["node_appearance_dim"] == 8 and cfg["color_residual_scale"] == .3,
        "fixed_gaussians": len(snapshot["graph"]["gaussian_node_indices"]) == 364500,
    }
    if args.branch == "a":
        checks["standard_fps"] = cfg.get("node_sampling", "fps") == "fps"
    else:
        checks["fps_plus_importance"] = (
            cfg["node_sampling"] == "fps_plus_importance" and
            cfg["node_importance_extra_nodes"] == 400 and
            cfg["node_sampling_gradient_source"] == "total")
    report = {"pass": all(checks.values()), "branch": args.branch,
              "checks": checks, "nodes": count}
    (args.model_path / "matched_fixed_audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
