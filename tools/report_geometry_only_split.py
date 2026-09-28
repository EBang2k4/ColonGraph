#!/usr/bin/env python3
"""Report the matched geometry-only adaptive-node test at eta=0.5."""
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCENE = "c1_sigmoid2_t4_v4"
RUNS = {
    "Standard FPS-2000": "adaptive_deformation_a_fps2000_eta05",
    "C shared geometry/appearance graph": "adaptive_deformation_c_split_eta05",
    "C fixed appearance graph": "adaptive_deformation_c_geometry_only_eta05",
}


def load(tag):
    path = ROOT / "output" / tag / SCENE
    result = json.loads((path / "results.json").read_text())["ours_30000"]
    summary = json.loads((path / "training_summary.json").read_text())
    benchmark = json.loads((path / "adaptive_render_benchmark.json").read_text())
    nodes = torch.load(
        path / "point_cloud/iteration_30000/deformation_nodes.pth",
        map_location="cpu", weights_only=True)
    return result, summary, benchmark, len(nodes["graph"]["node_xyz"])


def main():
    data = {name: load(tag) for name, tag in RUNS.items()}
    base = data["Standard FPS-2000"]
    fixed = data["C fixed appearance graph"]
    lines = [
        "# Does adaptive deformation-node splitting improve geometry?", "",
        "Matched eta=0.5 protocol. The fixed-appearance branch retains exactly the "
        "initial 2,000-node appearance K4 and 2,000x8D node feature table while only "
        "the geometry graph grows.", "",
        "| Branch | Geometry nodes | PSNR | SSIM | LPIPS | Depth MSE | Train min | FPS | VRAM MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, (result, summary, benchmark, nodes) in data.items():
        lines.append(
            f"| {name} | {nodes} | {result['PSNR']:.6f} | {result['SSIM']:.6f} | "
            f"{result['LPIPS']:.6f} | {result['DEPTH_MSE']:.8f} | "
            f"{summary['total_training_duration_s']/60:.2f} | "
            f"{benchmark['render_fps']:.2f} | {summary['peak_vram_mb']:.1f} |")
    br, bs, _, bn = base
    fr, fs, _, fn = fixed
    delta = {
        "geometry_nodes": fn - bn,
        "PSNR": fr["PSNR"] - br["PSNR"],
        "SSIM": fr["SSIM"] - br["SSIM"],
        "LPIPS": fr["LPIPS"] - br["LPIPS"],
        "DEPTH_MSE": fr["DEPTH_MSE"] - br["DEPTH_MSE"],
        "DEPTH_MSE_percent": 100 * (fr["DEPTH_MSE"] / br["DEPTH_MSE"] - 1),
        "training_time_percent": 100 * (
            fs["total_training_duration_s"] / bs["total_training_duration_s"] - 1),
    }
    lines += ["", "Fixed-appearance adaptive split minus Standard:", "",
              "```json", json.dumps(delta, indent=2), "```"]
    output = ROOT / "output/adaptive_deformation_geometry_only_comparison" / SCENE
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(delta, indent=2) + "\n")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
