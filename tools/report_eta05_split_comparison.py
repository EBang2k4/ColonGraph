#!/usr/bin/env python3
"""Matched Standard-FPS versus split-only comparison at eta=0.5."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCENE = "c1_sigmoid2_t4_v4"
RUNS = {
    "Standard FPS-2000, eta=0.5": "adaptive_deformation_a_fps2000_eta05",
    "C split-only, eta=0.5": "adaptive_deformation_c_split_eta05",
}


def load(tag):
    path = ROOT / "output" / tag / SCENE
    result = json.loads((path / "results.json").read_text())["ours_30000"]
    summary = json.loads((path / "training_summary.json").read_text())
    benchmark = json.loads((path / "adaptive_render_benchmark.json").read_text())
    snapshot = __import__("torch").load(
        path / "point_cloud" / "iteration_30000" / "deformation_nodes.pth",
        map_location="cpu", weights_only=True)
    return result, summary, benchmark, len(snapshot["graph"]["node_xyz"])


def main():
    data = {name: load(tag) for name, tag in RUNS.items()}
    base = data["Standard FPS-2000, eta=0.5"]
    split = data["C split-only, eta=0.5"]
    lines = [
        "# Matched eta=0.5: Standard FPS versus adaptive split-only", "",
        "Both runs use the same saved KNN-99 reference, seed, 100 coarse + 30k fine, "
        "B1-bbox geometry, R6 architecture/losses, and eta=0.5. The experimental "
        "variable is fixed 2,000 nodes versus split-only growth.", "",
        "| Branch | Nodes | PSNR | SSIM | LPIPS | Depth MSE | Train min | FPS | VRAM MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, (result, summary, benchmark, nodes) in data.items():
        lines.append(
            f"| {name} | {nodes} | {result['PSNR']:.6f} | {result['SSIM']:.6f} | "
            f"{result['LPIPS']:.6f} | {result['DEPTH_MSE']:.8f} | "
            f"{summary['total_training_duration_s']/60:.2f} | "
            f"{benchmark['render_fps']:.2f} | {summary['peak_vram_mb']:.1f} |")
    br, bs, _, bn = base
    sr, ss, _, sn = split
    delta = {
        "nodes": sn - bn,
        "PSNR": sr["PSNR"] - br["PSNR"],
        "SSIM": sr["SSIM"] - br["SSIM"],
        "LPIPS": sr["LPIPS"] - br["LPIPS"],
        "DEPTH_MSE": sr["DEPTH_MSE"] - br["DEPTH_MSE"],
        "training_time_percent": 100 * (
            ss["total_training_duration_s"] / bs["total_training_duration_s"] - 1),
    }
    lines += ["", "Split-only minus Standard:", "", "```json",
              json.dumps(delta, indent=2), "```"]
    output = ROOT / "output" / "adaptive_deformation_eta05_comparison" / SCENE
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(delta, indent=2) + "\n")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
