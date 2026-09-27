#!/usr/bin/env python3
"""Create the final split-only versus prune-late comparison."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCENE = "c1_sigmoid2_t4_v4"
RUNS = {
    "C split-only diagnostic": "adaptive_deformation_c_split_mechanism",
    "D2 concurrent conservative prune": "adaptive_deformation_d2_conservative_prune",
    "D3 split-first, prune-late": "adaptive_deformation_d3_prune_late",
}


def load(name):
    path = ROOT / "output" / name / SCENE
    result = json.loads((path / "results.json").read_text())["ours_30000"]
    summary = json.loads((path / "training_summary.json").read_text())
    bench = json.loads((path / "adaptive_render_benchmark.json").read_text())
    events = [json.loads(line) for line in
              (path / "adaptive_nodes" / "adaptive_events.jsonl").read_text().splitlines()]
    return path, result, summary, bench, events


def main():
    data = {label: load(name) for label, name in RUNS.items()}
    c = data["C split-only diagnostic"][1]
    d3 = data["D3 split-first, prune-late"][1]
    d3_events = data["D3 split-first, prune-late"][4]
    c_nodes = data["C split-only diagnostic"][4][-1]["nodes_after"]
    d3_nodes = d3_events[-1]["nodes_after"]
    depth_ratio = d3["DEPTH_MSE"] / c["DEPTH_MSE"]
    success = (d3_nodes < c_nodes and depth_ratio <= 1.05 and
               d3["PSNR"] >= c["PSNR"] - .05)
    lines = [
        "# Split mechanism and final prune-late experiment", "",
        "| Branch | Nodes | PSNR | SSIM | LPIPS | Depth MSE | Train min | FPS | VRAM MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, (_, result, summary, bench, events) in data.items():
        lines.append(
            f"| {label} | {events[-1]['nodes_after']} | {result['PSNR']:.6f} | "
            f"{result['SSIM']:.6f} | {result['LPIPS']:.6f} | "
            f"{result['DEPTH_MSE']:.8f} | {summary['total_training_duration_s']/60:.2f} | "
            f"{bench['render_fps']:.2f} | {summary['peak_vram_mb']:.1f} |")
    lines += [
        "", "## D3 schedule audit", "",
        "- 0–5k: warm-up.",
        "- 5k–20k: split-only using the unchanged C split rule.",
        "- 20k–22.5k: stabilization.",
        "- 22.5k–25k: child-only conservative prune.",
        "- 25k–30k: frozen graph.", "",
        f"D3 split/prune totals: `{sum(e.get('splits',0) for e in d3_events)}` / "
        f"`{sum(e.get('prunes',0) for e in d3_events)}`.",
        f"Depth ratio D3/C: `{depth_ratio:.6f}`; node reduction: `{c_nodes-d3_nodes}`.",
        f"Final success screen (Depth within 5%, PSNR loss <=0.05 dB, fewer nodes): "
        f"`{'PASS' if success else 'FAIL'}`.", "",
        "Mechanism details and figures are in C's `adaptive_nodes/` directory.",
    ]
    output = ROOT / "output" / "adaptive_deformation_prune_late_comparison" / SCENE
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.md").write_text("\n".join(lines) + "\n")
    (output / "comparison.json").write_text(json.dumps({
        "success": success, "depth_ratio_d3_over_c": depth_ratio,
        "node_reduction": c_nodes-d3_nodes,
    }, indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
