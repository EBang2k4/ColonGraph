#!/usr/bin/env python3
"""Aggregate three paired K0/A0 fixed-node seeds and apply A0 gates."""

import json
from pathlib import Path
import statistics


SCENE = "c1_sigmoid2_t4_v4"
ROOT = Path("output/arap_fixed_phase")


def load_branch(tag):
    path = Path("output") / tag / SCENE
    metrics = json.loads((path / "results.json").read_text())["ours_30000"]
    evaluation = json.loads((path / "arap_evaluation.json").read_text())
    timing = json.loads((path / "training_summary.json").read_text())
    benchmark = json.loads((path / "adaptive_render_benchmark.json").read_text())
    return {
        **metrics,
        "nodes": evaluation["geometry_nodes"],
        "motion_p95": evaluation["backbone_motion_magnitude"]["p95"],
        "stretch_median": evaluation["backbone_stretch"]["median"],
        "stretch_p95": evaluation["backbone_stretch"]["p95"],
        "stretch_max": evaluation["backbone_stretch"]["max"],
        "train_minutes": timing["total_training_duration_s"] / 60,
        "peak_vram_mb": timing["peak_vram_mb"],
        "fps": benchmark["render_fps"],
        "audit_pass": evaluation["pass"],
    }


def aggregate(rows, key):
    values = [row[key] for row in rows]
    return {"mean": statistics.mean(values), "std": statistics.stdev(values)}


def main():
    selection = json.loads((ROOT / "stability_selection.json").read_text())
    weight_text = selection["selected_weight_text"].replace(".", "p")
    k0, a0, pairs = [], [], []
    for seed in range(3):
        k = load_branch(f"arap_k0_seed{seed}")
        a = load_branch(f"arap_a0_w{weight_text}_seed{seed}")
        k0.append(k)
        a0.append(a)
        pairs.append({"seed": seed, **{
            key: a[key] - k[key]
            for key in ("PSNR", "SSIM", "LPIPS", "DEPTH_MSE", "motion_p95",
                        "stretch_median", "stretch_p95", "stretch_max",
                        "train_minutes", "fps", "peak_vram_mb")
        }})
    metric_keys = ("PSNR", "SSIM", "LPIPS", "DEPTH_MSE", "motion_p95",
                   "stretch_median", "stretch_p95", "stretch_max",
                   "train_minutes", "fps", "peak_vram_mb")
    summary = {
        branch: {key: aggregate(rows, key) for key in metric_keys}
        for branch, rows in (("K0", k0), ("A0", a0))
    }
    summary["paired_A0_minus_K0"] = {
        key: aggregate(pairs, key) for key in metric_keys
    }
    gates_by_seed = []
    for seed, (k, a) in enumerate(zip(k0, a0)):
        checks = {
            "audits_pass": k["audit_pass"] and a["audit_pass"],
            "motion_p95_at_least_80pct_K0": a["motion_p95"] >= 0.8 * k["motion_p95"],
            "PSNR_drop_at_most_0p1dB": a["PSNR"] >= k["PSNR"] - 0.1,
            "Depth_MSE_degradation_at_most_5pct": a["DEPTH_MSE"] <= 1.05 * k["DEPTH_MSE"],
            "stretch_p95_below_predefined_0p75": a["stretch_p95"] < 0.75,
            "stretch_max_below_predefined_3": a["stretch_max"] < 3.0,
        }
        gates_by_seed.append({"seed": seed, "checks": checks,
                              "pass": all(checks.values())})
    report = {
        "pass": all(item["pass"] for item in gates_by_seed),
        "arap_weight": selection["selected_weight"],
        "seeds": [0, 1, 2],
        "K0": k0, "A0": a0,
        "paired_A0_minus_K0": pairs,
        "summary_mean_std": summary,
        "predefined_A0_gates": gates_by_seed,
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "fixed_seed_report.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# Revised ARAP — fixed 2,000-node phase", "",
        f"Selected lambda_ARAP: `{selection['selected_weight_text']}` (gradient-ratio protocol).", "",
        "| Branch | PSNR | SSIM | LPIPS | Depth MSE | motion p95 | stretch p95 | Train min | FPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for branch in ("K0", "A0"):
        values = summary[branch]
        cell = lambda key, digits=6: f"{values[key]['mean']:.{digits}f} ± {values[key]['std']:.{digits}f}"
        lines.append(f"| {branch} | {cell('PSNR')} | {cell('SSIM')} | {cell('LPIPS')} | "
                     f"{cell('DEPTH_MSE', 8)} | {cell('motion_p95')} | "
                     f"{cell('stretch_p95')} | {cell('train_minutes', 2)} | {cell('fps', 2)} |")
    lines.extend(["", f"A0 stability gates: **{'PASS' if report['pass'] else 'FAIL'}**.", "",
                  "Adaptive A1 must not start unless this report passes and is reviewed."])
    (ROOT / "fixed_seed_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit("A0 failed one or more pre-defined stability gates")


if __name__ == "__main__":
    main()
