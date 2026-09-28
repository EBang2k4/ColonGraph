#!/usr/bin/env python3
"""Derive lambda from MLP-gradient ratio and gate stability diagnostics."""

import argparse
import json
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def weight_text(value):
    return format(float(value), ".8g")


def diagnostic_path(weight):
    tag = weight_text(weight).replace(".", "p")
    return Path(f"output/arap_a0_stability_w{tag}_seed0/c1_sigmoid2_t4_v4")


def propose(output):
    monitor = Path("output/arap_calibration_monitor_seed0/c1_sigmoid2_t4_v4")
    rows = read_jsonl(monitor / "arap_diagnostics.jsonl")
    by_iteration = {row["iteration"]: row for row in rows}
    if set(by_iteration) != {2000, 5000, 10000}:
        raise RuntimeError("Calibration must contain exactly 2k/5k/10k audits")
    ratio = by_iteration[5000]["raw_gradient_ratio"]
    if not ratio > 0:
        raise RuntimeError("Invalid raw gradient ratio at 5k")
    lambda0 = 0.1 / ratio
    plan = {
        "target_gradient_ratio": 0.1,
        "monitor_gradient_audits": rows,
        "lambda0": lambda0,
        "lambda0_text": weight_text(lambda0),
        "candidates": [
            {"factor": factor, "weight": lambda0 * factor,
             "weight_text": weight_text(lambda0 * factor)}
            for factor in (0.5, 1.0, 2.0)
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2))


def select(plan_path, output):
    plan = json.loads(plan_path.read_text())
    monitor_path = Path("output/arap_calibration_monitor_seed0/c1_sigmoid2_t4_v4")
    baseline = json.loads((monitor_path / "arap_evaluation.json").read_text())
    baseline_motion = baseline["backbone_motion_magnitude"]["p95"]
    candidates = []
    for candidate in plan["candidates"]:
        path = diagnostic_path(candidate["weight"])
        rows = read_jsonl(path / "arap_diagnostics.jsonl")
        evaluation = json.loads((path / "arap_evaluation.json").read_text())
        ratios = [row["weighted_gradient_ratio"] for row in rows]
        checks = {
            "three_gradient_audits": len(rows) == 3,
            "ratio_at_5k_at_most_0p2": next(
                row["weighted_gradient_ratio"] for row in rows
                if row["iteration"] == 5000) <= 0.2,
            "ratio_above_0p2_not_prolonged": sum(ratio > 0.2 for ratio in ratios) <= 1,
            "motion_p95_at_least_80pct_monitor": (
                evaluation["backbone_motion_magnitude"]["p95"] >= 0.8 * baseline_motion),
            "stretch_p95_below_predefined_0p75": (
                evaluation["backbone_stretch"]["p95"] < 0.75),
            "stretch_max_below_predefined_3": (
                evaluation["backbone_stretch"]["max"] < 3.0),
            "reload_pass": evaluation["pass"],
        }
        candidates.append({**candidate, "path": str(path), "checks": checks,
                           "pass": all(checks.values()),
                           "gradient_audits": rows, "evaluation": evaluation})

    # Weight choice is pre-registered: prefer lambda0; fall back only to half.
    base = next(item for item in candidates if item["factor"] == 1.0)
    half = next(item for item in candidates if item["factor"] == 0.5)
    selected = base if base["pass"] else half if half["pass"] else None
    report = {"plan": plan, "candidates": candidates,
              "selected_weight": None if selected is None else selected["weight"],
              "selected_weight_text": None if selected is None else selected["weight_text"],
              "pass": selected is not None}
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if selected is None:
        raise SystemExit("No pre-registered ARAP weight passed stability gates")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("propose", "select"))
    parser.add_argument("--plan", type=Path,
                        default=Path("output/arap_fixed_phase/calibration_plan.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "propose":
        propose(args.output or args.plan)
    else:
        select(args.plan, args.output or Path(
            "output/arap_fixed_phase/stability_selection.json"))


if __name__ == "__main__":
    main()
