#!/usr/bin/env python3
"""Plan and gate revised multi-pair ARAP calibration without 30k runs."""

import argparse
import json
import math
from pathlib import Path


ROOT = Path("output/arap_revised_phase")
SCENE = "c1_sigmoid2_t4_v4"


def weight_text(value):
    return format(float(value), ".8g")


def stability_path(weight):
    tag = weight_text(weight).replace(".", "p")
    return Path("output") / f"arap_revised_stability_w{tag}_seed0" / SCENE


def propose():
    path = Path("output/arap_revised_calibration_seed0") / SCENE
    audit = json.loads((path / "arap_multipair_gradient_audit.json").read_text())
    if not audit["pass"]:
        raise RuntimeError("Calibration gradients are not meaningful")
    by_iteration = {row["iteration"]: row for row in audit["rows"]}
    if set(by_iteration) != {2000, 5000, 10000}:
        raise RuntimeError("Expected multi-pair audits at 2k/5k/10k")
    ratio = by_iteration[5000]["raw_gradient_ratio"]
    if not math.isfinite(ratio) or ratio <= 0:
        raise RuntimeError("Invalid 5k multi-pair gradient ratio")
    lambda0 = 0.1 / ratio
    report = {
        "target_ratio": 0.1,
        "two_sided_gate": [0.05, 0.20],
        "pair_count": by_iteration[5000]["pair_count"],
        "calibration_audit": audit,
        "lambda0": lambda0,
        "lambda0_text": weight_text(lambda0),
        "candidates": [
            {"factor": factor, "weight": factor * lambda0,
             "weight_text": weight_text(factor * lambda0)}
            for factor in (0.5, 1.0, 2.0)
        ],
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "calibration_plan.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def select():
    plan = json.loads((ROOT / "calibration_plan.json").read_text())
    baseline_path = Path("output/arap_revised_calibration_seed0") / SCENE
    baseline = json.loads((baseline_path / "arap_evaluation.json").read_text())
    baseline_motion = baseline["backbone_motion_magnitude"]["p95"]
    candidates = []
    for candidate in plan["candidates"]:
        path = stability_path(candidate["weight"])
        audit = json.loads((path / "arap_multipair_gradient_audit.json").read_text())
        evaluation = json.loads((path / "arap_evaluation.json").read_text())
        ratios = [row["weighted_gradient_ratio"] for row in audit["rows"]]
        motion_ratio = (evaluation["backbone_motion_magnitude"]["p95"] /
                        max(baseline_motion, 1e-20))
        checks = {
            "gradient_meaningful_2k_5k_10k": audit["pass"],
            "all_gradient_ratios_in_0p05_to_0p20": all(
                0.05 <= ratio <= 0.20 for ratio in ratios),
            "motion_p95_preserved_0p8_to_1p25": 0.8 <= motion_ratio <= 1.25,
            "stretch_p95_below_predefined_0p75": (
                evaluation["backbone_stretch"]["p95"] < 0.75),
            "stretch_max_below_predefined_3": (
                evaluation["backbone_stretch"]["max"] < 3.0),
            "reload_pass": evaluation["pass"],
        }
        score = sum(abs(math.log(max(ratio, 1e-20) / 0.1)) for ratio in ratios)
        candidates.append({**candidate, "path": str(path), "checks": checks,
                           "pass": all(checks.values()), "motion_ratio": motion_ratio,
                           "weighted_gradient_ratios": ratios,
                           "target_distance": score,
                           "gradient_audit": audit, "evaluation": evaluation})
    passed = [candidate for candidate in candidates if candidate["pass"]]
    selected = min(passed, key=lambda item: item["target_distance"]) if passed else None
    report = {
        "pass": selected is not None,
        "plan": plan,
        "baseline_evaluation": baseline,
        "candidates": candidates,
        "selected_weight": None if selected is None else selected["weight"],
        "selected_weight_text": None if selected is None else selected["weight_text"],
        "explicit_stop_before_30k": True,
    }
    (ROOT / "stability_selection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if selected is None:
        raise SystemExit("No revised candidate passed all four stability gates")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("propose", "select"))
    args = parser.parse_args()
    propose() if args.mode == "propose" else select()


if __name__ == "__main__":
    main()
