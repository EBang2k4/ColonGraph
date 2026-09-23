"""Compare standard FPS, total-gradient Weighted FPS, and the additive-node control."""
import argparse
import json
from pathlib import Path


def metrics(path):
    data = json.loads((path / "results.json").read_text())
    return next(iter(data.values()))


def run(scene):
    root = Path("output")
    paths = {
        "standard_fps_2000": root / "joint_b1_bbox_r6" / scene,
        "total_gradient_weighted_2000_a050": root / "joint_b1_bbox_r6_weighted_fps_a050" / scene,
        "standard_2000_plus_importance_400": root / "joint_b1_bbox_r6_fps2000_plus_importance400" / scene,
    }
    rows = []
    baseline = metrics(paths["standard_fps_2000"])
    for name, path in paths.items():
        value = metrics(path)
        rows.append({
            "name": name,
            **{key: value[key] for key in ("PSNR", "SSIM", "LPIPS", "DEPTH_MSE")},
            "delta_vs_standard": {key: value[key] - baseline[key]
                                  for key in ("PSNR", "SSIM", "LPIPS", "DEPTH_MSE")},
        })
    audit = json.loads((paths["standard_2000_plus_importance_400"] /
                        "fps_plus_importance_audit.json").read_text())
    result = {
        "scene": scene, "standard_nodes": 2000, "importance_extra_nodes": 400,
        "gradient_source": "total", "rows": rows,
        "sampling": audit["distribution"],
        "success": (rows[-1]["DEPTH_MSE"] <= baseline["DEPTH_MSE"] * 1.01 and
                    rows[-1]["LPIPS"] < baseline["LPIPS"]),
    }
    output = root / "fps_plus_importance_comparison" / scene
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Standard FPS + additive importance nodes",
        "",
        "Fixed B1-bbox + R6 configuration. The hybrid retains all 2,000 standard nodes and adds 400 total-gradient nodes.",
        "",
        "| sampler | PSNR | SSIM | LPIPS | Depth MSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['name']} | {row['PSNR']:.6f} | {row['SSIM']:.6f} | "
                     f"{row['LPIPS']:.6f} | {row['DEPTH_MSE']:.8f} |")
    lines += ["", f"Success criterion (Depth within 1%, LPIPS lower): **{result['success']}**", ""]
    (output / "report.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="c1_sigmoid2_t4_v4")
    args = parser.parse_args()
    run(args.scene)
