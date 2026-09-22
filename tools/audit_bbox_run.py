"""Paired B1 raw/bbox audit; initial-only mode is safe while training runs."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from plyfile import PlyData
from audit_sparse_run import read_config


def xyz(path):
    vertex = PlyData.read(path)["vertex"]
    return torch.from_numpy(np.stack([vertex[k] for k in ("x", "y", "z")], 1).copy())


def audit(root, experiment, initial_only=False):
    out = root / "output" / experiment
    cfg = read_config(out / "cfg_args")
    raw = root / "output/baseline1_nodes2000" / Path(cfg["source_path"]).name
    raw_cfg = read_config(raw / "cfg_args")
    for config in (cfg, raw_cfg):
        config.setdefault("node_position_mode", "direct")
        config.setdefault("node_position_normalization", "none")
        config.setdefault("node_initialization_reference", "")
    assert cfg["node_position_mode"] == raw_cfg["node_position_mode"] == "direct"
    mode = cfg["node_position_normalization"]
    assert mode in ("bbox", "raw_only", "pe_only")
    assert raw_cfg["node_position_normalization"] == "none"
    assert cfg["num_nodes"] == 2000 and cfg["node_k"] == 4 and cfg["knn_weight"] == .01
    differences = {k: [raw_cfg.get(k), cfg.get(k)] for k in raw_cfg.keys() | cfg.keys()
                   if raw_cfg.get(k) != cfg.get(k)}
    allowed = {"node_position_normalization", "node_initialization_reference", "expname", "model_path"}
    if cfg["node_initialization_reference"]:
        assert (root / cfg["node_initialization_reference"]).resolve() == (raw / "point_cloud/iteration_1").resolve()
    if cfg["iterations"] != 30000:
        allowed |= {"iterations", "save_iterations"}
    assert not set(differences) - allowed, differences
    before_path = out / "point_cloud/iteration_1"
    raw_path = raw / "point_cloud/iteration_1"
    first = torch.load(before_path / "deformation_nodes.pth", map_location="cpu", weights_only=True)
    reference = torch.load(raw_path / "deformation_nodes.pth", map_location="cpu", weights_only=True)
    params = [k for k in first["mlp"] if k.startswith("net.")]
    assert all(torch.equal(first["mlp"][k], reference["mlp"][k]) for k in params), "Different initial MLP weights"
    first_xyz, raw_xyz = xyz(before_path / "point_cloud.ply"), xyz(raw_path / "point_cloud.ply")
    lo, hi = first_xyz.amin(0), first_xyz.amax(0)
    center, extent = (lo + hi) / 2, ((hi - lo).max() / 2).clamp_min(1e-8)
    torch.testing.assert_close(first["mlp"]["position_center"], center, rtol=0, atol=0)
    torch.testing.assert_close(first["mlp"]["position_scale"], extent, rtol=0, atol=0)
    graph_comparison = {}
    for k, value in first["graph"].items():
        ref = reference["graph"][k]
        equal = torch.equal(value, ref)
        graph_comparison[k] = {"bitwise_equal": equal, "max_abs_difference": (value-ref).abs().max().item()}
        # Re-running GPU coarse may introduce small floating-point differences.
        # Report these explicitly; never label close weights as bit-identical.
        if value.is_floating_point():
            torch.testing.assert_close(value, ref, rtol=1e-5, atol=1e-4)
        else:
            assert equal, f"Association indices changed: {k}"
        if cfg["node_initialization_reference"]:
            assert equal, f"Reference graph must be bit-identical: {k}"
    torch.testing.assert_close(first_xyz, raw_xyz, rtol=1e-5, atol=1e-4)
    if cfg["node_initialization_reference"]:
        assert torch.equal(first_xyz, raw_xyz), "Reference canonical xyz changed"
        assert np.array_equal(PlyData.read(before_path / "point_cloud.ply")["vertex"].data,
                              PlyData.read(raw_path / "point_cloud.ply")["vertex"].data), "Reference Gaussian attributes changed"
        for filename in ("deformation.pth", "deformation_table.pth"):
            a = torch.load(before_path / filename, map_location="cpu", weights_only=True)
            b = torch.load(raw_path / filename, map_location="cpu", weights_only=True)
            assert all(torch.equal(v, b[k]) for k, v in a.items()) if isinstance(a, dict) else torch.equal(a, b)
    result = {"experiment": experiment, "normalization_mode": mode, "raw_path": str(raw), "config_differences": differences,
              "full_schedule": cfg["iterations"] == 30000,
              "initial_mlp_parameters_bitwise_equal": True,
              "graph_comparison": graph_comparison,
              "initial_canonical_xyz_max_abs_difference": (first_xyz-raw_xyz).abs().max().item(),
              "normalization_center": center.tolist(), "normalization_half_extent": extent.item(),
              "checkpoint_buffer_names": ["position_center", "position_scale"],
              "seed_note": "Same train.py setup_seed(6666), followed by safe_state reseeding Python/NumPy/Torch to 0. Initial MLP parameters independently verified equal.",
              "initial_checks_passed": True}
    result["reference_all_gaussian_and_deformation_attributes_bitwise_equal"] = bool(cfg["node_initialization_reference"])
    if not initial_only:
        checks = json.loads((out / "step_checks.json").read_text())
        raw_checks = json.loads((raw / "step_checks.json").read_text())
        assert checks["data_current_sha256"] == raw_checks["data_current_sha256"]
        snapshot = out / f"point_cloud/iteration_{cfg['iterations']}"
        final = torch.load(snapshot / "deformation_nodes.pth", map_location="cpu", weights_only=True)
        assert all(torch.equal(v, final["graph"][k]) for k, v in first["graph"].items())
        assert all(torch.equal(first["mlp"][k], final["mlp"][k]) for k in ("position_center", "position_scale"))
        bench = json.loads((out / "b1_bbox_benchmark.json").read_text())
        assert bench["baseline"]["node_position_normalization"] == "none"
        assert bench["node"]["node_position_normalization"] == mode
        raw_metrics = json.loads((raw / "results.json").read_text())["ours_30000"]
        metrics = json.loads((out / "results.json").read_text())[f"ours_{cfg['iterations']}"]
        for stage in ("profile_bbox",):
            assert "Exit status: 0" in (root / "logs" / experiment / (stage + ".log")).read_text()
        result.update({"input_hash_identical": True, "graph_and_normalization_fixed": True,
                       "raw_metrics": raw_metrics, "bbox_metrics": metrics, "benchmark": bench,
                       "delta_bbox_minus_raw": {k: metrics[k]-raw_metrics[k] for k in metrics},
                       "depth_mse_reduction_percent": 100*(1-metrics["DEPTH_MSE"]/raw_metrics["DEPTH_MSE"]),
                       "quality_note": "No automatic default change: inspect Depth MSE and RGB tradeoff; smoke is not a quality conclusion."})
        lines = [f"# B1-raw vs B1-{mode}", "",
                 f"Full 30k schedule: {result['full_schedule']}. M=2000, K=4, Gaussian KNN loss=0.01, direct blending.", "",
                 f"| Metric | B1 raw | B1 {mode} | Variant - raw |", "|---|---:|---:|---:|"]
        for key in raw_metrics:
            lines.append(f"| {key} | {raw_metrics[key]:.6f} | {metrics[key]:.6f} | {metrics[key]-raw_metrics[key]:+.6f} |")
        a = bench["baseline"]["training_forward_deformation_ms"]["median_ms"]
        b = bench["node"]["training_forward_deformation_ms"]["median_ms"]
        lines += [f"| deformation_forward_ms | {a:.4f} | {b:.4f} | {b-a:+.4f} |", "",
                  f"Depth MSE reduction: {result['depth_mse_reduction_percent']:.2f}%.", "",
                  "Input hash matched. Same initial MLP weights. Graph/normalization fixed during fine. See JSON for any GPU coarse floating-point differences.", "",
                  "No ARAP, resampling, teacher warm-start or canonical freeze. Default remains none until quality is reviewed."]
        (out / "b1_bbox_comparison.md").write_text("\n".join(lines) + "\n")
    name = "bbox_initial_checks.json" if initial_only else "b1_bbox_comparison.json"
    (out / name).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    parser.add_argument("--initial-only", action="store_true")
    args = parser.parse_args()
    audit(Path(__file__).resolve().parents[1], args.experiment, args.initial_only)
