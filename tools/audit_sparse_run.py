"""Persist baseline evidence and check each completed experimental run."""
import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
import torch


def read_config(path):
    expression = ast.parse(path.read_text(), mode="eval").body
    return {x.arg: ast.literal_eval(x.value) for x in expression.keywords}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def audit(root, experiment):
    out = root / "output" / experiment
    cfg = read_config(out / "cfg_args")
    cfg.setdefault("node_position_normalization", "none")
    cfg.setdefault("node_initialization_reference", "")
    position_mode = cfg.get("node_position_mode", "direct")
    scene = Path(cfg["source_path"]).name
    baseline = root / "output/test" / f"{scene}_knn0.01_depth0.25_dcol0.005"
    base_cfg = read_config(baseline / "cfg_args")
    keys = ["coarse_iterations", "depth_weight", "dcol_weight", "col_smooth_weight", "tv_weight",
            "position_lr_init", "deformation_lr_init", "grid_lr_init", "densify_from_iter", "pruning_from_iter",
            "densify_from_iter_fine", "pruning_from_iter_fine", "eval", "depth_mode"]
    assert all(cfg[key] == base_cfg[key] for key in keys)
    data = root / "data/coloncrafter" / scene
    images, depths = sorted((data / "color").glob("*.png")), sorted((data / "depth").glob("*.npy"))
    poses = (data / "pose.txt").read_text().splitlines()
    assert len(images) == len(depths) == len(poses)
    assert [p.stem for p in images] == [p.stem for p in depths]
    snapshot = out / "point_cloud" / f"iteration_{cfg['iterations']}"
    state = torch.load(snapshot / "deformation_nodes.pth", map_location="cpu", weights_only=True)
    assert state["config"].get("node_position_mode", "direct") == position_mode
    assert state["config"].get("node_position_normalization", "none") == cfg["node_position_normalization"]
    if cfg["node_position_normalization"] != "none":
        assert state["mlp"]["position_scale"] > 0
    graph = state["graph"]
    w, idx = graph["gaussian_node_weights"], graph["gaussian_node_indices"]
    num_nodes = cfg["num_nodes"]
    assert cfg["node_k"] == 4
    assert tuple(graph["node_xyz"].shape) == (num_nodes, 3)
    assert tuple(w.shape) == tuple(idx.shape) and w.shape[1] == 4
    assert idx.dtype == torch.long and idx.min() >= 0 and idx.max() < num_nodes
    assert torch.isfinite(w).all() and (w >= 0).all()
    torch.testing.assert_close(w.sum(1), torch.ones(len(w)))
    assert all(torch.isfinite(x).all() for x in state["mlp"].values())
    initial_path = out / "point_cloud/iteration_1"
    initial = torch.load(initial_path / "deformation_nodes.pth", map_location="cpu", weights_only=True)
    if cfg["node_position_normalization"] != "none":
        assert all(torch.equal(state["mlp"][key], initial["mlp"][key])
                   for key in ("position_center", "position_scale")), "Input normalization changed during training"
    assert all(torch.equal(graph[key], initial["graph"][key]) for key in graph), "Graph changed during fine training"
    assert any(not torch.equal(value, initial["mlp"][key]) for key, value in state["mlp"].items()), "Node MLP did not update"
    before = torch.load(initial_path / "deformation.pth", map_location="cpu", weights_only=True)
    after = torch.load(snapshot / "deformation.pth", map_location="cpu", weights_only=True)
    geometry_keys = [key for key in before if any(head in key for head in ("pos_deform", "scales_deform", "rotations_deform"))]
    assert all(torch.equal(before[key], after[key]) for key in geometry_keys), "Original geometry heads changed"
    assert any(not torch.equal(before[key], after[key]) for key in before if "color_deform" in key), "Color branch did not update"
    from plyfile import PlyData
    first_xyz = PlyData.read(initial_path / "point_cloud.ply")["vertex"]["x"]
    final_xyz = PlyData.read(snapshot / "point_cloud.ply")["vertex"]["x"]
    assert (first_xyz != final_xyz).any(), "Canonical centers did not update"
    log_dir = root / "logs" / experiment
    log = (log_dir / "train.log").read_text()
    assert "100/100" in log and f"{cfg['iterations']}/{cfg['iterations']}" in log and "Training complete." in log
    assert log.index("100/100") < log.index("Initialized deformation nodes AFTER coarse")
    assert not re.search(r"Loss NaN/Inf|Traceback|RuntimeError", log)
    for stage in ("train", "render", "metrics", "render_lookat", "render_time_shift", "profile"):
        assert "Exit status: 0" in (log_dir / f"{stage}.log").read_text(), stage
    if position_mode == "local_frame":
        assert "Exit status: 0" in (log_dir / "profile_b1.log").read_text()
    profile_rows = (out / "deformation_profile.csv").read_text().splitlines()
    assert len(profile_rows) > 1
    data_hash = hashlib.sha256()
    for path in images + depths + [data / "pose.txt", data / "points3d.ply"]:
        data_hash.update(str(path.relative_to(data)).encode())
        data_hash.update(digest(path).encode())
    report = {
        "experiment": experiment, "baseline_path": str(baseline),
        "baseline_metrics": json.loads((baseline / "results.json").read_text()),
        "node_metrics": json.loads((out / "results.json").read_text()),
        "baseline_sha256": {str(p.relative_to(baseline)): digest(p) for p in
            [baseline / "cfg_args", baseline / "results.json"] + sorted((baseline / "point_cloud/iteration_30000").glob("*"))},
        "data_current_sha256": data_hash.hexdigest(), "frames": len(images),
        "train_frames": sum((i-1) % 8 != 0 for i in range(len(images))),
        "test_frames": sum((i-1) % 8 == 0 for i in range(len(images))),
        "N": len(w), "M": num_nodes, "K": 4, "knn_weight": cfg["knn_weight"],
        "node_position_mode": position_mode,
        "node_position_normalization": cfg["node_position_normalization"],
        "training_config_matches_baseline": keys,
        "full_schedule": cfg["iterations"] == base_cfg["iterations"] == 30000,
        "deformation_profile_rows": len(profile_rows) - 1,
        "structural_checks_passed": True,
        "fixed_graph_identical_to_iteration_1": True,
        "node_mlp_color_branch_canonical_centers_updated": True,
        "original_geometry_heads_unchanged": True,
        "benchmark": json.loads((out / "deformation_benchmark.json").read_text()),
        "note": "Smoke metrics do not establish final quality. No historical input hash exists for B0; current data identity is recorded here."}
    b0 = report["baseline_metrics"]["ours_30000"]
    b1 = report["node_metrics"][f"ours_{cfg['iterations']}"]
    report["metric_delta_node_minus_baseline"] = {key: b1[key] - b0[key] for key in b0}
    if position_mode == "local_frame":
        direct_path = root / "output/baseline1_nodes2000" / scene
        direct_cfg = read_config(direct_path / "cfg_args")
        direct_cfg.setdefault("node_position_mode", "direct")
        direct_cfg.setdefault("node_position_normalization", "none")
        direct_cfg.setdefault("node_initialization_reference", "")
        differences = {key: [direct_cfg.get(key), cfg.get(key)]
                       for key in direct_cfg.keys() | cfg.keys() if direct_cfg.get(key) != cfg.get(key)}
        allowed = {"node_position_mode", "expname", "model_path"}
        if cfg["iterations"] != 30000:  # Explicitly marked smoke experiment.
            allowed |= {"iterations", "save_iterations"}
        assert not set(differences) - allowed, differences
        assert cfg["num_nodes"] == 2000 and cfg["knn_weight"] == .01
        direct_checks = json.loads((direct_path / "step_checks.json").read_text())
        assert data_hash.hexdigest() == direct_checks["data_current_sha256"], "B1 input dataset changed"
        direct_metrics = json.loads((direct_path / "results.json").read_text())["ours_30000"]
        local_benchmark = json.loads((out / "b1_local_benchmark.json").read_text())
        assert local_benchmark["baseline"]["node_position_mode"] == "direct"
        assert local_benchmark["node"]["node_position_mode"] == "local_frame"
        pair = {"B1_path": str(direct_path), "B1_metrics": direct_metrics, "B1_Local_metrics": b1,
                "config_differences": differences, "input_hash_identical": True,
                "full_schedule": report["full_schedule"],
                "delta_local_minus_direct": {key: b1[key] - direct_metrics[key] for key in b1},
                "depth_mse_reduction_percent": 100 * (1 - b1["DEPTH_MSE"] / direct_metrics["DEPTH_MSE"]),
                "benchmark": local_benchmark}
        report["b1_local_comparison"] = pair
        (out / "b1_local_comparison.json").write_text(json.dumps(pair, indent=2) + "\n")
        comparison = [f"# B1 vs B1-Local — {scene}", "",
            f"Nodes=2000, K=4, Gaussian KNN loss=0.01. Full comparable schedule: {report['full_schedule']}.", "",
            "| Metric | B1 direct (30k) | B1-Local | Local − direct |", "|---|---:|---:|---:|"]
        for key in direct_metrics:
            comparison.append(f"| {key} | {direct_metrics[key]:.6f} | {b1[key]:.6f} | {b1[key]-direct_metrics[key]:+.6f} |")
        direct_ms = local_benchmark["baseline"]["training_forward_deformation_ms"]["median_ms"]
        local_ms = local_benchmark["node"]["training_forward_deformation_ms"]["median_ms"]
        comparison += [f"| deformation_forward_ms | {direct_ms:.4f} | {local_ms:.4f} | {local_ms-direct_ms:+.4f} |",
            "", f"Depth MSE reduction relative to B1: {pair['depth_mse_reduction_percent']:.2f}%.", "",
            "Config audit: only position formulation and output location differ for full runs; input hash identical.",
            "Smoke runs are not evidence for final quality. No ARAP or motion-aware sampling was added."]
        (out / "b1_local_comparison.md").write_text("\n".join(comparison) + "\n")
    (out / "step_checks.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [f"# Sparse nodes — {scene}", "", f"Fine iterations: {cfg['iterations']}; full schedule: {report['full_schedule']}.",
             f"Nodes: {num_nodes}; node K: 4; Gaussian KNN loss weight: {cfg['knn_weight']}.",
             "", "| Metric | Original (30k) | Node | Node − original |", "|---|---:|---:|---:|"]
    for key in b0:
        lines.append(f"| {key} | {b0[key]:.6f} | {b1[key]:.6f} | {b1[key] - b0[key]:+.6f} |")
    lines += ["", f"Deformation forward speedup: {report['benchmark']['forward_speedup']:.3f}x.", "",
              "Structural checks passed: fixed 4-NN graph, normalized weights, finite MLP, canonical/MLP/color updates, unchanged old geometry heads, all stages exit 0.",
              "", "Quality tolerance is not specified in spec.md; inspect metric deltas and rendered videos. Smoke runs cannot establish final quality."]
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    args = parser.parse_args()
    audit(Path(__file__).resolve().parents[1], args.experiment)
