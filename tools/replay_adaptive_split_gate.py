#!/usr/bin/env python3
"""Replay S1 from the passed S0 checkpoint without repeating its 5k warm-up."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lpips
import torch

from arguments import OptimizationParams
from gaussian_renderer import render
from tools.experiment_common import load_scene
from utils.adaptive_deformation_graph import AdaptiveNodeConfig, AdaptiveNodeManager


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=5000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source_checkpoint = args.source / "point_cloud" / f"iteration_{args.iteration}"
    source_files = sorted(path for path in source_checkpoint.iterdir() if path.is_file())
    source_hashes = {path.name: digest(path) for path in source_files}

    model, scene, pipe, background, cfg = load_scene(args.source, args.iteration)
    optimization = OptimizationParams(argparse.ArgumentParser()).extract(cfg)
    model.training_setup(optimization)
    adaptive_config = AdaptiveNodeConfig(
        mode="split_once", start=args.iteration, end=args.iteration, interval=2500,
        max_nodes=2400, min_nodes=1600, max_splits=1, score_decay=.99,
        running_decay=.9, split_z=1., min_coverage=32, min_age=5000,
        weight_threshold=.05, max_split_shock_ratio=.01)
    manager = AdaptiveNodeManager(model, adaptive_config,
                                  str(args.output / "adaptive_nodes"))
    state = torch.load(args.source / "adaptive_nodes" / "adaptive_state_latest.pth",
                       map_location="cpu", weights_only=False)
    state["config"] = asdict(adaptive_config)
    state["events"] = []
    manager.load_state_dict(state)
    camera = scene.getTrainCameras()[144]
    metric_model = lpips.LPIPS(net="vgg").cuda()
    with torch.no_grad():
        event = manager.adapt(args.iteration, camera, render, pipe, background, metric_model)
        reference = render(camera, model, pipe, background, stage="fine")["render"].detach()

    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.source / "cfg_args", args.output / "cfg_args")
    scene.model_path = str(args.output)
    scene.save(args.iteration, "fine")
    manager.save(args.output / "adaptive_nodes" / "adaptive_state_final.pth",
                 args.iteration)

    restored_model, restored_scene, restored_pipe, restored_background, _ = load_scene(
        args.output, args.iteration)
    restored_camera = restored_scene.getTrainCameras()[144]
    with torch.no_grad():
        restored = render(restored_camera, restored_model, restored_pipe,
                          restored_background, stage="fine")["render"]
    reload_max_error = float((reference - restored).abs().max())
    source_unchanged = source_hashes == {
        path.name: digest(path) for path in source_files}
    audit = {
        "pass": bool(event["splits"] == 1 and
                     event["shock_max"] <= event["shock_limit_world"] and
                     event["outside_affected_physical_k4_exact"] and
                     event["outside_affected_weights_exact"] and
                     reload_max_error <= 1e-6 and source_unchanged),
        "event": event,
        "reload_max_abs_render_error": reload_max_error,
        "source_checkpoint_unchanged": source_unchanged,
    }
    (args.output / "adaptive_nodes" / "s1_replay_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    if not audit["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
