#!/usr/bin/env python3
"""Matched CUDA render-throughput benchmark for adaptive graph branches."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from gaussian_renderer import render
from tools.experiment_common import load_scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    model, scene, pipe, background, _ = load_scene(args.model_path, args.iteration)
    cameras = scene.getTestCameras()
    with torch.no_grad():
        for camera in cameras[:min(5, len(cameras))]:
            render(camera, model, pipe, background, stage="fine")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.repeats):
            for camera in cameras:
                render(camera, model, pipe, background, stage="fine")
        end.record()
        torch.cuda.synchronize()
    total_frames = args.repeats * len(cameras)
    elapsed_ms = start.elapsed_time(end)
    report = {
        "iteration": args.iteration,
        "test_frames": len(cameras),
        "repeats": args.repeats,
        "render_ms_per_frame": elapsed_ms / total_frames,
        "render_fps": 1000.0 * total_frames / elapsed_ms,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
    }
    output = args.model_path / "adaptive_render_benchmark.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
