"""CUDA-event deformation-only benchmark; excludes rasterizer, losses and I/O."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import statistics
import torch
from arguments import ModelHiddenParams
from scene.gaussian_model import GaussianModel


def measure(fn, repeats):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        value = fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
        del value
    return {"median_ms": statistics.median(samples), "mean_ms": statistics.mean(samples),
            "min_ms": min(samples), "max_ms": max(samples), "samples": len(samples)}


def color_forward(model, time):
    if model.appearance_model is not None:
        graph = model.deformation_graph
        return model.appearance_model(
            model._xyz, time, model.features, model._deformation_table,
            graph.gaussian_node_indices, graph.gaussian_node_weights)
    return model._deformation.deformation_net.forward_color_only(
        model._xyz, model._opacity, model.features, time,
        model._deformation_table)


def benchmark(path, repeats):
    parser = argparse.ArgumentParser()
    hp = ModelHiddenParams(parser)
    model = GaussianModel(3, hp.extract(parser.parse_args([])))
    model.load_ply(str(Path(path) / "point_cloud.ply"))
    model.load_model(path)
    t = torch.full((len(model._xyz), 1), .5, device="cuda")
    inputs = (model._xyz, model._scaling, model._rotation, model._opacity,
              model.features, t, model._deformation_table)
    result = {"path": str(path), "mode": model.deformation_mode,
              "color_model": model.color_model, "N": len(model._xyz),
              "node_position_mode": model.args.node_position_mode,
              "node_position_normalization": model.args.node_position_normalization,
              "M": len(model.deformation_graph.node_xyz) if model.deformation_graph is not None else None}
    result["training_forward_deformation_ms"] = measure(lambda: model.deform(*inputs), repeats)
    with torch.no_grad():
        result["inference_deformation_ms"] = measure(lambda: model.deform(*inputs), repeats)
        if model.deformation_mode == "node":
            result["node_geometry_including_blend_ms"] = measure(lambda: model.deform_geometry_from_nodes(t), repeats)
            graph = model.deformation_graph
            result["node_mlp_only_ms"] = measure(lambda: model.node_mlp(graph.node_xyz, t), repeats)
        result["color_branch_ms"] = measure(lambda: color_forward(model, t), repeats)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    result = {"torch": torch.__version__, "cuda": torch.version.cuda,
              "device": torch.cuda.get_device_name(0), "warmup": 10,
              "note": "Profiles the checkpoint's active appearance branch; timing excludes backward, renderer and losses."}
    for name, path in (("baseline", args.baseline), ("node", args.node)):
        result[name] = benchmark(path, args.repeats)
        torch.cuda.empty_cache()
    result["forward_speedup"] = (result["baseline"]["training_forward_deformation_ms"]["median_ms"] /
                                   result["node"]["training_forward_deformation_ms"]["median_ms"])
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
