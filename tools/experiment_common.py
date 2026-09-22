"""Shared read-only checkpoint loader and artifact helpers for staged studies."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import torch
from arguments import ModelParams, ModelHiddenParams, OptimizationParams, PipelineParams
from scene import Scene
from scene.gaussian_model import GaussianModel
from tools.audit_sparse_run import read_config


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_scene(path, iteration=30000):
    parser = argparse.ArgumentParser()
    lp, hp, op, pp = ModelParams(parser), ModelHiddenParams(parser), OptimizationParams(parser), PipelineParams(parser)
    args = parser.parse_args([])
    for key, value in read_config(Path(path) / "cfg_args").items():
        setattr(args, key, value)
    args.model_path = str(path)
    model = GaussianModel(args.sh_degree, hp.extract(args))
    scene = Scene(lp.extract(args), model, load_iteration=iteration, shuffle=False)
    bg = torch.tensor([1., 1., 1.] if args.white_background else [0., 0., 0.], device="cuda")
    return model, scene, pp.extract(args), bg, args


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def geometry_hashes(model):
    values = {k: getattr(model, k) for k in ("_xyz", "_scaling", "_rotation", "_opacity")}
    values.update({"node_mlp." + k: v for k, v in model.node_mlp.state_dict().items()})
    values.update({"graph." + k: v for k, v in model.deformation_graph.state_dict().items()})
    for name in ("pos_deform", "scales_deform", "rotations_deform", "opacity_deform"):
        values.update({"legacy_geometry." + name + "." + k: v
                       for k,v in getattr(model._deformation.deformation_net,name).state_dict().items()})
    if hasattr(model, "closest_point_indices"):
        values["gaussian_knn"] = model.closest_point_indices
    return {k: tensor_digest(v) for k, v in values.items()}
