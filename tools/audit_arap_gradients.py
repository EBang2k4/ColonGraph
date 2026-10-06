#!/usr/bin/env python3
"""Multi-pair deformation-MLP gradient audit for ARAP calibration."""

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from gaussian_renderer import render
from tools.experiment_common import load_scene
from utils.loss_utils import TV_loss, canonical_xyz_laplacian, l1_loss


def gradient_tuple(loss, parameters):
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=False, allow_unused=True)
    return tuple(torch.zeros_like(parameter) if gradient is None else gradient.detach()
                 for parameter, gradient in zip(parameters, gradients))


def mean_gradient_norm(gradient_samples):
    count = len(gradient_samples)
    total = gradient_samples[0][0].new_zeros(())
    for parameter_index in range(len(gradient_samples[0])):
        mean_gradient = sum(
            (sample[parameter_index] for sample in gradient_samples),
            torch.zeros_like(gradient_samples[0][parameter_index])) / count
        total = total + mean_gradient.square().sum()
    return float(total.sqrt())


def reconstruction_loss(camera, model, pipe, background, depth_weight, tv_weight):
    output = render(camera, model, pipe, background, stage="fine")
    image = output["render"]
    gt_image = camera.original_image.cuda().float()
    mask = camera.mask
    if mask is not None:
        mask = mask.cuda()
    rgb = l1_loss(image, gt_image, mask)
    rendered_depth = output["depth"]
    gt_depth = camera.original_depth.cuda().float()
    if rendered_depth.ndim == 2:
        rendered_depth = rendered_depth.unsqueeze(0)
    if gt_depth.ndim == 2:
        gt_depth = gt_depth.unsqueeze(0)
    valid = (torch.isfinite(gt_depth) & (gt_depth > 0) &
             torch.isfinite(rendered_depth))
    if mask is not None:
        valid = valid & (mask != 0)
    if valid.any():
        normalized_rendered = torch.clamp(
            rendered_depth / (rendered_depth[valid].max() + 1e-6), 0, 1)
        normalized_gt = torch.clamp(
            gt_depth / (gt_depth[valid].max() + 1e-6), 0, 1)
        depth = depth_weight * torch.abs(
            normalized_rendered - normalized_gt)[valid].mean()
    else:
        depth = image.new_zeros(())
    tv = tv_weight * TV_loss(image.unsqueeze(0))
    # Color residual/smooth losses have no path to deformation MLP when the
    # appearance graph is fixed.  This is exactly the MLP-dependent render loss.
    return rgb + depth + tv, output


def gaussian_knn_raw_loss(model, time):
    """Return the unweighted legacy KNN-99 loss at one timestamp.

    This deliberately mirrors the fine-stage implementation in train.py so
    its gradient profile is directly comparable with the ARAP profile.
    """
    transformed_means = model.deform_geometry_from_nodes(time)[0]
    indices = model.closest_point_indices
    cluster_points = transformed_means[indices]
    cluster_means = cluster_points.mean(dim=1)
    center_points = transformed_means[
        torch.arange(cluster_means.shape[0], device=cluster_means.device)]
    return ((center_points - cluster_means) ** 2).sum(dim=1).mean()


def pair_indices(camera_count, pair_count, seed):
    if pair_count < 8 or pair_count > 16:
        raise ValueError("pair_count must be in [8,16]")
    t1 = torch.linspace(0, camera_count - 1, pair_count).round().long()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    t2 = torch.randint(camera_count, (pair_count,), generator=generator)
    t2 = torch.where(t2 == t1, (t2 + camera_count // 2) % camera_count, t2)
    return list(zip(t1.tolist(), t2.tolist()))


def audit_checkpoint(model_path, iteration, pair_count, include_knn_profile=False):
    model, scene, pipe, background, config = load_scene(model_path, iteration)
    if model.node_arap_graph is None:
        raise RuntimeError("Checkpoint does not contain a canonical ARAP graph")
    knn_path = Path(config.gaussian_knn_reference)
    if not knn_path.exists():
        raise FileNotFoundError("Missing matched Gaussian KNN-99 reference")
    canonical_indices = torch.load(
        knn_path, map_location=model._xyz.device, weights_only=True)
    if include_knn_profile:
        if canonical_indices.shape[0] != model._xyz.shape[0]:
            raise ValueError("KNN-99 reference does not match Gaussian count")
        model.closest_point_indices = canonical_indices
    parameters = tuple(model.node_mlp.parameters())
    cameras = scene.getTrainCameras()
    pairs = pair_indices(len(cameras), pair_count,
                         int(config.arap_seed) + 7919 * iteration)
    reconstruction_gradients, arap_gradients, knn_gradients, per_pair = [], [], [], []
    for t1_index, t2_index in pairs:
        reconstruction, output = reconstruction_loss(
            cameras[t1_index], model, pipe, background,
            float(config.depth_weight), float(config.tv_weight))
        reconstruction_gradient = gradient_tuple(reconstruction, parameters)
        reconstruction_gradients.append(reconstruction_gradient)
        reconstruction_pair_norm = math.sqrt(sum(
            float(value.square().sum()) for value in reconstruction_gradient))

        nodes_t1 = model.node_positions_at_time(cameras[t1_index].time)
        nodes_t2 = model.node_positions_at_time(cameras[t2_index].time)
        arap_raw = model.node_arap_graph(nodes_t1, nodes_t2)
        arap_gradient = gradient_tuple(arap_raw, parameters)
        arap_gradients.append(arap_gradient)
        arap_pair_norm = math.sqrt(sum(
            float(value.square().sum()) for value in arap_gradient))
        knn_raw_value = None
        knn_pair_norm = None
        if include_knn_profile:
            knn_raw = gaussian_knn_raw_loss(model, cameras[t1_index].time)
            knn_gradient = gradient_tuple(knn_raw, parameters)
            knn_gradients.append(knn_gradient)
            knn_raw_value = float(knn_raw.detach())
            knn_pair_norm = math.sqrt(sum(
                float(value.square().sum()) for value in knn_gradient))
        per_pair.append({
            "t1_index": t1_index, "t2_index": t2_index,
            "t1": float(cameras[t1_index].time),
            "t2": float(cameras[t2_index].time),
            "reconstruction_loss": float(reconstruction.detach()),
            "arap_raw_loss": float(arap_raw.detach()),
            "reconstruction_gradient_norm": reconstruction_pair_norm,
            "arap_gradient_norm": arap_pair_norm,
            "knn99_raw_loss": knn_raw_value,
            "knn99_gradient_norm": knn_pair_norm,
            "mean_alpha": float(output["alpha"].detach().mean()),
            "visible_gaussian_ratio": float(
                output["visibility_filter"].float().mean()),
        })

    reconstruction_norm = mean_gradient_norm(reconstruction_gradients)
    arap_norm = mean_gradient_norm(arap_gradients)
    raw_ratio = arap_norm / max(reconstruction_norm, 1e-20)
    knn_norm = mean_gradient_norm(knn_gradients) if include_knn_profile else None
    knn_raw_ratio = (knn_norm / max(reconstruction_norm, 1e-20)
                     if include_knn_profile else None)
    canonical = canonical_xyz_laplacian(model._xyz, canonical_indices)
    canonical_gradient, = torch.autograd.grad(canonical, model._xyz)
    valid_pairs = sum(
        item["reconstruction_gradient_norm"] > 1e-8 and
        item["mean_alpha"] > 1e-3 for item in per_pair)
    checks = {
        "pair_count_8_to_16": 8 <= pair_count <= 16,
        "reconstruction_gradient_finite_positive": (
            math.isfinite(reconstruction_norm) and reconstruction_norm > 1e-8),
        "arap_gradient_finite_positive": (
            math.isfinite(arap_norm) and arap_norm > 1e-8),
        "at_least_75pct_pairs_render_and_backprop": valid_pairs >= math.ceil(0.75 * pair_count),
        "canonical_laplacian_gradient_finite_positive": (
            torch.isfinite(canonical_gradient).all().item() and
            float(canonical_gradient.norm()) > 0),
    }
    if include_knn_profile:
        checks["knn99_gradient_finite_positive"] = (
            math.isfinite(knn_norm) and knn_norm > 1e-8)
    return {
        "iteration": iteration,
        "pair_count": pair_count,
        "checks": checks,
        "gradient_meaningful": all(checks.values()),
        "reconstruction_gradient_norm": reconstruction_norm,
        "arap_raw_gradient_norm": arap_norm,
        "raw_gradient_ratio": raw_ratio,
        "arap_weight": float(config.arap_weight),
        "weighted_gradient_ratio": float(config.arap_weight) * raw_ratio,
        "knn99_raw_gradient_norm": knn_norm,
        "knn99_raw_gradient_ratio": knn_raw_ratio,
        "knn99_weight": float(config.knn_weight),
        "knn99_weighted_gradient_ratio": (
            float(config.knn_weight) * knn_raw_ratio
            if include_knn_profile else None),
        "canonical_laplacian_raw": float(canonical.detach()),
        "canonical_laplacian_gradient_norm": float(canonical_gradient.norm()),
        "valid_reconstruction_pairs": valid_pairs,
        "mean_alpha": sum(item["mean_alpha"] for item in per_pair) / pair_count,
        "pairs": per_pair,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--iterations", nargs="+", type=int,
                        default=[2000, 5000, 10000])
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--include-knn-profile", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [audit_checkpoint(args.model_path, iteration, args.pairs,
                             args.include_knn_profile)
            for iteration in args.iterations]
    report = {
        "pass": all(row["gradient_meaningful"] for row in rows),
        "model_path": str(args.model_path),
        "rows": rows,
    }
    output = args.output or args.model_path / "arap_multipair_gradient_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit("ARAP multi-pair gradient audit is not meaningful")


if __name__ == "__main__":
    main()
