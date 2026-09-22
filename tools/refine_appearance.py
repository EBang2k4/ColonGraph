"""Appearance-only continuation; geometry/opacity are immutable and audited."""
from experiment_common import load_scene, save_json, geometry_hashes, tensor_digest
import argparse
import csv
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from plyfile import PlyData
from gaussian_renderer import render
from utils.loss_utils import l1_loss, TV_loss, ssim
from refit_appearance_model import residual_penalty_weight


def configure_appearance(model, args):
    if model.deformation_mode != "node" or args.use_do or args.no_dcol:
        raise ValueError("Requires node geometry, active color branch and disabled opacity deformation")
    for key in ("_xyz", "_scaling", "_rotation", "_opacity", "features", "_deformation_table"):
        getattr(model, key).requires_grad_(False)
    model.node_mlp.requires_grad_(False)
    model._deformation.requires_grad_(False)
    for module in (model._deformation.deformation_net.grid,
                   model._deformation.deformation_net.feature_out,
                   model._deformation.deformation_net.color_deform):
        module.requires_grad_(True)
    model.features.requires_grad_(True)
    model._deformation_table.requires_grad_(True)
    model.training_setup(args)
    model.optimizer.param_groups[:] = [dict(group, params=[p for p in group['params'] if p.requires_grad])
                                       for group in model.optimizer.param_groups
                                       if any(p.requires_grad for p in group['params'])]
    frozen_ids = {id(p) for p in model.node_mlp.parameters()} | {
        id(getattr(model, k)) for k in ("_xyz", "_scaling", "_rotation", "_opacity")}
    assert not frozen_ids & {id(p) for g in model.optimizer.param_groups for p in g['params']}


def appearance_hashes(model):
    field = model._deformation.deformation_net
    return {"base_color": tensor_digest(model.features), "color_embedding": tensor_digest(model._deformation_table),
            **{k: tensor_digest(torch.cat([p.detach().flatten() for p in getattr(field,k).parameters()]))
               for k in ("grid", "feature_out", "color_deform")}}


def make_lpips():
    import lpips
    # Loading pretrained weights constructs random layers internally; preserve
    # the training RNG so the only ablation change is the extra loss term.
    with torch.random.fork_rng():
        criterion = lpips.LPIPS(net='alex', version='0.1').cuda().eval()
    return criterion.requires_grad_(False)


def losses(pkg, view, args, knn_constant, perceptual=None):
    im = pkg['render'].unsqueeze(0)
    gt = view.original_image.cuda().float().unsqueeze(0)
    # This scene has no image mask; reject unsupported data instead of dropping it.
    if view.mask is not None:
        raise ValueError("This matched C3VD refinement expects no image mask")
    depth = pkg['depth'].reshape(1, 1, *pkg['depth'].shape[-2:])
    gt_depth = view.original_depth.cuda().float().reshape_as(depth)
    valid = torch.isfinite(gt_depth) & (gt_depth > 0) & torch.isfinite(depth)
    rgb = l1_loss(im, gt)
    if valid.any():
        dn = (depth/(depth[valid].max()+1e-6)).clamp(0,1)
        gn = (gt_depth/(gt_depth[valid].max()+1e-6)).clamp(0,1)
        dl = args.depth_weight*l1_loss(dn,gn,valid,True)
    else:
        dl = rgb.new_zeros(())
    dc = residual_penalty_weight(args)*pkg['dcol'].square().mean()
    c = pkg['transformed_color']
    cs = args.col_smooth_weight*(c-c.mean(0)).square().mean()
    tv = args.tv_weight*TV_loss(im)
    # Depth and KNN remain in total loss; both are constant w.r.t. appearance.
    terms = dict(rgb_l1=rgb, depth=dl, knn=knn_constant, dcolor=dc, color_smooth=cs, tv=tv)
    weight = getattr(args, 'appearance_dssim_weight', 0.)
    # Additive DSSIM ablation: do not reduce the existing RGB L1 coefficient.
    terms['dssim'] = weight * (1-ssim(im,gt))/2 if weight else rgb.new_zeros(())
    lpips_weight = getattr(args, 'appearance_lpips_weight', 0.)
    if weight and lpips_weight and not getattr(args, 'appearance_combined_losses', False):
        raise ValueError('DSSIM and LPIPS must be separate ablations unless combined losses are explicitly enabled')
    if lpips_weight and perceptual is None:
        raise ValueError('LPIPS loss requires a frozen perceptual network')
    # Do NOT use no_grad here: gradients must reach rendered RGB, not AlexNet.
    # Match evaluation normalization, without PNG quantization/clamping in train.
    terms['lpips'] = lpips_weight*perceptual(im,gt,normalize=True).mean() if lpips_weight else rgb.new_zeros(())
    return sum(terms.values()), terms


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output/'status.json', {"state":"RUNNING","stage":"load"})
    torch.set_num_threads(4)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    model, scene, pipe, bg, cfg = load_scene(args.source,30000)
    # Historical snapshots did not save Gaussian-Gaussian KNN. Reconstruct with
    # the same cuVS k=100 algorithm on ORIGINAL input XYZ, never on final XYZ.
    v = PlyData.read(Path(cfg.source_path)/'points3d.ply')['vertex']
    initial_xyz = torch.from_numpy(np.stack([v[k] for k in ('x','y','z')],1).copy()).cuda()
    canonical = model._xyz
    assert len(initial_xyz) == len(canonical)
    model._xyz = torch.nn.Parameter(initial_xyz, requires_grad=False)
    _, model.closest_point_indices = model.compute_point_cloud_normals(k=100)
    model._xyz = canonical
    configure_appearance(model,cfg)
    frozen_before = geometry_hashes(model)
    appearance_before = appearance_hashes(model)
    source_cfg = dict(vars(cfg))
    cfg.model_path = str(args.output); cfg.expname = str(args.output)
    cfg.iterations = 30000+args.steps
    cfg.save_iterations = sorted(set([30000,30000+args.steps]+list(range(31000,30000+args.steps+1,1000))))
    cfg.refinement_steps = args.steps; cfg.appearance_only = True
    cfg.refinement_source = str(args.source)
    cfg.appearance_dssim_weight = args.dssim_weight
    cfg.appearance_lpips_weight = args.lpips_weight
    cfg.appearance_combined_losses = args.allow_combined_losses
    perceptual = make_lpips() if args.lpips_weight else None
    (args.output/'cfg_args').write_text(str(argparse.Namespace(**vars(cfg))))
    scene.model_path = str(args.output)
    scene.save(30000,'fine')
    torch.save(model.closest_point_indices.cpu(),args.output/'gaussian_knn.pth')
    save_json(args.output/'protocol.json', {"source":str(args.source),"source_config":source_cfg,
        "steps":args.steps,"seed":0,"optimizer":"Fresh Adam, eps=1e-15; source optimizer unavailable",
        "learning_rates":"Continue original schedule at iterations 30001..35000 (no LR restart)",
        "losses":"RGB L1 + existing depth/KNN/dcolor/color smooth/TV, unchanged weights; additive DSSIM/LPIPS; combination requires explicit opt-in",
        "combined_losses":args.allow_combined_losses,
        "dssim_weight":args.dssim_weight,"dssim_definition":"(1-SSIM)/2, 11x11 Gaussian window sigma=1.5; unclipped float training RGB; L1 coefficient stays 1",
        "lpips_weight":args.lpips_weight,"lpips_definition":"LPIPS v0.1 AlexNet, pretrained frozen eval network; normalize=True maps RGB to [-1,1]; full-resolution unclipped float training RGB; no PNG quantization; L1 coefficient stays 1",
        "knn_note":"Node graph loaded unchanged; Gaussian k=100 reconstructed from input PLY because historical indices were not saved. No trainable effect in this frozen-geometry run.",
        "geometry_hashes_before":frozen_before,"appearance_hashes_before":appearance_before})
    # Float32, unnormalized depth + alpha invariant checks on ALL test cameras.
    test_before = []
    with torch.no_grad():
        for view in scene.getTestCameras():
            pkg = render(view,model,pipe,bg)
            test_before.append((pkg['depth'].cpu(),pkg['alpha'].cpu()))
    train_views = scene.getTrainCameras()
    cache = {}
    random.seed(0)
    start = time.monotonic()
    with (args.output/'training.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=['step','elapsed_s','total','rgb_l1','depth','knn','dcolor','color_smooth','tv','dssim','lpips'])
        writer.writeheader()
        for step in range(1,args.steps+1):
            model.update_learning_rate(30000+step,stage='fine')
            model.optimizer.zero_grad(set_to_none=True)
            index = random.randint(0,len(train_views)-1)
            pkg = render(train_views[index],model,pipe,bg)
            if index not in cache:
                with torch.no_grad():
                    means = pkg['transformed_points']
                    # Chunked mean avoids keeping N*99*3 temporary values alive.
                    total = means.new_zeros(())
                    for offset in range(0,len(means),4096):
                        idx = model.closest_point_indices[offset:offset+4096]
                        total += (means[offset:offset+len(idx)]-means[idx].mean(1)).square().sum()
                    cache[index] = cfg.knn_weight*total/len(means)
            loss,terms = losses(pkg,train_views[index],cfg,cache[index],perceptual)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite refinement loss")
            loss.backward()
            for group in model.optimizer.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        torch.nn.utils.clip_grad_value_(param,.5)
            assert all(getattr(model,k).grad is None for k in ('_xyz','_scaling','_rotation','_opacity'))
            assert all(p.grad is None for p in model.node_mlp.parameters())
            if perceptual is not None:
                assert not perceptual.training and all(p.grad is None and not p.requires_grad for p in perceptual.parameters())
            model.optimizer.step()
            if step==1 or step%100==0 or step==args.steps:
                row = dict(step=step,elapsed_s=time.monotonic()-start,total=loss.item(),**{k:v.item() for k,v in terms.items()})
                writer.writerow(row);stream.flush();print(row,flush=True)
                save_json(args.output/'status.json',dict(state='RUNNING',stage='appearance_training',**row))
            if step%1000==0:
                assert geometry_hashes(model)==frozen_before
                scene.save(30000+step,'fine')
    assert geometry_hashes(model)==frozen_before
    appearance_after = appearance_hashes(model)
    assert all(appearance_after[k] != v for k,v in appearance_before.items()), "An appearance parameter block did not update"
    scene.save(30000+args.steps,'fine')
    model.load_ply(str(args.output/f'point_cloud/iteration_{30000+args.steps}/point_cloud.ply'))
    model.load_model(str(args.output/f'point_cloud/iteration_{30000+args.steps}'))
    assert geometry_hashes(model)==frozen_before, "Geometry changed after checkpoint roundtrip"
    differences = []
    with torch.no_grad():
        for i,view in enumerate(scene.getTestCameras()):
            pkg = render(view,model,pipe,bg)
            d,a = test_before[i]
            dd = (pkg['depth'].cpu()-d).abs().max().item()
            da = (pkg['alpha'].cpu()-a).abs().max().item()
            torch.testing.assert_close(pkg['depth'].cpu(),d,rtol=1e-6,atol=1e-6)
            torch.testing.assert_close(pkg['alpha'].cpu(),a,rtol=1e-6,atol=1e-6)
            differences.append(dict(test_index=i,depth_max_abs_diff=dd,alpha_max_abs_diff=da))
    save_json(args.output/'freeze_audit.json', {"geometry_bitwise_unchanged":True,"checkpoint_roundtrip_passed":True,
        "all_appearance_blocks_updated":True,"appearance_hashes_after":appearance_after,
        "test_depth_alpha_invariants":differences,"optimizer_lrs":{g['name']:g['lr'] for g in model.optimizer.param_groups}})
    save_json(args.output/'status.json', {"state":"TRAINING_COMPLETED","steps":args.steps,"metrics_pending":True})


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=5000)
    p.add_argument('--dssim-weight',type=float,default=0.)
    p.add_argument('--lpips-weight',type=float,default=0.)
    p.add_argument('--allow-combined-losses',action='store_true')
    args=p.parse_args()
    if args.output.exists() or args.steps<1: p.error('Output must be new and steps positive')
    if not np.isfinite(args.dssim_weight) or args.dssim_weight<0: p.error('DSSIM weight must be finite and nonnegative')
    if not np.isfinite(args.lpips_weight) or args.lpips_weight<0: p.error('LPIPS weight must be finite and nonnegative')
    if args.dssim_weight and args.lpips_weight and not args.allow_combined_losses: p.error('Combined DSSIM/LPIPS requires --allow-combined-losses')
    try: run(args)
    except Exception as exc:
        if args.output.exists(): save_json(args.output/'status.json',dict(state='FAILED',error=repr(exc)))
        raise
