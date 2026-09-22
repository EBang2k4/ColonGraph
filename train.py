#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
import numpy as np
import random
import os
import csv
import json
import time as walltime
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, lpips_loss, TV_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from utils.timer import Timer
from utils import helper
from utils.node_sampling import (CoarseGradientImportance, weighted_farthest_point_sampling,
                                 build_gaussian_node_knn, validate_index_aligned_reference)
from scene.deformation_graph import SparseDeformationGraph


import lpips
from utils.scene_utils import render_training_image
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def _gradient_norm(parameters, reference):
    total = reference.new_zeros(())
    for parameter in parameters:
        if parameter.grad is not None:
            total = total + parameter.grad.detach().square().sum()
    return total.sqrt().item()


def _append_r6_diagnostic(model_path, row):
    path = os.path.join(model_path, "joint_r6_diagnostics.csv")
    exists = os.path.exists(path)
    with open(path, "a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations, 
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, stage, tb_writer, train_iter, timer):
    


    first_iter = 0
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    
    final_iter = train_iter
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    
    lpips_model = lpips.LPIPS(net="vgg").cuda()
    video_cams = scene.getVideoCameras()
    
    if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras()
    
    for iteration in range(first_iter, final_iter+1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer, ts = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer, stage="stage")["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration, stage=stage)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()
        if stage == 'coarse':
            # idx = 0
            idx = randint(0, len(viewpoint_stack)-1)
            # idx = iteration % len(viewpoint_stack)
        else:
            idx = randint(0, len(viewpoint_stack)-1)
            # idx = iteration % len(viewpoint_stack)
            
        viewpoint_cams = [viewpoint_stack[idx]]
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
            
        images = []
        depths = []
        gt_images = []
        gt_depths = []
        masks = []
        images_concealing = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        transformed_means_list = []
        transformed_color_list =[]
        
        for viewpoint_cam in viewpoint_cams:
            profile_interval = hyper.deformation_profile_interval
            gaussians.profile_deformation = stage == "fine" and profile_interval > 0 and iteration % profile_interval == 0
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage)
            if gaussians.profile_deformation:
                elapsed = gaussians.last_deformation_forward_ms
                print(f"[PROFILE] stage={stage} iteration={iteration} deformation_forward_ms={elapsed:.4f}")
                with open(os.path.join(scene.model_path, "deformation_profile.csv"), "a") as profile_file:
                    if profile_file.tell() == 0:
                        profile_file.write("iteration,deformation_forward_ms\n")
                    profile_file.write(f"{iteration},{elapsed:.6f}\n")
                if tb_writer:
                    tb_writer.add_scalar("timing/deformation_forward_ms", elapsed, iteration)
            gaussians.profile_deformation = False
            image, depth, viewspace_point_tensor, visibility_filter, radii, transformed_means, \
                transformed_color, dcol = \
                render_pkg["render"], render_pkg["depth"], render_pkg["viewspace_points"], \
                    render_pkg["visibility_filter"], render_pkg["radii"], \
                        render_pkg["transformed_points"], render_pkg["transformed_color"], \
                            render_pkg["dcol"]
                        
            gt_image = viewpoint_cam.original_image.cuda().float()
            gt_depth = viewpoint_cam.original_depth.cuda().float()
            mask = viewpoint_cam.mask
            if mask:
                mask = mask.cuda()
            
            image_concealing = render_pkg['render']
            transformed_means_list.append(transformed_means)
            transformed_color_list.append(transformed_color)
            images_concealing.append(image_concealing.unsqueeze(0))
            images.append(image.unsqueeze(0))
            if depth.ndim == 2:
                depth = depth.unsqueeze(0)
            if gt_depth.ndim == 2:
                gt_depth = gt_depth.unsqueeze(0)
            depths.append(depth.unsqueeze(0))
            gt_images.append(gt_image.unsqueeze(0))
            gt_depths.append(gt_depth.unsqueeze(0))
            if mask:
                masks.append(mask.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
            
        radii = torch.cat(radii_list,0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        rendered_images = torch.cat(images,0)
        rendered_depths = torch.cat(depths, 0)
        gt_images = torch.cat(gt_images,0)
        gt_depths = torch.cat(gt_depths, 0)
        transformed_means = torch.cat(transformed_means_list, 0)
        transformed_color = torch.cat(transformed_color_list, 0)
        images_concealing = torch.cat(images_concealing, 0)
        if mask:
            masks = torch.cat(masks, 0)
        
        loss = 0

        ## Ll1 loss
        if mask:
            Ll1 = l1_loss(images_concealing, gt_images, masks)
        else:
            Ll1 = l1_loss(images_concealing, gt_images)
        loss += Ll1

        ### DEPTH LOSS
        valid_depth_mask = (
            torch.isfinite(gt_depths)
            & (gt_depths > 0)
            & torch.isfinite(rendered_depths)
        )
        if mask:
            valid_depth_mask = valid_depth_mask & (masks != 0)

        if valid_depth_mask.any():
            rendered_depth_max = rendered_depths[valid_depth_mask].max()
            gt_depth_max = gt_depths[valid_depth_mask].max()
            normalized_rendered_depth = torch.clamp(
                rendered_depths / (rendered_depth_max + 1e-6), 0, 1
            )
            normalized_gt_depth = torch.clamp(
                gt_depths / (gt_depth_max + 1e-6), 0, 1
            )
            depth_loss = opt.depth_weight * l1_loss(
                normalized_rendered_depth,
                normalized_gt_depth,
                valid_depth_mask,
                True,
            )
        else:
            depth_loss = rendered_depths.new_zeros(())
        loss +=depth_loss

        ## XYZ CLUSTER LOSS
        idx = gaussians.closest_point_indices
        K = int(idx.max().item()) + 1

        cluster_pts = transformed_means[idx]
        cluster_means = cluster_pts.mean(dim=1)
        center_points = transformed_means[torch.arange(cluster_means.shape[0], device=cluster_means.device)]
        per_point_sq_dist = ((center_points - cluster_means) ** 2).sum(dim=1)

        loss_clusters = opt.knn_weight * per_point_sq_dist.mean()
        loss += loss_clusters

        ### MINIMIZE DCOLOR UPDATE - DCOL LOSS
        residual_weight = (opt.lambda_res if gaussians.color_model == "node_residual_mlp"
                           else opt.dcol_weight)
        loss_dcol = torch.mean(dcol ** 2) * residual_weight
        loss += loss_dcol

        ### MINIMIZE DIFFERENCES BETWEEN COLORS
        loss_color_smooth = opt.col_smooth_weight * ((transformed_color - transformed_color.mean(dim=0)) ** 2).mean()
        loss += loss_color_smooth


        ###  LOSS TV, breaks our colon so opt.tv_weight set to zero by default
        depth_tvloss = TV_loss(rendered_depths)
        img_tvloss = TV_loss(rendered_images)
        tv_loss = opt.tv_weight * (img_tvloss) # + depth_tvloss)
        loss += tv_loss

        if mask:
            psnr_ = psnr(images_concealing, gt_images, masks).mean().double()      
        else:
            psnr_ = psnr(images_concealing, gt_images).mean().double()  
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n❌ Loss NaN/Inf at iteration {iteration}")
            if gaussians.deformation_mode == "node":
                raise FloatingPointError(f"Non-finite {stage} loss at iteration {iteration}")
            
        loss.backward()
        if stage == "coarse" and getattr(gaussians, "coarse_gradient_importance", None) is not None:
            gaussians.coarse_gradient_importance.update(gaussians._xyz.grad, visibility_filter)
        r6_diagnostic = None
        if (stage == "fine" and gaussians.appearance_model is not None and
                (iteration == 1 or iteration % max(1, hyper.deformation_profile_interval) == 0 or
                 iteration == train_iter)):
            app = gaussians.appearance_model
            valid_depth_values = rendered_depths[valid_depth_mask]
            valid_gt_depth_values = gt_depths[valid_depth_mask]
            raw_depth_mse = ((valid_depth_values - valid_gt_depth_values).square().mean().item()
                             if valid_depth_values.numel() else float("nan"))
            eta = max(float(app.eta), 1e-12)
            r6_diagnostic = {
                "iteration": iteration,
                "loss": loss.item(),
                "rgb_l1": Ll1.item(),
                "depth_loss": depth_loss.item(),
                "raw_depth_mse": raw_depth_mse,
                "knn_loss": loss_clusters.item(),
                "residual_loss": loss_dcol.item(),
                "mean_abs_delta": dcol.detach().abs().mean().item(),
                "max_abs_delta": dcol.detach().abs().max().item(),
                "saturation_ratio": (dcol.detach().abs() > .9 * eta).float().mean().item(),
                "appearance_grad_norm": _gradient_norm(app.parameters(), loss),
                "node_appearance_grad_norm": _gradient_norm([app.node_appearance], loss),
                "embedding_grad_norm": _gradient_norm([gaussians._deformation_table], loss),
                "node_motion_grad_norm": _gradient_norm(gaussians.node_mlp.parameters(), loss),
                "xyz_grad_norm": _gradient_norm([gaussians._xyz], loss),
                "transformed_xyz_finite": int(torch.isfinite(transformed_means).all().item()),
                "depth_finite_ratio": torch.isfinite(rendered_depths).float().mean().item(),
                "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20,
            }
        for group in gaussians.optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    torch.nn.utils.clip_grad_value_(param, clip_value=0.5)

            # torch.nn.utils.clip_grad_norm_(param, max_norm=1)
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                string_dict = {"Loss": f"{ema_loss_for_log:.{7}f}",
                                "psnr": f"{psnr_:.{2}f}",
                                "point":f"{total_point}"}
                progress_bar.set_postfix(string_dict)
                progress_bar.update(10)
            if iteration == train_iter:
                progress_bar.close()

            # Log and save
            timer.pause()
            metrics = {'Ll1':Ll1, 'depth_loss':depth_loss, 'elapsed':iter_start.elapsed_time(iter_end), \
                'psnr':psnr_}
            if r6_diagnostic is not None:
                r6_diagnostic["iteration_ms"] = metrics["elapsed"]
                _append_r6_diagnostic(scene.model_path, r6_diagnostic)
                print("[R6_DIAGNOSTIC] " + json.dumps(r6_diagnostic), flush=True)
            
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, 
                            (pipe, background, 1.,))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, stage)
            if dataset.render_process:
                if (iteration < 1000 and iteration % 10 == 1) \
                    or (iteration < 3000 and iteration % 50 == 1) \
                        or (iteration < 10000 and iteration %  100 == 1) \
                            or (iteration < 60000 and iteration % 100 ==1):
                    render_training_image(scene, gaussians, video_cams, render, pipe, background, stage, iteration-1,timer.get_elapsed_time())
            timer.start()
            
            # Densification
            if iteration < opt.densify_until_iter :
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:    
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - \
                        opt.opacity_threshold_fine_after)/(opt.densify_until_iter)
                    densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - \
                        opt.densify_grad_threshold_after)/(opt.densify_until_iter )

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    # print('densify', iteration, opt.densification_interval, opt.densify_from_iter)
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                if iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0:
                    size_threshold = 40 if iteration > opt.opacity_reset_interval else None
                    # print('pruning', iteration, opt.pruning_interval, opt.pruning_from_iter)
                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    print("reset opacity")
                    gaussians.reset_opacity()
                    
            # Optimizer step
            if iteration < train_iter:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname, extra_mark):
    if hyper.node_sampling not in ("fps", "weighted_fps"):
        raise ValueError("node_sampling must be fps or weighted_fps")
    if hyper.node_sampling == "weighted_fps" and (hyper.deformation_mode != "node" or opt.coarse_iterations != 100):
        raise ValueError("Weighted FPS requires node mode and 100 coarse iterations")
    if hyper.node_sampling_importance_reference and hyper.node_sampling != "weighted_fps":
        raise ValueError("An importance reference is valid only for weighted_fps")
    if hyper.color_model == "node_residual_mlp":
        if hyper.deformation_mode != "node":
            raise ValueError("node_residual_mlp requires deformation_mode=node")
        if hyper.node_k != 4:
            raise ValueError("R6 joint training requires node_k=4")
        if hyper.no_dcol or hyper.use_do:
            raise ValueError("R6 joint training requires no_dcol=False and use_do=False")
    if hyper.deformation_mode == "node":
        if checkpoint or checkpoint_iterations:
            raise ValueError("Node mode uses Scene.save snapshots; optimizer resume is not supported")
        if min(opt.densify_from_iter, opt.pruning_from_iter, opt.densify_from_iter_fine,
               opt.pruning_from_iter_fine) < min(opt.iterations, opt.densify_until_iter):
            raise ValueError("Fixed node graph requires densification/pruning disabled as in run.sh")
    tb_writer = prepare_output_and_logger(expname)
    training_start = walltime.monotonic()
    torch.cuda.reset_peak_memory_stats()
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, load_coarse=args.no_fine)
    if hyper.node_sampling == "weighted_fps" and not hyper.node_sampling_importance_reference:
        gaussians.coarse_gradient_importance = CoarseGradientImportance(gaussians._xyz)
    timer.start()
    coarse_start = walltime.monotonic()
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                             checkpoint_iterations, checkpoint, debug_from,
                             gaussians, scene, "coarse", tb_writer, opt.coarse_iterations,timer)
    coarse_duration = walltime.monotonic() - coarse_start
    sampling_state = None
    if hyper.node_sampling == "weighted_fps":
        if hyper.node_sampling_importance_reference:
            reference_path = hyper.node_sampling_importance_reference
            sampling_state = torch.load(reference_path, map_location=gaussians._xyz.device,
                                        weights_only=True)
            required = {"grad_sum", "grad_sq_sum", "grad_count", "importance",
                        "coarse_steps", "observed_post_coarse_xyz", "canonical_xyz"}
            if not required.issubset(sampling_state):
                raise ValueError(f"Incomplete Weighted FPS importance reference: {reference_path}")
            if sampling_state["importance"].shape != (len(gaussians._xyz),):
                raise ValueError("Importance reference Gaussian topology does not match")
            sampling_state["current_post_coarse_xyz"] = gaussians._xyz.detach().clone()
            sampling_state["importance_reference_path"] = reference_path
            print(f"Weighted FPS: reusing audited coarse importance from {reference_path}", flush=True)
        else:
            sampling_state = gaussians.coarse_gradient_importance.finalize()
            sampling_state["observed_post_coarse_xyz"] = gaussians._xyz.detach().clone()
            sampling_state["importance_reference_path"] = ""
            del gaussians.coarse_gradient_importance
    fine_duration = 0.0
    if not args.no_fine:
        if hyper.deformation_mode == "node":
            gaussians.initialize_deformation_nodes()
            if hyper.node_initialization_reference:
                # Reuse an UNTRAINED fine iteration_1 snapshot to control GPU
                # coarse/FPS nondeterminism in paired normalization experiments.
                reference = hyper.node_initialization_reference
                if os.path.basename(os.path.normpath(reference)) != "iteration_1":
                    raise ValueError("Initialization reference must be an untrained iteration_1 snapshot")
                reference_config = torch.load(os.path.join(reference, "deformation_nodes.pth"),
                                              map_location="cpu", weights_only=True)["config"]
                for key in ("num_nodes", "node_k", "node_mlp_width", "node_mlp_depth",
                            "posebase_pe", "timebase_pe", "node_position_mode"):
                    expected = getattr(hyper, key)
                    if reference_config.get(key, "direct" if key == "node_position_mode" else None) != expected:
                        raise ValueError(f"Initialization reference config differs: {key}")
                normalization = hyper.node_position_normalization
                initial_mlp = {k: v.detach().clone() for k, v in gaussians.node_mlp.named_parameters()}
                active_sh_degree = gaussians.active_sh_degree
                cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
                gaussians.load_ply(os.path.join(reference, "point_cloud.ply"))
                gaussians.load_model(reference)
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state_all(cuda_rng)
                gaussians.active_sh_degree = active_sh_degree
                if not all(torch.equal(v, dict(gaussians.node_mlp.named_parameters())[k])
                           for k, v in initial_mlp.items()):
                    raise ValueError("Reference MLP differs from fresh initialization; warm-start is not allowed")
                gaussians.args.node_position_normalization = normalization
                gaussians.node_mlp.position_normalization = normalization
                gaussians.node_mlp.configure_position_normalization(gaussians._xyz.detach())
                print(f"Paired experiment: restored UNTRAINED post-coarse initialization from {reference}; RNG preserved; normalization={normalization}")
            if sampling_state is not None:
                # Preserve the existing paired post-coarse initialization. The
                # historical reference controls every Gaussian/MLP attribute;
                # only the FPS selection and the consequent K4 graph change.
                canonical = gaussians._xyz.detach()
                observed = sampling_state["observed_post_coarse_xyz"]
                if canonical.shape != observed.shape:
                    raise ValueError("Coarse/reference Gaussian identities do not match")
                if hyper.node_sampling_importance_reference:
                    source_canonical = sampling_state["canonical_xyz"]
                    if not torch.equal(source_canonical, canonical):
                        raise ValueError("Importance reference canonical XYZ differs from paired baseline")
                    current_alignment = validate_index_aligned_reference(
                        sampling_state["current_post_coarse_xyz"], canonical)
                    sampling_state["current_coarse_reference_alignment"] = current_alignment
                if hyper.node_initialization_reference:
                    alignment = validate_index_aligned_reference(observed, canonical)
                    sampling_state["reference_alignment"] = alignment
                    print("Weighted FPS reference alignment: " + json.dumps(alignment), flush=True)
                selected = weighted_farthest_point_sampling(
                    canonical, hyper.num_nodes, sampling_state["importance"], hyper.node_sampling_alpha)
                nodes = canonical[selected]
                indices, weights = build_gaussian_node_knn(canonical, nodes, hyper.node_k)
                gaussians.deformation_graph = SparseDeformationGraph(nodes, indices, weights)
                sampling_state.update({"selected_indices": selected, "canonical_xyz": canonical,
                                       "alpha": hyper.node_sampling_alpha, "first_node_index": 0,
                                       "reference_max_xyz_difference": (observed-canonical).abs().max().item()})
                torch.save({key: value.detach().cpu() if torch.is_tensor(value) else value
                            for key, value in sampling_state.items()},
                           os.path.join(scene.model_path, "weighted_fps_sampling.pth"))
                print(f"Weighted FPS: M={len(selected)} K={hyper.node_k} alpha={hyper.node_sampling_alpha}; coarse_steps={sampling_state['coarse_steps']}; fixed graph")
                del sampling_state, observed, canonical, selected, nodes, indices, weights
            if hyper.color_model == "node_residual_mlp":
                gaussians.initialize_appearance_model()
        
        opt.pruning_interval=opt.pruning_interval_fine
        opt.pruning_from_iter=opt.pruning_from_iter
        opt.densification_interval=opt.densification_interval_fine
        opt.densify_from_iter=opt.densify_from_iter_fine
        fine_start = walltime.monotonic()
        scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                            checkpoint_iterations, checkpoint, debug_from,
                            gaussians, scene, "fine", tb_writer, opt.iterations,timer)
        fine_duration = walltime.monotonic() - fine_start
    summary = {
        "color_model": hyper.color_model,
        "coarse_iterations": opt.coarse_iterations,
        "fine_iterations": 0 if args.no_fine else opt.iterations,
        "coarse_duration_s": coarse_duration,
        "fine_duration_s": fine_duration,
        "total_training_duration_s": walltime.monotonic() - training_start,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20,
    }
    with open(os.path.join(scene.model_path, "training_summary.json"), "w") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")

def prepare_output_and_logger(expname):    
    if not args.model_path:
        unique_str = expname
        args.model_path = os.path.join("./output/", unique_str)
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5000, 10000, 15000, 20000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1, 7000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    #network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, \
        args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname, args.extra_mark)

    # All done
    print("\nTraining complete.")
