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
from utils.loss_utils import (l1_loss, ssim, lpips_loss, TV_loss,
                             canonical_xyz_laplacian)
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
                                 fps_plus_importance_sampling,
                                 build_gaussian_node_knn, validate_index_aligned_reference,
                                 position_loss_gradient)
from scene.deformation_graph import SparseDeformationGraph
from utils.adaptive_deformation_graph import AdaptiveNodeConfig, AdaptiveNodeManager
from utils.adaptive_g3_graph import AdaptiveG3NodeManager


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


def _autograd_norm(loss, parameters):
    parameters = tuple(parameters)
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True)
    total = loss.new_zeros(())
    for gradient in gradients:
        if gradient is not None:
            total = total + gradient.detach().square().sum()
    return total.sqrt().item()


def _append_arap_diagnostic(model_path, row):
    path = os.path.join(model_path, "arap_diagnostics.jsonl")
    with open(path, "a") as stream:
        stream.write(json.dumps(row) + "\n")


def _append_r6_diagnostic(model_path, row):
    path = os.path.join(model_path, "joint_r6_diagnostics.csv")
    exists = os.path.exists(path)
    with open(path, "a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _append_node_edge_diagnostic(model_path, row):
    path = os.path.join(model_path, "node_edge_diagnostics.jsonl")
    with open(path, "a") as stream:
        stream.write(json.dumps(row) + "\n")


def _append_graph_forward_diagnostic(model_path, row):
    path = os.path.join(model_path, "graph_forward_diagnostics.jsonl")
    with open(path, "a") as stream:
        stream.write(json.dumps(row) + "\n")


def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations, 
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, stage, tb_writer, train_iter, timer):
    


    first_iter = 0
    gaussians.training_setup(opt)
    fine_resume_state = (getattr(gaussians, "fine_resume_state", None)
                         if stage == "fine" else None)
    if fine_resume_state is not None:
        if gaussians.surface_children is not None:
            gaussians.add_graph_residual_optimizer_group()
            from utils.surface_children import register_optimizer
            register_optimizer(gaussians)
        saved_group_count = len(fine_resume_state["optimizer"]["param_groups"])
        current_group_count = len(gaussians.optimizer.param_groups)
        if (gaussians.graph_motion_residual is not None and
                saved_group_count == current_group_count + 1):
            gaussians.add_graph_residual_optimizer_group()
            current_group_count += 1
        if saved_group_count != current_group_count:
            if not (gaussians.graph_motion_residual is not None and
                    saved_group_count == current_group_count):
                raise ValueError("Fine optimizer groups do not match graph-forward branch")
        gaussians.optimizer.load_state_dict(fine_resume_state["optimizer"])
        gaussians.add_graph_residual_optimizer_group()
        first_iter = int(fine_resume_state["iteration"])
    else:
        gaussians.add_graph_residual_optimizer_group()
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
    arap_audit_iterations = {
        int(value) for value in str(hyper.arap_gradient_audit_iterations).split(",")
        if value.strip()
    }
    arap_generator = torch.Generator(device="cpu").manual_seed(
        int(hyper.arap_seed) + 100003 * int(hyper.experiment_seed))
    isometry_generator = torch.Generator(device="cpu").manual_seed(
        int(hyper.local_isometry_seed) + 100003 * int(hyper.experiment_seed))
    node_edge_audit_iterations = {
        int(value) for value in str(hyper.node_edge_gradient_audit_iterations).split(",")
        if value.strip()
    }
    node_edge_generator = torch.Generator(device="cpu").manual_seed(
        int(hyper.node_edge_seed) + 100003 * int(hyper.experiment_seed))
    graph_forward_monitor_iterations = {
        int(value) for value in str(hyper.graph_forward_monitor_iterations).split(",")
        if value.strip()
    }
    matched_motion_reference = None
    if stage == "fine" and (hyper.graph_forward_mode == "g4" or hyper.graph_forward_adaptive):
        with open(hyper.graph_forward_motion_reference) as stream:
            motion_payload = json.load(stream)
        if int(motion_payload["seed"]) != int(hyper.experiment_seed):
            raise ValueError("Graph matched motion reference seed differs from run seed")
        if hyper.graph_forward_adaptive and motion_payload.get("reference_mode") != "g3":
            raise ValueError("Adaptive G3 motion safety must use matched fixed G3, not G0")
        matched_motion_reference = motion_payload["motion_reference_p95"]
    adaptive_manager = None
    if stage == "fine" and hyper.adaptive_node_mode != "none":
        adaptive_config = AdaptiveNodeConfig(
            mode=hyper.adaptive_node_mode,
            start=hyper.adaptive_node_start,
            end=hyper.adaptive_node_end,
            interval=hyper.adaptive_node_interval,
            split_end=hyper.adaptive_node_split_end,
            prune_start=hyper.adaptive_node_prune_start,
            max_nodes=hyper.adaptive_node_max,
            min_nodes=hyper.adaptive_node_min,
            max_splits=hyper.adaptive_node_max_splits,
            max_prunes=hyper.adaptive_node_max_prunes,
            prune_windows=hyper.adaptive_node_prune_windows,
            prune_quantile=hyper.adaptive_node_prune_quantile,
            prune_redundancy_distance_ratio=hyper.adaptive_node_prune_redundancy_distance_ratio,
            prune_redundancy_overlap=hyper.adaptive_node_prune_redundancy_overlap,
            prune_motion_error_ratio=hyper.adaptive_node_prune_motion_error_ratio,
            prune_safe_error_ratio=hyper.adaptive_node_prune_safe_error_ratio,
            score_decay=hyper.adaptive_node_score_decay,
            running_decay=hyper.adaptive_node_running_decay,
            split_z=hyper.adaptive_node_split_z,
            min_coverage=hyper.adaptive_node_min_coverage,
            min_age=hyper.adaptive_node_min_age,
            weight_threshold=hyper.adaptive_node_weight_threshold,
            max_split_shock_ratio=hyper.adaptive_node_max_split_shock_ratio,
        )
        if hyper.graph_forward_adaptive:
            manager_class = AdaptiveG3NodeManager
            manager_kwargs = {}
            if hyper.adaptive_g3_admission != "legacy":
                from utils.candidate_expansion import CandidateExpansionManager
                manager_class = CandidateExpansionManager
                manager_kwargs["admission"] = hyper.adaptive_g3_admission
            if hyper.adaptive_g3_signal != "legacy":
                from utils.persistent_split import PersistentSplitManager
                manager_class = PersistentSplitManager
                manager_kwargs.update(signal=hyper.adaptive_g3_signal,
                    quota_reference=hyper.adaptive_g3_quota_reference,
                    depth_weight=opt.depth_weight)
                if hyper.adaptive_g3_signal == "persistent_capacity":
                    from utils.capacity_split import CapacitySplitManager
                    manager_class = CapacitySplitManager
                    manager_kwargs['capacity_reference'] = hyper.rcap_reference
                elif hyper.adaptive_g3_signal == "persistent_underres":
                    from utils.underres_split import UnderresolutionSplitManager
                    manager_class = UnderresolutionSplitManager
                    manager_kwargs['capacity_reference'] = hyper.rcap_reference
            if hyper.surface_graph_refine:
                from utils.surface_graph_refine import SurfaceGraphRefineManager
                manager_class, manager_kwargs = SurfaceGraphRefineManager, {}
            if hyper.surface_child_mode != 'none':
                from utils.surface_children import SurfaceChildManager
                manager_class, manager_kwargs = SurfaceChildManager, dict(mode=hyper.surface_child_mode)
            adaptive_manager = manager_class(
                gaussians, adaptive_config, os.path.join(scene.model_path, "adaptive_nodes"),
                hyper.adaptive_g3_score, hyper.adaptive_g3_placement, **manager_kwargs)
            if hyper.adaptive_g3_state:
                adaptive_manager.load(hyper.adaptive_g3_state)
        else:
            adaptive_manager = AdaptiveNodeManager(
                gaussians, adaptive_config, os.path.join(scene.model_path, "adaptive_nodes"))
        gaussians.adaptive_node_manager = adaptive_manager
        if fine_resume_state is not None and fine_resume_state.get('pending_split'):
            if hyper.surface_child_mode != 'none':
                from utils.exact_split_branchpoint import digest, model_digest
                if model_digest(gaussians) != fine_resume_state['model_sha256']:
                    raise RuntimeError('Surface child source model reload is not exact')
                if digest(gaussians.optimizer.state_dict()) != fine_resume_state['optimizer_sha256']:
                    raise RuntimeError('Surface child source Adam reload is not exact')
            else:
                from utils.exact_split_branchpoint import restore_manager
                restore_manager(adaptive_manager, fine_resume_state)
    video_cams = scene.getVideoCameras()
    
    if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras()
    temporal_cameras = sorted(
        viewpoint_stack, key=lambda camera: float(torch.as_tensor(camera.time).reshape(-1)[0]))
    temporal_index = {id(camera): index for index, camera in enumerate(temporal_cameras)}
    if fine_resume_state is not None:
        camera_names = [camera.image_name for camera in viewpoint_stack]
        if camera_names != fine_resume_state["train_camera_order"]:
            raise ValueError("Fine warm-up camera order differs from saved state")
        torch.set_rng_state(fine_resume_state["torch_rng"])
        torch.cuda.set_rng_state_all(fine_resume_state["cuda_rng"])
        random.setstate(fine_resume_state["python_rng"])
        np.random.set_state(fine_resume_state["numpy_rng"])
        print("Resumed exact common fine state after iteration "
              f"{int(fine_resume_state['iteration'])}", flush=True)
        # Multi-seed matched continuations share the exact same model and
        # optimizer state at the branch point, but intentionally use a new
        # fine-stage sampling trajectory.  This reset must happen *after* the
        # checkpoint RNG restore above; doing it before scene_reconstruction
        # made every requested experiment seed silently follow seed 0.
        if hyper.reset_fine_rng:
            if fine_resume_state.get('pending_split'):
                raise ValueError('Exact pre-split branchpoint forbids RNG reset')
            setup_seed(int(hyper.experiment_seed))
            print("Reset fine RNG after exact resume: "
                  f"experiment_seed={int(hyper.experiment_seed)}", flush=True)

    if stage == 'fine' and hyper.connectivity_ablation_audit:
        if hyper.reset_fine_rng or (hyper.adaptive_node_mode != 'none' and not (
                hyper.surface_graph_refine and len(gaussians.deformation_graph.node_xyz) == 2000)):
            raise ValueError('Connectivity ablation forbids adaptive nodes and RNG reset')
        from utils.connectivity_ablation import record_initial
        record_initial(gaussians,fine_resume_state,viewpoint_stack,render,pipe,background,scene.model_path)

    if fine_resume_state is not None and fine_resume_state.get('pending_split'):
        from utils.exact_split_branchpoint import digest, model_digest
        pending=int(fine_resume_state['iteration'])
        camera=next(c for c in viewpoint_stack if c.image_name==fine_resume_state['pending_camera_name'])
        rng_hash=digest(dict(torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                            python_rng=random.getstate(),numpy_rng=np.random.get_state()))
        if rng_hash!=fine_resume_state['rng_sha256']:
            raise RuntimeError('Pre-split RNG restore failed')
        with torch.no_grad():
            initial=render(camera,gaussians,pipe,background,stage='fine')
            depth_row=adaptive_manager._depth_metrics(initial['depth'],camera.original_depth.cuda().float(),getattr(camera,'mask',None))
            audit=dict(phase='before_first_split',iteration=pending,model_sha256=model_digest(gaussians),
                       optimizer_sha256=digest(gaussians.optimizer.state_dict()),rng_sha256=rng_hash,
                       next_camera_names=fine_resume_state['next_camera_names'],depth=depth_row,
                       render=initial['render'].detach().cpu(),depth_tensor=initial['depth'].detach().cpu(),
                       alpha=initial['alpha'].detach().cpu())
            torch.save(audit,os.path.join(scene.model_path,'pre_split_reload_audit.pth'))
            del initial
            adaptive_manager.collect_event_signals(pending,scene.getTrainCameras(),render,pipe,background)
            adaptive_manager.adapt(pending,camera,render,pipe,background,lpips_model)
            scene.save(pending,'fine')
        print('[EXACT_PRE_SPLIT_RESUME] pending 5k split applied, no duplicate Adam step',flush=True)
    
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
        if adaptive_manager is not None and hasattr(adaptive_manager, 'begin_iteration'):
            adaptive_manager.begin_iteration(iteration)
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

        if stage == 'fine' and (hyper.connectivity_ablation_audit or hyper.surface_child_mode != 'none'):
            with open(os.path.join(scene.model_path,'connectivity_camera_sequence.jsonl'),'a') as stream:
                stream.write(json.dumps(dict(iteration=iteration,camera_name=viewpoint_stack[idx].image_name))+'\n')

        if (stage == "fine" and (hyper.reset_fine_rng or (fine_resume_state is not None and fine_resume_state.get('pending_split'))) and
                fine_resume_state is not None and
                iteration <= int(fine_resume_state["iteration"]) + 5):
            if fine_resume_state.get('pending_split'):
                offset = iteration - int(fine_resume_state['iteration']) - 1
                if viewpoint_stack[idx].image_name != fine_resume_state['next_camera_names'][offset]:
                    raise RuntimeError('Exact branchpoint camera sampling trajectory changed after pending split')
            seed_audit_path = os.path.join(scene.model_path, "fine_seed_audit.jsonl")
            with open(seed_audit_path, "a") as stream:
                stream.write(json.dumps({
                    "iteration": int(iteration),
                    "experiment_seed": int(hyper.experiment_seed),
                    "camera_index": int(idx),
                    "camera_name": viewpoint_stack[idx].image_name,
                }) + "\n")
            
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

        ## GEOMETRY REGULARIZER
        # Coarse retains the matched KNN-99 term for every branch.  The paired
        # fine-stage experiment replaces only that term with node-level ARAP.
        loss_clusters = transformed_means.new_zeros(())
        loss_arap_raw = transformed_means.new_zeros(())
        loss_arap = transformed_means.new_zeros(())
        loss_canonical_laplacian = transformed_means.new_zeros(())
        loss_isometry_raw = transformed_means.new_zeros(())
        loss_isometry = transformed_means.new_zeros(())
        loss_node_edge_raw = transformed_means.new_zeros(())
        loss_node_edge = transformed_means.new_zeros(())
        arap_diagnostics = None
        use_fine_knn = hyper.geometry_regularizer in (
            "knn99", "knn99_arap_monitor", "knn99_arap", "knn99_node_edge")
        if stage == "coarse" or use_fine_knn:
            idx = gaussians.closest_point_indices
            cluster_pts = transformed_means[idx]
            cluster_means = cluster_pts.mean(dim=1)
            center_points = transformed_means[
                torch.arange(cluster_means.shape[0], device=cluster_means.device)]
            per_point_sq_dist = ((center_points - cluster_means) ** 2).sum(dim=1)
            loss_clusters = opt.knn_weight * per_point_sq_dist.mean()
            loss += loss_clusters
        elif stage == "fine" and hyper.geometry_regularizer in ("arap", "arap_iso", "iso"):
            loss_canonical_laplacian = (
                float(hyper.canonical_laplacian_weight) *
                canonical_xyz_laplacian(
                    gaussians._xyz, gaussians.closest_point_indices))
            loss += loss_canonical_laplacian

        needs_arap = (stage == "fine" and (
            hyper.geometry_regularizer in ("arap", "knn99_arap", "arap_iso") or
            (hyper.geometry_regularizer == "knn99_arap_monitor" and
             iteration in arap_audit_iterations)))
        if needs_arap:
            if gaussians.node_arap_graph is None:
                raise RuntimeError("ARAP regularizer requested without an initialized graph")
            t2_index = int(torch.randint(
                len(viewpoint_stack), (1,), generator=arap_generator).item())
            t1 = viewpoint_cams[0].time
            t2 = viewpoint_stack[t2_index].time
            nodes_t1 = gaussians.node_positions_at_time(t1)
            nodes_t2 = gaussians.node_positions_at_time(t2)
            loss_arap_raw, arap_diagnostics = gaussians.node_arap_graph(
                nodes_t1, nodes_t2, return_diagnostics=True)
            if hyper.geometry_regularizer in ("arap", "knn99_arap", "arap_iso"):
                loss_arap = float(hyper.arap_weight) * loss_arap_raw
                loss += loss_arap

        if stage == "fine" and hyper.geometry_regularizer in ("arap_iso", "iso"):
            graph = gaussians.gaussian_isometry_graph
            if graph is None:
                raise RuntimeError("Local Gaussian isometry requested without graph")
            t1_index = idx
            window = int(hyper.local_isometry_temporal_window)
            candidates = [index for index in range(
                max(0, t1_index - window), min(len(viewpoint_stack), t1_index + window + 1))
                if index != t1_index]
            if not candidates:
                raise RuntimeError("No valid nearby timestamp for local isometry")
            selected = int(torch.randint(
                len(candidates), (1,), generator=isometry_generator).item())
            t2_iso = viewpoint_stack[candidates[selected]].time
            xyz_t2, _, _ = gaussians.deform_geometry_from_nodes(t2_iso)
            loss_isometry_raw = graph(transformed_means, xyz_t2)
            loss_isometry = float(hyper.local_isometry_weight) * loss_isometry_raw
            loss += loss_isometry

        needs_node_edge = (stage == "fine" and
                           hyper.geometry_regularizer == "knn99_node_edge")
        if needs_node_edge:
            graph = gaussians.node_edge_graph
            if graph is None:
                raise RuntimeError("Node-edge loss requested without a fixed graph")
            t1_temporal_index = temporal_index[id(viewpoint_cams[0])]
            window = 4
            candidates = [index for index in range(
                max(0, t1_temporal_index-window),
                min(len(temporal_cameras), t1_temporal_index+window+1))
                if index != t1_temporal_index]
            selected = int(torch.randint(
                len(candidates), (1,), generator=node_edge_generator).item())
            t1_edge = viewpoint_cams[0].time
            t2_edge = temporal_cameras[candidates[selected]].time
            nodes_t1_edge = gaussians.node_positions_at_time(t1_edge)
            nodes_t2_edge = gaussians.node_positions_at_time(t2_edge)
            loss_node_edge_raw = graph(nodes_t1_edge, nodes_t2_edge)
            if iteration > int(hyper.node_edge_start_iteration):
                loss_node_edge = float(hyper.node_edge_weight) * loss_node_edge_raw
                loss += loss_node_edge

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

        reconstruction_loss = (Ll1 + depth_loss + loss_dcol +
                               loss_color_smooth + tv_loss)
        if needs_node_edge and iteration in node_edge_audit_iterations:
            deformation_parameters = tuple(gaussians.node_mlp.parameters())
            reconstruction_gradient_norm = _autograd_norm(
                reconstruction_loss, deformation_parameters)
            knn99_gradient_norm = _autograd_norm(
                loss_clusters, deformation_parameters)
            edge_raw_gradient_norm = _autograd_norm(
                loss_node_edge_raw, deformation_parameters)
            reconstruction_denominator = max(reconstruction_gradient_norm, 1e-20)
            row = {
                "iteration": int(iteration),
                "topology": gaussians.node_edge_graph.topology,
                "t1": float(torch.as_tensor(t1_edge).reshape(-1)[0]),
                "t2": float(torch.as_tensor(t2_edge).reshape(-1)[0]),
                "frame_distance": abs(
                    candidates[selected] - t1_temporal_index),
                "reconstruction_gradient_norm": reconstruction_gradient_norm,
                "knn99_weighted_gradient_norm": knn99_gradient_norm,
                "knn99_weighted_gradient_ratio": (
                    knn99_gradient_norm / reconstruction_denominator),
                "node_edge_raw_loss": float(loss_node_edge_raw.detach()),
                "node_edge_weight": float(hyper.node_edge_weight),
                "node_edge_raw_gradient_norm": edge_raw_gradient_norm,
                "node_edge_weighted_gradient_ratio": (
                    float(hyper.node_edge_weight) * edge_raw_gradient_norm /
                    reconstruction_denominator),
            }
            _append_node_edge_diagnostic(scene.model_path, row)
            print("[NODE_EDGE_GRADIENT_AUDIT] " + json.dumps(row), flush=True)
        if needs_arap and iteration in arap_audit_iterations:
            deformation_parameters = tuple(gaussians.node_mlp.parameters())
            reconstruction_gradient_norm = _autograd_norm(
                reconstruction_loss, deformation_parameters)
            arap_raw_gradient_norm = _autograd_norm(
                loss_arap_raw, deformation_parameters)
            raw_ratio = arap_raw_gradient_norm / max(reconstruction_gradient_norm, 1e-20)
            row = {
                "iteration": int(iteration),
                "experiment_seed": int(hyper.experiment_seed),
                "regularizer": hyper.geometry_regularizer,
                "t1": float(torch.as_tensor(t1).reshape(-1)[0]),
                "t2": float(torch.as_tensor(t2).reshape(-1)[0]),
                "arap_raw_loss": float(loss_arap_raw.detach()),
                "arap_weight": float(hyper.arap_weight),
                "reconstruction_gradient_norm": reconstruction_gradient_norm,
                "arap_raw_gradient_norm": arap_raw_gradient_norm,
                "raw_gradient_ratio": raw_ratio,
                "weighted_gradient_ratio": float(hyper.arap_weight) * raw_ratio,
                "reflection_ratio": arap_diagnostics.reflection_ratio,
                "singular_value_min": float(arap_diagnostics.singular_values.min()),
                "singular_value_median": float(arap_diagnostics.singular_values.median()),
                "per_node_loss_p95": float(torch.quantile(
                    arap_diagnostics.per_node_loss, 0.95)),
            }
            _append_arap_diagnostic(scene.model_path, row)
            print("[ARAP_GRADIENT_AUDIT] " + json.dumps(row), flush=True)

        if mask:
            psnr_ = psnr(images_concealing, gt_images, masks).mean().double()      
        else:
            psnr_ = psnr(images_concealing, gt_images).mean().double()  
        
        importance_tracker = getattr(gaussians, "coarse_gradient_importance", None)
        depth_position_gradient = None
        if (stage == "coarse" and importance_tracker is not None and
                importance_tracker.gradient_source == "depth"):
            depth_position_gradient = position_loss_gradient(depth_loss, gaussians._xyz)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n❌ Loss NaN/Inf at iteration {iteration}")
            if gaussians.deformation_mode == "node":
                raise FloatingPointError(f"Non-finite {stage} loss at iteration {iteration}")

        if stage == "fine" and (hyper.graph_forward_mode == "g4" or hyper.graph_forward_adaptive):
            if (not torch.isfinite(rendered_images).all() or
                    not torch.isfinite(rendered_depths).all() or
                    not torch.isfinite(transformed_means).all()):
                raise FloatingPointError(f"Non-finite G4 forward at iteration {iteration}")
            if not visibility_filter.any():
                raise RuntimeError(f"G4 visibility collapse at iteration {iteration}")
            if (rendered_images.detach().abs().mean() < 1e-3 and
                    (rendered_images.detach().abs() < (1 / 255)).float().mean() > .999):
                raise RuntimeError(f"G4 black render at iteration {iteration}")

        adaptive_render_gradient = None
        if (adaptive_manager is not None and iteration <= adaptive_manager.config.end and
                getattr(adaptive_manager, 'collect_online_gradient', True)):
            signal_loss = (depth_loss if getattr(adaptive_manager, 'geometry_signal', '') == 'depth' else Ll1)
            if signal_loss.requires_grad:
                gradients = torch.autograd.grad(
                    signal_loss, transformed_means_list, retain_graph=True, allow_unused=False)
                adaptive_render_gradient = torch.stack(gradients, dim=0).mean(dim=0).detach()
            
        loss.backward()
        if adaptive_manager is not None and adaptive_render_gradient is not None:
            adaptive_manager.update_statistics(adaptive_render_gradient, visibility_filter)
        if stage == "coarse" and importance_tracker is not None:
            gradient = (depth_position_gradient if importance_tracker.gradient_source == "depth"
                        else gaussians._xyz.grad)
            importance_tracker.update(gradient, visibility_filter)
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
                "arap_raw_loss": loss_arap_raw.item(),
                "arap_weighted_loss": loss_arap.item(),
                "canonical_laplacian_loss": loss_canonical_laplacian.item(),
                "local_isometry_raw_loss": loss_isometry_raw.item(),
                "local_isometry_weighted_loss": loss_isometry.item(),
                "node_edge_raw_loss": loss_node_edge_raw.item(),
                "node_edge_weighted_loss": loss_node_edge.item(),
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
            if (stage == "fine" and hyper.graph_forward_mode != "none" and
                    iteration in graph_forward_monitor_iterations):
                reference = gaussians.graph_forward_reference_data
                if reference is None:
                    raise RuntimeError("Graph-forward monitoring requires its saved reference")
                graph_diagnostic = gaussians.graph_forward_diagnostics(
                    reference["trajectory_times"])
                mean_rgb = float(rendered_images.detach().abs().mean())
                black_fraction = float(
                    (rendered_images.detach().abs() < (1/255)).float().mean())
                visible_fraction = float(visibility_filter.float().mean())
                reference_motion = float(
                    (matched_motion_reference if matched_motion_reference is not None else
                     reference["motion_reference_p95"])[str(iteration)])
                graph_diagnostic.update({
                    "iteration": int(iteration),
                    "mode": hyper.graph_forward_mode,
                    "visible_gaussian_fraction": visible_fraction,
                    "mean_abs_render_rgb": mean_rgb,
                    "black_pixel_fraction": black_fraction,
                    "reference_motion_p95": reference_motion,
                    "motion_p95_ratio": (
                        graph_diagnostic["motion"]["p95"] / max(reference_motion, 1e-12)),
                })
                _append_graph_forward_diagnostic(scene.model_path, graph_diagnostic)
                print("[GRAPH_FORWARD_DIAGNOSTIC] " +
                      json.dumps(graph_diagnostic), flush=True)
                failures = []
                if not np.isfinite(graph_diagnostic["motion"]["p95"]):
                    failures.append("nonfinite_motion")
                if visible_fraction < 1e-5:
                    failures.append("visible_gaussian_collapse")
                if mean_rgb < 1e-3 and black_fraction > .999:
                    failures.append("almost_fully_black")
                if graph_diagnostic["motion_p95_ratio"] > 2:
                    failures.append("motion_p95_above_2x_reference")
                if failures:
                    raise RuntimeError(
                        "Graph-forward stability monitor aborted: " + ",".join(failures))
            if (stage == "fine" and (hyper.graph_forward_mode == "g4" or hyper.graph_forward_adaptive) and
                    (iteration == first_iter or iteration % 250 == 0)):
                # Safety only: intermediate G0 envelopes interpolate the saved
                # matched checkpoints; requested 5k/10k/20k/30k gates above are exact.
                keys = sorted(int(key) for key in matched_motion_reference)
                limits = [matched_motion_reference[str(key)] for key in keys]
                baseline_p95 = float(np.interp(iteration, keys, limits))
                with torch.no_grad():
                    rows = [gaussians._query_node_motion(t)[0] for t in
                            gaussians.graph_forward_reference_data["trajectory_times"]]
                    magnitudes = torch.stack(rows).norm(dim=-1).flatten()
                    actual_p95 = float(torch.quantile(magnitudes, .95))
                if not np.isfinite(actual_p95) or actual_p95 > 2 * baseline_p95:
                    raise RuntimeError(
                        f"G4 motion safety abort at {iteration}: {actual_p95} > 2x {baseline_p95}")
            
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, 
                            (pipe, background, 1.,))
            defer_adaptive_save = (adaptive_manager is not None and
                                   adaptive_manager.is_event(iteration))
            if (iteration in saving_iterations) and not defer_adaptive_save:
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
            if iteration < train_iter or hyper.surface_child_mode != 'none':
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if adaptive_manager is not None and adaptive_manager.is_event(iteration):
                if hyper.pre_split_branchpoint_stop:
                    from utils.exact_split_branchpoint import save_branchpoint
                    source_render=render(viewpoint_cams[0],gaussians,pipe,background,stage='fine')
                    save_branchpoint(hyper.pre_split_branchpoint_stop,scene,gaussians,
                        adaptive_manager,viewpoint_stack,viewpoint_cams[0],iteration,source_render)
                    del source_render
                    gaussians.pre_split_branchpoint_saved_at=iteration
                    progress_bar.close()
                    print('[EXACT_PRE_SPLIT_SAVED] stopped BEFORE split, full 30k LR/Adam schedule retained',flush=True)
                    return
                if hasattr(adaptive_manager, 'collect_event_signals'):
                    adaptive_manager.collect_event_signals(
                        iteration, scene.getTrainCameras(), render, pipe, background)
                adaptive_manager.adapt(
                    iteration, viewpoint_cams[0], render, pipe, background, lpips_model)
                if iteration in saving_iterations:
                    print("\n[ITER {}] Saving post-adaptation Gaussians".format(iteration))
                    scene.save(iteration, stage)

            if adaptive_manager is not None:
                adaptive_manager.record_motion_diagnostics(iteration)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    if adaptive_manager is not None:
        adaptive_manager.save(
            os.path.join(adaptive_manager.output_dir, "adaptive_state_final.pth"), train_iter)
    if stage == "fine" and hyper.fine_stage_state_save:
        state_path = hyper.fine_stage_state_save
        directory = os.path.dirname(state_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        exact_extra = {}
        if hyper.surface_child_mode != 'none':
            from utils.exact_split_branchpoint import RAW_NAMES
            exact_extra = dict(exact_raw={k:getattr(gaussians,k).detach().clone() for k in RAW_NAMES},
                               active_sh_degree=gaussians.active_sh_degree, spatial_lr_scale=gaussians.spatial_lr_scale)
        torch.save({
            **exact_extra,
            "format_version": 1,
            "iteration": int(train_iter),
            "optimizer": gaussians.optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "train_camera_order": [camera.image_name for camera in viewpoint_stack],
        }, state_path)
        print(f"Saved exact fine-stage optimizer/RNG state: {state_path}", flush=True)


def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname, extra_mark):
    if hyper.surface_child_mode not in ('none', 'control', 'adaptive', 'shadow'):
        raise ValueError('Unknown surface_child_mode')
    if hyper.surface_child_mode != 'none':
        if (not hyper.graph_forward_adaptive or hyper.surface_graph_refine or
                hyper.adaptive_node_max != 2535 or hyper.node_position_mode != 'direct'):
            raise ValueError('Surface children require G3 adaptive plumbing, direct mode, max2535; no edge splitting')
    importance_sampling = hyper.node_sampling in ("weighted_fps", "fps_plus_importance")
    if hyper.node_sampling not in ("fps", "weighted_fps", "fps_plus_importance"):
        raise ValueError("node_sampling must be fps, weighted_fps, or fps_plus_importance")
    if importance_sampling and (hyper.deformation_mode != "node" or opt.coarse_iterations != 100):
        raise ValueError("Importance sampling requires node mode and 100 coarse iterations")
    if hyper.node_sampling_importance_reference and not importance_sampling:
        raise ValueError("An importance reference requires an importance sampling mode")
    if hyper.node_sampling == "fps_plus_importance" and hyper.node_importance_extra_nodes <= 0:
        raise ValueError("fps_plus_importance requires node_importance_extra_nodes > 0")
    if hyper.node_sampling_gradient_source not in ("total", "depth"):
        raise ValueError("node_sampling_gradient_source must be total or depth")
    if hyper.geometry_regularizer not in (
            "knn99", "knn99_arap_monitor", "arap", "knn99_arap", "arap_iso", "iso",
            "knn99_node_edge"):
        raise ValueError("Unsupported geometry_regularizer")
    if hyper.geometry_regularizer != "knn99":
        if hyper.deformation_mode != "node":
            raise ValueError("Node ARAP requires deformation_mode=node")
        if hyper.arap_k != 6:
            raise ValueError("Revised ARAP v1 requires canonical K_train=6")
        if hyper.arap_weight < 0 or not np.isfinite(hyper.arap_weight):
            raise ValueError("arap_weight must be finite and nonnegative")
        if (hyper.canonical_laplacian_weight < 0 or
                not np.isfinite(hyper.canonical_laplacian_weight)):
            raise ValueError("canonical_laplacian_weight must be finite and nonnegative")
    if hyper.geometry_regularizer in ("arap", "knn99_arap", "arap_iso", "iso") and hyper.adaptive_node_mode != "none":
        raise ValueError("Run and gate fixed-node A0 before implementing adaptive A1 ARAP")
    if hyper.geometry_regularizer in ("arap_iso", "iso"):
        if hyper.local_isometry_k != 12 or hyper.local_isometry_min_degree != 4:
            raise ValueError("E2 requires Gaussian KNN-12 candidates and minimum degree 4")
        if hyper.local_isometry_temporal_window != 4:
            raise ValueError("E2 requires the fixed +/-4-frame temporal window")
        if hyper.local_isometry_weight < 0 or not np.isfinite(hyper.local_isometry_weight):
            raise ValueError("local_isometry_weight must be finite and nonnegative")
    if hyper.geometry_regularizer == "knn99_node_edge":
        if not hyper.node_edge_graph_reference:
            raise ValueError("knn99_node_edge requires a fixed graph reference")
        if hyper.node_edge_weight <= 0 or not np.isfinite(hyper.node_edge_weight):
            raise ValueError("node_edge_weight must be finite and positive")
        if hyper.node_edge_start_iteration != 2000:
            raise ValueError("Connectivity study requires the common 2k warm-up")
        if hyper.arap_weight != 0 or hyper.local_isometry_weight != 0:
            raise ValueError("Connectivity study disables ARAP and Gaussian isometry")
        if hyper.canonical_laplacian_weight != 0:
            raise ValueError("Connectivity study disables standalone canonical Laplacian")
        if not hyper.fine_warmup_reference or not hyper.fine_warmup_state:
            raise ValueError("Connectivity study requires the exact common 2k state")
    elif hyper.node_edge_graph_reference:
        raise ValueError("node_edge_graph_reference is only valid for knn99_node_edge")
    if hyper.graph_forward_mode not in ("none", "g1", "g2", "g3", "g4"):
        raise ValueError("graph_forward_mode must be none, g1, g2, g3 or g4")
    if hyper.graph_forward_mode != "none":
        if hyper.geometry_regularizer != "knn99":
            raise ValueError("Graph-forward study retains only KNN-99")
        if not hyper.graph_forward_reference:
            raise ValueError("Graph-forward study requires the fixed E4-derived reference")
        if not hyper.fine_warmup_reference or not hyper.fine_warmup_state:
            raise ValueError("Graph-forward study requires the exact common 2k state")
        if hyper.arap_weight != 0 or hyper.node_edge_weight != 0 or hyper.local_isometry_weight != 0:
            raise ValueError("Graph-forward study disables ARAP/edge/isometry losses")
        if hyper.canonical_laplacian_weight != 0:
            raise ValueError("Graph-forward study disables standalone canonical Laplacian")
        if hyper.adaptive_node_mode != "none" and not hyper.graph_forward_adaptive:
            raise ValueError("Graph-forward study disables adaptive nodes")
        if hyper.graph_forward_adaptive:
            if hyper.surface_graph_refine and hyper.adaptive_node_max != 2535:
                raise ValueError('G4-SurfRefine requires the fixed maximum2535 budget')
            if hyper.graph_forward_mode != "g3" or hyper.adaptive_node_mode != "split":
                raise ValueError("Adaptive graph-forward study requires G3 split-only")
            if not hyper.graph_forward_motion_reference:
                raise ValueError("Adaptive G3 requires matched fixed-G3 motion envelope")
            if (hyper.adaptive_node_start, hyper.adaptive_node_end,
                    hyper.adaptive_node_interval) != (5000, 20000, 2500):
                raise ValueError("Adaptive G3 keeps events 5k..20k every 2500")
            if hyper.color_residual_scale != .5:
                raise ValueError("Adaptive G3 retains R6 eta=.5")
            if hyper.pre_split_branchpoint_stop and (hyper.adaptive_g3_signal != 'rgb_matched' or hyper.adaptive_g3_admission != 'D3' or opt.iterations != 30000):
                raise ValueError('Common pre-split source must use R0/D3 and the original 30k schedule')
            if hyper.adaptive_g3_signal in ('persistent_capacity', 'persistent_underres') and not hyper.rcap_reference:
                raise ValueError('Rcap requires the frozen train-only capacity calibration')
        if hyper.node_appearance_graph_mode != "fixed_initial":
            raise ValueError("Graph-forward study requires the fixed 2k appearance graph")
        if hyper.graph_forward_hidden_dim != 64:
            raise ValueError("G2/G3 use the preregistered 64-wide residual decoder")
        if hyper.graph_forward_mode == "g4":
            if hyper.graph_forward_cancellation_alpha != 0.10:
                raise ValueError("G4 uses preregistered cancellation alpha=0.10")
            if not hyper.graph_forward_motion_reference:
                raise ValueError("G4 requires its seed-matched G0 motion reference")
    elif hyper.graph_forward_reference:
        raise ValueError("graph_forward_reference requires G1, G2 or G3")
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
    if hyper.adaptive_node_mode != "none":
        if hyper.deformation_mode != "node" or hyper.node_sampling != "fps":
            raise ValueError("Adaptive graph must start from the 2,000-node Standard FPS backbone")
        if hyper.num_nodes != 2000 or hyper.node_k != 4:
            raise ValueError("Adaptive graph requires exactly 2,000 initial nodes and K=4")
        if hyper.color_model != "node_residual_mlp" or hyper.node_appearance_dim != 8:
            raise ValueError("Adaptive graph experiment requires the fixed R6 8-D appearance")
    tb_writer = prepare_output_and_logger(expname)
    training_start = walltime.monotonic()
    torch.cuda.reset_peak_memory_stats()
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, load_coarse=args.no_fine)
    if hyper.gaussian_knn_reference:
        knn = torch.load(hyper.gaussian_knn_reference, map_location=gaussians._xyz.device,
                         weights_only=True)
        if knn.shape != gaussians.closest_point_indices.shape:
            raise ValueError("Gaussian KNN-99 reference shape does not match this scene")
        gaussians.closest_point_indices = knn.to(
            device=gaussians._xyz.device, dtype=gaussians.closest_point_indices.dtype)
        print(f"Gaussian KNN-99: loaded matched reference {hyper.gaussian_knn_reference}", flush=True)
    if hyper.gaussian_knn_save:
        directory = os.path.dirname(hyper.gaussian_knn_save)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(gaussians.closest_point_indices.detach().cpu(), hyper.gaussian_knn_save)
        print(f"Gaussian KNN-99: saved matched reference {hyper.gaussian_knn_save}", flush=True)
    if importance_sampling and not hyper.node_sampling_importance_reference:
        gaussians.coarse_gradient_importance = CoarseGradientImportance(
            gaussians._xyz, hyper.node_sampling_gradient_source)
    timer.start()
    coarse_start = walltime.monotonic()
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                             checkpoint_iterations, checkpoint, debug_from,
                             gaussians, scene, "coarse", tb_writer, opt.coarse_iterations,timer)
    coarse_duration = walltime.monotonic() - coarse_start
    sampling_state = None
    if importance_sampling:
        if hyper.node_sampling_importance_reference:
            reference_path = hyper.node_sampling_importance_reference
            sampling_state = torch.load(reference_path, map_location=gaussians._xyz.device,
                                        weights_only=True)
            required = {"grad_sum", "grad_sq_sum", "grad_count", "importance",
                        "coarse_steps", "observed_post_coarse_xyz", "canonical_xyz"}
            if hyper.node_sampling == "fps_plus_importance":
                required.add("raw_score")
            if not required.issubset(sampling_state):
                raise ValueError(f"Incomplete Weighted FPS importance reference: {reference_path}")
            if sampling_state["importance"].shape != (len(gaussians._xyz),):
                raise ValueError("Importance reference Gaussian topology does not match")
            source = sampling_state.get("gradient_source", "total")
            if source != hyper.node_sampling_gradient_source:
                raise ValueError(f"Importance reference source is {source}, requested {hyper.node_sampling_gradient_source}")
            sampling_state["gradient_source"] = source
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
                if hyper.node_sampling == "weighted_fps":
                    selected = weighted_farthest_point_sampling(
                        canonical, hyper.num_nodes, sampling_state["importance"], hyper.node_sampling_alpha)
                else:
                    selected = fps_plus_importance_sampling(
                        canonical, hyper.num_nodes, hyper.node_importance_extra_nodes,
                        sampling_state["importance"], sampling_state["raw_score"])
                nodes = canonical[selected]
                indices, weights = build_gaussian_node_knn(canonical, nodes, hyper.node_k)
                gaussians.deformation_graph = SparseDeformationGraph(nodes, indices, weights)
                sampling_state.update({"selected_indices": selected, "canonical_xyz": canonical,
                                       "alpha": hyper.node_sampling_alpha, "first_node_index": 0,
                                       "standard_nodes": hyper.num_nodes,
                                       "importance_extra_nodes": hyper.node_importance_extra_nodes,
                                       "effective_num_nodes": len(selected),
                                       "reference_max_xyz_difference": (observed-canonical).abs().max().item()})
                torch.save({key: value.detach().cpu() if torch.is_tensor(value) else value
                            for key, value in sampling_state.items()},
                           os.path.join(scene.model_path, "weighted_fps_sampling.pth"))
                print(f"Node sampling {hyper.node_sampling}: M={len(selected)} K={hyper.node_k} "
                      f"alpha={hyper.node_sampling_alpha}; coarse_steps={sampling_state['coarse_steps']}; fixed graph")
                del sampling_state, observed, canonical, selected, nodes, indices, weights
            if hyper.color_model == "node_residual_mlp":
                gaussians.initialize_appearance_model()
            if hyper.fine_warmup_reference:
                reference = hyper.fine_warmup_reference
                reference_name = os.path.basename(os.path.normpath(reference))
                if not reference_name.startswith("iteration_"):
                    raise ValueError("Fine resume reference must be an iteration snapshot")
                reference_iteration = int(reference_name.split("_", 1)[1])
                requested_config = dict(vars(hyper))
                gaussians.load_ply(os.path.join(reference, "point_cloud.ply"))
                gaussians.load_model(reference)
                for key, value in requested_config.items():
                    setattr(hyper, key, value)
                if len(gaussians.deformation_graph.node_xyz) != 2000 and not (
                        hyper.graph_forward_adaptive and hyper.adaptive_g3_state):
                    raise ValueError("Common warm-up must contain exactly 2000 nodes")
                if (gaussians.appearance_graph is None or
                        len(gaussians.appearance_graph.node_xyz) != 2000):
                    raise ValueError("Common warm-up appearance graph must remain fixed at 2000")
                expected_knn = torch.load(
                    hyper.gaussian_knn_reference, map_location=gaussians._xyz.device,
                    weights_only=True)
                loaded_knn = getattr(gaussians, "closest_point_indices", None)
                if loaded_knn is not None and not torch.equal(loaded_knn, expected_knn):
                    raise ValueError("Common warm-up Gaussian KNN-99 differs from matched reference")
                gaussians.closest_point_indices = expected_knn
                gaussians.fine_resume_state = torch.load(
                    hyper.fine_warmup_state, map_location="cpu",
                    weights_only=False)
                if hyper.adaptive_g3_signal in ('persistent_capacity', 'persistent_underres') and not gaussians.fine_resume_state.get('pending_split'):
                    raise ValueError('Rcap must branch from the exact pre-first-split state')
                if gaussians.fine_resume_state.get('pending_split'):
                    from utils.exact_split_branchpoint import restore_raw
                    restore_raw(gaussians,gaussians.fine_resume_state)
                elif hyper.surface_child_mode != 'none':
                    state = gaussians.fine_resume_state
                    with torch.no_grad():
                        for key, value in state['exact_raw'].items():
                            getattr(gaussians,key).copy_(value.to(getattr(gaussians,key)))
                    gaussians.active_sh_degree = state['active_sh_degree']
                    gaussians.spatial_lr_scale = state['spatial_lr_scale']
                if int(gaussians.fine_resume_state["iteration"]) != reference_iteration:
                    raise ValueError("Fine optimizer/RNG state does not match snapshot iteration")
                print(f"Loaded exact fine model/optimizer/RNG state at {reference_iteration} from {reference}",
                      flush=True)
            if hyper.geometry_regularizer == "knn99_node_edge":
                gaussians.load_node_edge_graph(hyper.node_edge_graph_reference)
            if hyper.graph_forward_mode != "none":
                gaussians.initialize_graph_forward(hyper.graph_forward_reference)
            if hyper.geometry_regularizer in (
                    "knn99_arap_monitor", "arap", "knn99_arap", "arap_iso"):
                gaussians.initialize_arap_graph(hyper.arap_k)
            if hyper.geometry_regularizer in ("arap_iso", "iso"):
                gaussians.initialize_gaussian_isometry_graph(
                    k=hyper.local_isometry_k,
                    minimum_degree=hyper.local_isometry_min_degree)
        
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
        "fine_iterations": 0 if args.no_fine else getattr(gaussians,'pre_split_branchpoint_saved_at',opt.iterations),
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
