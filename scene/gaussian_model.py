#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from operator import index
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from random import randint
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, cdist
from scene.deformation import (deform_network, NodeDeformationMLP,
                               NodeConditionedResidualAppearance)
from scene.deformation_graph import SparseDeformationGraph
from scene.regulation import compute_plane_smoothness
from cuvs.neighbors import ivf_flat

from gaussian_renderer import render
from utils.loss_utils import l1_loss
from tqdm import tqdm
import time
# from gaussian_norms import compute_gauss_norm

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = lambda x: torch.exp(torch.clamp(x, -10, 10)) #torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        self.faiss_index = None
        self.games_flatten_axis = 2

    def __init__(self, sh_degree : int, args):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self.args = args
        self.color_emb_dim = 16
        self.args.color_emb_dim = self.color_emb_dim
        self._deformation = deform_network(args)
        self.deformation_mode = getattr(args, "deformation_mode", "original")
        self.color_model = getattr(args, "color_model", "hexplane")
        if self.color_model not in ("hexplane", "node_residual_mlp"):
            raise ValueError("Full training supports color_model=hexplane or node_residual_mlp")
        self.args.node_position_mode = getattr(args, "node_position_mode", "direct")
        self.args.node_position_normalization = getattr(args, "node_position_normalization", "none")
        if self.args.node_position_normalization not in ("none", "bbox", "raw_only", "pe_only"):
            raise ValueError("node_position_normalization must be none, bbox, raw_only or pe_only")
        if self.args.node_position_mode not in ("direct", "local_frame"):
            raise ValueError("node_position_mode must be direct or local_frame")
        if self.deformation_mode not in ("original", "node"):
            raise ValueError("deformation_mode must be original or node")
        self.deformation_graph = None
        self.appearance_graph = None
        self.node_mlp = None
        self.appearance_model = None
        self.profile_deformation = False
        self.last_deformation_forward_ms = None
        self.deformation_perc = args.deformation_perc

        self._deformation_table = torch.empty(0) #stores color embeddings per gaussian
        # self._features_dc = torch.empty(0)
        # self._features_rest = torch.empty(0)
        self.features = torch.empty(0)
        
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def initialize_deformation_nodes(self, num_nodes=None, k=None):
        if self.deformation_graph is not None:
            raise RuntimeError("Deformation nodes are already initialized")
        num_nodes = num_nodes or self.args.num_nodes
        k = k or self.args.node_k
        self.deformation_graph = SparseDeformationGraph.from_gaussians(self._xyz, num_nodes, k)
        self.node_mlp = NodeDeformationMLP(
            self.args.node_mlp_width, self.args.node_mlp_depth,
            self.args.posebase_pe, self.args.timebase_pe,
            self.args.node_position_normalization).to(self._xyz.device)
        self.node_mlp.configure_position_normalization(self._xyz.detach())
        print(f"Initialized deformation nodes AFTER coarse: N={len(self._xyz)}, M={num_nodes}, K={k}; fixed canonical associations")

    def initialize_appearance_model(self):
        """Create the optional fine-stage appearance branch after graph setup."""
        if self.color_model == "hexplane":
            return
        if self.color_model != "node_residual_mlp":
            raise ValueError(f"Unsupported color model: {self.color_model}")
        if self.deformation_mode != "node" or self.deformation_graph is None:
            raise ValueError("node_residual_mlp requires node deformation mode and an initialized graph")
        if self.deformation_graph.gaussian_node_indices.shape[1] != 4:
            raise ValueError("R6 requires the existing K=4 Gaussian-to-node graph")
        if self.args.use_do or self.args.no_dcol:
            raise ValueError("R6 requires use_do=False and no_dcol=False")
        graph_mode = getattr(self.args, "node_appearance_graph_mode", "shared")
        if graph_mode not in ("shared", "fixed_initial"):
            raise ValueError("node_appearance_graph_mode must be shared or fixed_initial")
        if graph_mode == "fixed_initial" and self.appearance_graph is None:
            graph = self.deformation_graph
            self.appearance_graph = SparseDeformationGraph(
                graph.node_xyz, graph.gaussian_node_indices, graph.gaussian_node_weights)
        appearance_graph = (self.appearance_graph if graph_mode == "fixed_initial"
                            else self.deformation_graph)
        self.appearance_model = NodeConditionedResidualAppearance(
            self.args, len(appearance_graph.node_xyz)).to(self._xyz.device)
        print("Initialized R6 appearance: node_dim={} input=37 eta={} tanh={} seed={} graph={}".format(
            self.args.node_appearance_dim, self.args.color_residual_scale,
            self.args.color_residual_tanh, self.args.appearance_seed, graph_mode))

    def _query_node_motion(self, time):
        if self.deformation_graph is None:
            raise RuntimeError("Initialize nodes after coarse before fine rendering")
        if len(self._xyz) != len(self.deformation_graph.gaussian_node_indices):
            raise RuntimeError("Fixed node associations require unchanged Gaussian topology")
        return self.node_mlp(self.deformation_graph.node_xyz, time)

    def deform_from_nodes(self, time):
        return self.deformation_graph.blend(*self._query_node_motion(time))

    def deform_geometry_from_nodes(self, time):
        node_motion = self._query_node_motion(time)
        dx, dr, ds = self.deformation_graph.blend(*node_motion)
        if self.args.no_dx:
            xyz = self._xyz
        elif self.args.node_position_mode == "local_frame":
            xyz = self.deformation_graph.deform_positions_local(self._xyz, node_motion[0], node_motion[1])
        else:
            xyz = self._xyz + dx
        scales = self._scaling if self.args.no_ds else self._scaling + torch.clamp(
            ds, -0.1 * self._scaling.abs(), 0.1 * self._scaling.abs())
        rotations = self._rotation if self.args.no_dr else self._rotation + torch.clamp(
            dr, -0.1 * self._rotation.abs(), 0.1 * self._rotation.abs())
        # Renderer applies the existing normalize / exp / scale clamp afterwards.
        return xyz, scales, rotations

    def deform(self, points, scales, rotations, opacity, colors, time, color_embedding):
        if self.profile_deformation:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        if self.deformation_mode == "node":
            xyz, scales, rotations = self.deform_geometry_from_nodes(time)
            if self.color_model == "node_residual_mlp":
                if self.appearance_model is None:
                    raise RuntimeError("R6 appearance was not initialized or loaded")
                appearance_graph = (self.appearance_graph
                                    if self.appearance_graph is not None
                                    else self.deformation_graph)
                appearance = self.appearance_model(
                    points, time, colors, color_embedding,
                    appearance_graph.gaussian_node_indices,
                    appearance_graph.gaussian_node_weights)
            else:
                opacity, appearance = self._deformation.deformation_net.forward_color_only(
                    points, opacity, colors, time, color_embedding)
            result = xyz, scales, rotations, opacity, appearance
        else:
            result = self._deformation(points, scales, rotations, opacity, colors, time, color_embedding)
        if self.profile_deformation:
            end.record()
            end.synchronize()
            self.last_deformation_forward_ms = start.elapsed_time(end)
        return result

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._deformation.state_dict(),
            self._deformation_table,
            # self.grid,
            # self._features_dc,
            # self._features_rest,
            self.features,
            
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.percent_dense,
            self.spatial_lr_scale,
        )
        
    def get_param(self):
        
        return [self._xyz,
            # self._features_dc,
            # self._features_rest,
            self.features,
            self._scaling,
            self._rotation,
            self._opacity,
            self._deformation.get_grid_parameters(),
            self._deformation.get_mlp_parameters()]
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
            self._xyz, 
            self._deformation_table,
            self._deformation,
            # self.grid,
            # self._features_dc, 
            # self._features_rest,
            self.features,
            
            self._scaling, 
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            xyz_gradient_accum, 
            denom,
            opt_dict, 
            self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        # features_dc = self._features_dc
        # features_rest = self._features_rest
        # return torch.cat((features_dc, features_rest), dim=1)
        return self.features
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_deformation_table(self):
        return self._deformation_table

    # def get_gaussian_normal(self, q, s, view_dir = None):
    #     """
    #     Compute the normal of the 3D Gaussian and orient it towards camera.
    #     """
    #     input_rotation_normalized = q / q.norm(p=2, dim=1, keepdim=True)
    #     cuda_normal = compute_gauss_norm(s, input_rotation_normalized, 1.0) #1.0 scale modifier
    #     N = torch.nn.functional.normalize(cuda_normal, dim=1) 
        
    #     if view_dir is not None:
    #         V = -torch.nn.functional.normalize(view_dir, dim=1)
            
    #         # normals always towards camera
    #         N_dot_V = torch.sum(N * V, dim=1, keepdim=True)  # [N, 1]
    #         N = torch.where(N_dot_V < 0, -N, N)  # Flip N if N_dot_V < 0

    #     return N

    def rerun_knn(self, new_points, k = 100, from_scratch=False):
        if from_scratch:
            xyz = self.get_xyz.detach()
            build_params = ivf_flat.IndexParams(metric="sqeuclidean")
            self.cuvs_index = ivf_flat.build(build_params, xyz)
        else:
            new_idxs = (new_points.shape[0] + torch.arange(self.get_xyz.shape[0], device='cuda'))
            self.cuvs_index = ivf_flat.extend(self.cuvs_index, new_points.detach(), new_idxs)

        _, indices = ivf_flat.search(ivf_flat.SearchParams(),
                                     self.cuvs_index, new_points.detach(),
                                     k)
        indices = torch.as_tensor(indices, device='cuda')[:, 1:]
        if from_scratch:
            self.closest_point_indices = indices
        else:
            self.closest_point_indices = torch.cat((self.closest_point_indices, indices), dim=0).cuda()

        #return self.closest_point_indices
    

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, time_line: int):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        # fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        # features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        # features[:, :3, 0 ] = fused_color
        # features[:, 3:, 1:] = 0.0
        features = torch.tensor(np.asarray(pcd.colors)).float().cuda()
        

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._deformation = self._deformation.to("cuda") 
        _deformation_table = torch.randn((self.get_xyz.shape[0], self.color_emb_dim), device="cuda")*0.01


             
        self.features = nn.Parameter(features.requires_grad_(True))
        # self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        # self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self.features = nn.Parameter(features.requires_grad_(True))

        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        #self._deformation_table = torch.gt(torch.ones((self.get_xyz.shape[0]),device="cuda"),0)

        self._deformation_table = nn.Parameter(_deformation_table.requires_grad_(True))
        self.original_normals, self.closest_point_indices = self.compute_point_cloud_normals(k=100)



    def compute_point_cloud_normals(self, k=10):
        start = time.time()

        xyz = self.get_xyz.contiguous().cuda().detach()
        
        print("Performing k-NN search...")
        start = time.time()
        build_params = ivf_flat.IndexParams(metric="sqeuclidean")
        index = ivf_flat.build(build_params, xyz)
        distances, indices = ivf_flat.search(ivf_flat.SearchParams(),
                                       index, xyz,
                                       k)
        end = time.time()
        print(f"k-NN search completed in {end - start:.2f} seconds.")

        
        k_neighbors_indices = torch.as_tensor(indices, device='cuda' )[:, 1:]
        
        # neighbors = xyz[k_neighbors_indices.long()]

        # neighbors = xyz[k_neighbors_indices] # [N, k, 3]
        # neighbors_centered = neighbors - neighbors.mean(dim=1, keepdim=True)
        
        # covariance = torch.matmul(neighbors_centered.transpose(-2, -1), neighbors_centered) / (k - 1)
        # _, eigenvectors = torch.linalg.eigh(covariance)

        # normals = eigenvectors[..., 0]
        # normals_normalized = torch.nn.functional.normalize(normals, dim=1, eps=1e-8)
        
        # all_normals = torch.cat((xyz, normals_normalized), dim=1)
        all_normals = None
        self.cuvs_index = index 
            
        return all_normals, k_neighbors_indices

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        
        if self.color_model == "node_residual_mlp" and self.appearance_model is not None:
            # Fine-stage R6 replaces all HexPlane color parameters.  Geometry
            # node MLP and R6 share the existing deformation LR schedule.
            deformation_parameters = (list(self.node_mlp.parameters()) +
                                      list(self.appearance_model.parameters()))
            grid_parameters = []
        else:
            deformation_parameters = list(self._deformation.get_mlp_parameters())
            if self.node_mlp is not None:
                deformation_parameters.extend(self.node_mlp.parameters())
            grid_parameters = list(self._deformation.get_grid_parameters())

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init*self.spatial_lr_scale, "name": "xyz"},
            {'params': deformation_parameters, 'lr': training_args.deformation_lr_init, "name": "deformation"},
            #{'params': list(self.region.parameters()), 'lr': training_args.region_lr, "name": "region"},  
            #{'params': list(self.spatial.parameters()), 'lr': training_args.spatial_lr, "name": "spatial"},            
                      
            # {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            # {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self.features], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._deformation_table], 'lr': training_args.feature_lr, "name": "deformation_table"}
        ]
        if grid_parameters:
            l.insert(2, {'params': grid_parameters, 'lr': training_args.grid_lr_init, "name": "grid"})
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.deformation_scheduler_args = get_expon_lr_func(lr_init=training_args.deformation_lr_init,
                                                    lr_final=training_args.deformation_lr_final,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps) 
        self.grid_scheduler_args = get_expon_lr_func(lr_init=training_args.grid_lr_init,
                                                    lr_final=training_args.grid_lr_final,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps) 
        
    
    def update_lr(self, name, lr):
        for g in self.optimizer.param_groups:
            if g['name']==name:
                g['lr'] = lr
        
            

    
    def _resize_parameter(self, name, shape):
        tensor = getattr(self, name)
        new_tensor = torch.zeros(shape, device=tensor.device, dtype=tensor.dtype)
        new_tensor[:tensor.shape[0]] = tensor
        new_param = nn.Parameter(new_tensor.requires_grad_(True))
        if self.optimizer is not None:
            for group in self.optimizer.param_groups:
                if group["name"] == name:
                    stored_state = self.optimizer.state.get(group['params'][0], None)
                    if stored_state is not None:
                        stored_state["exp_avg"] = torch.zeros_like(new_tensor)
                        stored_state["exp_avg_sq"] = torch.zeros_like(new_tensor)
                        del self.optimizer.state[group['params'][0]]
                        self.optimizer.state[new_param] = stored_state  # type: ignore
                    group["params"][0] = new_param
                    break
            else:
                raise ValueError(f"Parameter {name} not found in optimizer")
        setattr(self, name, new_param)
        
    def update_learning_rate(self, iteration, stage='coarse'):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr
            if  "grid" in param_group["name"]:
                lr = self.grid_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr
            elif param_group["name"] == "deformation":
                lr = self.deformation_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr
                

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        # for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
        #     l.append('f_dc_{}'.format(i))
        # for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
        #     l.append('f_rest_{}'.format(i))
        for i in range(self.features.shape[1]):
            l.append('fea_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
    
    def compute_deformation(self,time):    
        deform = self._deformation[:,:,:time].sum(dim=-1)
        xyz = self._xyz + deform
        return xyz

    def load_model(self, path):
        print("loading model from exists{}".format(path))
        node_path = os.path.join(path, "deformation_nodes.pth")
        if os.path.exists(node_path):
            state = torch.load(node_path, map_location="cuda", weights_only=True)
            # Legacy B1 checkpoints always used direct translation blending.
            self.args.node_position_mode = state["config"].get("node_position_mode", "direct")
            self.args.node_position_normalization = state["config"].get("node_position_normalization", "none")
            for key, value in state["config"].items():
                setattr(self.args, key, value)
            self.deformation_mode = "node"
            self.color_model = state["config"].get("color_model", self.color_model)
            graph = state["graph"]
            self.deformation_graph = SparseDeformationGraph(
                graph["node_xyz"], graph["gaussian_node_indices"], graph["gaussian_node_weights"])
            appearance_graph = state.get("appearance_graph")
            self.appearance_graph = (None if appearance_graph is None else
                SparseDeformationGraph(
                    appearance_graph["node_xyz"],
                    appearance_graph["gaussian_node_indices"],
                    appearance_graph["gaussian_node_weights"]))
            self.node_mlp = NodeDeformationMLP(self.args.node_mlp_width, self.args.node_mlp_depth,
                                              self.args.posebase_pe, self.args.timebase_pe,
                                              self.args.node_position_normalization).cuda()
            self.node_mlp.load_state_dict(state["mlp"])
        elif self.deformation_mode == "node":
            raise FileNotFoundError(f"Missing fixed graph checkpoint: {node_path}")
        
        weight_dict = torch.load(os.path.join(path,"deformation.pth"),map_location="cuda")
        self._deformation.load_state_dict(weight_dict)
        self._deformation = self._deformation.to("cuda")
        
        self._deformation_table = torch.gt(torch.ones((self.get_xyz.shape[0]),device="cuda"),0)
        if os.path.exists(os.path.join(path, "deformation_table.pth")):
            self._deformation_table = torch.load(os.path.join(path, "deformation_table.pth"),map_location="cuda")
            

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        if os.path.exists(os.path.join(path, "closest_point_indices.pth")):
            self.closest_point_indices = torch.load(os.path.join(path,"closest_point_indices.pth"),map_location="cuda")

        appearance_path = os.path.join(path, "appearance.pth")
        if os.path.exists(appearance_path):
            appearance_state = torch.load(appearance_path, map_location="cuda", weights_only=True)
            config = appearance_state["config"]
            self.color_model = config["color_model"]
            for key, value in config.items():
                setattr(self.args, key, value)
            self.initialize_appearance_model()
            self.appearance_model.load_state_dict(appearance_state["state_dict"], strict=True)
        

    def save_deformation(self, path):
        torch.save(self._deformation.state_dict(),os.path.join(path, "deformation.pth"))
        torch.save(self._deformation_table,os.path.join(path, "deformation_table.pth"))
        if self.deformation_graph is not None:
            node_config_keys = (
                "num_nodes", "node_k", "node_mlp_width", "node_mlp_depth",
                "posebase_pe", "timebase_pe", "node_position_mode",
                "node_position_normalization", "color_model",
                "color_residual_scale", "color_residual_tanh",
                "node_appearance_dim", "node_appearance_graph_mode", "appearance_seed",
                "node_sampling", "node_sampling_alpha",
                "node_sampling_importance_reference",
                "node_sampling_gradient_source",
                "node_importance_extra_nodes",
            )
            torch.save({"graph": self.deformation_graph.state_dict(),
                        "appearance_graph": (None if self.appearance_graph is None else
                                             self.appearance_graph.state_dict()),
                        "mlp": self.node_mlp.state_dict(),
                        "config": {key: getattr(self.args, key) for key in node_config_keys}},
                       os.path.join(path, "deformation_nodes.pth"))
        if self.appearance_model is not None:
            appearance_config = {key: getattr(self.args, key) for key in (
                "color_model", "color_residual_scale", "color_residual_tanh",
                "node_appearance_dim", "node_appearance_graph_mode", "appearance_seed")}
            torch.save({"format_version": 1, "config": appearance_config,
                        "state_dict": self.appearance_model.state_dict()},
                       os.path.join(path, "appearance.pth"))

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        # features_dc = np.zeros((xyz.shape[0], 3, 1))
        # features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        # features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        # features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        # extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        # assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        # features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        # for idx, attr_name in enumerate(extra_f_names):
        #     features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        # features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        
        fea_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("fea_")]
        fea_names = sorted(fea_names, key = lambda x: int(x.split('_')[-1]))
        features = np.zeros((xyz.shape[0], len(fea_names)))
        for idx, attr_name in enumerate(fea_names):
            features[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        # self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        # self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self.features = nn.Parameter(torch.tensor(features, dtype=torch.float, device="cuda").requires_grad_(True))
        
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        # f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        fea = self.features.detach().cpu().numpy()
        
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, fea, opacities, scale, rotation), axis=1)
        
        # attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        
    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
    
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1:
                continue
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            
            if group["name"] == "deformation_table":
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state
                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
                continue
                
            if len(group["params"]) > 1:
                continue
                
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self.features = optimizable_tensors["f_dc"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._deformation_table = optimizable_tensors["deformation_table"]
        
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
                
            if group["name"] == "deformation_table":
                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state
                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
                continue
                
            if len(group["params"]) > 1:
                continue
                
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_fea, new_opacities, new_scaling, new_rotation, new_deformation_table):
        d = {"xyz": new_xyz,
            "f_dc": new_fea,
            "opacity": new_opacities,
            "scaling" : new_scaling,
            "rotation" : new_rotation,
            "deformation_table": new_deformation_table
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self.features = optimizable_tensors["f_dc"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._deformation_table = optimizable_tensors["deformation_table"]
        
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        # if new_xyz.shape[0]>0:
        #     self.rerun_knn(new_xyz, k=100)

        # if new_xyz.shape[0]>0:
        #     new_closest_point_indices = self.find_closest_indices(new_xyz)
        #     self.closest_point_indices = torch.cat((self.closest_point_indices, new_closest_point_indices), dim=0)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        if not selected_pts_mask.any():
            return
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        # new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        # new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_features = self.features[selected_pts_mask].repeat(N, 1)
        
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_deformation_table = self._deformation_table[selected_pts_mask].repeat(N, 1)
        self.densification_postfix(new_xyz, new_features, new_opacity, new_scaling, new_rotation, new_deformation_table)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask] 
        # new_features_dc = self._features_dc[selected_pts_mask]
        # new_features_rest = self._features_rest[selected_pts_mask]
        new_features = self.features[selected_pts_mask]
        
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_deformation_table = self._deformation_table[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features, new_opacities, new_scaling, new_rotation, new_deformation_table)
    
    def prune(self, max_grad, min_opacity, extent, max_screen_size):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        self.rerun_knn(self.get_xyz, k=100, from_scratch=True)
        torch.cuda.empty_cache()

    def densify(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)
    
    def standard_constaint(self):
        
        means3D = self._xyz.detach()
        scales = self._scaling.detach()
        rotations = self._rotation.detach()
        opacity = self._opacity.detach()
        time =  torch.tensor(0).to("cuda").repeat(means3D.shape[0],1)
        means3D_deform, scales_deform, rotations_deform, _ = self._deformation(means3D, scales, rotations, opacity, time)
        position_error = (means3D_deform - means3D)**2
        rotation_error = (rotations_deform - rotations)**2 
        scaling_erorr = (scales_deform - scales)**2
        return position_error.mean() + rotation_error.mean() + scaling_erorr.mean()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def _plane_regulation(self):
        multi_res_grids = self._deformation.deformation_net.grid.grids
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids =  [0,1,3]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total

    def _time_regulation(self):
        multi_res_grids = self._deformation.deformation_net.grid.grids
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids =[2, 4, 5]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total
    
    def _l1_regulation(self):
                # model.grids is 6 x [1, rank * F_dim, reso, reso]
        multi_res_grids = self._deformation.deformation_net.grid.grids

        total = 0.0
        for grids in multi_res_grids:
            if len(grids) == 3:
                continue
            else:
                # These are the spatiotemporal grids
                spatiotemporal_grids = [2, 4, 5]
            for grid_id in spatiotemporal_grids:
                total += torch.abs(1 - grids[grid_id]).mean()
        return total

    def compute_regulation(self, time_smoothness_weight, l1_time_planes_weight, plane_tv_weight):
        return plane_tv_weight * self._plane_regulation() + time_smoothness_weight * self._time_regulation() + l1_time_planes_weight * self._l1_regulation()
