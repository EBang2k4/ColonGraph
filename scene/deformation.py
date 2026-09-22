import torch
import torch.nn as nn
import torch.nn.init as init
import math
from scene.hexplane import HexPlaneField


class NodeDeformationMLP(nn.Module):
    def __init__(self, width=128, depth=3, position_pe=10, time_pe=4,
                 position_normalization="none"):
        super().__init__()
        if width < 1 or depth < 1:
            raise ValueError("MLP width and depth must be positive")
        if position_normalization not in ("none", "bbox", "raw_only", "pe_only"):
            raise ValueError("position_normalization must be none, bbox, raw_only or pe_only")
        self.position_normalization = position_normalization
        self.register_buffer("position_center", torch.zeros(3))
        self.register_buffer("position_scale", torch.ones(()))
        self.register_buffer("position_frequencies", 2.0 ** torch.arange(position_pe))
        self.register_buffer("time_frequencies", 2.0 ** torch.arange(time_pe))
        in_dim = 3 * (1 + 2 * position_pe) + 1 + 2 * time_pe
        layers = []
        for _ in range(depth):
            layers.extend([nn.Linear(in_dim, width), nn.ReLU()])
            in_dim = width
        layers.append(nn.Linear(width, 10))
        self.net = nn.Sequential(*layers)
        nn.init.normal_(self.net[-1].weight, std=1e-5)
        nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def encode(value, frequencies):
        phase = (value.unsqueeze(-1) * frequencies).flatten(1)
        return torch.cat((value, phase.sin(), phase.cos()), -1)

    @torch.no_grad()
    def configure_position_normalization(self, canonical_xyz):
        """Fit once after coarse; isotropic scaling changes MLP inputs only.

        Graph centers, distances, weights and output displacement units remain
        world-space. Both raw xyz and its positional encoding use this transform.
        """
        if canonical_xyz.ndim != 2 or canonical_xyz.shape[1] != 3 or not len(canonical_xyz):
            raise ValueError("canonical_xyz must be nonempty [N,3]")
        if not torch.isfinite(canonical_xyz).all():
            raise ValueError("canonical_xyz must be finite")
        lo, hi = canonical_xyz.amin(0), canonical_xyz.amax(0)
        self.position_center.copy_((lo + hi) / 2)
        self.position_scale.copy_(((hi - lo).amax() / 2).clamp_min(1e-8))

    def position_features(self, node_xyz):
        if self.position_normalization == "none":
            return self.encode(node_xyz, self.position_frequencies)
        u = (node_xyz - self.position_center) / self.position_scale
        if self.position_normalization == "bbox":
            return self.encode(u, self.position_frequencies)
        if self.position_normalization == "raw_only":
            return torch.cat((u, self.encode(node_xyz, self.position_frequencies)[:, 3:]), -1)
        return torch.cat((node_xyz, self.encode(u, self.position_frequencies)[:, 3:]), -1)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Old B1 snapshots have no normalization buffers. Keep strict loading for
        # every other key; missing buffers are allowed only in legacy raw mode.
        if self.position_normalization == "none":
            for key in ("position_center", "position_scale"):
                state_dict.setdefault(prefix + key, getattr(self, key).detach().clone())
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, node_xyz, time):
        # Each render has one timestamp; query exactly M nodes.
        time = torch.as_tensor(time, device=node_xyz.device, dtype=node_xyz.dtype)
        time = time.reshape(-1, 1)[:1].expand(len(node_xyz), 1)
        features = torch.cat((self.position_features(node_xyz),
                              self.encode(time, self.time_frequencies)), -1)
        delta = self.net(features)
        return delta[:, :3], delta[:, 3:7], delta[:, 7:10]


class NodeConditionedResidualAppearance(nn.Module):
    """R6 appearance branch: Gaussian embedding + blended node context + time."""

    def __init__(self, cfg, num_nodes):
        super().__init__()
        if getattr(cfg, "color_emb_dim", 16) != 16:
            raise ValueError("Node residual appearance requires the existing 16-D embedding")
        if num_nodes is None or num_nodes < 1:
            raise ValueError("Node residual appearance requires a nonempty node graph")
        self.eta = float(getattr(cfg, "color_residual_scale", 0.1))
        self.use_tanh = bool(getattr(cfg, "color_residual_tanh", True))
        self.node_appearance_dim = int(getattr(cfg, "node_appearance_dim", 8))
        if not math.isfinite(self.eta) or self.eta < 0:
            raise ValueError("color_residual_scale must be finite and nonnegative")
        if self.node_appearance_dim < 1:
            raise ValueError("node_appearance_dim must be positive")

        self.node_appearance = nn.Parameter(torch.empty(num_nodes, self.node_appearance_dim))
        self.register_buffer("time_frequencies", 2.0 ** torch.arange(6))
        input_dim = 16 + self.node_appearance_dim + 13

        # Keep construction isolated from the global training RNG.  Changing
        # the color branch must not change coarse training, FPS, or frame draws.
        with torch.random.fork_rng(devices=[]):
            self.residual_mlp = nn.Sequential(
                nn.Linear(input_dim, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, 3),
            )
            generator = torch.Generator(device="cpu").manual_seed(
                int(getattr(cfg, "appearance_seed", 0)))
            for layer in (self.residual_mlp[0], self.residual_mlp[2]):
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5), generator=generator)
                bound = 1 / math.sqrt(layer.weight.shape[1])
                nn.init.uniform_(layer.bias, -bound, bound, generator=generator)
            nn.init.normal_(self.node_appearance, mean=0.0, std=0.01, generator=generator)
            # R6 starts as the exact multiplicative identity.
            nn.init.zeros_(self.residual_mlp[-1].weight)
            nn.init.zeros_(self.residual_mlp[-1].bias)

    def node_context(self, node_indices, node_weights):
        if node_indices is None or node_weights is None:
            raise ValueError("Node residual appearance requires Gaussian-to-node associations")
        if node_indices.shape != node_weights.shape or node_indices.ndim != 2 or node_indices.shape[1] != 4:
            raise ValueError("Expected matching [N,4] node indices and weights")
        return (self.node_appearance[node_indices] * node_weights.unsqueeze(-1)).sum(1)

    def forward(self, points, times, colors, embedding, node_indices=None, node_weights=None):
        del points  # R6 has no direct XYZ input.
        times = torch.as_tensor(times, device=embedding.device, dtype=embedding.dtype).reshape(-1, 1)
        if len(times) == 1:
            times = times.expand(len(embedding), 1)
        if len(times) != len(embedding):
            raise ValueError("Expected one time or one time per Gaussian")
        node_feature = self.node_context(node_indices, node_weights)
        gamma = NodeDeformationMLP.encode(times, self.time_frequencies)
        raw = self.residual_mlp(torch.cat((embedding, node_feature, gamma), -1))
        delta = self.eta * (raw.tanh() if self.use_tanh else raw)
        return colors * (1 + delta), delta

class Deformation(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, input_ch_time=9, skips=[], args=None):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.no_grid = args.no_grid
        
        self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires)
        self.args = args
        self.create_net()
        
    def create_net(self):
        mlp_out_dim = 0
        if self.no_grid:
            self.feature_out = [nn.Linear(4,self.W)]
        else:
            self.feature_out = [nn.Linear(mlp_out_dim + self.grid.feat_dim ,self.W)]
        for i in range(self.D-1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W,self.W))
        
        self.feature_out = nn.Sequential(*self.feature_out)
        self.pos_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        self.scales_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        self.rotations_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 4))
        self.opacity_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))

        if self.args.no_color_emb:
            self.use_color_emb = False
            self.color_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        else:
            self.use_color_emb = True
            in_dim = self.W + self.args.color_emb_dim 
            self.color_deform = nn.Sequential(nn.Linear(in_dim, self.W), nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 3))
        
        for name, param in self.feature_out.named_parameters():
            if 'weight' in name:
                # init.ones_(param)
                # init.zeros_(param)
                init.uniform_(param, a=0.1, b=0.5)
        
        params_list = [self.pos_deform, self.scales_deform, self.rotations_deform, self.opacity_deform]
        for params in params_list:
            for name, param in params.named_parameters():
                if 'weight' in name:
                    # init.ones_(param)
                    init.zeros_(param)
                    # init.uniform_(param, a=0.1, b=0.5)
                if 'bias' in name: 
                    init.constant_(param, val=0)
    
    def query_time(self, rays_pts_emb, time_emb):
        if self.no_grid:
            h = torch.cat([rays_pts_emb[:,:3],time_emb[:,:1]],-1)
        else:
            grid_feature = self.grid(rays_pts_emb[:,:3], time_emb[:,:1])
            h = grid_feature
        h = self.feature_out(h)
        return h

    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity = None, color_emb = None, time_emb=None, use_color_emb = None):
        if time_emb is None:
            return self.forward_static(rays_pts_emb[:,:3])
        else:
            return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, opacity, color_emb, time_emb, use_color_emb)

    def forward_static(self, rays_pts_emb):
        grid_feature = self.grid(rays_pts_emb[:,:3])
        dx = self.static_mlp(grid_feature)
        return rays_pts_emb[:, :3] + dx

    def forward_dynamic(self,rays_pts_emb, scales_emb, rotations_emb, opacity_emb, color_emb, time_emb, use_color_emb):
        hidden = self.query_time(rays_pts_emb, time_emb).float()
        
        if self.args.no_dx:
            pts = rays_pts_emb[:, :3]
        else:
            dx = self.pos_deform(hidden)
            pts = rays_pts_emb[:, :3] + dx
        
        MAX_REL = 0.1
        # ----- SCALE -----
        if self.args.no_ds:
            scales = scales_emb[:, :3]
        else:
            ds = self.scales_deform(hidden)

            base = scales_emb[:, :3]

            # limit delta to MAX_REL of base value
            max_delta = MAX_REL * base.abs()
            ds = torch.clamp(ds, -max_delta, max_delta)

            scales = base + ds


        # ----- ROTATION -----
        if self.args.no_dr:
            rotations = rotations_emb[:, :4]
        else:
            dr = self.rotations_deform(hidden)

            base = rotations_emb[:, :4]

            # limit delta to MAX_REL of base value
            max_delta = MAX_REL * base.abs()
            dr = torch.clamp(dr, -max_delta, max_delta)

            rotations = base + dr
            
        if self.args.use_do:
            do = self.opacity_deform(hidden) 
            opacity = opacity_emb[:,:1] + do
        else:
            opacity = opacity_emb[:,:1] 
        
        ## COLOR 
        if self.args.no_dcol:
            color = color_emb[:,:3] 
            dcol = color_emb[:,:3]*0
        else:
            if self.use_color_emb:
                dcol = self.color_deform(torch.cat([hidden, use_color_emb],-1))
            else:
                dcol = self.color_deform(hidden)
            color = color_emb[:,:3] * (1+dcol)

        return pts, scales, rotations, opacity, (color, dcol)

    def forward_color_only(self, points, opacity, colors, time, color_embedding):
        """Original appearance branch, without any geometry head evaluation."""
        hidden = None
        if not self.args.no_dcol or self.args.use_do:
            hidden = self.query_time(points, time).float()
        if self.args.use_do:
            opacity = opacity + self.opacity_deform(hidden)
        if self.args.no_dcol:
            dcol = colors * 0
        elif self.use_color_emb:
            dcol = self.color_deform(torch.cat((hidden, color_embedding), -1))
        else:
            dcol = self.color_deform(hidden)
        return opacity, (colors * (1 + dcol), dcol)
    
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name:
                parameter_list.append(param)
        return parameter_list
    
    def get_grid_parameters(self):
        return list(self.grid.parameters()) 

class deform_network(nn.Module):
    def __init__(self, args) :
        super(deform_network, self).__init__()
        net_width = args.net_width
        timebase_pe = args.timebase_pe
        defor_depth= args.defor_depth
        posbase_pe= args.posebase_pe
        scale_rotation_pe = args.scale_rotation_pe
        opacity_pe = args.opacity_pe
        
        timenet_width = args.timenet_width
        timenet_output = args.timenet_output
        times_ch = 2*timebase_pe+1
        self.timenet = nn.Sequential(
            nn.Linear(times_ch, timenet_width), nn.ReLU(),
            nn.Linear(timenet_width, timenet_output))
        
        self.deformation_net = Deformation(W=net_width, D=defor_depth, input_ch=(4+3)+((4+3)*scale_rotation_pe)*2, input_ch_time=timenet_output, args=args)
        
        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(opacity_pe)]))
        self.apply(initialize_weights)
    
    def forward(self, point, scales=None, rotations=None, opacity=None, colors = None, times_sel=None, use_color_emb=None):
        if times_sel is not None:
            return self.forward_dynamic(point, scales, rotations, opacity, colors, times_sel, use_color_emb)
        else:
            return self.forward_static(point)
        
    def forward_static(self, points):
        points = self.deformation_net(points)
        return points

    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, colors=None, times_sel=None, use_color_emb=None):
        # times_emb = poc_fre(times_sel, self.time_poc)
        means3D, scales, rotations, opacity, colors = self.deformation_net( point,
                                                scales,
                                                rotations,
                                                opacity,
                                                colors,
                                                times_sel,
                                                use_color_emb)
        return means3D, scales, rotations, opacity, colors
    
    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() + list(self.timenet.parameters())
    
    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # init.constant_(m.weight, 0)
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
            # init.constant_(m.bias, 0)
