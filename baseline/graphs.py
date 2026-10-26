"""Create G3, J8 and M2 references from the current training run only."""
import torch,numpy as np
from baseline.common import write
from utils.exact_split_branchpoint import digest

@torch.no_grad()
def build_g3(m,cameras,out):
    from baseline.topology import build_e4
    from utils.graph_forward import repair_gaussian_k4_minimal
    g=m.deformation_graph;x=m._xyz.detach();z=g.node_xyz
    graph=build_e4(x.cpu(),z.cpu(),g.gaussian_node_indices.cpu(),g.gaussian_node_weights.cpu(),m.closest_point_indices.cpu())
    edges=graph.edge_indices.to(x.device);weights=graph.edge_weights.to(x.device)
    ids,w,changed,fallback,audit=repair_gaussian_k4_minimal(x,z,g.gaussian_node_indices,g.gaussian_node_weights,edges)
    cameras=sorted(cameras,key=lambda cam:float(cam.time));selected=np.rint(np.linspace(0,len(cameras)-1,16)).astype(int)
    times=x.new_tensor([float(cameras[j].time) for j in selected]);motion=torch.stack([m.node_positions_at_time(t)-z for t in times])
    delta=motion[:,edges[0]]-motion[:,edges[1]];distance=delta.square().sum(-1).mean(0).sqrt();sigma=distance.median().clamp_min(1e-8)
    gated=(weights*torch.exp(-distance.square()/(2*sigma.square()))).clamp_min(1e-12)
    payload=dict(format_version=1,canonical_node_xyz=z,original_k4_indices=g.gaussian_node_indices,original_k4_weights=g.gaussian_node_weights,
        graph_k4_indices=ids,graph_k4_weights=w,changed_mask=changed,fallback_mask=fallback,surface_edge_indices=edges,
        surface_edge_weights=weights,trajectory_gated_edge_weights=gated,trajectory_times=times,motion_reference_p95={},audit=audit)
    path=out/'setup/g3.pth';path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({k:v.cpu() if torch.is_tensor(v) else v for k,v in payload.items()},path)
    m.args.graph_forward_mode='g3';m.args.graph_forward_reference=str(path);m.initialize_graph_forward(str(path));m.add_graph_residual_optimizer_group()
    write(out/'setup/g3_audit.json',dict(TRAIN_only=True,source='current2k state',nodes=len(z),edges=edges.shape[1],**audit))

@torch.no_grad()
def build_j8(m,out,seed):
    from utils.graph_forward import _dense_adjacency
    from utils.geometry_blending import graph_spacing
    from utils.ggas_v2 import connected_growth,warm_start_prior
    from utils.ggas import initialize
    g=m.deformation_graph;x=m._xyz.detach();nodes=g.node_xyz;e=m.graph_motion_residual.edge_indices
    adj=_dense_adjacency(e,len(nodes));two=adj|(adj.float()@adj.float()>0);two.fill_diagonal_(True)
    h,_=graph_spacing(nodes,e);ids=g.gaussian_node_indices;w=g.gaussian_node_weights;anchor=ids.gather(1,w.argmax(1)[:,None])[:,0]
    indices=[];prior=[]
    for start in range(0,len(x),512):
        dist=torch.cdist(x[start:start+512],nodes,compute_mode='donot_use_mm_for_euclid_dist');score=dist.square()/(2*h[anchor[start:start+512],None].square())
        selected=connected_growth(ids[start:start+512],adj,two,score);indices.append(selected.cpu());prior.append(warm_start_prior(score,selected,w[start:start+512]).cpu())
    q=dict(format_version=1,spatial_graph_indices=torch.cat(indices),spatial_graph_prior=torch.cat(prior),source_xyz_sha256=digest(x),
        original_indices_sha256=digest(ids),original_weights_sha256=digest(w))
    path=out/'setup/j8.pth';torch.save(q,path)
    m.args.ggas_candidates=str(path);m.args.ggas_variant='J_NO_MOTION_SELECTION';m.args.ggas_study_version=2
    initialize(m,out)
    write(out/'setup/j8_audit.json',dict(TRAIN_only=True,source='current5k state',K=8,spatial_only=True,nodes=len(nodes)))

@torch.no_grad()
def build_m2(m,boundary,out):
    from utils.stage14 import edges,isotropic_weights
    x=m._xyz.detach();original=m.closest_point_indices.long();ids=torch.empty((len(x),16),device=x.device,dtype=torch.long)
    for start in range(0,len(x),4096):
        z=slice(start,start+4096);distance=(x[original[z]]-x[z,None]).square().sum(-1);order=distance.argsort(dim=1,stable=True)[:,:16];ids[z]=original[z].gather(1,order)
    ref=edges(x,ids);gamma=torch.ones(len(x),device=x.device)
    if boundary is not None:gamma[boundary]=0
    q=dict(indices=ids.int(),reference_edges=ref,iso=isotropic_weights(ref),gamma=gamma)
    torch.save({k:v.cpu() for k,v in q.items()},out/'setup/m2.pth');m.stage14_ref=q;m.stage14_capture=True
    write(out/'setup/m2_audit.json',dict(TRAIN_only=True,source='current10k state',candidate_set='nearest16 within frozen initial99',lambda_M2=.01,Dynamic_boundary_gamma=0,C3VD_gamma=1))
