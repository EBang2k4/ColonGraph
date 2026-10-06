"""Prepare two fixed-G3 references only after the preceding study completes."""
import argparse
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from tools.experiment_common import load_scene
from utils.connectivity_ablation import euclidean_reference
from tools.build_node_connectivity_graphs import build_e4


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    scene='c1_sigmoid2_t4_v4'
    surface=torch.load(f'output/graph_forward_reference/{scene}/graph_forward_reference.pth',map_location='cpu',weights_only=True)
    model,_,_,_,_=load_scene(f'output/node_connectivity_common_warmup/{scene}',2000)
    if model.graph_motion_residual is not None:raise RuntimeError('Branch source must precede G3 training')
    if not torch.equal(model.deformation_graph.node_xyz.cpu(),surface['canonical_node_xyz']):
        raise RuntimeError('Common canonical node identities differ')
    displacement=torch.stack([model.node_positions_at_time(t)-model.deformation_graph.node_xyz
                              for t in surface['trajectory_times'].cuda()]).cpu()
    source,target=surface['surface_edge_indices']
    original_distance=(displacement[:,source]-displacement[:,target]).square().sum(-1).mean(0).sqrt()
    if not torch.allclose(original_distance,surface['trajectory_distance'],rtol=1e-5,atol=1e-4):
        raise RuntimeError('Common trajectory does not reproduce frozen G3 gating reference')
    knn=torch.load('output/adaptive_deformation_matched_knn99.pth',map_location='cpu',weights_only=True)
    rebuilt,geodesic=build_e4(model._xyz.detach().cpu(),surface['canonical_node_xyz'],
        surface['original_k4_indices'],surface['original_k4_weights'],knn,return_geodesic=True)
    if not torch.equal(rebuilt.edge_indices,surface['surface_edge_indices']):
        raise RuntimeError('Rebuilt surface-proxy topology differs from original frozen reference')
    if not torch.allclose(rebuilt.edge_weights,surface['surface_edge_weights'],rtol=1e-5,atol=1e-7):
        raise RuntimeError('Rebuilt spatial kernel differs from original frozen reference')
    spatial_sigma=float(geodesic[source,target].median().clamp_min(1e-8))
    euclidean=euclidean_reference(surface,displacement,geodesic,spatial_sigma)
    for name,payload in [('G3-Euc',euclidean),('G3-Surf',copy.deepcopy(surface))]:
        for key in ('canonical_node_xyz','original_k4_indices','original_k4_weights','graph_k4_indices',
                    'graph_k4_weights','changed_mask','fallback_mask','trajectory_times'):
            if not torch.equal(payload[key],surface[key]):raise RuntimeError(f'Connectivity altered {key}')
        torch.save(payload,a.output/f'{name}.pth')
    audit=dict(pass_=True,source='common pre-G3 2k',nodes=2000,k=6,adaptive=False,
        trajectory_sigma=float(surface['audit']['trajectory_sigma']),gate_sigma_fixed=True,spatial_sigma_fixed=spatial_sigma,
        surface_reference_numerically_unchanged=True,geometry_and_appearance_k4_identical=True,
        euclidean=euclidean['audit']['connectivity_ablation'],
        surface_edges=int(surface['surface_edge_indices'].shape[1]))
    (a.output/'reference_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
