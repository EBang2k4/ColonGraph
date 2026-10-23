"""Stage 2: full fixed-K4 audit and provenance-preserving weight caches."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import time
import numpy as np
import torch
from plyfile import PlyData
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from utils.geometry_blending import compute_weights, graph_spacing, validate_cache
from utils.exact_split_branchpoint import digest
from tools.experiment_common import load_scene
from tools.evaluate_texture_experiment import file_hash, write_json

ROOT = Path(__file__).resolve().parents[1]
SCENE = 'c1_sigmoid2_t4_v4'
OUT = ROOT/'output/experiments/geometry_blending'/SCENE
SETUP = ROOT.parent/'ColonGraph/output/g3_surf_fresh_setup'/SCENE
BASE = ROOT/'output/zero_mean_texture'/SCENE/'A'
SOURCE = ROOT/'output/zero_mean_texture'/SCENE/'common_5000'
VARIANTS = {'A_idw': ('idw', 2.5), 'B_idw2': ('idw_squared', 2.5),
            'C_beta2': ('wendland_c2', 2.), 'C_beta2_5': ('wendland_c2', 2.5)}


def stats(value):
    x = torch.as_tensor(value).detach().double().flatten()
    if not x.numel():
        return {'count': 0}
    if not torch.isfinite(x).all():
        raise ValueError('Nonfinite audit statistic')
    qs = torch.quantile(x, x.new_tensor([0., .01, .05, .5, .9, .95, .99, 1.]))
    return dict(count=len(x), mean=float(x.mean()), **dict(zip(
        ['min', 'p01', 'p05', 'median', 'p90', 'p95', 'p99', 'max'], qs.tolist())))


def correlation(a, b):
    a, b = torch.as_tensor(a).double().flatten(), torch.as_tensor(b).double().flatten()
    a, b = a-a.mean(), b-b.mean()
    denom = a.square().sum().sqrt()*b.square().sum().sqrt()
    return float((a*b).sum()/denom) if denom > 0 else None


def ply_xyz(path):
    vertex = PlyData.read(path)['vertex']
    return torch.from_numpy(np.stack([vertex[k] for k in ('x','y','z')], axis=1)).cuda()


def hist(value):
    x = torch.as_tensor(value).detach().cpu().numpy().ravel()
    counts, bins = np.histogram(x, bins=64)
    return {'counts': counts.tolist(), 'edges': bins.tolist()}


@torch.no_grad()
def audit():
    started = time.monotonic()
    directory = OUT/'audit'; directory.mkdir(parents=True, exist_ok=True)
    caches = OUT/'caches'; caches.mkdir(exist_ok=True)
    model, scene, pipe, bg, args = load_scene(BASE, 20000)
    model.profile_deformation = False
    ref_path = SETUP/'references/graph_forward_reference.pth'
    ref = torch.load(ref_path, map_location='cuda', weights_only=True)
    g = model.deformation_graph
    assert len(model._xyz) == 364500 and len(g.node_xyz) == 2000
    assert torch.equal(g.gaussian_node_indices, ref['graph_k4_indices'])
    assert torch.equal(g.gaussian_node_weights, ref['graph_k4_weights'])
    assert torch.equal(model.graph_motion_residual.edge_indices, ref['surface_edge_indices'])
    assert torch.equal(model.graph_motion_residual.edge_weights, ref['trajectory_gated_edge_weights'])
    assert args.color_model == 'node_residual_mlp' and not args.texture_enabled
    # Reconstruct initial cached distances, then the 23 repaired rows' 2k origin.
    origin1 = SETUP/'warmup/point_cloud/iteration_1/point_cloud.ply'
    origin2 = SETUP/'warmup/point_cloud/iteration_2000/point_cloud.ply'
    origin = ply_xyz(origin1)
    orig_d = (origin[:,None]-g.node_xyz[ref['original_k4_indices']]).norm(dim=-1)
    raw = (orig_d+1e-8).reciprocal(); reconstructed = raw/raw.sum(-1,keepdim=True)
    if not torch.equal(reconstructed, ref['original_k4_weights']):
        raise RuntimeError('Canonical cache-origin reconstruction is not exact')
    origin[ref['changed_mask']] = ply_xyz(origin2)[ref['changed_mask']]
    ids, base = g.gaussian_node_indices, g.gaussian_node_weights
    d = (origin[:,None]-g.node_xyz[ids]).norm(dim=-1)
    # Original repair intentionally ran CPU norm/normalization on its 23 rows.
    repair_distances = (origin[ref['changed_mask']].cpu()[:,None]-g.node_xyz.cpu()[ids[ref['changed_mask']].cpu()]).norm(dim=-1)
    d[ref['changed_mask']] = repair_distances.cuda()
    raw = (d+1e-8).reciprocal(); reconstructed = raw/raw.sum(-1,keepdim=True)
    repaired_raw = (repair_distances+1e-8).reciprocal()
    reconstructed[ref['changed_mask']] = (repaired_raw/repaired_raw.sum(-1,keepdim=True)).cuda()
    if not torch.equal(reconstructed, base):
        raise RuntimeError('Repaired canonical cache-origin reconstruction is not exact')
    current_d = (model._xyz[:,None]-g.node_xyz[ids]).norm(dim=-1)
    precompute = time.monotonic()
    h, spacing_audit = graph_spacing(g.node_xyz, ref['surface_edge_indices'])
    spacing_seconds = time.monotonic()-precompute
    if spacing_audit['isolated_nodes']:
        raise RuntimeError('Reference graph has isolated spacing nodes')
    invariants = dict(nodes=digest(g.node_xyz), indices=digest(ids),
        appearance=digest(model.appearance_graph.state_dict()),
        g3_graph=digest({k:v for k,v in model.graph_motion_residual.state_dict().items() if not k.startswith('decoder.')}))
    provenance = dict(reference_sha256=file_hash(ref_path), cache_origins={str(p):file_hash(p) for p in (origin1,origin2)},
        origin_policy='initial fine-1 pre-Adam coordinates; 23 repaired rows use fine-2000 coordinates',
        exact_original_and_repaired_IDW_reconstruction=True, epsilon=1e-8,
        precision_policy='Original rows use CUDA float32; repair rows reproduce historical CPU float32 norms/normalization. B/C kernels normalize float64.')
    arrays = dict(origin=origin.cpu().numpy(), trained_xyz=model._xyz.detach().cpu().numpy(),
        node_xyz=g.node_xyz.cpu().numpy(), indices=ids.cpu().numpy(), cache_distances=d.cpu().numpy(),
        trained_distances=current_d.cpu().numpy(), node_spacing=h.cpu().numpy())
    results, weights, infos, histograms = {}, {}, {}, {}
    for name, (mode,beta) in VARIANTS.items():
        before = time.monotonic()
        w, info = compute_weights(d, base, mode, ids, h, beta)
        torch.cuda.synchronize(); compute_seconds = time.monotonic()-before
        weights[name], infos[name] = w, info
        fallback = info['fallback']
        cache = dict(format_version=1, mode=mode, beta=beta, weights=w.cpu(),
            weights_sha256=digest(w), baseline_weights_sha256=digest(base), invariants=invariants,
            node_spacing=h.cpu(), node_radius=(h*beta).cpu(), fallback=fallback.cpu(), provenance=provenance)
        validate_cache(model, cache)
        torch.save(cache, caches/(name+'.pth'))
        keff = w.square().sum(-1).reciprocal()
        summary = dict(weights=stats(w), weight_columns=[stats(w[:,j]) for j in range(4)],
            max_weight=stats(w.max(-1).values), min_weight=stats(w.min(-1).values), effective_K=stats(keff),
            highly_concentrated_max_weight_gt_095=float((w.max(-1).values>.95).float().mean()),
            highly_concentrated_Keff_lt_15=float((keff<1.5).float().mean()),
            zero_support_count=info['zero_support'], numerical_fallback_count=info['numerical_fallback'],
            fallback_fraction=float(fallback.float().mean()), fallback_weight_mass_fraction=float(w[fallback].sum()/len(w)),
            precompute_seconds=compute_seconds, cache_bytes=(caches/(name+'.pth')).stat().st_size,
            radius=stats(h*beta), max_sum_error=float((w.sum(-1)-1).abs().max()))
        for label,dist in [('cache_origin',d),('trained_20k',current_d)]:
            rho = dist.double()/h[ids]
            summary[label] = dict(distances=stats(dist), distance_columns=[stats(dist[:,j]) for j in range(4)],
                min_distance=stats(dist.min(-1).values), max_distance=stats(dist.max(-1).values),
                max_min_ratio=stats(dist.max(-1).values/dist.min(-1).values.clamp_min(1e-8)),
                weighted_distance=stats((w*dist).sum(-1)), distance_over_spacing=stats(rho),
                distant_weight={str(t):stats((w*(rho>t)).sum(-1)) for t in (1.5,2.,2.5)})
        if mode == 'wendland_c2':
            summary['active_count'] = {str(k):int((info['active_count']==k).sum()) for k in range(5)}
            summary['raw_denominator'] = stats(info['raw_denominator'])
            boundary = {}
            for sign in (-1,1):
                perturbed = (d.double()+sign*1e-4*h[ids]).clamp_min(0)
                pw, pi = compute_weights(perturbed, base, mode, ids, h, beta)
                boundary[str(sign)] = dict(weight_L1=stats((pw-w).abs().sum(-1)),
                    fallback_transitions=int((pi['fallback']!=fallback).sum()),
                    L1_gt_05_fraction=float(((pw-w).abs().sum(-1)>.5).float().mean()))
            summary['boundary_distance_perturbation_1e_4_spacing'] = boundary
            fxyz = origin[fallback].cpu().numpy()
            np.save(directory/(name+'_fallback_xyz.npy'), fxyz)
            summary['fallback_spatial_clustering'] = dict(count=len(fxyz),
                min=fxyz.min(0).tolist() if len(fxyz) else None, max=fxyz.max(0).tolist() if len(fxyz) else None,
                note='Full canonical fallback locations saved; no support fallback iff this array is empty.')
        arrays[name+'_weights'] = w.cpu().numpy()
        arrays[name+'_fallback'] = fallback.cpu().numpy()
        histograms[name] = {'effective_K':hist(keff),'max_weight':hist(w.max(-1).values),
            'weighted_distance':hist((w*d).sum(-1)), 'rho':hist(d.double()/h[ids]),
            'distant_weight_rho2':hist((w*(d.double()/h[ids]>2)).sum(-1))}
        results[name] = summary
        print(name, 'fallback', summary['fallback_fraction'], 'Keff', summary['effective_K']['mean'], flush=True)
    # Paired motion diagnostic uses test times only; never selects beta by depth/RGB.
    cameras = sorted(scene.getTestCameras(),key=lambda c:int(c.image_name))
    write_json(OUT/'protocol.json', dict(scene=SCENE, baseline_render_checkpoint=str(BASE), iteration=20000,
        short_source=str(SOURCE), short_iterations=[5001,10000], beta_selection='offline coverage/numerics only',
        evaluation_frames=[c.image_name for c in cameras], evaluation_times=[float(c.time) for c in cameras],
        motion_role='36 test timestamps for locality/disagreement diagnostics, no metric-based beta tuning',
        visual_selection='Top three GT-only mean depth-gradient test views, selected before rendering variants',
        scheduler='absolute original 30k schedule retained; stop separately at 10k after optimizer',
        fold_metric='Historical GT-P80 implementation unavailable locally; new GT-P80 high-gradient-region depth-error diagnostic, not historical official Fold',
        shared_max='All prediction variants use baseline A valid max for each matched view; GT uses common GT valid max'))
    motion = []; variance_sum = {name:torch.zeros(len(d),device='cuda') for name in weights}
    by_time = []
    distant = (base*(d.double()/h[ids]>2)).sum(-1)
    for camera in cameras:
        u, _, _ = model._query_node_motion(float(camera.time)); motion.append(u.cpu())
        local = u[ids]; row = dict(frame=camera.image_name,time=float(camera.time),variants={})
        for name,w in weights.items():
            mean = (local*w[...,None]).sum(1)
            v = (((local-mean[:,None])**2).sum(-1)*w).sum(-1)
            variance_sum[name] += v
            row['variants'][name] = dict(variance=stats(v), correlation_baseline_distant_rho2=correlation(v,distant))
        by_time.append(row)
    for name in weights:
        v = variance_sum[name]/len(cameras)
        results[name]['motion_mean_over_times'] = dict(variance=stats(v),
            correlation_baseline_distant_rho2=correlation(v,distant))
        arrays[name+'_mean_motion_variance'] = v.cpu().numpy()
    torch.save(dict(frame_ids=[c.image_name for c in cameras], translations=torch.stack(motion)), directory/'corrected_node_motion.pth')
    # Connectivity and shortest-path diagnostics on existing graph only.
    edges = ref['surface_edge_indices'].cpu().numpy(); m = len(g.node_xyz)
    el = (g.node_xyz[ref['surface_edge_indices'][0]]-g.node_xyz[ref['surface_edge_indices'][1]]).norm(dim=-1)
    a,b = np.concatenate((edges,edges[::-1]),axis=1)
    graph = csr_matrix((np.ones(len(a)),(a,b)),shape=(m,m))
    hop = torch.from_numpy(shortest_path(graph,directed=False,unweighted=True)).cuda()
    lengths = np.tile(el.cpu().numpy(),2)
    path = torch.from_numpy(shortest_path(csr_matrix((lengths,(a,b)),shape=(m,m)),directed=False)).cuda()
    adjacency = torch.from_numpy(graph.toarray()>0).cuda()
    within = adjacency[ids[:,:,None],ids[:,None,:]] | torch.eye(4,device='cuda',dtype=torch.bool)[None]
    reach = within.clone()
    for _ in range(3):
        reach = reach | (reach[:,:,:,None] & reach[:,None,:,:]).any(2)
    pairs = torch.triu_indices(4,4,offset=1,device='cuda')
    pair_hops = hop[ids[:,pairs[0]],ids[:,pairs[1]]]
    pair_paths = path[ids[:,pairs[0]],ids[:,pairs[1]]]
    node_pair_d = (g.node_xyz[ids[:,pairs[0]]]-g.node_xyz[ids[:,pairs[1]]]).norm(dim=-1)
    cutoff = torch.quantile(el,.99)
    long_adj = torch.zeros((m,m),device='cuda',dtype=torch.bool)
    e = ref['surface_edge_indices'][:,el>cutoff]; long_adj[e[0],e[1]] = True; long_adj[e[1],e[0]] = True
    long_k4 = long_adj[ids[:,:,None],ids[:,None,:]].any((1,2))
    dominant = ids.gather(1,base.argmax(-1)[:,None])[:,0]
    knn = torch.load(SETUP/'gaussian_knn99.pth',map_location='cuda',weights_only=True)[:,:12]
    consistent = torch.zeros(len(ids),device='cuda'); contained = consistent.clone()
    for start in range(0,len(ids),32768):
        end = min(start+32768,len(ids)); ng = knn[start:end]
        consistent[start:end] = (dominant[ng]==dominant[start:end,None]).float().mean(-1)
        contained[start:end] = (ids[ng]==dominant[start:end,None,None]).any(-1).float().mean(-1)
    connectivity = dict(induced_K4_connected_fraction=float(reach.all((1,2)).float().mean()),
        all_pair_hops=stats(pair_hops), max_pair_hops=stats(pair_hops.max(-1).values),
        all_pair_graph_Euclidean_path=stats(pair_paths), all_pair_Euclidean=stats(node_pair_d),
        graph_path_to_Euclidean_ratio=stats(pair_paths/node_pair_d.clamp_min(1e-8)),
        surface_edge_lengths=stats(el), long_edge_gt_p99_threshold=float(cutoff),
        K4_contains_long_edge_fraction=float(long_k4.float().mean()),
        dominant_same_in_12_cached_Gaussian_neighbors=stats(consistent),
        dominant_contained_in_neighbor_K4=stats(contained),
        warning='Graph is a surface proxy partly built from original K4, so these tests are not independent anatomical evidence.')
    arrays.update(max_K4_pair_hops=pair_hops.max(-1).values.cpu().numpy(),
        dominant_neighbor_consistency=consistent.cpu().numpy(), distant_weight_rho2=distant.cpu().numpy())
    np.savez_compressed(directory/'distributions.npz', **arrays)
    # Choose radius using coverage/concentration only, before any render metrics.
    selected = 'C_beta2_5'
    choice = dict(selected_variant=selected, beta=2.5, based_on='Offline coverage and less one-node concentration at larger radius; no rendered metrics consulted',
        coverage={name:{key:results[name][key] for key in ('fallback_fraction','active_count','highly_concentrated_Keff_lt_15')}
            for name in ('C_beta2','C_beta2_5')},
        exploratory=results['A_idw']['cache_origin']['distant_weight']['2.0']['mean']<.01,
        fallback_limitation=results[selected]['fallback_fraction']>.001)
    write_json(directory/'beta_selection.json',choice)
    write_json(directory/'motion_by_time.json',by_time)
    write_json(directory/'histograms.json',histograms)
    report = dict(provenance=provenance, invariants=invariants, spacing=stats(h), spacing_audit=spacing_audit,
        spacing_precompute_seconds=spacing_seconds, variants=results, connectivity=connectivity,
        beta_selection=choice, audit_wall_seconds=time.monotonic()-started)
    write_json(directory/'summary.json',report)
    print('AUDIT_COMPLETE',choice,flush=True)


if __name__ == '__main__':
    audit()
