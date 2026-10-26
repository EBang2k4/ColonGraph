"""Pure E4 surface topology builder; no saved reference inputs."""
import itertools
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components,dijkstra
from utils.node_edge_graph import NodeEdgeConsistencyGraph


def quantiles(values):
    values = torch.as_tensor(values).float().flatten()
    return {"mean": float(values.mean()), "median": float(values.median()),
            "p90": float(torch.quantile(values, .9)),
            "p95": float(torch.quantile(values, .95)),
            "max": float(values.max())}

def aggregate(keys, values, reduce="sum"):
    order = torch.argsort(keys)
    keys, values = keys[order], values[order]
    unique, inverse = torch.unique_consecutive(keys, return_inverse=True)
    if reduce == "sum":
        result = torch.zeros(len(unique), dtype=values.dtype)
        result.scatter_add_(0, inverse, values)
    elif reduce == "amin":
        result = torch.full((len(unique),), float("inf"), dtype=values.dtype)
        result.scatter_reduce_(0, inverse, values, reduce="amin", include_self=True)
    elif reduce == "amax":
        result = torch.full((len(unique),), -float("inf"), dtype=values.dtype)
        result.scatter_reduce_(0, inverse, values, reduce="amax", include_self=True)
    else:
        raise ValueError(reduce)
    return unique, result

def edge_tensors(keys, node_count):
    source = torch.div(keys, node_count, rounding_mode="floor")
    target = keys.remainder(node_count)
    return torch.stack((source, target)).long()

def ensure_connected(keys, scores, node_xyz, score_from_distance=True):
    node_count = len(node_xyz)
    edges = edge_tensors(keys, node_count)
    row = torch.cat((edges[0], edges[1])).numpy()
    col = torch.cat((edges[1], edges[0])).numpy()
    graph = coo_matrix((np.ones(len(row)), (row, col)), shape=(node_count, node_count))
    count, labels = connected_components(graph, directed=False)
    if count == 1:
        return keys, scores, 0
    distances = torch.cdist(node_xyz, node_xyz)
    additions = []
    labels = torch.from_numpy(labels)
    base = 0
    for component in range(1, count):
        left = (labels == base).nonzero().flatten()
        right = (labels == component).nonzero().flatten()
        block = distances[left][:, right]
        flat = int(block.argmin())
        i, j = int(left[flat // len(right)]), int(right[flat % len(right)])
        additions.append(min(i, j)*node_count + max(i, j))
        labels[labels == component] = base
    add_keys = torch.tensor(additions, dtype=torch.long)
    add_edges = edge_tensors(add_keys, node_count)
    lengths = (node_xyz[add_edges[0]]-node_xyz[add_edges[1]]).norm(dim=1)
    sigma = float(lengths.median().clamp_min(1e-8))
    add_scores = (torch.exp(-lengths.square()/(2*sigma*sigma))
                  if score_from_distance else torch.ones_like(lengths))
    all_keys, inverse = torch.unique(torch.cat((keys, add_keys)),
                                     sorted=True, return_inverse=True)
    all_scores = torch.zeros(len(all_keys))
    all_scores.scatter_reduce_(0, inverse, torch.cat((scores, add_scores)),
                               reduce="amax", include_self=False)
    return all_keys, all_scores.clamp_min(1e-12), len(additions)

def make_graph(node_xyz, keys, weights, topology, metadata):
    edges = edge_tensors(keys, len(node_xyz))
    lengths = (node_xyz[edges[0]]-node_xyz[edges[1]]).norm(dim=1).clamp_min(1e-8)
    return NodeEdgeConsistencyGraph(
        node_xyz, edges, weights.clamp_min(1e-12), lengths,
        topology, metadata)

def build_gaussian_surface(xyz, gaussian_knn, k4):
    node_count = int(k4.max())+1
    candidate = gaussian_knn[:, :12].long()
    rows = torch.arange(len(xyz))[:, None].expand_as(candidate)
    overlap = torch.empty_like(candidate)
    chunk = 32768
    for start in range(0, len(xyz), chunk):
        end = min(len(xyz), start+chunk)
        overlap[start:end] = (
            k4[start:end, None, :, None] ==
            k4[candidate[start:end]][:, :, None, :]).sum((2, 3))
    keep = overlap >= 1
    low, high = torch.minimum(rows[keep], candidate[keep]), torch.maximum(rows[keep], candidate[keep])
    valid = low != high
    keys = low[valid]*len(xyz)+high[valid]
    values = overlap[keep][valid].float()
    keys, values = aggregate(keys, values, "amax")
    edges = edge_tensors(keys, len(xyz))
    return edges, values.long()

def build_e4(xyz, node_xyz, k4, k4_weights, gaussian_knn, return_geodesic=False):
    node_count = len(node_xyz)
    surface_edges, overlap = build_gaussian_surface(xyz, gaussian_knn, k4)
    dominant = k4.gather(1, k4_weights.argmax(1, keepdim=True)).squeeze(1)
    left, right = dominant[surface_edges[0]], dominant[surface_edges[1]]
    valid = left != right
    low, high = torch.minimum(left[valid], right[valid]), torch.maximum(left[valid], right[valid])
    gaussian_left = xyz[surface_edges[0, valid]]
    gaussian_right = xyz[surface_edges[1, valid]]
    surface_length = (gaussian_left-gaussian_right).norm(dim=1)
    surface_cost = ((node_xyz[left[valid]]-gaussian_left).norm(dim=1) +
                    surface_length*(5-overlap[valid].float()) +
                    (node_xyz[right[valid]]-gaussian_right).norm(dim=1))
    mapped_keys, mapped_cost = aggregate(
        low*node_count+high, surface_cost, "amin")

    support_keys, support_costs = [], []
    for a, b in itertools.combinations(range(4), 2):
        na, nb = k4[:, a], k4[:, b]
        low, high = torch.minimum(na, nb), torch.maximum(na, nb)
        cost = ((node_xyz[na]-xyz).norm(dim=1) +
                (node_xyz[nb]-xyz).norm(dim=1)).clamp_min(1e-8)
        support_keys.append(low*node_count+high)
        support_costs.append(cost)
    support_keys, support_cost = aggregate(
        torch.cat(support_keys), torch.cat(support_costs), "amin")
    coarse_keys, coarse_cost = aggregate(
        torch.cat((mapped_keys, support_keys)),
        torch.cat((mapped_cost, support_cost)), "amin")
    coarse_edges = edge_tensors(coarse_keys, node_count)
    row = torch.cat((coarse_edges[0], coarse_edges[1])).numpy()
    col = torch.cat((coarse_edges[1], coarse_edges[0])).numpy()
    data = torch.cat((coarse_cost, coarse_cost)).double().numpy()
    coarse = coo_matrix((data, (row, col)), shape=(node_count, node_count)).tocsr()
    components, _ = connected_components(coarse, directed=False)
    if components != 1:
        raise RuntimeError(f"surface proxy node graph has {components} components")
    geodesic = torch.from_numpy(dijkstra(coarse, directed=False)).float()
    geodesic.fill_diagonal_(float("inf"))
    selected_distance, selected = geodesic.topk(6, largest=False)
    rows = torch.arange(node_count)[:, None].expand_as(selected)
    low, high = torch.minimum(rows, selected), torch.maximum(rows, selected)
    keys, distances = aggregate(
        (low*node_count+high).flatten(), selected_distance.flatten(), "amin")
    sigma = float(distances.median().clamp_min(1e-8))
    weights = torch.exp(-distances.square()/(2*sigma*sigma)).clamp_min(1e-12)

    canonical_distance = torch.cdist(node_xyz, node_xyz)
    canonical_distance.fill_diagonal_(float("inf"))
    canonical_knn = canonical_distance.topk(6, largest=False).indices
    canonical_rows = torch.arange(node_count)[:, None].expand_as(canonical_knn)
    c_low, c_high = torch.minimum(canonical_rows, canonical_knn), torch.maximum(canonical_rows, canonical_knn)
    canonical_keys = torch.unique((c_low*node_count+c_high).flatten())
    retained = torch.isin(canonical_keys, keys)
    final_edges = edge_tensors(keys, node_count)
    euclidean = (node_xyz[final_edges[0]]-node_xyz[final_edges[1]]).norm(dim=1)
    metadata = {
        "gaussian_candidate_k": 12, "minimum_shared_k4": 1,
        "gaussian_surface_edges": int(surface_edges.shape[1]),
        "gaussian_surface_shared_k4": {
            str(value): float((overlap == value).float().mean()) for value in range(1, 5)},
        "surface_proxy_node_edges": int(len(coarse_keys)),
        "directed_target_k": 6,
        "canonical_knn6_edges": int(len(canonical_keys)),
        "canonical_shortcut_edges_rejected": int((~retained).sum()),
        "canonical_shortcut_rejection_ratio": float((~retained).float().mean()),
        "selected_geodesic_distance": quantiles(distances),
        "selected_euclidean_distance": quantiles(euclidean),
        "geodesic_to_euclidean_ratio": quantiles(distances/euclidean.clamp_min(1e-8)),
    }
    result=make_graph(node_xyz, keys, weights, "surface_aware", metadata)
    return (result,geodesic) if return_geodesic else result
