"""Precomputed prior rules shared by the controlled GGAS-v2 ablations."""
import torch


def warm_start_prior(score, indices, original_weights, mass_cap=.20):
    # Preserve v1 exactly: original4 relative IDW, new4 normalized exp support.
    q=torch.softmax((-score.gather(1,indices)).double(),1).clamp_min(1e-30)
    mass=mass_cap*q[:,4:].sum(1,keepdim=True)/q.sum(1,keepdim=True)
    return torch.cat((original_weights.double()*(1-mass),
        mass*q[:,4:]/q[:,4:].sum(1,keepdim=True)),1).float()


def connected_growth(original, adjacency, two_hop, score):
    raw=two_hop[original].any(1);selected=original.clone()
    for _ in range(4):
        eligible=adjacency[selected].any(1)&raw;eligible.scatter_(1,selected,False)
        if not eligible.any(1).all():raise RuntimeError('Insufficient connected candidates; no silent fallback')
        chosen=score.masked_fill(~eligible,torch.inf).argmin(1)
        selected=torch.cat((selected,chosen[:,None]),1)
    return selected
