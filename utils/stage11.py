import torch

def compensated_knn(x,indices,reference,k,boundary):
    ramp=min(1.,max(0.,(k-10000)/1000.))
    gamma=.75*ramp*(~boundary).to(x.dtype)
    return .01*((x-x[indices].mean(1))-gamma[:,None]*reference).square().sum(-1).mean()
