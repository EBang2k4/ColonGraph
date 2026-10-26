import torch

def fold_mask(gtdepth,valid):
    """TEST diagnostic only. Top10% fixed GT/prior depth-gradient pixels, stable ID ties."""
    d=gtdepth.squeeze();v=valid.bool().squeeze();g=torch.zeros_like(d)
    dx=(d[:,1:]-d[:,:-1]).abs()*(v[:,1:]&v[:,:-1])
    dy=(d[1:]-d[:-1]).abs()*(v[1:]&v[:-1])
    g[:,:-1]+=dx;g[:,1:]+=dx;g[:-1]+=dy;g[1:]+=dy
    ids=v.flatten().nonzero().flatten();k=max(1,int(len(ids)*.1))
    order=torch.argsort(g.flatten()[ids],descending=True,stable=True)
    mask=torch.zeros_like(v).flatten();mask[ids[order[:k]]]=True
    return mask.reshape_as(v)
