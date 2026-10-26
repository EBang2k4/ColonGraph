"""Stage14 G2G: exact squared-world-unit energies, immutable reference tensors."""
import torch


def edges(x,indices):return x[:,None,:]-x[indices.long()]


def isotropic_weights(reference):
    # Literal first16 column subset. sigma8 = column8 distance, audited against ordering.
    distance=reference.norm(dim=-1)
    sigma=distance[:,7].clamp_min(1e-12)
    w=torch.exp(-reference.square().sum(-1)/(2*sigma[:,None].square()))
    return w/(w.sum(1,keepdim=True)+1e-8)


def covariance(scales,rotation_matrix):
    return (rotation_matrix*scales.square()[:,None,:])@rotation_matrix.transpose(1,2)


def anisotropic_reference(reference,si,sj):
    eye=torch.eye(3,device=reference.device,dtype=reference.dtype)
    avg=.5*(si[:,None]+sj)+1e-6*eye
    val,vec=torch.linalg.eigh(avg)
    assert (val>0).all() and torch.isfinite(val).all()
    proj=(vec.transpose(-1,-2)@reference[...,None]).squeeze(-1)
    mahal=(proj.square()/val).sum(-1).clamp_min(0)
    median=mahal.median(dim=1).values
    w=torch.exp(-mahal/(2*median[:,None]+1e-6));w=w/(w.sum(1,keepdim=True)+1e-8)
    precision=(vec*(val.mean(-1,keepdim=True)/val).clamp(.25,4)[...,None,:])@vec.transpose(-1,-2)
    packed=torch.stack([precision[...,0,0],precision[...,1,1],precision[...,2,2],precision[...,0,1],precision[...,0,2],precision[...,1,2]],-1)
    return w,packed,mahal


def quadratic(e,p):
    return (e.square()*p[...,:3]).sum(-1)+2*(e[...,0]*e[...,1]*p[...,3]+e[...,0]*e[...,2]*p[...,4]+e[...,1]*e[...,2]*p[...,5])


def components(method,x,ref,indices,weights,gamma,node=None,micro=None,base=None,scales=None,precision=None):
    """Per-edge components (all length²); batch may represent a fixed row chunk."""
    current=edges(x,indices) if method != "M2" else None
    if method=='M1':
        return dict(vector=(current-gamma[:,None,None]*ref).square().sum(-1),
                    spring=.5*(current.norm(dim=-1)-ref.norm(dim=-1)).square())
    if method=='M2':
        return dict(node=edges(node,indices).square().sum(-1),micro=edges(micro,indices).square().sum(-1),
                    base=(edges(base,indices)-gamma[:,None,None]*ref).square().sum(-1))
    if method=='M3':
        radius=scales.max(-1).values
        e=current-gamma[:,None,None]*ref
        return dict(precision=quadratic(e,precision),gap=.25*torch.relu(current.norm(dim=-1)-1.5*(radius[:,None]+radius[indices.long()])).square())
    raise ValueError(method)


def energy(method,x,q,node=None,micro=None,base=None,scales=None,return_components=False):
    # FullN16 vectorized path; noN99 tensors, no KNN searches or covariance inversions in loop.
    parts=components(method,x,q['reference_edges'],q['indices'],q['iso'],q['gamma'],node,micro,base,scales,q.get('precision'))
    w=q['maha'] if method=='M3' else q['iso']
    losses={key:.01*(value*w).sum(1).mean() for key,value in parts.items()}
    total=sum(losses.values())
    return (total,losses) if return_components else total
