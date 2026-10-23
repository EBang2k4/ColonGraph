"""Stage9 equations. No claims of anatomical or metric invariance.

Ray projection depends on the supplied camera; geometry evaluation must supply
and record that same camera instead of evaluating a different deformation field.
"""
import math
import torch


def ramp(k,start,end):
    return min(1.,max(0.,(k-start)/(end-start)))


def cosine(k,start=15000,end=30000,floor=.06):
    return floor+(1-floor)*.5*(1+math.cos(math.pi*ramp(k,start,end)))


def reference_offset(x,indices):
    assert indices.ndim==2 and indices.shape==(len(x),99)
    with torch.no_grad():return (x-x[indices].mean(1)).detach().clone()


def compensated_knn(x,indices,reference,k,start=10000,end=11000):
    assert reference.shape==x.shape and not reference.requires_grad
    gamma=.85*ramp(k,start,end)
    return .01*((x-x[indices].mean(1))-gamma*reference).square().sum(-1).mean()


def depth_loss(pred,gt,valid,weight=.25,eta=.60):
    valid=valid.bool() & torch.isfinite(gt) & (gt>0) & torch.isfinite(pred)
    if not valid.any():return pred.sum()*0
    scale_pred=pred[valid].max().detach()+1e-6
    scale_gt=gt[valid].max().detach()+1e-6
    self_normalized=((pred/scale_pred).clamp(0,1)-(gt/scale_gt).clamp(0,1)).abs()[valid].mean()
    metric=((pred[valid]-gt[valid]).square()+1e-8).sqrt().mean()/scale_gt
    return weight*((1-eta)*self_normalized+eta*metric)


def ray_tangent(delta,xyz,camera_center):
    direction=(xyz-camera_center.to(xyz).reshape(1,3)).detach()
    direction=direction/direction.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    return delta-(delta*direction).sum(-1,keepdim=True)*direction
