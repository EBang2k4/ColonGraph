"""Predeclared Stage10 schedules and native-autograd scale-gated depth."""
import math
import torch
from utils.loss_utils import l1_loss


def ramp(k,start,end):return min(1.,max(0.,(k-start)/(end-start)))


def cosine(k,start,end,floor):
    return floor+(1-floor)*.5*(1+math.cos(math.pi*ramp(k,start,end)))


def photo_weights(k):
    t=ramp(k,18000,30000)
    return .80-.55*t,.15,2.5+7.5*t


def depth_loss(pred,gt,valid,weight=.25,metric_weight=0.):
    # This is the original train.py normalized term, including max autograd.
    if valid.any():
        pmax=pred[valid].max();gmax=gt[valid].max()
        pn=torch.clamp(pred/(pmax+1e-6),0,1)
        gn=torch.clamp(gt/(gmax+1e-6),0,1)
        norm=weight*l1_loss(pn,gn,valid,True)
        # Do not even evaluate incompatible raw-depth differences if gate is off.
        metric=(metric_weight*((pred[valid]-gt[valid]).square()+1e-8).sqrt().mean()/(gmax+1e-6)
                if metric_weight else pred.new_zeros(()))
    else:norm=metric=pred.new_zeros(())
    return (norm+metric if metric_weight else norm),norm,metric
