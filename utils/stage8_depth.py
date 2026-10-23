"""Exact historical F0/U2 depth functions, without renderer/training imports."""
import torch


def depth_valid_mask(depth, alpha):
    """Dataset-only foreground validity; exact Blender background sentinel."""
    import torch
    return torch.isfinite(depth) & (depth > 0) & (depth != 1e10) & (alpha > 0)


def depth_statistics(pred,gt,alpha):
    from utils.adaptive_deformation_graph import AdaptiveNodeManager
    pred=pred.squeeze().float();gt=gt.squeeze().float()
    valid=depth_valid_mask(gt,alpha)
    metric=AdaptiveNodeManager._depth_metrics(pred,gt,valid)
    ngt=(gt/gt[valid].max().clamp_min(1e-6)).clamp(0,1)
    npr=(pred/pred[valid].max().clamp_min(1e-6)).clamp(0,1)
    gv=valid[1:,1:]&valid[1:,:-1]&valid[:-1,1:]
    mag=((ngt[1:,1:]-ngt[1:,:-1]).square()+(ngt[1:,1:]-ngt[:-1,1:]).square()).sqrt()
    threshold=torch.quantile(mag[gv],.8)
    fold=gv&(mag>=threshold)
    return dict(Depth_MSE_raw=metric['raw_mse'],Depth_MSE_normalized=metric['normalized_mse'],
        Fold_Depth=float((npr-ngt).square()[1:,1:][fold].mean()),
        Fold_Depth_raw=float((pred-gt).square()[1:,1:][fold].mean()),
        fold_pixels=int(fold.sum()),valid_depth_pixels=int(valid.sum()))
