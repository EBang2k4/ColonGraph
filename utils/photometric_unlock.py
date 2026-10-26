"""Explicit fine-stage loss; DSSIM convention is (1-SSIM)/2."""
import torch
from utils.loss_utils import ssim


def handoff_weights(iteration, profile='J'):
    from utils.stage5_schedule import photometric
    return photometric(iteration, profile)


def rgb_loss(rendered, target, l1, mode='l1', audit=False, iteration=0, profile='J'):
    if mode == 'l1':
        return l1, {}
    if mode not in ('hybrid_psnr_ssim', 'hybrid_psnr_ssim_handoff', 'charbonnier_psnr_ssim_handoff'):
        raise ValueError('Unknown RGB loss ' + mode)
    mse = (rendered-target).square().mean()
    dssim = (1-ssim(rendered, target))/2
    l1_weight, dssim_weight, mse_weight = (handoff_weights(iteration, profile)
        if mode in ('hybrid_psnr_ssim_handoff','charbonnier_psnr_ssim_handoff') else (.8, .2, 2.))
    robust=(rendered-target).square().add(1e-6).sqrt().mean() if mode=='charbonnier_psnr_ssim_handoff' else l1
    robust_name='Charb' if mode=='charbonnier_psnr_ssim_handoff' else 'L1'
    parts = {robust_name: l1_weight*robust, 'DSSIM': dssim_weight*dssim, 'MSE': mse_weight*mse}
    loss = sum(parts.values())
    diagnostic = {}
    if audit:
        diagnostic = dict(raw_L1=float(l1.detach()), raw_MSE=float(mse.detach()), raw_DSSIM=float(dssim.detach()),
                          weighted_RGB=float(loss.detach()), DSSIM_definition='(1-SSIM)/2',
                          iteration=int(iteration),lambda_L1=l1_weight,lambda_DSSIM=dssim_weight,lambda_MSE=mse_weight)
        if robust_name=='Charb':
            diagnostic.update(raw_Charb=float(robust.detach()),lambda_Charb=l1_weight,Charb_epsilon_squared=1e-6,
                              MAE_used_as_loss=False,RGB_robust_term='Charb')
            diagnostic.pop('lambda_L1')
        for name, value in parts.items():
            grad, = torch.autograd.grad(value, rendered, retain_graph=True)
            assert torch.isfinite(grad).all()
            diagnostic[name+'_weighted_value'] = float(value.detach())
            diagnostic[name+'_image_gradient_norm'] = float(grad.detach().norm())
        assert all(diagnostic[k+'_image_gradient_norm'] > 0 for k in parts)
    return loss, diagnostic
