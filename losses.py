import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim


def image_gradients(x):
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    return dx, dy


def speckle_physics_loss(clean_sar, raw_sar, window=7):
    """Physics prior for the despeckling stage (no clean reference needed).

    1. Radiometric preservation: speckle has unit mean (E[F] = 1), so the local
       multi-look mean of the observation is an unbiased estimate of the true
       reflectivity. The despeckled image must keep the same local mean:
           L_rad = || AvgPool_w(X_hat) - AvgPool_w(Y) ||_1
    2. Log-domain total variation: in the homomorphic domain speckle is additive
       and high-frequency, so the clean log-reflectivity should be piecewise smooth:
           L_tv  = || grad log(X_hat + eps) ||_1
    Returns (L_rad, L_tv).
    """
    pad = window // 2
    m_clean = F.avg_pool2d(F.pad(clean_sar, (pad,) * 4, mode='reflect'), window, stride=1)
    m_raw = F.avg_pool2d(F.pad(raw_sar, (pad,) * 4, mode='reflect'), window, stride=1)
    l_rad = (m_clean - m_raw).abs().mean()
    dx, dy = image_gradients(torch.log(clean_sar + 0.01))
    l_tv = dx.abs().mean() + dy.abs().mean()
    return l_rad, l_tv


class PhysicsInformedCompositeLoss(nn.Module):
    """Generator objective.

    L = l1*L1 + ssim*(1-SSIM) + grad*L_grad + adv*L_GAN + rad*L_rad + tv*L_tv

    L_grad is a gradient-difference loss between prediction and optical target
    (keeps object boundaries sharp). L_rad / L_tv supervise the despeckling stage
    with the multiplicative speckle model (see speckle_physics_loss). Any weight
    can be set to 0 to obtain the baselines / ablations.
    """
    def __init__(self, lambda_l1=100.0, lambda_ssim=10.0, lambda_grad=10.0, lambda_adv=1.0,
                 lambda_rad=10.0, lambda_tv=1.0):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_grad = lambda_grad
        self.lambda_adv = lambda_adv
        self.lambda_rad = lambda_rad
        self.lambda_tv = lambda_tv
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred_rgb, target_rgb, clean_sar, raw_sar, d_pred_fake=None):
        zero = pred_rgb.new_zeros(())
        terms = {}
        terms['l1'] = (pred_rgb - target_rgb).abs().mean()
        terms['ssim'] = 1.0 - ssim(pred_rgb.float(), target_rgb.float(), data_range=1.0, win_size=7) \
            if self.lambda_ssim > 0 else zero
        if self.lambda_grad > 0:
            pdx, pdy = image_gradients(pred_rgb)
            tdx, tdy = image_gradients(target_rgb)
            terms['grad'] = (pdx - tdx).abs().mean() + (pdy - tdy).abs().mean()
        else:
            terms['grad'] = zero
        terms['adv'] = self.bce(d_pred_fake, torch.ones_like(d_pred_fake)) \
            if (d_pred_fake is not None and self.lambda_adv > 0) else zero
        if self.lambda_rad > 0 or self.lambda_tv > 0:
            terms['rad'], terms['tv'] = speckle_physics_loss(clean_sar.float(), raw_sar.float())
        else:
            terms['rad'], terms['tv'] = zero, zero

        total = (self.lambda_l1 * terms['l1'] + self.lambda_ssim * terms['ssim'] +
                 self.lambda_grad * terms['grad'] + self.lambda_adv * terms['adv'] +
                 self.lambda_rad * terms['rad'] + self.lambda_tv * terms['tv'])
        return total, terms
