"""
Structure-guided post-fusion of the despeckled SAR with the predicted RGB.

Two variants are evaluated in the paper:

1. luminance_replacement (original prototype): L* = a*SAR + (1-a)*L*_pred.
   Injects SAR geometry but also SAR *brightness*, which differs from optical
   brightness (e.g. water is dark in SAR but not necessarily in RGB).

2. detail_injection (proposed): only the high-frequency SAR component is added,
       L* = L*_pred + b * g * (S - G_sigma * S),   g = std(L*_pred) / std(S)
   in the spirit of component-substitution pansharpening. Low-frequency
   brightness and the chroma (a*, b*) stay those predicted by the network.
"""
import cv2
import numpy as np


def _stretch(sar):
    p2, p98 = np.percentile(sar, (2, 98))
    return np.clip((sar - p2) / (p98 - p2 + 1e-6), 0.0, 1.0)


def _to_lab(rgb):
    u8 = (np.clip(rgb, 0, 1) * 255.0).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_RGB2LAB).astype(np.float32)


def _to_rgb(lab):
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0


def luminance_replacement(clean_sar, pred_rgb, alpha=0.7):
    lab = _to_lab(pred_rgb)
    s = _stretch(clean_sar) * 255.0
    lab[..., 0] = alpha * s + (1 - alpha) * lab[..., 0]
    return _to_rgb(lab)


def detail_injection(clean_sar, pred_rgb, beta=0.5, sigma=2.0):
    lab = _to_lab(pred_rgb)
    s = _stretch(clean_sar) * 255.0
    hp = s - cv2.GaussianBlur(s, (0, 0), sigma)
    gain = lab[..., 0].std() / (s.std() + 1e-6)
    lab[..., 0] = lab[..., 0] + beta * gain * hp
    return _to_rgb(lab)


def chroma_boost(rgb, vibrancy=1.0):
    """Display-only saturation control (not used for any reported metric)."""
    if abs(vibrancy - 1.0) < 1e-6:
        return rgb
    lab = _to_lab(rgb)
    lab[..., 1:] = 128.0 + (lab[..., 1:] - 128.0) * vibrancy
    return _to_rgb(lab)
