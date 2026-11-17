"""
Evaluation metrics for SAR-to-optical colorization and SAR despeckling.

All functions return the raw measured value. No scaling, clipping-to-target or
fallback constants are applied (the previous version clipped EPI into
[0.72, 0.85] and returned a hard-coded 0.822, which invalidated the numbers).

Images are float arrays in [0, 1]; RGB arrays are [H, W, 3].
"""
import cv2
import numpy as np
from scipy.stats import spearmanr
from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity as _ssim


# --------------------------------------------------------------------------- #
# Full-reference metrics (prediction vs. Sentinel-2 ground truth)
# --------------------------------------------------------------------------- #
def calculate_psnr(gt, pred):
    return float(_psnr(gt, pred, data_range=1.0))


def calculate_ssim(gt, pred):
    return float(_ssim(gt, pred, data_range=1.0, channel_axis=2 if gt.ndim == 3 else None))


def calculate_sam(pred, gt, eps=1e-8):
    """Spectral Angle Mapper (Kruse et al., 1993), mean angle in degrees."""
    p = pred.reshape(-1, pred.shape[-1]).astype(np.float64)
    g = gt.reshape(-1, gt.shape[-1]).astype(np.float64)
    denom = np.linalg.norm(p, axis=1) * np.linalg.norm(g, axis=1)
    valid = denom > eps
    cos = np.clip((p[valid] * g[valid]).sum(1) / denom[valid], -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)).mean())


def _luma(rgb):
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)


def calculate_gmsd(gt, pred, c=0.0026):
    """Gradient Magnitude Similarity Deviation (Xue et al., 2014). Lower is better.
    Measures how well edges/structure of the prediction match the reference."""
    g, p = _luma(gt), _luma(pred)
    k = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], np.float32) / 3.0
    def gm(x):
        return np.sqrt(cv2.filter2D(x, -1, k) ** 2 + cv2.filter2D(x, -1, k.T) ** 2)
    gg, gp = gm(g), gm(p)
    gms = (2 * gg * gp + c) / (gg ** 2 + gp ** 2 + c)
    return float(gms.std())


# --------------------------------------------------------------------------- #
# No-reference despeckling metrics (despeckled SAR vs. raw SAR)
# --------------------------------------------------------------------------- #
def homogeneous_blocks(raw, block=16, frac=0.10, min_mean=0.05):
    """Pick the `frac` most homogeneous blocks (lowest coefficient of variation)
    of the *raw* image. The same blocks are then used for raw and filtered
    images so ENL values are directly comparable."""
    h, w = raw.shape
    cands = []
    for y in range(0, h - block + 1, block):
        for x in range(0, w - block + 1, block):
            p = raw[y:y + block, x:x + block]
            m = p.mean()
            if m > min_mean:
                cands.append((p.std() / m, y, x))
    cands.sort()
    n = max(1, int(len(cands) * frac))
    return [(y, x) for _, y, x in cands[:n]]


def calculate_enl(img, blocks=None, block=16):
    """Equivalent Number of Looks, ENL = mean^2 / var, averaged over homogeneous
    blocks. Higher means stronger speckle suppression."""
    img = img.astype(np.float64)
    if img.max() > 1.0:
        img = img / 255.0
    if img.ndim == 3:
        img = img[..., 0]
    if blocks is None:
        blocks = homogeneous_blocks(img, block)
    vals = []
    for y, x in blocks:
        p = img[y:y + block, x:x + block]
        v = p.var()
        if v > 1e-10:
            vals.append(p.mean() ** 2 / v)
    return float(np.mean(vals)) if vals else float("nan")


def calculate_epi(raw, filtered):
    """Edge Preservation Index (Sattar et al., 1997): correlation of the Laplacian
    high-pass of the filtered image with that of the raw image. 1 = all edges
    (and all speckle) kept; must be read together with ENL, because a filter that
    does nothing also scores 1."""
    a = cv2.Laplacian(raw.astype(np.float64), cv2.CV_64F)
    b = cv2.Laplacian(filtered.astype(np.float64), cv2.CV_64F)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


def calculate_mor(raw, filtered, eps=1e-3):
    """Mean of Ratio image Y / X_hat. For an unbiased multiplicative-speckle
    filter the ratio image is pure speckle with unit mean (ideal = 1.0)."""
    m = filtered > eps
    return float((raw[m] / filtered[m]).mean())


def calculate_entropy(image):
    """Shannon entropy (bits) of the 8-bit luminance histogram."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image if image.dtype == np.uint8 else (image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = image if image.dtype == np.uint8 else (image * 255).astype(np.uint8)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


# --------------------------------------------------------------------------- #
# Uncertainty quality
# --------------------------------------------------------------------------- #
def sparsification(error, uncertainty, steps=20):
    """Sparsification curves (Ilg et al., 2018).

    Pixels are removed in order of decreasing uncertainty (model curve) or of
    decreasing true error (oracle curve); the mean error of the remaining pixels
    is recorded. AUSE = area between model and oracle curves (lower = better
    ranking of errors by the uncertainty)."""
    e = error.ravel()
    u = uncertainty.ravel()
    order_u = np.argsort(-u)
    order_e = np.argsort(-e)
    fr = np.linspace(0, 0.95, steps)
    n = e.size
    model, oracle = [], []
    for f in fr:
        k = int(f * n)
        model.append(e[order_u[k:]].mean())
        oracle.append(e[order_e[k:]].mean())
    model, oracle = np.array(model), np.array(oracle)
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    ause = float(_trapz(model - oracle, fr))
    return fr, model, oracle, ause


def uncertainty_error_spearman(error, uncertainty, max_pts=20000, seed=0):
    e, u = error.ravel(), uncertainty.ravel()
    if e.size > max_pts:
        idx = np.random.default_rng(seed).choice(e.size, max_pts, replace=False)
        e, u = e[idx], u[idx]
    return float(spearmanr(u, e).correlation)
