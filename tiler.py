import numpy as np
import torch


def generate_hann_2d(size):
    w = np.outer(np.hanning(size), np.hanning(size))
    return (w / (w.max() + 1e-7)).astype(np.float32) + 1e-3  # +eps so image borders keep weight


@torch.no_grad()
def process_large_swath_seamless(model, large_sar_array, device, tile_size=256, overlap=64):
    """Colorize a large SAR scene tile-by-tile and blend tiles with a 2-D Hann window.

    large_sar_array: float32 [H, W] in [0, 1] (same scaling as training).
    Returns RGB float32 [3, H, W] in [0, 1].
    """
    model.eval()
    h, w = large_sar_array.shape
    stride = tile_size - overlap
    pad_h = max(0, tile_size - h) + (stride - (max(h, tile_size) - tile_size) % stride) % stride
    pad_w = max(0, tile_size - w) + (stride - (max(w, tile_size) - tile_size) % stride) % stride
    padded = np.pad(large_sar_array, ((0, pad_h), (0, pad_w)), mode='reflect')
    ph, pw = padded.shape

    out = np.zeros((3, ph, pw), dtype=np.float32)
    weight = np.zeros((ph, pw), dtype=np.float32)
    win = generate_hann_2d(tile_size)

    for y in range(0, ph - tile_size + 1, stride):
        for x in range(0, pw - tile_size + 1, stride):
            patch = torch.from_numpy(padded[y:y + tile_size, x:x + tile_size].copy())[None, None].to(device)
            _, pred = model(patch)  # model input and output are both in [0, 1]
            out[:, y:y + tile_size, x:x + tile_size] += pred[0].cpu().numpy() * win
            weight[y:y + tile_size, x:x + tile_size] += win

    out /= np.maximum(weight, 1e-6)
    return np.clip(out[:, :h, :w], 0.0, 1.0)
