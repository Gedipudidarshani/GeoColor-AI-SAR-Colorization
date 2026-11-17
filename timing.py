"""Inference cost per 256x256 tile (batch 1, after warm-up). Writes results/timing.json."""
import json
import time

import torch

from train import VARIANTS, build_generator, prepare_input


@torch.no_grad()
def measure(variant, device, n=100, warmup=20):
    g = build_generator(variant, pretrained=False).to(device).eval()
    x = torch.rand(1, 1, 256, 256, device=device)
    c = torch.zeros(1, dtype=torch.long, device=device) if VARIANTS[variant].get("cond") else None
    for _ in range(warmup):
        g(prepare_input(variant, x), c)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        g(prepare_input(variant, x), c)
    torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t) / n, sum(p.numel() for p in g.parameters()) / 1e6


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda")
    out = {}
    for v in ["unet_l1", "pix2pix", "lee_pix2pix", "ours", "ours_noadv", "ours_r18_cond_noadv"]:
        ms, p = measure(v, dev)
        out[v] = {"ms": ms, "params_m": p}
        print(f"{v:22s} {p:5.2f} M  {ms:6.1f} ms/tile")
    json.dump({"device": torch.cuda.get_device_name(0), **out}, open("results/timing.json", "w"), indent=1)
