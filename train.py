"""
Train one model variant under an identical budget (iterations, batch, crop, optimiser).

    python train.py --variant ours --iters 4000

Variants (baselines + ablations, see VARIANTS):
    unet_l1      U-Net regression, L1 only                       (no GAN, no despeckling)
    pix2pix      U-Net + PatchGAN, L1 + adversarial [Isola'17]    (standard baseline)
    lee_pix2pix  classical Lee filter -> Pix2Pix                 (hand-crafted despeckling baseline)
    ours_nodesp  proposed loss, no despeckling stage             (ablation)
    ours_nophys  proposed, despeckler trained end-to-end only    (ablation: no physics prior)
    ours         proposed: homomorphic despeckler + physics prior + composite loss
"""
import argparse
import csv
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import PairedSARDataset
from losses import PhysicsInformedCompositeLoss
from models import SARColorizerGenerator, PatchGANDiscriminator, lee_filter

VARIANTS = {
    "unet_l1":     dict(despeckle=False, lee=False, gan=False, w=dict(lambda_l1=100, lambda_ssim=0, lambda_grad=0, lambda_adv=0, lambda_rad=0, lambda_tv=0)),
    "pix2pix":     dict(despeckle=False, lee=False, gan=True,  w=dict(lambda_l1=100, lambda_ssim=0, lambda_grad=0, lambda_adv=1, lambda_rad=0, lambda_tv=0)),
    "lee_pix2pix": dict(despeckle=False, lee=True,  gan=True,  w=dict(lambda_l1=100, lambda_ssim=0, lambda_grad=0, lambda_adv=1, lambda_rad=0, lambda_tv=0)),
    "ours_nodesp": dict(despeckle=False, lee=False, gan=True,  w=dict(lambda_l1=100, lambda_ssim=10, lambda_grad=10, lambda_adv=1, lambda_rad=0, lambda_tv=0)),
    "ours_nophys": dict(despeckle=True,  lee=False, gan=True,  w=dict(lambda_l1=100, lambda_ssim=10, lambda_grad=10, lambda_adv=1, lambda_rad=0, lambda_tv=0)),
    "ours":        dict(despeckle=True,  lee=False, gan=True,  w=dict(lambda_l1=100, lambda_ssim=10, lambda_grad=10, lambda_adv=1, lambda_rad=10, lambda_tv=0.5)),
}
# Improvements (v2): terrain-conditioned FiLM and an ImageNet-pretrained ResNet-18 encoder.
VARIANTS["ours_cond"] = dict(VARIANTS["ours"], cond=True)
VARIANTS["ours_r18"] = dict(VARIANTS["ours"], encoder="resnet18")
VARIANTS["ours_r18_cond"] = dict(VARIANTS["ours"], encoder="resnet18", cond=True)
# Non-adversarial versions: the interim benchmark showed the GAN term costs ~2 dB PSNR on unseen scenes.
_NOADV_W = dict(VARIANTS["ours"]["w"], lambda_adv=0)
VARIANTS["ours_noadv"] = dict(VARIANTS["ours"], gan=False, w=_NOADV_W)
VARIANTS["ours_r18_cond_noadv"] = dict(VARIANTS["ours"], gan=False, w=_NOADV_W, encoder="resnet18", cond=True)

CLASSES = ["agri", "barrenland", "grassland", "urban"]
LABEL_DROPOUT = 0.2  # train the "unknown terrain" embedding so the model also works without a label


def class_index(names, device):
    return torch.tensor([CLASSES.index(n) for n in names], dtype=torch.long, device=device)


def cond_labels(variant, names, device, train=False):
    """Terrain labels for conditioned variants (None otherwise)."""
    if not VARIANTS[variant].get("cond"):
        return None
    c = class_index(names, device)
    if train:
        drop = torch.rand(c.shape[0], device=device) < LABEL_DROPOUT
        c = torch.where(drop, torch.full_like(c, len(CLASSES)), c)
    return c


def build_generator(variant, pretrained=True):
    cfg = VARIANTS[variant]
    return SARColorizerGenerator(use_despeckle=cfg["despeckle"], encoder=cfg.get("encoder", "unet"),
                                 n_classes=len(CLASSES) if cfg.get("cond") else 0, pretrained=pretrained)


def prepare_input(variant, sar):
    return lee_filter(sar) if VARIANTS[variant]["lee"] else sar


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def validate(gen, variant, loader, device):
    gen.eval(); gen.set_mc_dropout(False)
    l1, psnr, n = 0.0, 0.0, 0
    for b in loader:
        sar = prepare_input(variant, b["sar"].to(device))
        opt = b["optical"].to(device)
        _, pred = gen(sar, cond_labels(variant, b["cls"], device))
        mse = ((pred - opt) ** 2).mean(dim=(1, 2, 3))
        l1 += (pred - opt).abs().mean(dim=(1, 2, 3)).sum().item()
        psnr += (10 * torch.log10(1.0 / mse.clamp_min(1e-10))).sum().item()
        n += sar.shape[0]
    gen.train()
    return l1 / n, psnr / n


def train(variant, iters=4000, batch_size=8, crop=128, lr=2e-4, val_every=500, seed=42):
    seed_all(seed)
    cfg = VARIANTS[variant]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] {variant} on {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'} | cfg={cfg}")

    train_set = PairedSARDataset("train", crop=crop, augment=True)
    val_set = PairedSARDataset("val", per_class=40, seed=1)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True,
                              num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=0)
    print(f"[*] train tiles={len(train_set)}  val tiles={len(val_set)}")

    gen = build_generator(variant).to(device)
    disc = PatchGANDiscriminator().to(device) if cfg["gan"] else None
    opt_g = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999)) if disc else None
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=iters, eta_min=lr * 0.05)
    crit = PhysicsInformedCompositeLoss(**cfg["w"])
    bce = torch.nn.BCEWithLogitsLoss()
    # fp16 autocast is ~20x slower and numerically unstable on the MX550, so train in fp32
    use_amp = False
    torch.backends.cudnn.benchmark = True
    scaler = torch.amp.GradScaler(enabled=use_amp)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(f"runs/{variant}", exist_ok=True)
    state_path = f"checkpoints/{variant}_last.pt"
    best, it, elapsed0 = float("inf"), 0, 0.0
    if os.path.exists(state_path):  # resume after an interruption / GPU stall
        st = torch.load(state_path, map_location=device, weights_only=False)
        gen.load_state_dict(st["gen"]); opt_g.load_state_dict(st["opt_g"]); sched_g.load_state_dict(st["sched_g"])
        if disc:
            disc.load_state_dict(st["disc"]); opt_d.load_state_dict(st["opt_d"])
        best, it, elapsed0 = st["best"], st["it"], st["elapsed"]
        random.setstate(st["py_rng"]); torch.set_rng_state(st["torch_rng"].cpu())
        print(f"[*] resumed {variant} at iteration {it}")
        log_f = open(f"runs/{variant}/log.csv", "a", newline="")
        log = csv.writer(log_f)
    else:
        log_f = open(f"runs/{variant}/log.csv", "w", newline="")
        log = csv.writer(log_f)
        log.writerow(["iter", "train_l1", "val_l1", "val_psnr", "elapsed_s"])

    def heartbeat():
        with open(f"runs/{variant}/heartbeat", "w") as hb:
            hb.write(str(it))

    t0, run_l1 = time.time() - elapsed0, []
    gen.train()
    while it < iters:
        for b in train_loader:
            if it >= iters:
                break
            raw = b["sar"].to(device, non_blocking=True)
            sar = prepare_input(variant, raw)
            opt = b["optical"].to(device, non_blocking=True)

            # ---- generator ----
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                clean, fake = gen(sar, cond_labels(variant, b["cls"], device, train=True))
                d_fake = disc(sar, fake) if disc else None
                loss_g, terms = crit(fake, opt, clean, sar, d_pred_fake=d_fake)
            opt_g.zero_grad(set_to_none=True)
            scaler.scale(loss_g).backward()
            scaler.step(opt_g)
            sched_g.step()

            # ---- discriminator ----
            if disc:
                with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                    d_real = disc(sar, opt)
                    d_fake = disc(sar, fake.detach())
                    loss_d = 0.5 * (bce(d_real, torch.full_like(d_real, 0.9)) + bce(d_fake, torch.zeros_like(d_fake)))
                opt_d.zero_grad(set_to_none=True)
                scaler.scale(loss_d).backward()
                scaler.step(opt_d)
            scaler.update()

            run_l1.append(terms["l1"].item())
            it += 1
            if it % 50 == 0:
                heartbeat()
            if it % 100 == 0:
                el = time.time() - t0
                print(f"  it {it}/{iters}  L1 {np.mean(run_l1[-100:]):.4f}  "
                      f"rad {float(terms['rad']):.4f} tv {float(terms['tv']):.3f}  {el / 60:.1f} min", flush=True)
            if it % val_every == 0 or it == iters:
                v_l1, v_psnr = validate(gen, variant, val_loader, device)
                log.writerow([it, float(np.mean(run_l1[-val_every:])), v_l1, v_psnr, time.time() - t0]); log_f.flush()
                print(f"  --> val L1 {v_l1:.4f}  PSNR {v_psnr:.2f} dB", flush=True)
                if v_l1 < best:
                    best = v_l1
                    torch.save(gen.state_dict(), f"checkpoints/{variant}.pth")
                torch.save(dict(gen=gen.state_dict(), opt_g=opt_g.state_dict(), sched_g=sched_g.state_dict(),
                                disc=disc.state_dict() if disc else None, opt_d=opt_d.state_dict() if disc else None,
                                best=best, it=it, elapsed=time.time() - t0, py_rng=random.getstate(),
                                torch_rng=torch.get_rng_state()), state_path)
    log_f.close()
    open(f"runs/{variant}/done", "w").write(f"{best}")
    print(f"[+] {variant}: best val L1 {best:.4f}, total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS) + ["all"], default="ours")
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--crop", type=int, default=128)
    args = ap.parse_args()
    names = list(VARIANTS) if args.variant == "all" else [args.variant]
    for v in names:
        train(v, iters=args.iters, batch_size=args.batch_size, crop=args.crop)
