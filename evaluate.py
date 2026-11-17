"""
Honest benchmark on the held-out test scenes (splits.json, no scene overlap with training).

    python evaluate.py --per_class 150 --mc_passes 10

Writes results/metrics.json, results/per_image.csv and results/qual/*.png.
Every number is computed directly from the model output; nothing is rescaled.
"""
import argparse
import csv
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from scipy.stats import wilcoxon
from torch.utils.data import DataLoader

from dataset import PairedSARDataset
from fusion import detail_injection, luminance_replacement
from metrics import (calculate_enl, calculate_epi, calculate_gmsd, calculate_mor, calculate_psnr,
                     calculate_sam, calculate_ssim, homogeneous_blocks, sparsification,
                     uncertainty_error_spearman)
from models import lee_filter
from train import VARIANTS, build_generator, cond_labels, prepare_input

DISPLAY = {
    "sar_gray": "SAR as grey (floor)",
    "mean_color": "Mean-colour prior (floor)",
    "unet_l1": "U-Net (L1)",
    "pix2pix": "Pix2Pix [Isola'17]",
    "lee_pix2pix": "Lee filter + Pix2Pix",
    "ours_nodesp": "Ours w/o despeckling stage",
    "ours_nophys": "Ours w/o physics prior",
    "ours": "Ours (full)",
    "ours_cond": "Ours + terrain FiLM",
    "ours_cond@nolabel": "Ours + terrain FiLM (no label)",
    "ours_r18": "Ours + pretrained ResNet-18",
    "ours_r18_cond": "Ours + ResNet-18 + terrain FiLM",
    "ours_r18_cond@nolabel": "Ours + ResNet-18 + FiLM (no label)",
    "ours_noadv": "Ours w/o GAN",
    "ours_r18_cond_noadv": "Ours w/o GAN + ResNet-18 + FiLM",
    "ours_r18_cond_noadv@nolabel": "Ours w/o GAN + R18 + FiLM (no label)",
    "ours+lum": "Ours + luminance replacement (val-tuned a)",
    "ours+detail": "Ours + detail injection (val-tuned b)",
}


def rgb_metrics(gt, pred):
    return dict(psnr=calculate_psnr(gt, pred), ssim=calculate_ssim(gt, pred),
                sam=calculate_sam(pred, gt), gmsd=calculate_gmsd(gt, pred))


def train_mean_color(n=400):
    ds = PairedSARDataset("train", per_class=n // 4, seed=3)
    acc = np.zeros(3)
    for i in range(len(ds)):
        acc += ds[i]["optical"].mean(dim=(1, 2)).numpy()
    return acc / len(ds)


def to_np(t):
    return t.detach().cpu().permute(0, 2, 3, 1).numpy() if t.ndim == 4 else t.detach().cpu().numpy()


def save_png(arr, path):
    a = np.clip(arr, 0, 1)
    Image.fromarray((a * 255).astype(np.uint8)).save(path)


@torch.no_grad()
def run(per_class=150, mc_passes=10, mc_per_class=40, qual_per_class=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    os.makedirs("results/qual", exist_ok=True)

    test = PairedSARDataset("test", per_class=per_class, seed=7)
    loader = DataLoader(test, batch_size=8, shuffle=False, num_workers=0)
    print(f"[*] test tiles: {len(test)} (scenes unseen in training)")

    # only models whose training has finished (runs/<v>/done); never score a half-trained checkpoint
    variants = [v for v in VARIANTS if os.path.exists(f"checkpoints/{v}.pth") and os.path.exists(f"runs/{v}/done")]
    print(f"[*] trained variants found: {variants}")
    # main model = proposed variant with the lowest *validation* L1 (test set never used for selection)
    cands = [v for v in variants if v.startswith("ours") and v not in ("ours_nodesp", "ours_nophys")
             and os.path.exists(f"runs/{v}/done")]
    MAIN = min(cands, key=lambda v: float(open(f"runs/{v}/done").read())) if cands else "ours"
    print(f"[*] main model (selected on validation): {MAIN}")
    models = {}
    for v in variants:
        g = build_generator(v, pretrained=False).to(device)
        g.load_state_dict(torch.load(f"checkpoints/{v}.pth", map_location=device, weights_only=True))
        g.eval(); g.set_mc_dropout(False)
        models[v] = g

    mean_rgb = train_mean_color()

    # fusion strength chosen on the VALIDATION scenes (never on test)
    ALPHAS, BETAS = [0.1, 0.2, 0.3, 0.5, 0.7], [0.25, 0.5, 1.0]
    alpha_best, beta_best, val_fusion = 0.7, 0.5, {}
    if MAIN in models:
        vset = PairedSARDataset("val", per_class=40, seed=1)
        acc = defaultdict(list)
        for j in range(len(vset)):
            s_ = vset[j]
            x_ = s_["sar"][None].to(device)
            cl_, pr_ = models[MAIN](prepare_input(MAIN, x_), cond_labels(MAIN, [s_["cls"]], device))
            c_np, p_np = cl_[0, 0].cpu().numpy(), pr_[0].permute(1, 2, 0).cpu().numpy()
            g_np = s_["optical"].permute(1, 2, 0).numpy()
            acc["none"].append(calculate_psnr(g_np, p_np))
            for a_ in ALPHAS:
                acc[f"lum_{a_}"].append(calculate_psnr(g_np, luminance_replacement(c_np, p_np, a_)))
            for b_ in BETAS:
                acc[f"detail_{b_}"].append(calculate_psnr(g_np, detail_injection(c_np, p_np, b_)))
        val_fusion = {k: float(np.mean(v)) for k, v in acc.items()}
        alpha_best = max(ALPHAS, key=lambda a_: val_fusion[f"lum_{a_}"])
        beta_best = max(BETAS, key=lambda b_: val_fusion[f"detail_{b_}"])
        print(f"[*] fusion chosen on validation: alpha={alpha_best}, beta={beta_best}  (val PSNR {val_fusion})")
    rows = []                      # per-image metric rows
    desp = defaultdict(lambda: defaultdict(list))
    fusion_sweep = defaultdict(list)
    qual_count = defaultdict(int)
    timing = {}

    idx = 0
    for b in loader:
        raw = b["sar"].to(device)
        gt = to_np(b["optical"])
        raw_np = raw.cpu().numpy()[:, 0]
        preds, cleans = {}, {}
        for v, g in models.items():
            torch.cuda.synchronize() if device.type == "cuda" else None
            t0 = time.time()
            clean, pred = g(prepare_input(v, raw), cond_labels(v, b["cls"], device))
            torch.cuda.synchronize() if device.type == "cuda" else None
            timing.setdefault(v, []).append((time.time() - t0) / raw.shape[0])
            preds[v] = to_np(pred)
            cleans[v] = clean.cpu().numpy()[:, 0]
            if VARIANTS[v].get("cond"):  # same model without the terrain label
                _, pred_nl = g(prepare_input(v, raw), None)
                preds[v + "@nolabel"] = to_np(pred_nl)
        lee_np = lee_filter(raw).cpu().numpy()[:, 0]

        for i in range(raw.shape[0]):
            cls, fname = b["cls"][i], b["filename"][i]
            g_i = gt[i]
            outs = {"sar_gray": np.repeat(raw_np[i][..., None], 3, 2),
                    "mean_color": np.broadcast_to(mean_rgb, g_i.shape).astype(np.float32)}
            for v in preds:
                outs[v] = preds[v][i]
            if MAIN in models:
                outs["ours+lum"] = luminance_replacement(cleans[MAIN][i], preds[MAIN][i], alpha_best)
                outs["ours+detail"] = detail_injection(cleans[MAIN][i], preds[MAIN][i], beta_best)
                for a in [0.1, 0.3, 0.5, 0.7, 0.9]:
                    fusion_sweep[f"lum_{a}"].append(rgb_metrics(g_i, luminance_replacement(cleans[MAIN][i], preds[MAIN][i], a)))
                for be in [0.25, 0.5, 1.0, 1.5]:
                    fusion_sweep[f"detail_{be}"].append(rgb_metrics(g_i, detail_injection(cleans[MAIN][i], preds[MAIN][i], be)))
            for m, p in outs.items():
                rows.append(dict(method=m, cls=cls, file=fname, **rgb_metrics(g_i, p)))

            # despeckling quality on the SAME homogeneous blocks of the raw tile
            blocks = homogeneous_blocks(raw_np[i])
            filt = {"raw": raw_np[i], "lee": lee_np[i]}
            for v in [x for x in models if VARIANTS[x]["despeckle"]]:
                if v in cleans:
                    filt[v] = cleans[v][i]
            for k, f in filt.items():
                desp[k]["enl"].append(calculate_enl(f, blocks))
                desp[k]["epi"].append(calculate_epi(raw_np[i], f))
                desp[k]["mor"].append(calculate_mor(raw_np[i], f))

            if qual_count[cls] < qual_per_class:
                qual_count[cls] += 1
                stem = f"{cls}_{qual_count[cls]}"
                save_png(raw_np[i], f"results/qual/{stem}_sar.png")
                save_png(lee_np[i], f"results/qual/{stem}_lee.png")
                save_png(g_i, f"results/qual/{stem}_gt.png")
                for m in ["pix2pix", "lee_pix2pix", "unet_l1", "ours_nodesp", "ours", MAIN, "ours+detail", "ours+lum"]:
                    if m in outs:
                        save_png(outs[m], f"results/qual/{stem}_{m}.png")
                if MAIN in cleans:
                    save_png(cleans[MAIN][i], f"results/qual/{stem}_ours_clean.png")
                    save_png(preds[MAIN][i], f"results/qual/{stem}_main.png")
            idx += 1
        print(f"  {idx}/{len(test)}", end="\r", flush=True)
    print()

    # ---------------- aggregate ----------------
    methods = list(dict.fromkeys(r["method"] for r in rows))
    keys = ["psnr", "ssim", "sam", "gmsd"]
    summary, per_class = {}, {}
    for m in methods:
        rs = [r for r in rows if r["method"] == m]
        summary[m] = {k: float(np.mean([r[k] for r in rs])) for k in keys}
        summary[m].update({k + "_std": float(np.std([r[k] for r in rs])) for k in keys})
        per_class[m] = {c: {k: float(np.mean([r[k] for r in rs if r["cls"] == c])) for k in keys}
                        for c in sorted({r["cls"] for r in rs})}

    # paired Wilcoxon signed-rank test: ours vs every other method
    signif = {}
    if MAIN in methods:
        base = {r["file"] + r["cls"]: r for r in rows if r["method"] == MAIN}
        for m in methods:
            if m == MAIN:
                continue
            other = {r["file"] + r["cls"]: r for r in rows if r["method"] == m}
            common = sorted(set(base) & set(other))
            signif[m] = {}
            for k in ["psnr", "ssim", "sam"]:
                a = np.array([base[c][k] for c in common]); o = np.array([other[c][k] for c in common])
                try:
                    signif[m][k] = float(wilcoxon(a, o).pvalue)
                except ValueError:
                    signif[m][k] = float("nan")

    # median over tiles: one near-constant block can make a single tile's ENL explode
    desp_summary = {k: {m: float(np.nanmedian(v)) for m, v in d.items()} for k, d in desp.items()}
    fusion_summary = {k: {m: float(np.mean([r[m] for r in v])) for m in keys} for k, v in fusion_sweep.items()}

    # ---------------- MC-dropout uncertainty ----------------
    unc = {}
    if MAIN in models:
        g = models[MAIN]
        sub = PairedSARDataset("test", per_class=mc_per_class, seed=7)
        curves_m, curves_o, auses, rhos, mvars, rand_curve = [], [], [], [], [], []
        mc_psnr_gain = []
        for i in range(len(sub)):
            s = sub[i]
            x = s["sar"][None].to(device)
            gt_i = s["optical"].permute(1, 2, 0).numpy()
            c = cond_labels(MAIN, [s["cls"]], device)
            _, det, _ = g.predict_with_epistemic_uncertainty(x, passes=1, cls=c)
            _, mean, var = g.predict_with_epistemic_uncertainty(x, passes=mc_passes, cls=c)
            mean_np = mean[0].permute(1, 2, 0).cpu().numpy()
            err = np.abs(mean_np - gt_i).mean(-1)
            u = var[0, 0].cpu().numpy()
            fr, cm, co, ause = sparsification(err, u)
            curves_m.append(cm); curves_o.append(co); auses.append(ause)
            rand_curve.append(np.full_like(cm, err.mean()))
            rhos.append(uncertainty_error_spearman(err, u))
            mvars.append(float(u.mean()))
            mc_psnr_gain.append(calculate_psnr(gt_i, mean_np) - calculate_psnr(gt_i, det[0].permute(1, 2, 0).cpu().numpy()))
            if i % (mc_per_class) == 0:
                save_png(s["sar"][0].numpy(), f"results/qual/unc_{s['cls']}_sar.png")
                save_png(mean_np, f"results/qual/unc_{s['cls']}_pred.png")
                save_png(gt_i, f"results/qual/unc_{s['cls']}_gt.png")
                np.save(f"results/qual/unc_{s['cls']}_std.npy", np.sqrt(u))
                np.save(f"results/qual/unc_{s['cls']}_err.npy", err)
                save_png(u / (np.percentile(u, 99) + 1e-12), f"results/qual/unc_{s['cls']}.png")
                save_png(err / (np.percentile(err, 99) + 1e-12), f"results/qual/err_{s['cls']}.png")
        unc = dict(passes=mc_passes, n_tiles=len(sub), fractions=fr.tolist(),
                   model_curve=np.mean(curves_m, 0).tolist(), oracle_curve=np.mean(curves_o, 0).tolist(),
                   random_curve=np.mean(rand_curve, 0).tolist(), ause=float(np.mean(auses)),
                   spearman=float(np.mean(rhos)), mean_variance=float(np.mean(mvars)),
                   mc_mean_psnr_gain=float(np.mean(mc_psnr_gain)))

    params = {v: sum(p.numel() for p in g.parameters()) / 1e6 for v, g in models.items()}
    ms = {v: 1000 * float(np.median(t[1:] if len(t) > 1 else t)) for v, t in timing.items()}

    out = dict(n_test_tiles=len(test), main=MAIN, fusion_alpha=alpha_best, fusion_beta=beta_best, val_fusion=val_fusion, display=DISPLAY, summary=summary, per_class=per_class,
               wilcoxon_vs_ours=signif, despeckling=desp_summary, fusion_sweep=fusion_summary,
               uncertainty=unc, params_m=params, ms_per_tile=ms)
    with open("results/metrics.json", "w") as f:
        json.dump(out, f, indent=1)
    with open("results/per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print("\n" + "=" * 84)
    print(f"{'Method':40s} {'PSNR':>8s} {'SSIM':>7s} {'SAM(deg)':>9s} {'GMSD':>7s}")
    print("-" * 84)
    for m in methods:
        s = summary[m]
        print(f"{DISPLAY.get(m, m):40s} {s['psnr']:8.2f} {s['ssim']:7.3f} {s['sam']:9.2f} {s['gmsd']:7.3f}")
    print("=" * 84)
    print("Despeckling (raw / Lee / ours):", json.dumps(desp_summary, indent=1))
    if unc:
        print(f"MC-dropout: AUSE={unc['ause']:.4f}  Spearman={unc['spearman']:.3f}  mean var={unc['mean_variance']:.2e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_class", type=int, default=150)
    ap.add_argument("--mc_passes", type=int, default=10)
    ap.add_argument("--mc_per_class", type=int, default=40)
    a = ap.parse_args()
    run(a.per_class, a.mc_passes, a.mc_per_class)
