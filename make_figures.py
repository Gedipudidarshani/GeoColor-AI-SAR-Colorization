"""
Generate every figure and LaTeX table of the paper from results/metrics.json,
runs/*/log.csv and results/qual/*. Nothing here is typed in by hand.

    python make_figures.py      ->  paper/figs/*.pdf|png, paper/tables/*.tex
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT_F, OUT_T = "paper/figs", "paper/tables"
os.makedirs(OUT_F, exist_ok=True); os.makedirs(OUT_T, exist_ok=True)

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
GREY, GREY_LIGHT = "#8d8c87", "#c3c2b7"
# fixed entity -> colour mapping (validated categorical order), same in every figure
COLOR = {"pix2pix": "#eb6834", "lee_pix2pix": "#1baf7a", "unet_l1": "#eda100", "ours": "#6250d6"}
MARK = {"pix2pix": "s", "lee_pix2pix": "^", "unet_l1": "D", "ours": "v"}
MAIN_COLOR = "#2a78d6"
SHORT = {"sar_gray": "SAR as grey", "mean_color": "Mean colour", "unet_l1": "U-Net (L1)", "pix2pix": "Pix2Pix",
         "lee_pix2pix": "Lee + Pix2Pix", "ours_nodesp": "Ours w/o despeckle", "ours_nophys": "Ours w/o physics prior",
         "ours": "Ours (v1)", "ours+lum": "Ours + lum. repl.", "ours+detail": "Ours + detail inj.",
         "ours_cond": "Ours + FiLM", "ours_cond@nolabel": "Ours + FiLM (no label)", "ours_r18": "Ours + ResNet-18",
         "ours_r18_cond": "Ours + R18 + FiLM", "ours_r18_cond@nolabel": "Ours + R18 + FiLM (no label)",
         "ours_noadv": "Ours w/o GAN", "ours_r18_cond_noadv": "Ours w/o GAN + R18 + FiLM",
         "ours_r18_cond_noadv@nolabel": "Ours w/o GAN + R18 + FiLM (no label)"}
CLS = ["agri", "barrenland", "grassland", "urban"]
CLS_NAME = {"agri": "Agriculture", "barrenland": "Barren", "grassland": "Grassland", "urban": "Urban"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.facecolor": SURF, "axes.facecolor": SURF,
})


def save(fig, name):
    fig.savefig(f"{OUT_F}/{name}.pdf"); fig.savefig(f"{OUT_F}/{name}.png")
    plt.close(fig)


R = json.load(open("results/metrics.json"))
S = R["summary"]
MAIN = R.get("main", "ours")
COLOR[MAIN] = MAIN_COLOR
MARK[MAIN] = "o"
SHORT[MAIN] = "Ours (final)" if MAIN != "ours" else "Ours"
ORDER = [m for m in dict.fromkeys(["sar_gray", "mean_color", "unet_l1", "pix2pix", "lee_pix2pix", "ours_nodesp", "ours_nophys",
                                   "ours", "ours_cond", "ours_r18", "ours_r18_cond", "ours_noadv", "ours_r18_cond_noadv",
                                   MAIN + "@nolabel", MAIN, "ours+lum", "ours+detail"]) if m in S]


# --------------------------------------------------------------------- Fig: comparison
def fig_comparison():
    methods = list(ORDER)
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 0.6 + 0.19 * len(methods)), sharey=True)
    for ax, (k, lab, better) in zip(axes, [("psnr", "PSNR (dB)  higher is better", "hi"),
                                           ("ssim", "SSIM  higher is better", "hi"),
                                           ("sam", "SAM (°)  lower is better", "lo")]):
        vals = [S[m][k] for m in methods]
        cols = [MAIN_COLOR if m in (MAIN, "ours+detail", "ours+lum") else
                (GREY_LIGHT if m in ("sar_gray", "mean_color") else GREY) for m in methods]
        y = np.arange(len(methods))[::-1]
        ax.barh(y, vals, color=cols, height=0.62, edgecolor=SURF, linewidth=1.0)
        lo = min(vals) * (0.9 if k != "sam" else 0.95)
        ax.set_xlim(lo if k == "psnr" else 0, max(vals) * 1.13)
        for yi, v in zip(y, vals):
            ax.text(v, yi, f" {v:.2f}" if k != "ssim" else f" {v:.3f}", va="center", fontsize=6.5, color=INK)
        ax.set_xlabel(lab); ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(np.arange(len(methods))[::-1]); axes[0].set_yticklabels([SHORT[m] for m in methods])
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_comparison")


# --------------------------------------------------------------------- Fig: per terrain
def fig_per_terrain():
    methods = [m for m in dict.fromkeys(["unet_l1", "pix2pix", "lee_pix2pix", MAIN]) if m in R["per_class"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.1))
    for ax, (k, lab) in zip(axes, [("psnr", "PSNR (dB)"), ("sam", "SAM (°)")]):
        w = 0.8 / len(methods)
        x = np.arange(len(CLS))
        for j, m in enumerate(methods):
            vals = [R["per_class"][m][c][k] for c in CLS]
            xs = x - 0.4 + w / 2 + j * w
            ax.bar(xs, vals, w * 0.92, color=COLOR[m], label=SHORT[m], edgecolor=SURF, linewidth=0.8)
            for xi, v in zip(xs, vals):
                ax.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=5.8, color=INK)
        ax.set_xticks(x); ax.set_xticklabels([CLS_NAME[c] for c in CLS]); ax.set_ylabel(lab)
        ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
        if k == "psnr":
            allv = [R["per_class"][m][c][k] for m in methods for c in CLS]
            ax.set_ylim(min(allv) - 2, max(allv) + 1.2)
    axes[0].legend(ncol=4, loc="upper left", bbox_to_anchor=(0, 1.2), fontsize=7)
    fig.tight_layout()
    save(fig, "fig_per_terrain")


# --------------------------------------------------------------------- Fig: training curves
def fig_training():
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    for m in dict.fromkeys(["unet_l1", "pix2pix", "lee_pix2pix", "ours", MAIN]):
        p = f"runs/{m}/log.csv"
        if not os.path.exists(p):
            continue
        rows = list(csv.DictReader(open(p)))
        if not rows:
            continue
        it = [int(r["iter"]) for r in rows]; ps = [float(r["val_psnr"]) for r in rows]
        ax.plot(it, ps, color=COLOR[m], lw=2 if m == MAIN else 1.2, marker=MARK[m], ms=3.5,
                label=SHORT[m], markeredgecolor=SURF, markeredgewidth=0.6, zorder=3 if m == MAIN else 2)
    ax.set_xlabel("Training iteration (batch 8, 128×128 crops)"); ax.set_ylabel("Validation PSNR (dB)")
    ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.legend(fontsize=6, ncol=2, loc="lower right")
    fig.tight_layout(); save(fig, "fig_training")


# --------------------------------------------------------------------- Fig: sparsification
def fig_sparsification():
    U = R.get("uncertainty")
    if not U:
        return
    fr = np.array(U["fractions"]) * 100
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    ax.plot(fr, U["random_curve"], color=GREY, lw=1.2, ls="--", label="Random removal")
    ax.plot(fr, U["model_curve"], color=MAIN_COLOR, lw=2, marker="o", ms=3, markeredgecolor=SURF, label="MC-dropout uncertainty")
    ax.plot(fr, U["oracle_curve"], color=INK, lw=1.2, label="Oracle (true error)")
    ax.fill_between(fr, U["oracle_curve"], U["model_curve"], color=MAIN_COLOR, alpha=0.12, lw=0)
    ax.text(fr[len(fr) // 2], np.mean([U["model_curve"][len(fr) // 2], U["oracle_curve"][len(fr) // 2]]),
            f"AUSE = {U['ause']:.4f}", fontsize=7, color=INK, ha="center")
    ax.set_xlabel("Pixels removed, most uncertain first (%)"); ax.set_ylabel("Mean abs. error of remaining")
    ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True); ax.legend(fontsize=6.5)
    fig.tight_layout(); save(fig, "fig_sparsification")


# --------------------------------------------------------------------- Fig: fusion sweep
def fig_fusion():
    F = R.get("fusion_sweep")
    if not F:
        return
    base = S[MAIN]
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9))
    for ax, (k, lab) in zip(axes, [("psnr", "PSNR (dB)"), ("gmsd", "GMSD (lower better)")]):
        for pref, col, mk, name in [("lum", COLOR["pix2pix"], "s", "Luminance repl."), ("detail", MAIN_COLOR, "o", "Detail injection")]:
            ks = sorted([x for x in F if x.startswith(pref)], key=lambda s: float(s.split("_")[1]))
            xs = [0.0] + [float(x.split("_")[1]) for x in ks]
            ys = [base[k]] + [F[x][k] for x in ks]
            ax.plot(xs, ys, color=col, marker=mk, ms=3.5, lw=1.5, label=name, markeredgecolor=SURF)
        ax.set_xlabel("Fusion strength (α or β)"); ax.set_ylabel(lab, fontsize=7)
        ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    axes[0].legend(fontsize=6, loc="lower left")
    fig.tight_layout(); save(fig, "fig_fusion")


# --------------------------------------------------------------------- Fig: qualitative
def _img(p):
    return np.asarray(Image.open(p)) if os.path.exists(p) else None


def fig_qualitative():
    cols = [("sar", "SAR input"), ("lee", "Lee filter"), ("ours_clean", "Ours stage 1"), ("unet_l1", "U-Net (L1)"), ("pix2pix", "Pix2Pix"),
            ("lee_pix2pix", "Lee+Pix2Pix"), ("main", "Ours"), ("ours+detail", "Ours+detail"), ("gt", "Sentinel-2 GT")]
    cols = [c for c in cols if os.path.exists(f"results/qual/agri_1_{c[0]}.png")]
    fig, axes = plt.subplots(len(CLS), len(cols), figsize=(7.16, 7.16 * len(CLS) / len(cols) + 0.25))
    for r, c in enumerate(CLS):
        for j, (k, title) in enumerate(cols):
            ax = axes[r, j]; im = _img(f"results/qual/{c}_1_{k}.png")
            if im is not None:
                ax.imshow(im, cmap="gray" if im.ndim == 2 else None, vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(title, fontsize=7)
            if j == 0:
                ax.set_ylabel(CLS_NAME[c], fontsize=7)
    fig.subplots_adjust(wspace=0.03, hspace=0.03)
    save(fig, "fig_qualitative")


def fig_uncertainty():
    if not os.path.exists("results/qual/unc_urban_std.npy"):
        return
    fig, axes = plt.subplots(len(CLS), 4, figsize=(3.5, 3.6))
    for r, c in enumerate(CLS):
        std = np.load(f"results/qual/unc_{c}_std.npy"); err = np.load(f"results/qual/unc_{c}_err.npy")
        panels = [(_img(f"results/qual/unc_{c}_sar.png"), "gray", "SAR"), (_img(f"results/qual/unc_{c}_pred.png"), None, "Ours"),
                  (err, "magma", "|error|"), (std, "magma", "MC std")]
        for j, (im, cmap, t) in enumerate(panels):
            ax = axes[r, j]
            vmax = np.percentile(im, 99) if cmap == "magma" else None
            ax.imshow(im, cmap=cmap, vmin=0 if cmap else None, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(t, fontsize=7)
            if j == 0:
                ax.set_ylabel(CLS_NAME[c], fontsize=7)
    fig.subplots_adjust(wspace=0.03, hspace=0.03)
    save(fig, "fig_uncertainty")


# --------------------------------------------------------------------- LaTeX tables
def best_idx(vals, hi=True):
    return int(np.argmax(vals) if hi else np.argmin(vals))


def table_main():
    methods = [m for m in dict.fromkeys(["sar_gray", "mean_color", "unet_l1", "pix2pix", "lee_pix2pix", "ours", MAIN, MAIN + "@nolabel", "ours+lum", "ours+detail"]) if m in S]
    W = R.get("wilcoxon_vs_ours", {})
    learned = [m for m in methods if m not in ("sar_gray", "mean_color")]
    cols = {k: [S[m][k] for m in learned] for k in ["psnr", "ssim", "sam", "gmsd"]}
    best = {"psnr": learned[best_idx(cols["psnr"])], "ssim": learned[best_idx(cols["ssim"])],
            "sam": learned[best_idx(cols["sam"], False)], "gmsd": learned[best_idx(cols["gmsd"], False)]}
    L = [r"\begin{tabular}{lcccc}", r"\toprule",
         r"Method & PSNR (dB)$\uparrow$ & SSIM$\uparrow$ & SAM ($^\circ$)$\downarrow$ & GMSD$\downarrow$ \\", r"\midrule"]
    for m in methods:
        if m == "unet_l1":
            L.append(r"\midrule")
        if m == MAIN:
            L.append(r"\midrule")
        s = S[m]
        def f(k, fmt):
            v = fmt.format(s[k]); sd = fmt.format(s[k + "_std"])
            star = "" if m == MAIN or m not in W or W[m].get(k, 1) < 0.05 or k == "gmsd" else r"$^\dagger$"
            v = rf"\textbf{{{v}}}" if best.get(k) == m else v
            return f"{v}$\\pm${sd}{star}"
        L.append(f"{SHORT[m]} & {f('psnr', '{:.2f}')} & {f('ssim', '{:.3f}')} & {f('sam', '{:.2f}')} & {f('gmsd', '{:.3f}')} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT_T}/main.tex", "w").write("\n".join(L))


def table_ablation():
    ck, no = r"\checkmark", r"--"
    #        variant                S1  Lphys Ls,g  GAN  R18  FiLM
    rows = [("unet_l1",             no, no,   no,   no,  no,  no),
            ("pix2pix",             no, no,   no,   ck,  no,  no),
            ("ours_nodesp",         no, no,   ck,   ck,  no,  no),
            ("ours_nophys",         ck, no,   ck,   ck,  no,  no),
            ("ours",                ck, ck,   ck,   ck,  no,  no),
            ("ours_cond",           ck, ck,   ck,   ck,  no,  ck),
            ("ours_r18",            ck, ck,   ck,   ck,  ck,  no),
            ("ours_r18_cond",       ck, ck,   ck,   ck,  ck,  ck),
            ("ours_noadv",          ck, ck,   ck,   no,  no,  no),
            ("ours_r18_cond_noadv", ck, ck,   ck,   no,  ck,  ck)]
    D = R["despeckling"]
    L = [r"\begin{tabular}{lcccccccccc}", r"\toprule",
         r"Variant & S1 & $\mathcal{L}_{phys}$ & $\mathcal{L}_{s,g}$ & GAN & R18 & FiLM & PSNR$\uparrow$ & SSIM$\uparrow$ & SAM$\downarrow$ & MoR \\", r"\midrule"]
    for m, *flags in rows:
        if m not in S:
            continue
        mor = f"{D[m]['mor']:.3f}" if m in D else "--"
        name = rf"\textbf{{{SHORT[m]}}}" if m == MAIN else SHORT[m]
        L.append(f"{name} & " + " & ".join(flags) + f" & {S[m]['psnr']:.2f} & {S[m]['ssim']:.3f} & {S[m]['sam']:.2f} & {mor} " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT_T}/ablation.tex", "w").write("\n".join(L))


def table_despeckle():
    D = R["despeckling"]
    names = {"raw": "Raw SAR (no filter)", "lee": "Lee filter ($7{\\times}7$, $L{=}40$)",
             "ours_nophys": "Stage 1, no physics prior", "ours": "Stage 1 + physics prior (v1)",
             "ours_cond": "Stage 1 + prior (FiLM)", "ours_r18": "Stage 1 + prior (R18)", "ours_r18_cond": "Stage 1 + prior (R18+FiLM)",
             "ours_noadv": "Stage 1 + prior, no GAN", "ours_r18_cond_noadv": "Stage 1 + prior, no GAN (R18+FiLM)"}
    L = [r"\begin{tabular}{lccc}", r"\toprule", r"Filter & ENL$\uparrow$ & EPI$\uparrow$ & MoR (ideal 1) \\", r"\midrule"]
    for k in ["raw", "lee", "ours_nophys", "ours", "ours_cond", "ours_r18", "ours_r18_cond", "ours_noadv", "ours_r18_cond_noadv"]:
        if k in D:
            L.append(f"{names[k]} & {D[k]['enl']:.1f} & {D[k]['epi']:.3f} & {D[k]['mor']:.3f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT_T}/despeckle.tex", "w").write("\n".join(L))


def table_per_terrain():
    methods = [m for m in dict.fromkeys(["unet_l1", "pix2pix", "lee_pix2pix", MAIN]) if m in R["per_class"]]
    L = [r"\begin{tabular}{l" + "cc" * len(CLS) + "}", r"\toprule",
         " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{CLS_NAME[c]}}}" for c in CLS) + r" \\",
         " & " + " & ".join(["PSNR & SAM"] * len(CLS)) + r" \\", r"\midrule"]
    for m in methods:
        cells = []
        for c in CLS:
            ps = R["per_class"][m][c]["psnr"]; sa = R["per_class"][m][c]["sam"]
            bp = max(R["per_class"][x][c]["psnr"] for x in methods); bs = min(R["per_class"][x][c]["sam"] for x in methods)
            cells.append((rf"\textbf{{{ps:.2f}}}" if ps == bp else f"{ps:.2f}"))
            cells.append((rf"\textbf{{{sa:.2f}}}" if sa == bs else f"{sa:.2f}"))
        L.append(f"{SHORT[m]} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT_T}/per_terrain.tex", "w").write("\n".join(L))


def table_cost():
    T = json.load(open("results/timing.json")) if os.path.exists("results/timing.json") else {}
    L = [r"\begin{tabular}{lcc}", r"\toprule", r"Model & Params (M) & ms / tile$^\ast$ \\", r"\midrule"]
    for m in dict.fromkeys(["unet_l1", "pix2pix", "lee_pix2pix", "ours", "ours_noadv", MAIN]):
        if m in T:
            L.append(f"{SHORT[m]} & {T[m]['params_m']:.2f} & {T[m]['ms']:.1f} " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT_T}/cost.tex", "w").write("\n".join(L))


if __name__ == "__main__":
    for fn in [fig_comparison, fig_per_terrain, fig_training, fig_sparsification, fig_fusion, fig_qualitative,
               fig_uncertainty, table_main, table_ablation, table_despeckle, table_per_terrain, table_cost]:
        fn(); print("ok", fn.__name__)


# --------------------------------------------------------------------- numbers for the prose
def numbers_tex():
    """\newcommand macros so every number quoted in the text comes from metrics.json."""
    import re
    def name(s):
        s = s.replace("@nolabel", "NoLabel").replace("+", "Plus")
        parts = re.split(r"[_\W]+", s)
        out = "".join(p.capitalize() for p in parts if p)
        return re.sub(r"\d", lambda m: "ABCDEFGHIJ"[int(m.group())], out)
    L = [f"% auto-generated by make_figures.py from results/metrics.json -- do not edit",
         rf"\newcommand{{\nTest}}{{{R['n_test_tiles']}}}",
         rf"\newcommand{{\mainName}}{{{SHORT.get(MAIN, MAIN)}}}"]
    for m, s in S.items():
        n = name(m)
        L += [rf"\newcommand{{\psnr{n}}}{{{s['psnr']:.2f}}}", rf"\newcommand{{\ssim{n}}}{{{s['ssim']:.3f}}}",
              rf"\newcommand{{\sam{n}}}{{{s['sam']:.2f}}}", rf"\newcommand{{\gmsd{n}}}{{{s['gmsd']:.3f}}}"]
    for k, d in R["despeckling"].items():
        n = name(k)
        L += [rf"\newcommand{{\enl{n}}}{{{d['enl']:.1f}}}", rf"\newcommand{{\epi{n}}}{{{d['epi']:.3f}}}",
              rf"\newcommand{{\mor{n}}}{{{d['mor']:.3f}}}"]
    U = R.get("uncertainty") or {}
    if U:
        L += [rf"\newcommand{{\ause}}{{{U['ause']:.4f}}}", rf"\newcommand{{\uncRho}}{{{U['spearman']:.3f}}}",
              rf"\newcommand{{\mcT}}{{{U['passes']}}}", rf"\newcommand{{\mcGain}}{{{U['mc_mean_psnr_gain']:+.2f}}}",
              rf"\newcommand{{\nMC}}{{{U['n_tiles']}}}"]
    open(f"{OUT_T}/numbers.tex", "w").write("\n".join(L) + "\n")


numbers_tex()
print("ok numbers_tex")


def derived_tex():
    """Differences and p-values quoted in the text (appended to numbers.tex)."""
    W = R.get("wilcoxon_vs_ours", {})
    def d(a, b, k="psnr"):
        return S[a][k] - S[b][k] if a in S and b in S else float("nan")
    def p(m, k):
        v = W.get(m, {}).get(k, float("nan"))
        if v != v:
            return "n/a"
        if v < 1e-3:
            e = int(np.floor(np.log10(v)))
            return rf"\ensuremath{{<10^{{{e + 1}}}}}"
        return f"{v:.3f}"
    nl = MAIN + "@nolabel"
    L = [rf"\newcommand{{\dGanCost}}{{{d('ours_noadv', 'ours'):.2f}}}",
         rf"\newcommand{{\dPixVsUnet}}{{{d('unet_l1', 'pix2pix'):.2f}}}",
         rf"\newcommand{{\dFinalVsUnet}}{{{d(MAIN, 'unet_l1'):.2f}}}",
         rf"\newcommand{{\dFinalVsPix}}{{{d(MAIN, 'pix2pix'):.2f}}}",
         rf"\newcommand{{\dFinalVsMean}}{{{d(MAIN, 'mean_color'):.2f}}}",
         rf"\newcommand{{\dNoLabelVsUnet}}{{{d(nl, 'unet_l1'):.2f}}}",
         rf"\newcommand{{\dNoLabelVsFinal}}{{{d(nl, MAIN):.2f}}}",
         rf"\newcommand{{\dSsimFinalVsUnet}}{{{d(MAIN, 'unet_l1', 'ssim'):.3f}}}",
         rf"\newcommand{{\pFinalUnetPsnr}}{{{p('unet_l1', 'psnr')}}}",
         rf"\newcommand{{\pFinalUnetSsim}}{{{p('unet_l1', 'ssim')}}}",
         rf"\newcommand{{\pFinalUnetSam}}{{{p('unet_l1', 'sam')}}}",
         rf"\newcommand{{\pFinalPixPsnr}}{{{p('pix2pix', 'psnr')}}}",
         rf"\newcommand{{\pFinalMeanPsnr}}{{{p('mean_color', 'psnr')}}}",
         rf"\newcommand{{\pNoLabelPsnr}}{{{p(nl, 'psnr')}}}",
         rf"\newcommand{{\fusionAlpha}}{{{R.get('fusion_alpha', float('nan'))}}}",
         rf"\newcommand{{\fusionBeta}}{{{R.get('fusion_beta', float('nan'))}}}"]
    open(f"{OUT_T}/numbers.tex", "a").write("\n".join(L) + "\n")


derived_tex()
print("ok derived_tex")
