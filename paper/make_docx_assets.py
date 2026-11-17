"""Image assets for the Word version of the paper: architecture diagram and equation images."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.join(os.path.dirname(__file__), "figs")
EQ = os.path.join(OUT, "eq")
os.makedirs(EQ, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 9})


def architecture():
    fig, ax = plt.subplots(figsize=(7.16, 2.15))
    ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")

    def box(x, y, w, h, text, fc="#ffffff", ls="-", ec="#444444"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8",
                                    fc=fc, ec=ec, lw=0.9, ls=ls))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.6, linespacing=1.25)
        return (x, y, w, h)

    def arrow(p, q, style="-|>", ls="-", color="#333333", text=None, toff=(0, 1.2)):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=8, lw=0.9, ls=ls, color=color))
        if text:
            ax.text((p[0] + q[0]) / 2 + toff[0], (p[1] + q[1]) / 2 + toff[1], text, ha="center", va="bottom",
                    fontsize=7, style="italic", color="#555555")

    blue = "#e8f0fb"
    y0, h = 12, 9
    sar = box(0.5, y0, 10, h, "SAR $Y$\n1×256×256")
    s1 = box(15, y0, 19, h, "Stage 1\nhomomorphic\ndespeckler", fc=blue)
    s2 = box(39.5, y0, 24, h, "Stage 2\nResNet-18 encoder +\nU-Net decoder (MC dropout)", fc=blue)
    s3 = box(69, y0, 15, h, "Stage 3\noptional\nfusion")
    out = box(89, y0, 10.5, h, "RGB $\\hat I$")
    mid = y0 + h / 2
    arrow((11.2, mid), (14.6, mid))
    arrow((34.4, mid), (39.1, mid), text="$\\hat X$")
    arrow((63.9, mid), (68.6, mid), text="$\\hat I_0$")
    arrow((84.4, mid), (88.6, mid))
    # despeckled SAR to fusion: explicit polyline below the boxes
    yb = 8.6
    ax.plot([29.5, 29.5, 79.5], [y0 - 0.4, yb, yb], color="#666666", lw=0.8)
    arrow((79.5, yb), (79.5, y0 - 0.4), color="#666666")
    ax.text(66, yb + 0.5, "despeckled $\\hat X$", ha="center", va="bottom", fontsize=7, style="italic", color="#555555")
    # FiLM
    film = box(39.5, 24.6, 24, 4.6, "terrain $c$ → FiLM $(\\gamma_c, \\beta_c)$", fc="#f2f2f2")
    arrow((51.5, 24.2), (51.5, 21.4), color="#666666")
    # losses
    lp = box(10, 0.6, 29, 4.6, "$\\mathcal{L}_{phys}=\\lambda_r\\mathcal{L}_{rad}+\\lambda_{tv}\\mathcal{L}_{tv}$",
             fc="#fdf1e6", ls="--", ec="#b07a3c")
    lc = box(41, 0.6, 33, 4.6, "$\\lambda_1\\mathcal{L}_1+\\lambda_s\\mathcal{L}_{SSIM}+\\lambda_g\\mathcal{L}_{grad}$",
             fc="#fdf1e6", ls="--", ec="#b07a3c")
    arrow((20.0, 5.6), (20.0, 11.6), ls="--", color="#b07a3c")
    arrow((50.0, 5.6), (50.0, 11.6), ls="--", color="#b07a3c")
    fig.savefig(os.path.join(OUT, "fig_architecture.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


EQUATIONS = {
    "eq1": r"$Y = X\cdot F,\qquad \mathbb{E}[F]=1$",
    "eq2": r"$\log Y = \log X + \log F$",
    "eq3": r"$\mathcal{L}_{rad} = \Vert \mathcal{A}_w(\hat X) - \mathcal{A}_w(Y) \Vert_1$",
    "eq4": r"$\mathcal{L}_{tv} = \Vert \nabla \log(\hat X+\epsilon) \Vert_1$",
    "eq5": r"$h \mapsto h\,(1+\gamma_c)+\beta_c$",
    "eq6": r"$\mathcal{L} = \lambda_1\mathcal{L}_1 + \lambda_s(1-\mathrm{SSIM}) + \lambda_g\mathcal{L}_{grad}"
           r" + \lambda_a\mathcal{L}_{GAN} + \lambda_r\mathcal{L}_{rad} + \lambda_{tv}\mathcal{L}_{tv}$",
    "eq7": r"$L^* = \alpha S + (1-\alpha)\,L^*_{pred}$",
    "eq8": r"$L^* = L^*_{pred} + \beta\,\frac{\sigma(L^*_{pred})}{\sigma(S)}\,(S - G_\sigma * S)$",
    "eq9": r"$\mu = \frac{1}{T}\sum_{t=1}^{T}\hat y_t,\qquad \sigma^2 = \frac{1}{T-1}\sum_{t=1}^{T}(\hat y_t-\mu)^2$",
}


def equations():
    for name, tex in EQUATIONS.items():
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, tex, fontsize=10)
        fig.savefig(os.path.join(EQ, f"{name}.png"), dpi=400, bbox_inches="tight", pad_inches=0.02,
                    facecolor="white")
        plt.close(fig)


def architecture_column():
    """Vertical version of the README (GitHub) mermaid flowchart, sized for one IEEE column."""
    fig, ax = plt.subplots(figsize=(3.4, 4.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 132); ax.axis("off")
    blue, orange, grey, ink = "#e8f0fb", "#fdf1e6", "#f2f2f2", "#333333"

    def box(cx, cy, w, h, text, fc="#ffffff", ls="-", ec="#444444", fs=7.2):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.4,rounding_size=1.5",
                                    fc=fc, ec=ec, lw=0.9, ls=ls))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, linespacing=1.2)

    def arrow(p, q, ls="-", color=ink):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=0.9, ls=ls, color=color))

    def label(x, y, text, ha="left"):
        ax.text(x, y, text, ha=ha, va="center", fontsize=6.6, style="italic", color="#555555")

    X, W = 50, 44                       # main column centre and width (edges at 28 and 72)
    box(X, 124, 40, 11, "Sentinel-1 SAR tile\n1×256×256")
    box(X, 98, W, 15, "Stage 1\nHomomorphic despeckler\n(log-domain residual CNN)", fc=blue)
    box(X, 68, W, 15, "Stage 2\nResNet-18 encoder +\nU-Net decoder (MC dropout)", fc=blue)
    box(X, 38, W, 12, "Stage 3 (optional)\nstructure-guided fusion")
    box(X, 12, 36, 9, "Optical-like RGB")
    arrow((X, 118.2), (X, 106.2))
    arrow((X, 89.8), (X, 76.2)); label(X + 1.5, 83, "despeckled $\\hat X$")
    arrow((X, 59.8), (X, 44.6)); label(X + 1.5, 52, "RGB $\\hat I_0$")
    arrow((X, 31.4), (X, 17.2))
    # left inputs
    box(11.5, 98, 19, 13, "Physics\nprior\n$L_{rad}+L_{tv}$", fc=orange, ls="--", ec="#b07a3c", fs=6.4)
    arrow((21.6, 98), (27.4, 98), ls="--", color="#b07a3c")
    box(11.5, 68, 19, 13, "Terrain\nclass $c$\n(FiLM)", fc=grey, fs=6.4)
    arrow((21.6, 68), (27.4, 68), color="#666666")
    # right input: training loss
    box(84, 68, 17, 15, "$L_1$ + SSIM\n+ gradient\nloss", fc=orange, ls="--", ec="#b07a3c", fs=6.4)
    arrow((75.1, 68), (72.6, 68), ls="--", color="#b07a3c")
    # high-pass of X̂ from Stage 1 to Stage 3 (routed along the right edge)
    ax.plot([72.6, 97, 97, 76], [98, 98, 38, 38], color="#666666", lw=0.8, ls=":")
    arrow((76, 38), (72.6, 38), ls=":", color="#666666")
    label(96, 84, "high-pass\nof $\\hat X$", ha="right")
    fig.savefig(os.path.join(OUT, "fig_architecture_column.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    architecture_column()
    equations()
    print("assets written")
