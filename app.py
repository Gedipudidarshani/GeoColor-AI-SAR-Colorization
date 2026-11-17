import io
import json
import os

import matplotlib.cm as cm
import numpy as np
import streamlit as st
import torch
from PIL import Image

from fusion import chroma_boost, detail_injection, luminance_replacement
from metrics import calculate_enl, calculate_entropy, calculate_epi, homogeneous_blocks
from train import CLASSES, VARIANTS, build_generator

st.set_page_config(page_title="GeoColor AI: SAR Colorization", page_icon="🛰️", layout="wide")

def pick_variant():
    """Model chosen on the validation set by evaluate.py; otherwise the best available."""
    if os.path.exists("results/metrics.json"):
        with open("results/metrics.json") as f:
            main = json.load(f).get("main")
        if main and os.path.exists(f"checkpoints/{main}.pth"):
            return main
    for v in ["ours_r18_cond", "ours_r18", "ours_cond", "ours"]:
        if os.path.exists(f"checkpoints/{v}.pth"):
            return v
    return None


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variant = pick_variant()
    if variant is None:
        return None, device, None
    model = build_generator(variant, pretrained=False)
    model.load_state_dict(torch.load(f"checkpoints/{variant}.pth", map_location=device, weights_only=True))
    model.to(device).eval()
    return model, device, variant


@st.cache_data
def load_benchmark():
    if os.path.exists("results/metrics.json"):
        with open("results/metrics.json") as f:
            return json.load(f)
    return None


model, device, variant = load_model()
ckpt_used = f"checkpoints/{variant}.pth" if variant else None
is_cond = bool(variant and VARIANTS[variant].get("cond"))

st.title("🛰️ GeoColor AI: Physics-Guided SAR-to-Optical Colorization")
st.caption("Homomorphic despeckling → U-Net colorization → structure-guided fusion → Monte Carlo dropout uncertainty")
if ckpt_used is None:
    st.error("No trained checkpoint found. Run `python train.py --variant ours` first.")
    st.stop()
st.sidebar.success(f"Model: `{ckpt_used}`")

with st.sidebar:
    st.header("Inference settings")
    uploaded = st.file_uploader("Sentinel-1 SAR tile (grayscale)", type=["png", "jpg", "jpeg", "tif", "tiff"])
    terrain = st.selectbox("Terrain", ["Unknown"] + CLASSES, disabled=not is_cond,
                           help="Conditions the colour palette (FiLM). 'Unknown' uses the label-free embedding."
                           if is_cond else "The loaded model is not terrain-conditioned.")
    mc_passes = st.slider("Monte Carlo dropout passes", 2, 20, 10,
                          help="Number of stochastic forward passes. The colour shown is their mean; "
                               "the uncertainty map is their per-pixel variance.")
    fusion_mode = st.radio("Structure-guided fusion", ["Detail injection (proposed)", "Luminance replacement", "None (raw network output)"])
    strength = st.slider("Fusion strength (β or α)", 0.0, 1.0, 0.25, 0.05,
                         help="0.25 (β) and 0.1 (α) were selected on the validation scenes.")
    vibrancy = st.slider("Display saturation", 0.5, 2.0, 1.0, 0.05,
                         help="Display-only. Values other than 1.0 change colours for viewing and are not part of the evaluated method.")
    unc_threshold = st.number_input("Uncertainty flag threshold (std)", 0.0, 0.5, 0.05, 0.01, format="%.2f")

if not uploaded:
    st.info("Upload a Sentinel-1 tile, e.g. from `dataset/v_2/<terrain>/s1/`. Tiles from test scenes (see `splits.json`) were never seen in training.")
    bench = load_benchmark()
    if bench:
        st.subheader("Held-out benchmark (from `results/metrics.json`)")
        rows = [{"Method": bench["display"].get(m, m), "PSNR (dB)": round(s["psnr"], 2), "SSIM": round(s["ssim"], 3),
                 "SAM (°)": round(s["sam"], 2), "GMSD": round(s["gmsd"], 3)} for m, s in bench["summary"].items()]
        st.dataframe(rows, hide_index=True)
        st.caption(f"{bench['n_test_tiles']} test tiles from scenes not used for training.")
    st.stop()

raw_pil = Image.open(uploaded).convert("L")
if raw_pil.size != (256, 256):
    st.warning(f"Input is {raw_pil.size[0]}×{raw_pil.size[1]}; resized to 256×256 (training tile size).")
    raw_pil = raw_pil.resize((256, 256), Image.Resampling.BILINEAR)
raw = np.asarray(raw_pil, dtype=np.float32) / 255.0
x = torch.from_numpy(raw)[None, None].to(device)

with st.spinner(f"Running {mc_passes} Monte Carlo passes…"):
    cls = torch.tensor([CLASSES.index(terrain)], device=device) if (is_cond and terrain != "Unknown") else None
    clean_t, mean_t, var_t = model.predict_with_epistemic_uncertainty(x, passes=mc_passes, cls=cls)

clean = clean_t[0, 0].cpu().numpy()
pred = mean_t[0].permute(1, 2, 0).cpu().numpy()
std_map = np.sqrt(var_t[0, 0].cpu().numpy())

if fusion_mode.startswith("Detail"):
    rgb = detail_injection(clean, pred, beta=strength)
elif fusion_mode.startswith("Luminance"):
    rgb = luminance_replacement(clean, pred, alpha=strength)
else:
    rgb = pred
rgb_display = (np.clip(chroma_boost(rgb, vibrancy), 0, 1) * 255).astype(np.uint8)

unc_vis = std_map / (np.percentile(std_map, 99) + 1e-8)
unc_heat = (cm.inferno(np.clip(unc_vis, 0, 1))[..., :3] * 255).astype(np.uint8)

c1, c2, c3, c4 = st.columns(4)
c1.image(raw_pil, caption="1. Raw SAR backscatter", width="stretch")
c2.image((clean * 255).astype(np.uint8), caption="2. Despeckled reflectivity (stage 1)", width="stretch")
c3.image(rgb_display, caption="3. Colorized optical-like RGB", width="stretch")
c4.image(unc_heat, caption="4. Epistemic uncertainty (MC-dropout std)", width="stretch")

st.divider()
st.subheader("Quality indicators for this tile (no ground truth needed)")
blocks = homogeneous_blocks(raw)
enl_raw, enl_clean = calculate_enl(raw, blocks), calculate_enl(clean, blocks)
flagged = float((std_map > unc_threshold).mean() * 100)
m1, m2, m3, m4 = st.columns(4)
m1.metric("ENL (homogeneous areas)", f"{enl_clean:.1f}", f"{enl_clean - enl_raw:+.1f} vs raw",
          help="Equivalent Number of Looks on the same homogeneous blocks of raw and despeckled SAR. Higher = less speckle.")
m2.metric("Edge Preservation Index", f"{calculate_epi(raw, clean):.3f}",
          help="Laplacian correlation with the raw SAR (1 = all edges kept). Read together with ENL.")
m3.metric("Output entropy", f"{calculate_entropy(rgb_display):.2f} bits")
m4.metric("Pixels flagged uncertain", f"{flagged:.1f}%", help=f"Share of pixels with MC-dropout std > {unc_threshold:.2f}.")
st.caption(f"Mean predictive std: {std_map.mean():.4f}. Colours are a learned estimate of the optical appearance, "
           "not a measurement; treat high-uncertainty regions with caution.")

buf = io.BytesIO()
Image.fromarray(rgb_display).save(buf, format="PNG")
st.download_button("Download colorized PNG", buf.getvalue(), file_name=f"colorized_{os.path.splitext(uploaded.name)[0]}.png", mime="image/png")
