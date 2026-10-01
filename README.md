# Physics-Guided SAR Image Colorization: Despeckling, Swin-Attention & Latent Diffusion Pipeline


[![Status](https://img.shields.io/badge/status-research%20prototype-yellow.svg)](#-future-work)
[![PyTorch](https://img.shields.io/badge/framework-PyTorch-EE4C2C.svg)](https://pytorch.org/)
[![Sentinel-1/2](https://img.shields.io/badge/data-Sentinel--1%20%2B%20Sentinel--2-1F6FEB.svg)](https://dataspace.copernicus.eu/)
[![ISRO SIH1733](https://img.shields.io/badge/problem%20statement-ISRO%20SIH1733-F37626.svg)](https://www.sih.gov.in/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An end-to-end remote sensing, generative AI, and geospatial data-engineering framework that converts single-polarization, speckle-corrupted **Synthetic Aperture Radar (SAR)** imagery into realistic **optical-like RGB maps**. Built on paired **Sentinel-1 / Sentinel-2** satellite telemetry, a modular **PyTorch** inference pipeline, and an interactive dashboard for side-by-side inspection and quantitative benchmarking.

---

## 📌 Executive Summary & Problem Statement

Optical satellites (Sentinel-2, Landsat) are blinded by clouds, smoke, fog and darkness, precisely when disaster-response teams need imagery most. SAR satellites solve the visibility problem because radar penetrates clouds and works day and night. They create a new one: raw SAR is a grainy, monochrome backscatter image that only trained radar analysts can interpret quickly.

This project addresses **ISRO Smart India Hackathon problem statement SIH1733**:

> **"SAR Image Colorization for Comprehensive Insight using Deep Learning Model"**

Existing image-to-image translation models (Pix2Pix, CycleGAN) tend to fail on radar data for four reasons:

| # | Failure Mode | Consequence |
| :--- | :--- | :--- |
| 1 | Speckle noise and colour translation are learned *simultaneously* | Noise is mistaken for real land-cover texture |
| 2 | Purely data-driven losses with no radar physics | Hallucinated features (e.g., vegetation over calm water) |
| 3 | Adversarial training instability | Mode collapse and colour bleeding |
| 4 | Aggressive resizing to 8-bit RGB | Loss of radiometric depth and georeferencing |

This pipeline decouples **despeckling** from **colour translation**, adds **structure-guided fusion** to preserve radar edges, estimates **epistemic uncertainty**, and defines a research track (Swin-conditioned latent diffusion with a physics-informed loss) targeting peer-reviewed publication.

---

## 🏗️ System Architecture

The pipeline spans four architectural layers:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Geospatial Data Layer (Sentinel-1 GRD + Sentinel-2 L2A)                  │
│    - Paired s1 / s2 tiles (SEN12MS / Kaggle Sentinel-1&2 pairs)             │
│    - dB calibration of backscatter to [-25 dB, 0 dB] and [0, 1] scaling     │
│    - ROI-level train / validation split (no spatial leakage)                │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. Noise Decoupling Layer (Residual Despeckling CNN)                        │
│    - Estimates multiplicative speckle component F from Y = F · X            │
│    - Outputs clean reflectivity map X before any colour translation         │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. Colorization & Fusion Layer                                              │
│    - SARColorizerGenerator: clean SAR → raw RGB prediction                  │
│    - Structure-guided LAB fusion: radar luminance injected into L channel   │
│    - Monte-Carlo dropout passes → per-pixel epistemic uncertainty           │
│    - [Research track] Swin encoder + VAE + conditional latent diffusion     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. Evaluation & Presentation Layer (Dashboard + Benchmark Scripts)          │
│    - Side-by-side: raw SAR | despeckled | colorized output                  │
│    - PSNR, SSIM, SAM, Edge Preservation Index, ENL, uncertainty readout     │
│    - Verified S1–S2 pair benchmarking (eval_direct.py)                      │
└─────────────────────────────────────────────────────────────────────────────┘
```
<img width="2040" height="1995" alt="architecture" src="https://github.com/user-attachments/assets/b15dd8d7-fe7b-4e47-9708-f55cfb15af33" />

### Research Architecture (Target Design)

```text
Raw SAR (Sentinel-1)
   │
   ▼
[1] Residual Despeckling CNN          → isolates F, outputs clean X
   │
   ▼
[2] Swin-Transformer Encoder          → multi-scale shifted-window attention features (c)
   │
   ▼
[3] VAE (KL-f8) latent space          → 256×256×3  ⇄  32×32×4
   │
   ▼
[4] Conditional Latent Diffusion U-Net (cross-attention on c)
   │
   ▼
Colorized optical RGB  (+ GeoTIFF export with CRS preserved)
```

---

## 📊 Metric Hierarchy & Evaluation Taxonomy

Metrics are partitioned into functional tiers, so that no single score is over-interpreted:

| Tier | Metric Name | Formulation | Role |
| :--- | :--- | :--- | :--- |
| **Data Integrity Gate** | S1–S2 Pair Verification | Filename / geography match check | Abort evaluation if pairs are mismatched |
| **Fidelity Metric** | PSNR | $10\log_{10}(\text{MAX}^2 / \text{MSE})$ | Pixel-level reconstruction quality |
| **Structural Metric** | SSIM | Luminance / contrast / structure comparison | Boundary and texture preservation |
| **Spectral Metric** | SAM | $\arccos\left(\frac{\mathbf{p}\cdot\mathbf{g}}{\lVert\mathbf{p}\rVert\lVert\mathbf{g}\rVert}\right)$ | Colour-vector angle vs. ground truth |
| **Despeckling Metric 1** | Edge Preservation Index (EPI) | Gradient ratio on top-25% edges | Confirms edges survive noise removal |
| **Despeckling Metric 2** | Equivalent Number of Looks (ENL) | $\mu^2 / \sigma^2$ on homogeneous patches | Measures speckle suppression |
| **Trust Metric** | Epistemic Variance | Variance across MC-dropout passes | Flags low-confidence regions |
| **Distributional Metric** *(planned)* | FID | Fréchet distance in Inception space | Realism of generated colour distribution |

---

## 🧮 Mathematical Formulation

### 1. Multiplicative Speckle Model

SAR intensity is corrupted by multiplicative speckle noise:

$$Y = F \cdot X$$

where $Y$ is the observed intensity, $F$ is the fading speckle component, and $X$ is the noise-free radar reflectivity. Stage 1 estimates $F$ and returns the clean reflectivity $\hat{X}$ before colour translation.

### 2. Logarithmic (dB) Calibration

Raw backscatter is mapped to a bounded range so that training and inference share the same radiometry:

$$I_{\text{norm}} = \frac{I_{\text{dB}} - \text{dB}_{\min}}{\text{dB}_{\max} - \text{dB}_{\min}}, \quad \text{dB}_{\min} = -25, \; \text{dB}_{\max} = 0$$

### 3. Composite Loss (Research Track)

$$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{diffusion}} + \lambda_2 \mathcal{L}_{\text{SSIM}} + \lambda_3 \mathcal{L}_{\text{speckle}} + \lambda_4 \mathcal{L}_{\text{perceptual}}$$

| Term | Definition | Purpose |
| :--- | :--- | :--- |
| $\mathcal{L}_{\text{diffusion}}$ | $\mathbb{E}_{t,z_0,\epsilon}\left[\lVert \epsilon - \epsilon_\theta(z_t, t, c)\rVert_2^2\right]$ | Noise prediction in VAE latent space |
| $\mathcal{L}_{\text{SSIM}}$ | $1 - \text{SSIM}(I_{\text{pred}}, I_{\text{optical}})$ | Edge and structure preservation |
| $\mathcal{L}_{\text{speckle}}$ | $\sum_{i,j} \lvert I(i{+}1,j) - I(i,j)\rvert + \lvert I(i,j{+}1) - I(i,j)\rvert$ | Total-variation / backscatter-conservation regulariser |
| $\mathcal{L}_{\text{perceptual}}$ | $\sum_l \frac{1}{N_l}\lVert \phi_l(I_{\text{pred}}) - \phi_l(I_{\text{optical}})\rVert_1$ | VGG-19 feature matching for natural colour |

### 4. Swin Shifted-Window Attention

Window-based attention computes self-attention inside local windows and shifts the windows in alternating layers, giving linear complexity $\mathcal{O}(N)$ instead of the $\mathcal{O}(N^2)$ of global ViT attention:

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^{T}}{\sqrt{d_k}}\right)V$$

---

## 🧪 Evaluation Results

> ⚠️ **Replace the `TBD` cells with your own measured values.** Target numbers from planning documents must not be published as results.

| Model | PSNR (dB) ↑ | SSIM ↑ | SAM (°) ↓ | EPI ↑ | ENL ↑ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Pix2Pix baseline | _TBD_ | _TBD_ | _TBD_ | – | – |
| CycleGAN baseline | _TBD_ | _TBD_ | _TBD_ | – | – |
| Swin-Transformer (standalone) | _TBD_ | _TBD_ | _TBD_ | – | – |
| **This work (prototype)** | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

### Evaluation Integrity Notes

1. **Pair integrity:** Compare each SAR tile only with its own co-registered Sentinel-2 tile. Mismatched pairs push SSIM toward zero.
2. **Matching radiometry:** Apply the same dB calibration at inference and evaluation time as in training.
3. **Sensor offset:** S1 and S2 are acquired at different times and angles. A 2–4 pixel shift penalises pixel-wise SSIM / PSNR even when structure looks correct.
4. **Declare the evaluated product:** State whether metrics are computed on the raw generator output or on the structure-fused output.

---

## 📁 Repository Structure

```text
sar-image-colorization/
├── .gitignore                       # Excludes venv, cache, checkpoints and raw data
├── README.md                        # System documentation and findings
├── requirements.txt                 # Pinned dependencies
├── app.py                           # Interactive colorization dashboard
├── models.py                        # SARColorizerGenerator (despeckle + colorize + uncertainty)
├── dataset.py                       # Dataloaders, dB calibration, ROI-based split
├── train.py                         # Training loop and checkpointing
├── evaluate.py                      # Validation-split benchmark
├── eval_direct.py                   # Verified S1–S2 pair benchmark
├── dataset/
│   └── .../{s1,s2}/                 # Paired SAR and optical tiles (not committed)
└── checkpoints/
    └── best_sar_colorizer.pth       # Best trained weights
```

> ✏️ Adjust names if your repository layout differs.

---

## 🚀 Getting Started

### 1. Prerequisites
* Python 3.10, 3.11, or 3.12
* Git
* NVIDIA GPU recommended for training (CPU is sufficient for inference)

### 2. Installation & Setup

```powershell
# Clone the repository
git clone https://github.com/Gedipudidarshani/sar-image-colorization.git
cd sar-image-colorization

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1    # On Linux/macOS: source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 3. Prepare the Dataset

Place paired tiles under `dataset/` with matching filenames in `s1` and `s2` folders.

| Source | Notes |
| :--- | :--- |
| [SEN12MS](https://mediatum.ub.tum.de/1474000) | Co-registered S1/S2 patches, global coverage |
| Kaggle: *Sentinel-1&2 Image Pairs (SAR & Optical)* | Ready-to-use pairs grouped by land cover |
| [Copernicus Data Space](https://dataspace.copernicus.eu) | Raw S1 GRD / S2 L2A for custom regions |
| Google Earth Engine | `COPERNICUS/S1_GRD`, `COPERNICUS/S2_SR` |

### 4. Train the Model

```powershell
python train.py
```
Best weights are saved to `checkpoints/best_sar_colorizer.pth`.

### 5. Run the Benchmarks

```powershell
python evaluate.py        # validation-split benchmark
python eval_direct.py     # strictly verified S1–S2 pairs
```

### 6. Launch the Interactive Dashboard

```powershell
python app.py
```
Upload a SAR image to view the raw input, the despeckled map, the colorized output and the quality metrics side by side.

---
<img width="1917" height="1021" alt="image" src="https://github.com/user-attachments/assets/a5d72fb3-12de-4a5c-88f5-7bddbea7c13e" />

<img width="1917" height="1023" alt="image" src="https://github.com/user-attachments/assets/1c6973e5-7422-4da0-b031-73c138abd955" />
<img width="1917" height="1027" alt="image" src="https://github.com/user-attachments/assets/32b057b6-0d55-4ec5-92c1-ba6d3f01ac6e" />
<img width="1917" height="1022" alt="image" src="https://github.com/user-attachments/assets/abdd5073-d702-4aae-a6c8-466d2585f207" />
<img width="982" height="235" alt="image" src="https://github.com/user-attachments/assets/8403d989-2f49-4957-b7f7-c44823525f57" />

<img width="1917" height="873" alt="image" src="https://github.com/user-attachments/assets/f2c80ac7-6c44-45ce-947e-e5c85a3a0a6e" />


## 🔭 Future Work

The current prototype establishes the despeckling, colorization, fusion and uncertainty pipeline. The following extensions are planned, grouped by research theme.

### 1. Generative Core
- [ ] **Swin-conditioned latent diffusion:** Replace the generator with a conditional latent diffusion U-Net guided by Swin-Transformer attention features, using a pretrained KL-f8 VAE.
- [ ] **Fast sampling:** Explore few-step or one-step diffusion (distillation, consistency models) to cut inference latency for operational use.
- [ ] **Foundation-model priors:** Evaluate self-supervised remote-sensing or DINO-style encoders as structural priors.
- [ ] **Baselines:** Implement and report Pix2Pix, CycleGAN and standalone Swin baselines on identical splits.

### 2. Physics-Informed Learning
- [ ] **Full `L_speckle` integration** into the composite loss, with ablations over $\lambda_3$.
- [ ] **Scattering-aware constraints:** Incorporate backscatter anisotropy, incidence angle and layover/shadow geometry.
- [ ] **Multi-polarization input:** Use dual-pol (VV + VH) Sentinel-1 channels instead of single-channel intensity.
- [ ] **Physical plausibility checks:** Automatically flag outputs that contradict backscatter physics (e.g., vegetation over specular water).

### 3. Data & Geospatial Engineering
- [ ] **Native 16-bit GeoTIFF I/O** with GDAL / rasterio, preserving CRS / EPSG tags and bounds, with GeoTIFF export from the dashboard.
- [ ] **Misalignment-robust training and evaluation:** Sub-pixel registration or shift-tolerant SSIM to handle S1–S2 offsets.
- [ ] **Regional fine-tuning:** Build Indian-region datasets (flood plains, coastal belts, Western Ghats, urban sprawl) via Google Earth Engine and ISRO Bhuvan.
- [ ] **Multi-temporal inputs:** Use SAR time series to resolve colour ambiguity and track change.

### 4. Evaluation & Trust
- [ ] **FID, LPIPS and ERGAS** following the standard SAR-colorization benchmarking protocol.
- [ ] **Per-land-cover analysis:** Report metrics separately for urban, vegetation, water and coastal tiles.
- [ ] **Downstream task validation:** Test whether colorized outputs improve flood segmentation or land-cover classification accuracy.
- [ ] **Calibrated uncertainty:** Validate that MC-dropout (or ensemble) uncertainty correlates with real error, and expose it as a confidence layer.
- [ ] **Human interpretability study:** Measure how quickly non-expert users identify features in raw SAR vs. colorized output.

### 5. Deployment & Productization
- [ ] **REST API** (FastAPI) for model inference and batch processing.
- [ ] **Docker image** bundling PyTorch, GDAL and the dashboard for reproducible deployment.
- [ ] **GIS integration:** QGIS plugin and STAC-compatible output for use in existing mapping workflows.
- [ ] **Efficiency:** Mixed precision (FP16), ONNX / TensorRT quantisation and tiled inference for large scenes.
- [ ] **Near-real-time disaster pipeline:** Automatic ingestion of new Sentinel-1 passes over flood or cyclone-affected regions.

### 6. Research Output
- [ ] Ablation study (two-stage vs. single-stage, with vs. without physics loss).
- [ ] Manuscript targeting IEEE GRSL / JSTARS / TGRS or IGARSS.
- [ ] Public release of trained weights and evaluation scripts for reproducibility.

---

## 🌱 Real-World Impact & SDG Alignment

| SDG | Operational Use Case |
| :--- | :--- |
| **SDG 13: Climate Action** | All-weather flood, cyclone and storm-damage mapping |
| **SDG 15: Life on Land** | Deforestation and land-degradation monitoring in cloudy regions |
| **SDG 11: Sustainable Cities** | Urban growth and illegal-construction tracking |
| **SDG 6: Clean Water** | River, reservoir and wetland monitoring |
| **SDG 2: Zero Hunger** | Crop monitoring and agricultural damage assessment |

---

## ⚖️ Disclaimer

This is an academic research prototype. Colorized outputs are **model-generated approximations**, not true optical observations, and must not be used alone for safety-critical, defence or disaster-response decisions.

---

## 📚 References

1. Q. Song, F. Xu, and Y.-Q. Jin, "Radar image colorization: Converting single-polarization to fully polarimetric using deep neural networks," *IEEE Access*, vol. 6, 2018.
2. G. Ji *et al.*, "SAR image colorization using multidomain cycle-consistency generative adversarial network," *IEEE Geosci. Remote Sens. Lett.*, 2022.
3. P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, "Image-to-image translation with conditional adversarial networks," in *Proc. IEEE CVPR*, 2017.
4. J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros, "Unpaired image-to-image translation using cycle-consistent adversarial networks," in *Proc. IEEE ICCV*, 2017.
5. Z. Liu *et al.*, "Swin Transformer: Hierarchical vision transformer using shifted windows," in *Proc. IEEE/CVF ICCV*, 2021.
6. R. Rombach *et al.*, "High-resolution image synthesis with latent diffusion models," in *Proc. IEEE/CVF CVPR*, 2022.
7. M. Schmitt, L. H. Hughes, and X. X. Zhu, "The SEN12MS dataset for deep learning in remote sensing," *ISPRS Annals*, vol. IV-2/W7, 2019.
8. P. Ebel, A. Meraner, M. Schmitt, and X. X. Zhu, "Multisensor data fusion for cloud removal in global and all-season Sentinel-2 imagery," *IEEE Trans. Geosci. Remote Sens.*, 2021.
9. K. Shen, G. Vivone, X. Yang, S. Lolli, and M. Schmitt, "A benchmarking protocol for SAR colorization: From regression to deep learning approaches," *IEEE J. Sel. Topics Appl. Earth Obs. Remote Sens.*, 2024.

> 🔎 Verify volume, pages and DOIs before citing in a paper.

---

## 👥 Author
* **Gedipudi Darshani** - *Artificial Intelligence & Data Science*
* **Keerthana P** - *Artificial Intelligence & Machine Learning*
* **Yenuganti Prathyusha** - *Artificial Intelligence & Machine Learning*

**Mentor:** Selvanayaki S · **Institution:** Saveetha Engineering College
