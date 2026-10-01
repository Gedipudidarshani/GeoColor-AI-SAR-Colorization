# GeoColor-AI-SAR-Colorization
## Physics-Guided SAR-to-Optical Colorization Engine & Guardrail Telemetry Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/framework-PyTorch%202.0+-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Remote Sensing Standards](https://img.shields.io/badge/benchmarks-IEEE%20GRSS-success.svg)](https://www.grss-ieee.org/)

An end-to-end physics-guided computer vision, deep generative translation, and operational remote-sensing pipeline developed for **ISRO SIH Problem Statement 1733** (*SAR Image Colorization for Comprehensive Insight Using Deep Learning*). Translates single-polarization **Sentinel-1** Synthetic Aperture Radar (SAR) into calibrated **Sentinel-2** multi-spectral optical representations using homomorphic log-domain despeckling, multi-scale structural guidance, and Bayesian epistemic uncertainty guardrails adhering to **IEEE Geoscience and Remote Sensing Society (GRSS)** benchmarking standards.

---

## 📌 Executive Summary & Problem Statement

Synthetic Aperture Radar (SAR) remote sensing delivers continuous, all-weather Earth observation by penetrating dense clouds, adverse atmospheric conditions, and nocturnal darkness. However, single-polarization C-band backscatter ($Y = F \cdot X$) produces monochromatic rasters corrupted by multiplicative Rayleigh-distributed speckle noise and radar-specific geometric distortions (foreshortening, layover, and shadowing). These phenomena present critical cognitive interpretation barriers for human analysts and induce catastrophic hallucinations in standard computer vision models.

Naive cross-domain generative models (such as standard Pix2Pix or CycleGAN) fail on cross-modal satellite pairs:
* Multiplicative speckle gradients induce false edges, boundary smearing, and mode collapse.
* Lack of physical backscatter constraints leads to extreme neon-cyan/magenta hallucinations and unstable land-cover synthesis.
* Black-box predictions omit uncertainty estimation, making autonomous tactical deployment unsafe.

This end-to-end engineering pipeline solves cross-modal translation by decoupling noise reduction from synthesis:
1. **Stage 1 (Homomorphic Log-Domain Despeckling):** Converts multiplicative fading noise into an additive model, suppressing speckle while isolating true ground reflectivity.
2. **Stage 2 (Multi-Scale Structural Colorization):** Synthesizes multi-spectral chromaticity using a custom U-Net core with structural guidance, preserving road networks, water bodies, and building footprints 1-to-1.
3. **Stage 3 (Operational Uncertainty Guardrail):** Estimates test-time Bayesian epistemic variance via Monte Carlo sampling, providing active hallucination risk telemetry and quantitative confidence maps.

---

## 🏗️ System Architecture

The pipeline spans four distinct architectural layers:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Ingestion & Radiometric Preprocessing Layer (GDAL / Rasterio / NumPy)    │
│    - Decibel (dB) dynamic range normalization ([-25.0 dB, 0.0 dB] to [0, 1])│
│    - Multi-scale spatial tile extraction (256x256 co-registered patches)    │
│    - EPSG / CRS geographic projection and 16-bit GeoTIFF metadata retention  │
│    - Co-registered Sentinel-1 (C-band SAR) & Sentinel-2 (MSI Optical) pairs │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. Physics-Guided Generative Translation Core (PyTorch / CUDA)              │
│    - Homomorphic Log-Domain Decoupling: ln(Y) = ln(X) + ln(F)               │
│    - Modified U-Net with skip-connected multi-scale feature bottlenecks      │
│    - PatchGAN spatial discriminator enforcing local textural consistency    │
│    - Composite Physics Loss: L_total = L1 + lambda_adv*L_GAN + lambda_tv*L_TV│
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. Automated Guardrail Engine & Bayesian Inference (SciPy / OpenCV)         │
│    - Test-time Monte Carlo Dropout (M = 10 stochastic forward passes)        │
│    - Pixel-wise Epistemic Variance mapping: sigma^2(x, y)                   │
│    - Structure-Guided LAB Chrominance Fusion & Corner-Reflector Clamping    │
│    - Edge-gated Sobel gradient verification on structural contours          │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. Operational Telemetry & Benchmarking Layer (Streamlit / Plotly)          │
│    - 4-Panel Synchronized Geo-Visualizer (SAR, Despeckled, RGB, Uncertainty)│
│    - Real-time vibrancy modulation and dynamic contrast stretching          │
│    - Automated IEEE standard evaluation suite (SSIM, PSNR, SAM, EPI, ENL)   │
└─────────────────────────────────────────────────────────────────────────────┘
