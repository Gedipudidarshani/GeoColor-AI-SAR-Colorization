| Method | PSNR (dB) ↑ | SSIM ↑ | SAM (°) ↓ | GMSD ↓ |
|---|---|---|---|---|
| Mean-colour prior (trivial floor) | 13.96 | 0.238 | 9.88 | 0.288 |
| U-Net, L1 only | 14.14 | 0.237 | 11.23 | 0.274 |
| Pix2Pix (Isola et al., 2017) | 12.58 | 0.149 | 16.96 | 0.241 |
| Lee filter + Pix2Pix | 12.64 | 0.133 | 12.42 | 0.247 |
| Ours v1 (with GAN) | 12.14 | 0.161 | 14.45 | 0.249 |
| Ours without GAN | 14.47 | 0.247 | 11.74 | 0.279 |
| **Ours (final)**: no GAN + ResNet-18 + terrain FiLM | 14.33 | 0.248 | 11.81 | 0.278 |
| Ours (final), run without terrain label | 14.55 | 0.249 | 11.26 | 0.279 |