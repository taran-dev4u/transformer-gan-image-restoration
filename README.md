# Dual-Stage Transformer-GAN Architecture for High-Resolution Facial Image Restoration

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Interactive%20Portfolio-brightgreen?logo=googlechrome)](https://taran-dev4u.github.io/taran-portfolio/#projects)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c?logo=pytorch)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Transformers-yellow)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An end-to-end deep learning framework combining **Vision Transformers (ViT)** for global structural prior recovery with **Patch-based Generative Adversarial Networks (PatchGAN)** for photorealistic texture synthesis on severely degraded facial portraits.

---

## 🎯 Architecture Overview

```
Degraded Input (128x128) ───► [ViT Structural Encoder] ───► Global Structure Map (Z_struct)
                                                                      │
Reference Style Latent  ───► [Multi-Head Cross-Attention] ───────────┘
                                       │
                                       ▼
                      [PatchGAN Generator & Discriminator]
                                       │
                                       ▼
                       Restored Output (512x512 High-Res)
```

1. **Global Structural Priors:** 12-layer Vision Transformer captures long-range dependencies and facial geometry (eyes, nose, jawline) under extreme occlusions.
2. **PatchGAN Texture Discriminator:** 70x70 receptive field discriminator penalizes high-frequency structural hallucinations.
3. **Compound Loss Function:** Joint optimization combining $\mathcal{L}_{L1}$, Perceptual VGG-19 loss $\mathcal{L}_{perc}$, Adversarial WGAN-GP loss $\mathcal{L}_{adv}$, and Identity Preserving ArcFace cosine loss $\mathcal{L}_{id}$.

---

## 📊 Benchmark Results

Evaluated on the **CelebA-HQ** and **FFHQ** test benchmarks under 50% random structural mask occlusion:

| Model Architecture | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ |
| :--- | :---: | :---: | :---: | :---: |
| Standard U-Net | 26.4 | 0.812 | 0.245 | 42.1 |
| Pix2PixHD | 28.9 | 0.874 | 0.162 | 28.4 |
| DeepFill v2 | 30.1 | 0.898 | 0.128 | 21.7 |
| **Our Transformer-GAN (Proposed)** | **33.2** | **0.948** | **0.068** | **12.3** |

---

## 🚀 Quickstart & Inference

```bash
# Clone the repository
git clone https://github.com/taran-dev4u/transformer-gan-image-restoration.git
cd transformer-gan-image-restoration

# Install dependencies
pip install torch torchvision transformers einops pillow

# Run inference on degraded input
python inference.py --input examples/degraded_face.png --output outputs/restored_face.png --checkpoint weights/best_model.pth
```

---

## 🌐 Live Interactive Demonstration

Explore the live model architecture, interactive comparisons, and inference walkthrough at:
🔗 **[https://taran-dev4u.github.io/taran-portfolio/#projects](https://taran-dev4u.github.io/taran-portfolio/#projects)**
