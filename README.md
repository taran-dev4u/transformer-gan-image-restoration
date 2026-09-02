# Image Inpainting & Restoration using Transformer Structure Prior & GAN Synthesis

[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red.svg)](https://pytorch.org/)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Transformers%20%2B%20GANs-purple.svg)](https://arxiv.org/)
[![Project Type](https://img.shields.io/badge/Project-B.Tech%20Capstone%20Project-blue.svg)](https://vit.ac.in/)
[![Evaluation](https://img.shields.io/badge/Metrics-PSNR%20%7C%20SSIM%20%7C%20FID-green.svg)](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio)

---

## 📌 Executive Summary & Project Genesis

This repository contains the complete research and engineering implementation of an **Image Inpainting and Structural Restoration Framework** developed as a **B.Tech Capstone Project** in Computer Science & Engineering.

Traditional convolutional image inpainting often suffers from blurry artifacts or structural distortion when filling large corrupted regions because CNNs lack long-range contextual dependencies. This project addresses the challenge by combining:
1. A **Vision Transformer (ViT) Structure Prior Branch** that captures global topological coherence and semantic outlines across large missing regions.
2. A **Generative Adversarial Network (GAN) Texture Synthesis Branch** with Spectral Normalization and PatchGAN Discriminator to recover high-frequency textures, natural edges, and realistic pixel details.

```
+----------------+      Global Context       +-------------------------+
| Corrupted Image| ------------------------> | Multi-Head Self-Attn    | --+
|   + Mask (M)   |                           | Structural Prior Branch |   |  Feature
+----------------+                           +-------------------------+   |  Fusion
        |                                                                  v
        |               Local Detail         +-------------------------+ +---------------+     +------------------+
        +----------------------------------> | PatchGAN Texture        | | Inpainted     | --> | Discriminator &  |
                                             | Synthesis Generator     | | Reconstructed |     | Adversarial Loss |
                                             +-------------------------+ +---------------+     +------------------+
```

---

## 🚀 Key Architectural Innovations

### 1. Dual-Branch Synergy
- **Structural Branch:** Computes multi-scale self-attention over non-local patch tokens, predicting coarse structural boundaries and low-frequency semantic maps.
- **Texture Branch:** Uses gated convolutions and dilated residual blocks conditioned on the structural prior to synthesize fine-grained photorealistic details.

### 2. Multi-Objective Loss Formulation
$$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\ell_1} + \lambda_{\text{adv}} \mathcal{L}_{\text{GAN}} + \lambda_{\text{perc}} \mathcal{L}_{\text{perceptual}} + \lambda_{\text{style}} \mathcal{L}_{\text{style}}$$

- **Reconstruction Loss ($\\mathcal{L}_{\\ell_1}$):** Measures pixel-wise $L_1$ distance across valid and masked regions.
- **Perceptual Loss ($\\mathcal{L}_{\\text{perceptual}}$):** Computes deep feature activations across layers of a pre-trained VGG-19 network.
- **Style Loss ($\\mathcal{L}_{\\text{style}}$):** Matches Gram matrices $G(\phi_i)$ of deep representations to ensure texture distribution consistency.
- **Hinge Adversarial Loss ($\\mathcal{L}_{\\text{GAN}}$):** Stabilized with Spectral Normalization in the PatchGAN discriminator.

---

## 📂 Repository Structure

```
transformer-gan-image-restoration/
├── deep-learning-model/             # Deep learning architecture and training scripts
│   ├── project_taranmam_*.ipynb     # End-to-end model training, loss curves, and visual testing
│   ├── project_report_*.pdf         # Comprehensive technical capstone report
│   ├── project_ppt_*.pptx           # Capstone defense presentation slides
│   ├── project_datasets_*.txt       # Dataset documentation (CelebA, Places2, ImageNet subsets)
│   ├── project_deployment_*.txt     # Model deployment and inference pipeline specs
│   └── project_weights_*.txt        # Checkpoint download pointers & pretrained model registry
├── project-files/                   # Supplementary course and review documentation
└── README.md                        # Project documentation
```

---

## 📊 Quantitative Benchmarks & Results

| Mask Coverage (%) | Model Architecture | PSNR (dB) ↑ | SSIM ↑ | FID ↓ |
| :--- | :--- | :--- | :--- | :--- |
| **10% - 20%** | Baseline Partial Convolutions (PConv) | 28.4 | 0.942 | 14.8 |
| **10% - 20%** | **Ours (Transformer + GAN Dual-Branch)** | **32.1** | **0.978** | **8.2** |
| **30% - 50%** | Baseline Global-Local GAN | 21.6 | 0.812 | 28.5 |
| **30% - 50%** | **Ours (Transformer + GAN Dual-Branch)** | **26.8** | **0.915** | **15.3** |

---

## 👨‍💻 Author & Academic Attribution
- **Author:** Taran Mamidala
- **Degree:** Bachelor of Technology in Computer Science & Engineering (B.Tech CSE)
- **Institution:** Vellore Institute of Technology (VIT)
- **Project:** Capstone Senior Design Project
