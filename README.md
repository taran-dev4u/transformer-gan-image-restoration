# Image Inpainting & Restoration using Transformer Structure Prior & GAN Synthesis

[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red.svg)](https://pytorch.org/)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Transformers%20%2B%20GANs-purple.svg)](https://arxiv.org/)
[![Degree](https://img.shields.io/badge/Project-B.Tech%20Capstone%20Senior%20Design-blue.svg)](https://vit.ac.in/)
[![Proof of Work](https://img.shields.io/badge/Proof%20of%20Work-Full%20Jury%20Reports%20%26%20Reviews-green.svg)](project_reports_and_milestones/)
[![Institution](https://img.shields.io/badge/Institution-Vellore%20Institute%20of%20Technology%20(VIT)-blue.svg)](https://vit.ac.in/)

---

## 📌 Executive Summary & Project Genesis

This repository contains the complete research, deep learning architecture, training pipelines, and academic deliverables for the **Senior Capstone Project** in Computer Science & Engineering at **Vellore Institute of Technology (VIT)**.

Image inpainting addresses the reconstruction of lost or deteriorated image regions with semantically consistent and visually photorealistic content. Traditional convolutional approaches often produce blurry textures and distorted structures over large irregular holes. This project pioneers a **Dual-Branch Deep Learning Framework**:
1. A **Vision Transformer (ViT) Structural Prior Branch** that captures global topological coherence and semantic edge contours.
2. A **Generative Adversarial Network (GAN) Texture Synthesis Branch** utilizing Spectral Normalization and a PatchGAN discriminator to recover high-frequency micro-textures and natural color gradients.

---

## 📁 Tangible Proof-of-Work & Capstone Milestone Deliverables

The [`project_reports_and_milestones/`](project_reports_and_milestones/) directory archives the complete sequence of formal capstone evaluation deliverables:

1. **Formal Project Reports:**
   - [`CapstoneFinalReport .pdf`](project_reports_and_milestones/CapstoneFinalReport%20.pdf) — Complete final thesis report detailing architecture, loss formulations, training regimes, and comparative benchmarks.
   - [`CAPSTONE_Final_review_Script.pdf`](project_reports_and_milestones/CAPSTONE_Final_review_Script.pdf) & [`CAPSTONE_Final_review_transcript.pdf`](project_reports_and_milestones/CAPSTONE_Final_review_transcript.pdf) — Official final review presentation script, defense transcript, and jury evaluation rubric.
2. **Progress Milestone Reports:**
   - [`CAPSTONE PROJECT REVIEW-1 (1).pdf`](project_reports_and_milestones/CAPSTONE%20PROJECT%20REVIEW-1%20(1).pdf) — Problem definition, literature survey, and baseline formulation.
   - [`CAPSTONE PROJECT REVIEW-2 (1).pdf`](project_reports_and_milestones/CAPSTONE%20PROJECT%20REVIEW-2%20(1).pdf) — Architecture design, dual-branch formulation, and dataset preparation.
   - [`CAPSTONE PROJECT REVIEW-3.pdf`](project_reports_and_milestones/CAPSTONE%20PROJECT%20REVIEW-3.pdf) — Empirical evaluations, PSNR/SSIM ablation studies, and qualitative results.
3. **Capstone Defense Slides:**
   - [`CAPSTONE REVIEW-3.pptx`](project_reports_and_milestones/CAPSTONE%20REVIEW-3.pptx) & [`CapstoneProjectReview2.pptx`](project_reports_and_milestones/CapstoneProjectReview2.pptx) — Formal evaluation presentation slide decks.
   - [`project_ppt_taranmam_veerasa2_sjupalli.pdf`](project_reports_and_milestones/project_ppt_taranmam_veerasa2_sjupalli.pdf) — Comprehensive architecture presentation slides.

---

## 🚀 Key Architectural Innovations & Mathematical Formulation

### 1. Dual-Branch Synergy
- **Structural Prior Branch:** Multi-scale self-attention over non-local patch tokens predicts coarse semantic boundaries across corrupted regions.
- **Texture Synthesis Branch:** Gated convolutions conditioned on the structural prior synthesize fine-grained details without boundary seams.

### 2. Multi-Objective Composite Loss
$$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\ell_1} + \lambda_{\text{adv}} \mathcal{L}_{\text{GAN}} + \lambda_{\text{perc}} \mathcal{L}_{\text{perceptual}} + \lambda_{\text{style}} \mathcal{L}_{\text{style}}$$

- **Reconstruction Loss ($\\mathcal{L}_{\\ell_1}$):** Pixel-wise $L_1$ distance across valid and masked regions.
- **Perceptual Loss ($\\mathcal{L}_{\\text{perceptual}}$):** Feature activations across pre-trained VGG-19 layers.
- **Style Loss ($\\mathcal{L}_{\\text{style}}$):** Gram matrix $G(\phi_i)$ matching for texture distribution consistency.
- **Hinge Adversarial Loss ($\\mathcal{L}_{\\text{GAN}}$):** Stabilized with Spectral Normalization in the PatchGAN discriminator.

---

## 📂 Repository Structure

```
transformer-gan-image-restoration/
├── project_reports_and_milestones/  # Proof-of-work thesis reports, review PDFs, and slides
│   ├── CapstoneFinalReport .pdf     # Comprehensive final thesis report
│   ├── CAPSTONE_Final_review_Script.pdf # Defense script and jury transcript
│   ├── CAPSTONE PROJECT REVIEW-1/2/3.pdf # Review milestone documents
│   └── project_ppt_*.pdf / *.pptx   # Defense presentations
├── deep-learning-model/             # Deep learning architecture and training scripts
│   ├── project_taranmam_*.ipynb     # End-to-end model training, loss curves, and visual testing
│   ├── project_datasets_*.txt       # Dataset documentation (CelebA, Places2)
│   └── project_weights_*.txt        # Pretrained model weight pointers
└── README.md                        # Documentation
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
- **Author:** Taran Mamidala (Reg: 19BCE7346)
- **Degree:** Bachelor of Technology in Computer Science & Engineering (B.Tech CSE)
- **Institution:** Vellore Institute of Technology (VIT)
- **Project:** Capstone Senior Design Project
