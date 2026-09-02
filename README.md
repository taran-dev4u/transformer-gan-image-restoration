# Image Inpainting with Transformer Structure Prior & GAN Texture Synthesis

Senior capstone project implementing a dual-branch deep learning framework for large-hole image inpainting and structural restoration. Combines Vision Transformer (ViT) attention for global semantic structure recovery with PatchGAN texture synthesis for high-frequency detail generation.

## Architecture

```
Corrupted Image + Mask
         │
         ├───► Vision Transformer Branch ────► Coarse Structural Prior ──┐
         │                                                               │
         └───► PatchGAN Texture Branch   ◄───────────────────────────────┘
                     │
                     ▼
             Inpainted Output ───► Discriminator (Hinge Loss + Spectral Norm)
```

- **Structural Prior Branch:** Uses multi-head self-attention over image patch tokens to reconstruct global boundaries and low-frequency edge contours across missing regions.
- **Texture Synthesis Branch:** Gated convolutions conditioned on the structural prior to generate fine photorealistic textures.
- **Loss Functions:** Composite objective balancing $L_1$ reconstruction loss, VGG-19 perceptual feature loss, Gram matrix style loss, and adversarial hinge loss.

## Project Deliverables & Reports

All milestone documentation and review materials are in `project_reports_and_milestones/`:
- `CapstoneFinalReport .pdf` — Complete final thesis report detailing architecture, loss formulations, and benchmarks.
- `CAPSTONE_Final_review_Script.pdf` / `CAPSTONE_Final_review_transcript.pdf` — Defense script and jury review transcript.
- `CAPSTONE PROJECT REVIEW-1/2/3.pdf` — Milestone progress reports.
- `project_ppt_*.pdf` / `*.pptx` — Presentation decks used during project reviews.

## Setup & Training

```bash
pip install torch torchvision numpy scipy matplotlib pillow tqdm

# Launch the end-to-end training notebook
jupyter notebook deep-learning-model/project_taranmam_veerasa2_sjupalli.ipynb
```
