# Image Inpainting with Transformer Structure Prior & GAN Texture Synthesis

A dual-branch deep learning framework for large-hole image inpainting and structural restoration. Combines Vision Transformer (ViT) self-attention for global semantic structure recovery with a gated PatchGAN texture synthesis network for high-frequency detail generation.

## Architecture

```text
Corrupted Image + Mask
         │
         ├───► Vision Transformer Branch ────► Coarse Structural Prior ──┐
         │                                                               │
         └───► PatchGAN Texture Branch   ◄───────────────────────────────┘
                     │
                     ▼
             Inpainted Output ───► Discriminator (Hinge Loss + Spectral Norm)
```

- **Structural Prior Branch:** Multi-head self-attention over image patch tokens to reconstruct global boundaries and low-frequency edge contours across missing regions.
- **Texture Synthesis Branch:** Gated convolutions conditioned on the structural prior to synthesize photorealistic textures.
- **Composite Loss:** Combines $L_1$ reconstruction loss, VGG-19 perceptual feature loss, Gram matrix style loss, and adversarial hinge loss.
- **Evaluation Metrics:** Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), and Mean Absolute Error (MAE).

## Installation

```bash
git clone https://github.com/taran-dev4u/transformer-gan-image-restoration.git
cd transformer-gan-image-restoration
pip install -r requirements.txt
pip install -e .
```

## Python API Usage

```python
import torch
from image_restoration import InpaintingPipeline, StructuralTransformerPrior, TextureGANGenerator

# Initialize pipeline
pipeline = InpaintingPipeline()

# Prepare normalized tensors [B, C, H, W] in [0, 1]
image = torch.rand(1, 3, 256, 256)
mask = torch.zeros(1, 1, 256, 256)
mask[:, :, 80:176, 80:176] = 1.0  # Mask center region

# Run inpainting
inpainted, coarse_prior = pipeline.inpaint(image, mask)

# Evaluate metrics
metrics = pipeline.evaluate(ground_truth=image, mask=mask)
print(f"PSNR: {metrics['psnr_db']} dB | SSIM: {metrics['ssim']}")
```

## Interactive Streamlit App

Launch the interactive drawing canvas to test inpainting on custom masks:

```bash
cd streamlit-web-app
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Running Tests

```bash
pytest -v
```

## Training Notebook & Reports

- Training pipeline: `deep-learning-model/image_restoration_training.ipynb`
- Milestone reports and architecture write-ups: `project_reports_and_milestones/`
