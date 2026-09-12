from typing import Dict, Optional, Tuple
import torch

from .metrics import compute_mae, compute_psnr, compute_ssim
from .models import StructuralTransformerPrior, TextureGANGenerator


class InpaintingPipeline:
    """Orchestrates structural prior extraction and GAN-based texture synthesis."""

    def __init__(
        self,
        transformer: Optional[StructuralTransformerPrior] = None,
        generator: Optional[TextureGANGenerator] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cpu")
        self.transformer = (transformer or StructuralTransformerPrior()).to(self.device)
        self.generator = (generator or TextureGANGenerator()).to(self.device)
        self.transformer.eval()
        self.generator.eval()

    def inpaint(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs the two-stage inpainting pass over normalized image and mask tensors in [0, 1]."""
        image = image.to(self.device)
        mask = mask.to(self.device)

        with torch.no_grad():
            # Stage 1: Coarse structural prior from Vision Transformer
            coarse_prior = self.transformer(image, mask)

            # Stage 2: Fine texture generation from Gated GAN Generator
            final_output = self.generator(image, mask, coarse_prior)

        return final_output, coarse_prior

    def evaluate(self, ground_truth: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
        """Corrupts ground truth with mask, runs inpainting, and computes reconstruction metrics."""
        ground_truth = ground_truth.to(self.device)
        mask = mask.to(self.device)
        corrupted = ground_truth * (1.0 - mask)

        inpainted, _ = self.inpaint(corrupted, mask)

        return {
            "psnr_db": round(compute_psnr(ground_truth, inpainted), 2),
            "ssim": round(compute_ssim(ground_truth, inpainted), 4),
            "mae": round(compute_mae(ground_truth, inpainted), 4),
        }
