import torch
from image_restoration.models import StructuralTransformerPrior, TextureGANGenerator
from image_restoration.pipeline import InpaintingPipeline


def test_inpainting_pipeline_inpaint_and_evaluate():
    transformer = StructuralTransformerPrior(
        in_channels=4, patch_size=16, embed_dim=32, num_heads=2, num_layers=1
    )
    generator = TextureGANGenerator(in_channels=7, out_channels=3, base_channels=8)
    pipeline = InpaintingPipeline(transformer=transformer, generator=generator)

    gt = torch.rand(1, 3, 64, 64)
    mask = torch.zeros(1, 1, 64, 64)
    mask[:, :, 20:40, 20:40] = 1.0

    inpainted, coarse = pipeline.inpaint(gt * (1.0 - mask), mask)
    assert inpainted.shape == (1, 3, 64, 64)
    assert coarse.shape == (1, 3, 64, 64)

    eval_results = pipeline.evaluate(gt, mask)
    assert "psnr_db" in eval_results
    assert "ssim" in eval_results
    assert "mae" in eval_results
    assert eval_results["psnr_db"] > 0.0
