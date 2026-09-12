import torch
from image_restoration.models import (
    PatchDiscriminator,
    StructuralTransformerPrior,
    TextureGANGenerator,
)


def test_structural_transformer_prior_forward():
    model = StructuralTransformerPrior(
        in_channels=4, patch_size=16, embed_dim=64, num_heads=2, num_layers=1
    )
    img = torch.rand(2, 3, 64, 64)
    mask = torch.zeros(2, 1, 64, 64)
    mask[:, :, 20:44, 20:44] = 1.0

    out = model(img, mask)
    assert out.shape == (2, 3, 64, 64)
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


def test_texture_gan_generator_forward():
    generator = TextureGANGenerator(in_channels=7, out_channels=3, base_channels=16)
    img = torch.rand(2, 3, 64, 64)
    mask = torch.zeros(2, 1, 64, 64)
    struct_prior = torch.rand(2, 3, 64, 64)

    out = generator(img, mask, struct_prior)
    assert out.shape == (2, 3, 64, 64)
    # Check that unmasked areas are preserved exactly from input
    unmasked_diff = torch.abs(out[:, :, 0:10, 0:10] - img[:, :, 0:10, 0:10])
    assert torch.all(unmasked_diff < 1e-5)


def test_patch_discriminator_forward():
    disc = PatchDiscriminator(in_channels=3, base_channels=16)
    img = torch.rand(2, 3, 64, 64)
    logits = disc(img)
    # Output spatial dimensions should be reduced by strides
    assert logits.dim() == 4
    assert logits.shape[0] == 2
    assert logits.shape[1] == 1
