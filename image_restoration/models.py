import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """Splits image into non-overlapping patches and projects to embedding dimension."""

    def __init__(self, in_channels: int = 4, patch_size: int = 16, embed_dim: int = 128):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B, C, H, W -> B, D, H/P, W/P -> B, N, D
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class StructuralTransformerPrior(nn.Module):
    """Vision Transformer branch recovering global low-frequency structural priors."""

    def __init__(
        self,
        in_channels: int = 4,
        patch_size: int = 16,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.out_conv = nn.Sequential(
            nn.Conv2d(embed_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Concatenate 3-channel image and 1-channel mask -> 4 channels
        x = torch.cat([image, mask], dim=1)
        b, _, h, w = x.shape
        tokens = self.patch_embed(x)
        encoded = self.transformer(tokens)

        # Reshape tokens back to 2D feature map
        grid_h = h // self.patch_size
        grid_w = w // self.patch_size
        features = encoded.transpose(1, 2).reshape(b, self.embed_dim, grid_h, grid_w)
        upsampled = F.interpolate(features, size=(h, w), mode="bilinear", align_corners=False)
        return self.out_conv(upsampled)


class TextureGANGenerator(nn.Module):
    """Gated convolutional generator conditioned on the coarse structural prior."""

    def __init__(self, in_channels: int = 7, out_channels: int = 3, base_channels: int = 32):
        super().__init__()
        # Input: image (3) + mask (1) + structural prior (3) = 7 channels
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor, mask: torch.Tensor, struct_prior: torch.Tensor) -> torch.Tensor:
        x = torch.cat([image, mask, struct_prior], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        b = self.bottleneck(e2)
        d1 = self.dec1(b)
        out = self.out_conv(d1)
        # Composite: retain original unmasked pixels, inpaint masked regions
        return image * (1.0 - mask) + out * mask


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator for local texture and edge realism."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
