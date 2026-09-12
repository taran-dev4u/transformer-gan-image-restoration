import torch
from image_restoration.metrics import compute_mae, compute_psnr, compute_ssim


def test_metrics_identical_images():
    img = torch.rand(1, 3, 64, 64)
    psnr = compute_psnr(img, img)
    ssim = compute_ssim(img, img)
    mae = compute_mae(img, img)

    assert psnr >= 99.0
    assert ssim >= 0.999
    assert mae == 0.0


def test_metrics_noisy_image():
    clean = torch.ones(1, 3, 32, 32) * 0.5
    noisy = clean + torch.randn(1, 3, 32, 32) * 0.1
    noisy = torch.clamp(noisy, 0.0, 1.0)

    psnr = compute_psnr(clean, noisy)
    ssim = compute_ssim(clean, noisy)
    mae = compute_mae(clean, noisy)

    assert 10.0 < psnr < 40.0
    assert 0.0 <= ssim <= 1.0
    assert mae > 0.0
