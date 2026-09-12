import math
import torch


def compute_psnr(target: torch.Tensor, prediction: torch.Tensor, max_val: float = 1.0) -> float:
    """Computes Peak Signal-to-Noise Ratio (PSNR) in dB between target and prediction."""
    mse = torch.mean((target - prediction) ** 2).item()
    if mse == 0.0:
        return 100.0  # Perfect reconstruction
    return 20.0 * math.log10(max_val / math.sqrt(mse))


def compute_mae(target: torch.Tensor, prediction: torch.Tensor) -> float:
    """Computes Mean Absolute Error ($L_1$ distance) between images."""
    return torch.mean(torch.abs(target - prediction)).item()


def compute_ssim(
    target: torch.Tensor, prediction: torch.Tensor, c1: float = 0.01**2, c2: float = 0.03**2
) -> float:
    """Computes global structural similarity index (SSIM) over image tensors in [0, 1]."""
    mu_x = torch.mean(target)
    mu_y = torch.mean(prediction)

    sigma_x_sq = torch.var(target, unbiased=False)
    sigma_y_sq = torch.var(prediction, unbiased=False)
    sigma_xy = torch.mean((target - mu_x) * (prediction - mu_y))

    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x_sq + sigma_y_sq + c2)

    ssim_val = numerator / denominator
    return float(torch.clamp(ssim_val, 0.0, 1.0).item())
