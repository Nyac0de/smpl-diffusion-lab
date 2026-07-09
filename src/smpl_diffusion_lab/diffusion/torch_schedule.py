from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TorchDiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor


def make_linear_torch_schedule(
    num_timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> TorchDiffusionSchedule:
    """Create a linear diffusion schedule using PyTorch tensors"""
    if not isinstance(num_timesteps, int):
        raise TypeError("num_timesteps must be an integer.")
    if num_timesteps <= 0:
        raise ValueError("num_timesteps must be a positive integer.")
    if not 0.0 < beta_start < 1.0:
        raise ValueError("beta_start must be in (0, 1)")
    if not 0.0 < beta_end < 1.0:
        raise ValueError("beta_end must be in (0, 1)")
    if beta_start > beta_end:
        raise ValueError("beta_start must be less than or equal to beta_end.")
    betas = torch.linspace(
        beta_start,
        beta_end,
        num_timesteps,
        dtype=dtype,
        device=device,
    )
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    return TorchDiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
    )
