import torch
from torch import nn
from torch.nn import functional as F

from smpl_diffusion_lab.diffusion.schedule import DiffusionSchedule
from smpl_diffusion_lab.diffusion.torch_forward import q_sample_torch


def noise_prediction_loss(
    model: nn.Module,
    x0: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    schedule: DiffusionSchedule,
) -> torch.Tensor:
    """Compute the DDPM noise-prediction loss."""

    xt = q_sample_torch(
        x0=x0,
        timesteps=timesteps,
        noise=noise,
        schedule=schedule,
    )

    noise_pred = model(xt, timesteps)

    loss = F.mse_loss(noise_pred,noise)

    return loss
