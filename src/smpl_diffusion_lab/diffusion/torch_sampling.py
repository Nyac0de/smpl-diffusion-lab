import torch
from torch import nn

from smpl_diffusion_lab.diffusion.schedule import DiffusionSchedule
from smpl_diffusion_lab.diffusion.torch_reverse import p_sample_torch


@torch.no_grad()
def sample_ddpm_torch(
    model: nn.Module,
    schedule: DiffusionSchedule,
    num_samples: int,
    data_dim: int,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Generate samples by running the DDPM reverse process."""

    num_timesteps = schedule.betas.shape[0]

    x_t = torch.randn(
        (num_samples, data_dim),
        generator=generator,
        device=device,
    )

    for timestep in reversed(range(num_timesteps)):
        timesteps = torch.full(
            size=(num_samples,),
            fill_value=timestep,
            dtype=torch.long,
            device=device,
        )

        noise_pred = model(x_t, timesteps)

        sampling_noise = torch.randn(
            x_t.shape,
            generator=generator,
            dtype=x_t.dtype,
            device=x_t.device
        )

        x_t = p_sample_torch(
            xt=x_t,
            timesteps=timesteps,
            noise_pred=noise_pred,
            sampling_noise=sampling_noise,
            schedule=schedule,
        )

    return x_t
