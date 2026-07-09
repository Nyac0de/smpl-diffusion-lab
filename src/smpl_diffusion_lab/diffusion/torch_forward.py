import torch

from smpl_diffusion_lab.diffusion.schedule import DiffusionSchedule


def q_sample_torch(
    x0: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    schedule: DiffusionSchedule,
) -> torch.Tensor:
    """Sample x_t from q(x_t | x_0) using PyTorch tensors."""
    if x0.shape != noise.shape:
        raise ValueError("x0 and noise must have the same shape")
    if not isinstance(timesteps, torch.Tensor):
        raise TypeError("timesteps must be a torch Tensor")
    if timesteps.ndim != 1:
        raise ValueError("timesteps must be a 1D tensor")
    if timesteps.dtype != torch.long:
        raise TypeError("timesteps must have dtype torch.long")
    if x0.ndim < 2:
        raise ValueError("x0 and noise must be at least 2D tensors.")
    if timesteps.shape[0] != x0.shape[0]:
        raise ValueError("timesteps must have the same length as the batch size of x0.")
    if torch.any(timesteps < 0) or torch.any(timesteps >= schedule.alpha_bars.shape[0]):
        raise ValueError("timesteps must be in the range [0, num_timesteps).")

    alpha_bars = torch.as_tensor(
        schedule.alpha_bars,
        dtype=x0.dtype,
        device=x0.device,
    )
    alpha_bars_t = alpha_bars[timesteps]

    broadcast_shape = (timesteps.shape[0],) + (1,) * (x0.ndim - 1)
    alpha_bars_t = alpha_bars_t.reshape(broadcast_shape)

    return torch.sqrt(alpha_bars_t) * x0 + torch.sqrt(1.0 - alpha_bars_t) * noise
