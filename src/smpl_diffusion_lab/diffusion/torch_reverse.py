import torch

from smpl_diffusion_lab.diffusion.schedule import DiffusionSchedule


def predict_x0_from_noise(
        xt: torch.Tensor,
        timesteps: torch.Tensor,
        noise_pred: torch.Tensor,
        schedule: DiffusionSchedule,
) -> torch.Tensor:
    """Recover predicted x0 from xt and predicted noise"""
    alpha_bars = torch.as_tensor(
        schedule.alpha_bars,
        dtype=xt.dtype,
        device=xt.device,
    )
    if xt.shape != noise_pred.shape:
        raise ValueError("xt and noise must have the same shape")
    if not isinstance(timesteps, torch.Tensor):
        raise TypeError("timesteps must be a torch tensor")
    if timesteps.ndim != 1:
        raise ValueError("timesteps must be a 1D tensor")
    if timesteps.dtype != torch.long:
        raise TypeError("timesteps must have dtype torch.long")
    if xt.ndim < 2:
        raise ValueError("xt and noise must be at least 2D tensors")
    if timesteps.shape[0] != xt.shape[0]:
        raise ValueError("timesteps and xt must have the same shape")
    if torch.any(timesteps < 0) or torch.any(timesteps >= schedule.alpha_bars.shape[0]):
        raise ValueError("timesteps must be in the range [0, num_timesteps).")
    alpha_bars_t = alpha_bars[timesteps]

    broadcast_shape = (timesteps.shape[0],) + (1,) * (xt.ndim - 1)
    alpha_bars_t = alpha_bars_t.reshape(broadcast_shape)

    return (xt - torch.sqrt(1.0 - alpha_bars_t) * noise_pred) / torch.sqrt(alpha_bars_t)


def predict_previous_mean_from_noise(
    xt: torch.Tensor,
    timesteps: torch.Tensor,
    noise_pred: torch.Tensor,
    schedule: DiffusionSchedule,
) -> torch.Tensor:
    """Compute the reverse-process mean for x_{t-1}"""

    betas = torch.as_tensor(
        schedule.betas,
        dtype=xt.dtype,
        device=xt.device,
    )
    alphas = torch.as_tensor(
        schedule.alphas,
        dtype=xt.dtype,
        device=xt.device,
    )
    alpha_bars = torch.as_tensor(
        schedule.alpha_bars,
        dtype=xt.dtype,
        device=xt.device,
    )

    betas_t = betas[timesteps]
    alphas_t = alphas[timesteps]
    alpha_bars_t = alpha_bars[timesteps]

    broadcast_shape = (timesteps.shape[0],) + (1,) * (xt.ndim - 1)
    betas_t = betas_t.reshape(broadcast_shape)
    alphas_t = alphas_t.reshape(broadcast_shape)
    alpha_bars_t = alpha_bars_t.reshape(broadcast_shape)

    mean_numerator = xt - betas_t / torch.sqrt(1.0 - alpha_bars_t) * noise_pred
    mean = mean_numerator / torch.sqrt(alphas_t)

    return mean


def make_posterior_variances_torch(
        schedule: DiffusionSchedule,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Compute posterior variance for all timesteps"""
    betas = torch.as_tensor(
        schedule.betas,
        dtype=dtype,
        device=device,
    )

    alpha_bars = torch.as_tensor(
        schedule.alpha_bars,
        dtype=dtype,
        device=device,
    )

    alpha_bars_previous = torch.cat(
        [
            torch.ones(1, dtype=dtype, device=device),
            alpha_bars[:-1],
        ]
    )

    posterior_variances = (
        betas
        * (1.0 - alpha_bars_previous)
        / (1.0 - alpha_bars)
    )

    return posterior_variances


def p_sample_torch(
    xt: torch.Tensor,
    timesteps: torch.Tensor,
    noise_pred: torch.Tensor,
    sampling_noise: torch.Tensor,
    schedule: DiffusionSchedule,
) -> torch.Tensor:
    """Sample x_{t-1} from x_t"""
    if xt.shape != noise_pred.shape:
        raise ValueError("xt and noise_pred must have the same shape")
    if xt.shape != sampling_noise.shape:
        raise ValueError("xt and sampling_noise must have the same shape")

    mean = predict_previous_mean_from_noise(
        xt=xt,
        timesteps=timesteps,
        noise_pred=noise_pred,
        schedule=schedule,
    )

    posterior_variances = make_posterior_variances_torch(
        schedule=schedule,
        dtype=xt.dtype,
        device=xt.device,
    )
    posterior_variances_t = posterior_variances[timesteps]

    broadcast_shape = (timesteps.shape[0],) + (1,) * (xt.ndim - 1)
    posterior_variances_t = posterior_variances_t.reshape(
        broadcast_shape
    )

    nonzero_mask = (timesteps != 0).to(dtype=xt.dtype)
    nonzero_mask = nonzero_mask.reshape(broadcast_shape)

    x_previous = mean + nonzero_mask * torch.sqrt(posterior_variances_t) * sampling_noise

    return x_previous
