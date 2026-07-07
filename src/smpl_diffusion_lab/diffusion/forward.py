import numpy as np

from smpl_diffusion_lab.diffusion.schedule import DiffusionSchedule


def q_sample(
    x0: np.ndarray,
    timesteps: np.ndarray,
    noise: np.ndarray,
    schedule: DiffusionSchedule,
) -> np.ndarray:
    """Sample x_t from q(x_t | x_0)."""
    if x0.shape != noise.shape:
        raise ValueError("x0 and noise must have the same shape.")
    if not isinstance(timesteps, np.ndarray):
        raise TypeError("timesteps must be a numpy array.")
    if timesteps.ndim != 1:
        raise ValueError("timesteps must be a 1D array.")
    if not np.issubdtype(timesteps.dtype, np.integer):
        raise TypeError("timesteps must be an array of integers.")
    if x0.ndim < 2:
        raise ValueError("x0 and noise must be at least 2D arrays.")
    if timesteps.shape[0] != x0.shape[0]:
        raise ValueError("timesteps must have the same length as the batch size of x0.")
    if np.any(timesteps < 0) or np.any(timesteps >= schedule.alpha_bars.shape[0]):
        raise ValueError("timesteps must be in the range [0, num_timesteps).")
    alpha_bars_t = schedule.alpha_bars[timesteps]
    broadcast_shape = (timesteps.shape[0],) + (1,) * (x0.ndim - 1)
    alpha_bars_t = alpha_bars_t.reshape(broadcast_shape)

    return np.sqrt(alpha_bars_t) * x0 + np.sqrt(1.0 - alpha_bars_t) * noise
