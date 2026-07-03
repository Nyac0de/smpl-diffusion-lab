from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DiffusionSchedule:
    betas: np.ndarray
    alphas: np.ndarray
    alpha_bars: np.ndarray


def make_linear_schedule(
    num_timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
) -> DiffusionSchedule:
    """Create a linear diffusion schedule."""
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
    betas = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float32)
    alphas = 1.0 - betas
    alpha_bars= np.cumprod(alphas, axis=0)
    return DiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
    )