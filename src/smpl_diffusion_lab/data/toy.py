import math

import torch


def sample_ring_mixture(
        num_samples: int,
        num_modes: int = 8,
        radius: float = 2.0,
        noise_std: float = 0.05,
        generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample 2D points from Gaussian clusters arranged on a ring"""

    mode_indices = torch.randint(
        low=0,
        high=num_modes,
        size=(num_samples,),
        generator=generator,
    )

    angles = (2.0 * math.pi * mode_indices.float()) / num_modes

    x_coordinates = radius * torch.cos(angles)
    y_coordinates = radius * torch.sin(angles)

    centers = torch.stack(
        [x_coordinates, y_coordinates],
        dim = 1,
    )

    noise = torch.randn(
        centers.shape,
        generator=generator,
        dtype=centers.dtype,
        device=centers.device,
    ) * noise_std
    samples = centers + noise

    return samples

