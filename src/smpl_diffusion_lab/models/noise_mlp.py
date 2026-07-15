import torch
from torch import nn


class NoisePredictMLP(nn.Module):
    """Predict noise from a noisy sample and its diffusion timestep."""


    def __init__(
            self,
            data_dim: int = 2,
            hidden_dim: int = 64,
            num_timesteps: int = 100,
    ) -> None:
        super().__init__()

        self.num_timesteps = num_timesteps

        self.network = nn.Sequential(
            nn.Linear(data_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
        )


    def forward(
            self,
            xt: torch.Tensor,
            timesteps: torch.Tensor,
    ) -> torch.Tensor:
        normalized_t = timesteps.to(
            dtype=xt.dtype,
            device=xt.device,
        ).reshape(-1, 1)

        normalized_t = normalized_t / (self.num_timesteps - 1)

        model_input = torch.cat(
            [xt, normalized_t],
            dim=1,
        )

        noise_pred = self.network(model_input)

        return noise_pred
