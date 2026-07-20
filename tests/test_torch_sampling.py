import torch
from torch import nn

from smpl_diffusion_lab.diffusion.schedule import make_linear_schedule
from smpl_diffusion_lab.diffusion.torch_sampling import sample_ddpm_torch


class ZeroNoiseModel(nn.Module):
    def forward(
            self,
            xt: torch.Tensor,
            timesteps: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros_like(xt)


def test_sample_ddpm_torch_has_expected_shape():
    schedule = make_linear_schedule(4, 0.1, 0.4)
    model = ZeroNoiseModel()

    samples = sample_ddpm_torch(
        model=model,
        schedule=schedule,
        num_samples=8,
        data_dim=2,
        generator=torch.Generator().manual_seed(0),
    )

    assert samples.shape == (8, 2)
    assert samples.dtype == torch.float32
    assert torch.isfinite(samples).all()
    assert samples.requires_grad is False
