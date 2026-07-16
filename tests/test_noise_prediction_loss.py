import torch
from torch import nn
from torch.nn import functional as F

from smpl_diffusion_lab.diffusion.schedule import make_linear_schedule
from smpl_diffusion_lab.diffusion.torch_forward import q_sample_torch
from smpl_diffusion_lab.training.loss import noise_prediction_loss
from smpl_diffusion_lab.models.noise_mlp import NoisePredictMLP


class IdentityNoiseModel(nn.Module):
    def forward(
        self,
        xt: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        return xt


def test_noise_prediction_noise_uses_noisy_input():
    schedule = make_linear_schedule(4, 0.1, 0.4)

    x0 = torch.tensor(
        [
            [1.0, 2.0],
            [-1.0, 0.5],
        ],
        dtype=torch.float32,
    )
    noise = torch.tensor(
        [
            [0.1, 0.2],
            [0.4, 0.0],
        ],
        dtype=torch.float32
    )

    timesteps = torch.tensor([0, 3], dtype=torch.long)

    model = IdentityNoiseModel()

    loss = noise_prediction_loss(
        model=model,
        x0=x0,
        timesteps=timesteps,
        noise=noise,
        schedule=schedule,
    )

    xt = q_sample_torch(
        x0=x0,
        timesteps=timesteps,
        noise=noise,
        schedule=schedule,
    )

    excepted = F.mse_loss(xt, noise)

    torch.testing.assert_close(loss, excepted)

    assert loss.shape == ()


def test_noise_prediction_loss_computes_model_gradients():
    schedule = make_linear_schedule(4, 0.1, 0.4)

    model = NoisePredictMLP(
        data_dim=2,
        hidden_dim=16,
        num_timesteps=4,
    )

    x0 = torch.tensor(
        [
            [1.0, 2.0],
            [-1.0, 0.5],
        ],
        dtype=torch.float32,
    )
    noise = torch.tensor(
        [
            [0.1, -0.2],
            [0.4, 0.0],
        ],
        dtype=torch.float32,
      )
    timesteps = torch.tensor([0, 3], dtype=torch.long)

    loss = noise_prediction_loss(
        model=model,
        x0=x0,
        timesteps=timesteps,
        noise=noise,
        schedule=schedule,
    )

    parameters = list(model.parameters())

    assert all(parameter.grad is None for parameter in parameters)

    loss.backward()

    assert all(parameter.grad is not None for parameter in parameters)
    assert all(
        parameter.grad.shape == parameter.shape
        for parameter in parameters
    )
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in parameters
    )
