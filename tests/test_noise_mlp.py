import torch

from smpl_diffusion_lab.models.noise_mlp import NoisePredictMLP


def test_noise_predictor_mlp_has_expected_output_shape():
    model = NoisePredictMLP(
        data_dim=2,
        hidden_dim=32,
        num_timesteps=100.
    )

    xt = torch.randn(8, 2)
    timesteps = torch.tensor(
        [0, 1, 10, 25, 50, 75, 98, 99],
        dtype=torch.long,
    )

    noise_pred = model(xt, timesteps)

    assert noise_pred.shape == xt.shape
    assert noise_pred.dtype == xt.dtype
    assert torch.isfinite(noise_pred).all()
