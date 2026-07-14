import torch

from smpl_diffusion_lab.diffusion.schedule import make_linear_schedule
from smpl_diffusion_lab.diffusion.torch_forward import q_sample_torch
from smpl_diffusion_lab.diffusion.torch_reverse import (
    make_posterior_variances_torch,
    predict_previous_mean_from_noise,
    predict_x0_from_noise,
    p_sample_torch,
)



def test_predict_x0_from_noise_recovers_original_x0():
    schedule = make_linear_schedule(4, 0.1, 0.4)

    x0 = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [-1.0, 0.5, 2.0],
        ]
    )
    noise = torch.tensor(
        [
            [0.1, -0.2, 0.3],
            [0.4, 0.0, -0.5],
        ]
    )
    timesteps = torch.tensor([0, 3], dtype=torch.long)

    xt = q_sample_torch(x0, timesteps, noise, schedule)

    recoverd_x0 = predict_x0_from_noise(
        xt=xt,
        timesteps=timesteps,
        noise_pred=noise,
        schedule=schedule,
    )

    torch.testing.assert_close(recoverd_x0, x0)


def test_reverse_mean_recovers_x0_at_timestep_zero():
    schedule = make_linear_schedule(4, 0.1, 0.4)

    x0 = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [-1.0, 0.5, 2.0],
        ],
        dtype=torch.float32,
    )
    noise = torch.tensor(
        [
            [0.1, -0.2, 0.3],
            [0.4, 0.0, -0.5],
        ],
        dtype=torch.float32,
    )
    timesteps = torch.zeros(2, dtype=torch.long)

    xt = q_sample_torch(x0, timesteps, noise, schedule)

    mean = predict_previous_mean_from_noise(
        xt=xt,
        timesteps=timesteps,
        noise_pred=noise,
        schedule=schedule,
    )

    torch.testing.assert_close(mean, x0)


def test_make_posterior_variances_torch_values():
    schedule = make_linear_schedule(4, 0.1, 0.4)

    posterior_variances = make_posterior_variances_torch(schedule)

    expected = torch.tensor(
        [
            0.0,
            0.07142857,
            0.16935484,
            0.28440368,
        ],
        dtype=torch.float32
    )

    torch.testing.assert_close(posterior_variances, expected)


def test_p_sample_torch_adds_noise_only_before_final_step():
    schedule = make_linear_schedule(4, 0.1, 0.4)

    xt = torch.tensor(
        [
        [1.0, 2.0, 3.0],
        [-1.0, 0.5, 2.0],
        ],
        dtype=torch.float32,
    )
    noise_pred = torch.tensor(
        [
        [0.1, -0.2, 0.3],
        [0.4, 0.0, -0.5],
        ],
        dtype=torch.float32,
    )
    sampling_noise = torch.tensor(
        [
        [10.0, 10.0, 10.0],
        [0.4, 0.0, -0.5],
        ],
        dtype=torch.float32,
    )
    timesteps = torch.tensor([0, 1], dtype=torch.long)

    mean = predict_previous_mean_from_noise(
        xt=xt,
        timesteps=timesteps,
        noise_pred=noise_pred,
        schedule=schedule,
    )
    x_previous = p_sample_torch(
        xt=xt,
        timesteps=timesteps,
        noise_pred=noise_pred,
        sampling_noise=sampling_noise,
        schedule=schedule,
    )

    posterior_variances = make_posterior_variances_torch(schedule)

    expected = mean.clone()
    expected[1] = (
        mean[1]
        + torch.sqrt(posterior_variances[1]) * sampling_noise[1]
    )

    torch.testing.assert_close(x_previous, expected)
