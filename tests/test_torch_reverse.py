import torch

from smpl_diffusion_lab.diffusion.schedule import make_linear_schedule
from smpl_diffusion_lab.diffusion.torch_forward import q_sample_torch
from smpl_diffusion_lab.diffusion.torch_reverse import (
    predict_previous_mean_from_noise,
    predict_x0_from_noise,
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
