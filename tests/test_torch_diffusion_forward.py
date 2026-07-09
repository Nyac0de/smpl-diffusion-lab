import pytest
import torch

from smpl_diffusion_lab.diffusion.schedule import make_linear_schedule
from smpl_diffusion_lab.diffusion.torch_forward import q_sample_torch


def test_q_sample_torch_scales_x0_when_noise_is_zero():
    schedule = make_linear_schedule(4, 0.1, 0.4)
    x0 = torch.ones((2, 3))
    noise = torch.zeros_like(x0)
    timesteps = torch.tensor([0, 3], dtype=torch.long)

    xt = q_sample_torch(x0, timesteps, noise, schedule)

    expected = torch.sqrt(
        torch.as_tensor(schedule.alpha_bars[timesteps.numpy()])
    ).reshape(2, 1) * x0

    torch.testing.assert_close(xt, expected)


def test_q_sample_torch_scales_noise_when_x0_is_zero():
    schedule = make_linear_schedule(4, 0.1, 0.4)

    x0 = torch.zeros((2, 3))
    noise = torch.ones_like(x0)
    timesteps = torch.tensor([0, 3], dtype=torch.long)

    xt = q_sample_torch(x0, timesteps, noise, schedule)

    expected = torch.sqrt(
        torch.as_tensor(1 - schedule.alpha_bars[timesteps.numpy()])
    ).reshape(2, 1) * noise

    torch.testing.assert_close(xt, expected)


def test_q_sample_torch_supports_pose_shape():
    schedule = make_linear_schedule(4, 0.1, 0.4)

    x0 = torch.ones(2, 17, 3)
    noise = torch.zeros_like(x0)
    timesteps = torch.tensor([0, 3], dtype=torch.long)

    xt = q_sample_torch(x0, timesteps, noise, schedule)

    assert xt.shape == x0.shape
    torch.testing.assert_close(
        xt[0],
        torch.sqrt(torch.as_tensor(schedule.alpha_bars[0])) * x0[0]
    )
    torch.testing.assert_close(
        xt[1],
        torch.sqrt(torch.as_tensor(schedule.alpha_bars[3])) * x0[1]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_q_sample_torch_supports_cuda():
    schedule = make_linear_schedule(4, 0.1, 0.4)

    x0 = torch.ones(2, 17, 3, device="cuda")
    noise = torch.zeros_like(x0)
    timesteps = torch.tensor([0, 3], dtype=torch.long, device="cuda:0")

    xt = q_sample_torch(x0, timesteps, noise, schedule)

    assert xt.device.type == "cuda"
    assert xt.shape == x0.shape
    torch.testing.assert_close(
        xt[0],
        torch.sqrt(torch.as_tensor(schedule.alpha_bars[0], device=x0.device)) * x0[0],
    )

    torch.testing.assert_close(
        xt[1],
        torch.sqrt(torch.as_tensor(schedule.alpha_bars[3], device=x0.device)) * x0[1],
    )


def test_q_sample_rejects_invalid_timestep_dtype():
    schedule = make_linear_schedule(4)

    x0 = torch.ones(2, 3)
    noise = torch.zeros_like(x0)
    timesteps = torch.tensor([0.1, 1.0])

    with pytest.raises(TypeError):
        q_sample_torch(x0, timesteps, noise, schedule)


def test_q_sample_rejects_invalid_timestep_dtype_value():
    schedule = make_linear_schedule(4)

    x0 = torch.ones(2, 3)
    noise = torch.zeros_like(x0)

    with pytest.raises(ValueError):
        q_sample_torch(x0, torch.tensor([-1, 1], dtype=torch.long), noise, schedule)

    with pytest.raises(ValueError):
        q_sample_torch(x0, torch.tensor([0, 4], dtype=torch.long), noise, schedule)
