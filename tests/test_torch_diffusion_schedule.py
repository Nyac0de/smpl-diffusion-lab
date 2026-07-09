import pytest
import torch

from smpl_diffusion_lab.diffusion.torch_schedule import (
    TorchDiffusionSchedule,
    make_linear_torch_schedule,
)


def test_linear_schedule_shapes():
    schedule = make_linear_torch_schedule(10, beta_start=0.1, beta_end=0.2)

    assert isinstance(schedule, TorchDiffusionSchedule)
    assert schedule.betas.shape == (10,)
    assert schedule.alphas.shape == (10,)


def test_linear_torch_schedule_values():
    schedule = make_linear_torch_schedule(4, beta_start=0.1, beta_end=0.4)

    torch.testing.assert_close(
        schedule.betas,
        torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32),
    )
    torch.testing.assert_close(
        schedule.alphas,
        torch.tensor([0.9, 0.8, 0.7, 0.6], dtype=torch.float32),
    )
    torch.testing.assert_close(
        schedule.alpha_bars,
        torch.tensor([0.9, 0.72, 0.504, 0.3024], dtype=torch.float32),
    )


def test_linear_torch_schedule_dtype():
    schedule = make_linear_torch_schedule(4, 0.1, 0.4, dtype=torch.float64)

    assert schedule.betas.dtype == torch.float64
    assert schedule.alphas.dtype == torch.float64
    assert schedule.alpha_bars.dtype == torch.float64


def test_linear_torch_schedule_rejects_invalid_inputs():
    with pytest.raises(TypeError):
        make_linear_torch_schedule(1.5)
    with pytest.raises(ValueError):
        make_linear_torch_schedule(0)
    with pytest.raises(ValueError):
        make_linear_torch_schedule(10, beta_start=0.0)
    with pytest.raises(ValueError):
        make_linear_torch_schedule(10, beta_end=1.0)
    with pytest.raises(ValueError):
        make_linear_torch_schedule(10, beta_start=0.2, beta_end=0.1)
