import numpy as np
import pytest

from smpl_diffusion_lab.diffusion.schedule import (
    DiffusionSchedule,
    make_linear_schedule,
)

# Test the input shape
def test_linear_schedule_shapes():
    schedule = make_linear_schedule(num_timesteps=10, beta_start=0.1, beta_end=0.2)
    assert isinstance(schedule, DiffusionSchedule)
    assert schedule.betas.shape == (10,)
    assert schedule.alphas.shape == (10,)
    assert schedule.alpha_bars.shape == (10,)

# Test the values of betas, alphas, and alpha_bars
def test_linear_schedule_values():
    schedule = make_linear_schedule(num_timesteps=4, beta_start=0.1, beta_end=0.4)
    np.testing.assert_allclose(
        schedule.betas,
        np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        schedule.alphas,
        np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        schedule.alpha_bars,
        np.array([0.9, 0.72, 0.504, 0.3024], dtype=np.float32),
        rtol=1e-6,
    )

# Test invalid inputs
def test_invalid_num_timesteps():
    with pytest.raises(ValueError):
        make_linear_schedule(0)
    with pytest.raises(TypeError):
        make_linear_schedule(1.5)
    with pytest.raises(ValueError):
        make_linear_schedule(10, beta_start=0.0)
    with pytest.raises(ValueError):
        make_linear_schedule(10, beta_end=1.0)

# Test invalid beta order
def test_invalid_beta_order():
    with pytest.raises(ValueError):
        make_linear_schedule(10, beta_start=0.2, beta_end=0.1)

# Test the alpha_bars are decreasing
def test_alpha_bars_decreasing():
    schedule = make_linear_schedule(num_timesteps=10, beta_start=0.1, beta_end=0.2)
    assert np.all(np.diff(schedule.alpha_bars) < 0)