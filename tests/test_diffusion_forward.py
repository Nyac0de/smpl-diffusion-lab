import numpy as np
import pytest

from smpl_diffusion_lab.diffusion.forward import q_sample
from smpl_diffusion_lab.diffusion.schedule import make_linear_schedule


def test_q_sample_scales_x0_when_noise_is_zero():
    schedule = make_linear_schedule(4, 0.1, 0.4)
    x0 = np.ones((2, 3), dtype=np.float32)
    noise = np.zeros_like(x0)
    timesteps = np.array([0, 3])

    xt = q_sample(x0, timesteps, noise, schedule)

    expected = np.sqrt(schedule.alpha_bars[timesteps]).reshape(2, 1) * x0
    np.testing.assert_allclose(xt, expected, rtol=1e-6)


def test_q_sample_scales_noise_when_x0_is_zero():
    schedule = make_linear_schedule(4, 0.1, 0.4)
    x0 = np.zeros((2, 3), dtype=np.float32)
    noise = np.ones_like(x0)
    timesteps = np.array([0, 3])

    xt = q_sample(x0, timesteps, noise, schedule)

    expected = np.sqrt(1.0 - schedule.alpha_bars[timesteps]).reshape(2, 1) * noise
    np.testing.assert_allclose(xt, expected, rtol=1e-6)


def test_q_sample_supports_pose_shape():
    schedule = make_linear_schedule(4, 0.1, 0.4)
    x0 = np.ones((2, 17, 3), dtype=np.float32)
    noise = np.zeros_like(x0)
    timesteps = np.array([0, 3])

    xt = q_sample(x0, timesteps, noise, schedule)

    assert xt.shape == x0.shape
    np.testing.assert_allclose(
        xt[0],
        np.sqrt(schedule.alpha_bars[0]) * x0[0],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        xt[1],
        np.sqrt(schedule.alpha_bars[3]) * x0[1],
        rtol=1e-6,
    )


def test_q_sample_rejects_mismatched_shape():
    schedule = make_linear_schedule(4, 0.1, 0.4)
    x0 = np.ones((2, 3), dtype=np.float32)
    noise = np.zeros((2, 4), dtype=np.float32)  # Mismatched shape
    timesteps = np.array([0, 1])

    with pytest.raises(ValueError):
        q_sample(x0, timesteps, noise, schedule)


def test_q_sample_rejects_invalid_timesteps():
    schedule = make_linear_schedule(4, 0.1, 0.4)
    x0 = np.ones((2, 3), dtype=np.float32)
    noise = np.zeros_like(x0)

    with pytest.raises(TypeError):
        q_sample(x0, [0, 1], noise, schedule)

    with pytest.raises(TypeError):
        q_sample(x0, np.array([0.0, 1.0]), noise, schedule)

    timesteps_neg = np.array([-1, 1])
    with pytest.raises(ValueError):
        q_sample(x0, timesteps_neg, noise, schedule)

    timesteps_exceed = np.array([0, 4])
    with pytest.raises(ValueError):
        q_sample(x0, timesteps_exceed, noise, schedule)

    with pytest.raises(ValueError):
        q_sample(x0, np.array([0]), noise, schedule)


def test_q_sample_rejects_low_dimensional_x0():
    schedule = make_linear_schedule(4)
    x0 = np.ones((2,), dtype=np.float32)
    noise = np.zeros_like(x0)
    timesteps = np.array([0, 1])

    with pytest.raises(ValueError):
        q_sample(x0, timesteps, noise, schedule)
