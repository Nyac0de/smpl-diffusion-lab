import torch

from smpl_diffusion_lab.data.toy import sample_ring_mixture


def test_sample_ring_mixture_has_expected_shape():
    samples = sample_ring_mixture(
        num_samples=32,
        generator=torch.Generator().manual_seed(0),
    )

    assert samples.shape == (32, 2)
    assert samples.dtype == torch.float32
    assert torch.isfinite(samples).all()


def test_sample_ring_mixture_without_noise_lies_on_radius():
    samples = sample_ring_mixture(
        num_samples=64,
        radius=2.0,
        noise_std=0.0,
        generator=torch.Generator().manual_seed(0),
    )

    distances = torch.linalg.vector_norm(samples, dim=1)

    assert torch.allclose(
        distances,
        torch.full((64,), 2.0),
        atol=1e-6,
    )
