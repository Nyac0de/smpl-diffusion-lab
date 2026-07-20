from pathlib import Path

import matplotlib.pyplot as plt
import torch
import math

from smpl_diffusion_lab.data.toy import sample_ring_mixture
from smpl_diffusion_lab.diffusion.schedule import make_linear_schedule
from smpl_diffusion_lab.diffusion.torch_sampling import sample_ddpm_torch
from smpl_diffusion_lab.models.noise_mlp import NoisePredictMLP


def distribution_diagnostics(
    samples: torch.Tensor,
    *,
    num_modes: int = 8,
    radius: float = 2.0,
) -> dict[str, float]:
    """Compute distances from samples to the nearest ring mode"""
    mode_indices = torch.arange(
        num_modes,
        dtype=samples.dtype,
        device=samples.device,
    )

    mode_angles = (
        2.0
        * math.pi
        * mode_indices
        / num_modes
    )

    mode_centers = radius * torch.stack(
        [
            torch.cos(mode_angles),
            torch.sin(mode_angles),
        ],
        dim=1,
    )

    distances_to_modes = torch.cdist(
        samples,
        mode_centers,
    )
    nearest_distances = distances_to_modes.min(dim=1).values

    sample_angles = torch.atan2(
        samples[:, 1],
        samples[:, 0],
    )
    sample_angles = torch.remainder(
        sample_angles,
        2.0 * math.pi,
    )

    angle_step = 2.0 * math.pi / num_modes
    nearest_mode_indices = torch.round(
        sample_angles / angle_step
    )
    nearest_mode_angles = nearest_mode_indices * angle_step

    angle_errors = torch.abs(
        sample_angles - nearest_mode_angles
    )
    angle_errors = torch.torch.minimum(
        angle_errors,
        2.0 * math.pi - angle_errors,
    )

    return {
        "nearest_mode_distance_mean": (
            nearest_distances.mean().item()
        ),
        "nearest_mode_distance_p90": (
            torch.quantile(nearest_distances, 0.9).item()
        ),
        "angle_error_mean_degrees": (
            torch.rad2deg(angle_errors).mean().item()
        ),
        "angle_error_p90_degrees": (
            torch.rad2deg(
                torch.quantile(angle_errors, 0.9)
            ).item()
        ),

    }


def main():
    checkpoint_path = Path("outputs/toy_noise_mlp.pt")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    num_timesteps = checkpoint["num_timesteps"]
    data_dim = checkpoint["data_dim"]
    hidden_dim = checkpoint["hidden_dim"]

    model = NoisePredictMLP(
        data_dim=data_dim,
        hidden_dim=hidden_dim,
        num_timesteps=num_timesteps,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    schedule = make_linear_schedule(
        num_timesteps,
        beta_start=1e-4,
        beta_end=2e-2,
    )

    num_samples = 2000

    real_samples = sample_ring_mixture(
        num_samples=num_samples,
        generator=torch.Generator().manual_seed(0),
    )

    generated_samples = sample_ddpm_torch(
        model=model,
        schedule=schedule,
        num_samples=num_samples,
        data_dim=data_dim,
        generator=torch.Generator().manual_seed(1),
    )

    print("real shape:", real_samples.shape)
    print("generated shape:", generated_samples.shape)

    print(
        "real finite:",
        torch.isfinite(real_samples).all().item(),
    )
    print(
        "generated finite:",
        torch.isfinite(generated_samples).all().item(),
    )

    real_radii = torch.linalg.vector_norm(
        real_samples,
        dim=1,
    )
    generated_radii = torch.linalg.vector_norm(
        generated_samples,
        dim=1,
    )

    real_diagnostics = distribution_diagnostics(real_samples)
    generated_diagnostics = distribution_diagnostics(generated_samples)

    print("\nReal distribution diagnostics:")
    for name, value in real_diagnostics.items():
        print(f"   {name}: {value:.6f}")
    print("\nGenerated distribution diagnostics:")
    for name,value in generated_diagnostics.items():
        print(f"    {name}: {value:.6f}")
    print("real mean radius:", real_radii.mean().item())
    print(
        "generated mean radius:",
        generated_radii.mean().item(),
    )

    real_numpy = real_samples.cpu().numpy()
    generated_numpy = generated_samples.cpu().numpy()

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10, 5),
        constrained_layout=True,
    )

    axes[0].scatter(
        real_numpy[:, 0],
        real_numpy[:, 1],
        s=8,
        alpha=0.45,
        color="#2563EB",
        edgecolors="none",
    )
    axes[0].set_title("Real ring mixture")

    axes[1].scatter(
        generated_numpy[:, 0],
        generated_numpy[:, 1],
        s=8,
        alpha=0.45,
        color="#D97706",
        edgecolors="none",
    )
    axes[1].set_title("DDPM generated samples")

    for axis in axes:
        axis.set_xlim(-3.5, 3.5)
        axis.set_ylim(-3.5, 3.5)
        axis.set_aspect("equal")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.grid(
            linewidth=0.5,
            alpha=0.2,
        )

    figure.suptitle(
        "Toy diffusion: real vs generated distribution"
    )

    output_path = Path("outputs/toy_samples.png")
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"saved figure to {output_path}")

if __name__ == "__main__":
    main()
