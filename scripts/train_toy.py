import torch

from smpl_diffusion_lab.data.toy import sample_ring_mixture
from smpl_diffusion_lab.diffusion.schedule import make_linear_schedule
from smpl_diffusion_lab.models.noise_mlp import NoisePredictMLP
from smpl_diffusion_lab.training.loss import noise_prediction_loss
from pathlib import Path


def main():
    torch.manual_seed(0)

    num_timesteps = 1000
    batch_size = 256
    num_training_steps = 5000

    schedule = make_linear_schedule(
        num_timesteps,
        beta_start=1e-4,
        beta_end=2e-2,
    )

    model = NoisePredictMLP(
        data_dim=2,
        hidden_dim=64,
        num_timesteps=num_timesteps,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    loss_sum = 0.0

    for step in range(1, num_training_steps + 1):
        x0 = sample_ring_mixture(
            num_samples=batch_size,
        )

        timesteps = torch.randint(
            low=0,
            high=num_timesteps,
            size=(batch_size,),
            dtype=torch.long,
        )

        noise = torch.randn_like(x0)

        optimizer.zero_grad()

        loss = noise_prediction_loss(
            model=model,
            x0=x0,
            timesteps=timesteps,
            noise=noise,
            schedule=schedule,
        )

        loss.backward()
        optimizer.step()
        loss_sum += loss.item()

        if step % 100 == 0:
            average_loss = loss_sum / 100
            print(f"step={step:04d} "
                  f"average_loss={average_loss:.6f}"
            )
            loss_sum = 0.0

    checkpoint_path = Path("outputs/toy_noise_mlp.pt")
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_timesteps": num_timesteps,
            "data_dim": 2,
            "hidden_dim": 64,
        },
        checkpoint_path,
    )

    print(f"saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    main()
