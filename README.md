# smpl-diffusion-lab

From-scratch learning project for 3D human pose estimation, SMPL, and diffusion models.

This repository is a learning-oriented rewrite inspired by DiffPose, but it does not copy the original implementation directly.

## Goals

- Understand diffusion models from a minimal implementation.
- Build a toy 3D pose diffusion model from scratch.
- Extend the project toward SMPL parameter learning and mesh recovery.
- Practice clean PyTorch project organization.

## Roadmap

```text

smpl-diffusion-lab

├─ Stage 0: project setup

├─ Stage 1: toy diffusion

├─ Stage 2: 3D skeleton diffusion

├─ Stage 3: conditional 2D-to-3D pose diffusion

├─ Stage 4: graph or transformer pose backbone

├─ Stage 5: SMPL forward model

└─ Stage 6: SMPL diffusion experiments
```

## References

- DiffPose: https://github.com/gongjia0208/diffpose
- DiffPose paper: https://arxiv.org/abs/2211.16940