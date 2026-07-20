# Mentoring State

## Sync

- Updated: 2026-07-20
- Branch: `main`
- Based on commit: `e9cdea1`
- Remote: `https://github.com/Nyac0de/smpl-diffusion-lab.git`

## Objective

Complete Stage 1 toy diffusion while developing the ability to organize already-understood PyTorch components into a training pipeline. Do not move to SMPL until toy training, full sampling, checkpointing, and basic visualization have been verified.

## Verified Progress

- Implemented and tested the linear schedule, forward noising, predicted `x0`, reverse mean, posterior variance, and one reverse sampling step.
- Implemented and tested a two-dimensional ring-mixture sampler and a timestep-conditioned noise MLP.
- Implemented the DDPM noise-prediction loss and verified finite parameter gradients.
- Executed one Adam update and verified that at least one model parameter changed.
- Project history reports 33 passing tests at commit `e9cdea1`; re-run them in the active machine before relying on that result.

These facts verify completed code and tests. They do not by themselves prove closed-book reimplementation ability.

## Current Exercise

- Status: `awaiting-attempt`
- Help level: `L2`
- Target files: `src/smpl_diffusion_lab/training/` and `tests/`
- Specification: derive the responsibilities and tensor data flow of one reusable `train_step` before writing Python.
- Learner's latest reasoning: none yet
- Blocking issue: none

The learner should describe the sequence from `x0` to one optimizer update, including `timesteps`, `noise`, `xt`, `noise_pred`, scalar `loss`, parameter gradients, shape, dtype, device, and the choice among returning `loss`, `loss.detach()`, or `loss.item()`.

## Skill Evidence

- Demonstrated: can follow shape-guided implementation, use tests, inspect tracebacks, distinguish `backward()` from `optimizer.step()`, and preserve parameter snapshots with `detach().clone()`.
- Developing: independently retrieving PyTorch syntax; organizing several correct components into a reusable training workflow; predicting broadcasting without step-by-step prompting.
- Recurring errors: misspelled APIs and variable names, missing commas or indentation, function object versus function call, method assignment versus method call, premature variable use, and confusion between full-schedule tensors and per-batch indexed coefficients.

## Retrieval Queue

- Next suitable session: explain and reimplement the coefficient extraction and broadcast-shape logic used by `q_sample_torch` without viewing its body.
- After completing `train_step`: modify it from two-dimensional data to a generic `(B, ...)` batch without changing its training semantics.

## Next Action

Ask the learner for the ordered, shape-annotated data flow of one `train_step`. Review concept and autograd ordering at `L3` before asking for a function signature or Python implementation.

## Environment Notes

- Project requires Python 3.10 or newer.
- Recorded training environment: Conda environment `smpldiff`, Python 3.11.15, PyTorch 2.11.0+cu128, eight NVIDIA GeForce RTX 4090D GPUs.
- Do not version datasets, checkpoints, outputs, credentials, or absolute machine paths.
