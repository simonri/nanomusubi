"""Timestep sampling density helpers."""

import logging

import torch

logger = logging.getLogger(__name__)


def compute_density_for_timestep_sampling(batch_size: int):
  """Uniform random density over [0, 1)."""
  return torch.rand(size=(batch_size,), device="cpu")


def get_sigmas(noise_scheduler, timesteps, device, n_dim=4, dtype=torch.float32):
  sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
  schedule_timesteps = noise_scheduler.timesteps.to(device)
  timesteps = timesteps.to(device)

  if any([(schedule_timesteps == t).sum() == 0 for t in timesteps]):
    logger.warning("Some timesteps are not in the schedule / 一部のtimestepsがスケジュールに含まれていません")
    step_indices = [torch.argmin(torch.abs(schedule_timesteps - t)).item() for t in timesteps]
  else:
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

  sigma = sigmas[step_indices].flatten()
  while len(sigma.shape) < n_dim:
    sigma = sigma.unsqueeze(-1)
  return sigma
