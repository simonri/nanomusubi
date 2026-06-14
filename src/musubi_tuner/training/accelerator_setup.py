"""Accelerator/device setup utilities (single-GPU)."""

import argparse
import time

import torch
from accelerate import Accelerator
from accelerate.utils import DynamoBackend, TorchDynamoPlugin


# for collate_fn: epoch and step is multiprocessing.Value
class collator_class:
  def __init__(self, epoch, dataset):
    self.current_epoch = epoch
    self.dataset = dataset  # not used if worker_info is not None, in case of multiprocessing

  def __call__(self, examples):
    worker_info = torch.utils.data.get_worker_info()
    # worker_info is None in the main process
    if worker_info is not None:
      dataset = worker_info.dataset
    else:
      dataset = self.dataset

    # set epoch for validation
    dataset.set_current_epoch(self.current_epoch.value)
    return examples[0]  # batch size is always 1, so we unwrap it here


def prepare_accelerator(cfg: argparse.Namespace) -> Accelerator:
  """Build a single-GPU Accelerator with mixed-precision, dynamo, and logging configured."""
  if cfg.logging_dir is None:
    logging_dir = None
  else:
    log_prefix = "" if cfg.log_prefix is None else cfg.log_prefix
    logging_dir = cfg.logging_dir + "/" + log_prefix + time.strftime("%Y%m%d%H%M%S", time.localtime())

  if cfg.log_with is None:
    log_with = "tensorboard" if logging_dir is not None else None
  else:
    log_with = cfg.log_with
    if log_with == "tensorboard":
      if logging_dir is None:
        raise ValueError("logging_dir is required when log_with is tensorboard")

  dynamo_plugin = None
  if cfg.dynamo_backend.upper() != "NO":
    dynamo_plugin = TorchDynamoPlugin(
      backend=DynamoBackend(cfg.dynamo_backend.upper()),
      mode=cfg.dynamo_mode,
      fullgraph=cfg.dynamo_fullgraph,
      dynamic=cfg.dynamo_dynamic,
    )

  accelerator = Accelerator(
    gradient_accumulation_steps=cfg.gradient_accumulation_steps,
    mixed_precision=cfg.mixed_precision if cfg.mixed_precision else None,
    log_with=log_with,
    project_dir=logging_dir,
    dynamo_plugin=dynamo_plugin,
  )
  print("accelerator device:", accelerator.device)
  return accelerator
