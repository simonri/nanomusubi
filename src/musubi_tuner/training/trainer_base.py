"""NetworkTrainer for Wan 2.2 I2V LoRA training."""

import argparse
import ast
import datetime
import importlib
import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field
from multiprocessing import Value

import numpy as np
import toml
import torch
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors.torch import load_file
from tqdm import tqdm

import musubi_tuner.lora_wan as lora_wan_module
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_WAN
from musubi_tuner.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from musubi_tuner.training.accelerator_setup import collator_class, prepare_accelerator
from musubi_tuner.training.timesteps import compute_density_for_timestep_sampling, get_sigmas
from musubi_tuner.utils import model_utils
from musubi_tuner.utils.model_utils import clean_memory_on_device
from musubi_tuner.wan.modules.model import WanModel, detect_wan_model_config, detect_wan_sd_dtype, load_wan_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# --- inlined from sai_model_spec.py ---

_BASE_METADATA = {
  "modelspec.sai_model_spec": "1.0.0",
  "modelspec.architecture": None,
  "modelspec.implementation": None,
  "modelspec.title": None,
  "modelspec.resolution": None,
  "modelspec.description": None,
  "modelspec.author": None,
  "modelspec.date": None,
  "modelspec.license": None,
  "modelspec.tags": None,
  "modelspec.merged_from": None,
  "modelspec.prediction_type": None,
  "modelspec.timestep_range": None,
  "modelspec.encoder_layer": None,
}
_MODELSPEC_TITLE = "modelspec.title"
_ARCH_WAN = "wan2.2"
_ADAPTER_LORA = "lora"
_IMPL_WAN = "https://github.com/Wan-Video/Wan2.2"


def build_metadata(timestamp: float, timesteps: tuple[int, int] | None = None):
  metadata = {}
  metadata.update(_BASE_METADATA)

  metadata["modelspec.architecture"] = f"{_ARCH_WAN}/{_ADAPTER_LORA}"
  metadata["modelspec.implementation"] = _IMPL_WAN
  metadata[_MODELSPEC_TITLE] = f"LoRA@{timestamp}"
  metadata["modelspec.date"] = datetime.datetime.fromtimestamp(int(timestamp)).isoformat()
  metadata["modelspec.resolution"] = "1280x720"

  for key in ["modelspec.author", "modelspec.description", "modelspec.merged_from",
              "modelspec.license", "modelspec.tags", "modelspec.prediction_type",
              "modelspec.encoder_layer"]:
    del metadata[key]

  if timesteps is not None:
    metadata["modelspec.timestep_range"] = f"{timesteps[0]},{timesteps[1]}"
  else:
    del metadata["modelspec.timestep_range"]

  return metadata


# --- inlined from train_utils.py ---

EPOCH_FILE_NAME = "{}-{:06d}"


def get_sanitized_config_or_none(cfg: argparse.Namespace):
  if not cfg.log_config:
    return None

  sensitive_path_args = [
    "dit", "text_encoder1", "text_encoder2", "image_encoder",
    "base_weights", "network_weights", "output_dir", "logging_dir",
  ]
  filtered_args = {}
  for k, v in vars(cfg).items():
    if k not in sensitive_path_args:
      if v is None or isinstance(v, (bool, str, float, int)):
        filtered_args[k] = v
      elif isinstance(v, list):
        filtered_args[k] = f"{v}"
      elif isinstance(v, object):
        filtered_args[k] = f"{v}"
  return filtered_args


class LossRecorder:
  def __init__(self):
    self.loss_list: list[float] = []
    self.loss_total: float = 0.0

  def add(self, *, epoch: int, step: int, loss: float) -> None:
    if epoch == 0:
      self.loss_list.append(loss)
    else:
      while len(self.loss_list) <= step:
        self.loss_list.append(0.0)
      self.loss_total -= self.loss_list[step]
      self.loss_list[step] = loss
    self.loss_total += loss

  @property
  def moving_average(self) -> float:
    return self.loss_total / len(self.loss_list)


def get_epoch_ckpt_name(model_name, epoch_no: int):
  return EPOCH_FILE_NAME.format(model_name, epoch_no) + ".safetensors"


def get_last_ckpt_name(model_name):
  return model_name + ".safetensors"


# --- end inline ---



SS_METADATA_KEY_BASE_MODEL_VERSION = "ss_base_model_version"
SS_METADATA_KEY_NETWORK_MODULE = "ss_network_module"

# Wan-only fork: only one LoRA network module is supported.
NETWORK_MODULE_NAME = "musubi_tuner.lora_wan"
SS_METADATA_KEY_NETWORK_DIM = "ss_network_dim"
SS_METADATA_KEY_NETWORK_ALPHA = "ss_network_alpha"
SS_METADATA_KEY_NETWORK_ARGS = "ss_network_args"

SS_METADATA_MINIMUM_KEYS = [
  SS_METADATA_KEY_BASE_MODEL_VERSION,
  SS_METADATA_KEY_NETWORK_MODULE,
  SS_METADATA_KEY_NETWORK_DIM,
  SS_METADATA_KEY_NETWORK_ALPHA,
  SS_METADATA_KEY_NETWORK_ARGS,
]


@dataclass
class DiTOutput:
  """Return type for ``NetworkTrainer.call_dit``.

  Internal extension point — no API stability guarantees. Vanilla flow only
  needs ``pred`` and ``target``; extension subclasses can stash arbitrary
  additional outputs (e.g. hidden features for representation-alignment
  losses) in the ``extra`` dict without breaking the base signature.
  """

  pred: torch.Tensor
  target: torch.Tensor
  extra: dict = field(default_factory=dict)


class NetworkTrainer:
  def __init__(self):
    self.timestep_range_pool = []
    self.num_timestep_buckets: int | None = None  # for get_bucketed_timestep()
    self.vae_frame_stride = 4
    self.default_discrete_flow_shift = 14.5
    self.default_guidance_scale: float = 1.0

  # TODO 他のスクリプトと共通化する
  def generate_step_logs(
    self,
    cfg: argparse.Namespace,
    current_loss,
    avr_loss,
    lr_scheduler,
    lr_descriptions,
    optimizer=None,
    keys_scaled=None,
    mean_norm=None,
    maximum_norm=None,
  ):
    logs = {"loss/current": current_loss, "loss/average": avr_loss}

    if keys_scaled is not None:
      logs["max_norm/keys_scaled"] = keys_scaled
      logs["max_norm/average_key_norm"] = mean_norm
      logs["max_norm/max_key_norm"] = maximum_norm

    lrs = lr_scheduler.get_last_lr()
    for i, lr in enumerate(lrs):
      if lr_descriptions is not None:
        lr_desc = lr_descriptions[i]
      else:
        if len(lrs) > 2:
          lr_desc = f"group{i}"
        else:
          lr_desc = "unet"

      logs[f"lr/{lr_desc}"] = lr

      if cfg.optimizer_type.lower().startswith("DAdapt".lower()) or cfg.optimizer_type.lower().endswith("Prodigy".lower()):
        # tracking d*lr value
        logs[f"lr/d*lr/{lr_desc}"] = lr_scheduler.optimizers[-1].param_groups[i]["d"] * lr_scheduler.optimizers[-1].param_groups[i]["lr"]

      if cfg.optimizer_type.lower().endswith("ProdigyPlusScheduleFree".lower()) and optimizer is not None:
        # tracking d*lr value of unet.
        logs[f"lr/d*lr/{lr_desc}"] = optimizer.param_groups[i]["d"] * optimizer.param_groups[i]["lr"]
        if "effective_lr" in optimizer.param_groups[i]:
          logs[f"lr/d*eff_lr/{lr_desc}"] = optimizer.param_groups[i]["d"] * optimizer.param_groups[i]["effective_lr"]

    return logs

  def get_optimizer(self, cfg, trainable_params: list[torch.nn.Parameter]) -> tuple[str, str, torch.optim.Optimizer]:
    # adamw, adamw8bit, adafactor

    optimizer_type = cfg.optimizer_type.lower()

    # split optimizer_type and optimizer_args
    optimizer_kwargs = {}
    if cfg.optimizer_args is not None and len(cfg.optimizer_args) > 0:
      for arg in cfg.optimizer_args:
        key, value = arg.split("=")
        value = ast.literal_eval(value)
        optimizer_kwargs[key] = value

    lr = cfg.learning_rate
    optimizer = None
    optimizer_class = None

    if optimizer_type.endswith("8bit".lower()):
      try:
        import bitsandbytes as bnb
      except ImportError:
        raise ImportError("No bitsandbytes / bitsandbytesがインストールされていないようです")

      if optimizer_type == "AdamW8bit".lower():
        logger.info(f"use 8-bit AdamW optimizer | {optimizer_kwargs}")
        optimizer_class = bnb.optim.AdamW8bit
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "Adafactor".lower():
      # Adafactor: check relative_step and warmup_init
      if "relative_step" not in optimizer_kwargs:
        optimizer_kwargs["relative_step"] = True  # default
      if not optimizer_kwargs["relative_step"] and optimizer_kwargs.get("warmup_init", False):
        logger.info("set relative_step to True because warmup_init is True / warmup_initがTrueのためrelative_stepをTrueにします")
        optimizer_kwargs["relative_step"] = True
      logger.info(f"use Adafactor optimizer | {optimizer_kwargs}")

      if optimizer_kwargs["relative_step"]:
        logger.info("relative_step is true / relative_stepがtrueです")
        if lr != 0.0:
          logger.warning("learning rate is used as initial_lr / 指定したlearning rateはinitial_lrとして使用されます")
        cfg.learning_rate = None

        lr = None
      else:
        if cfg.max_grad_norm != 0.0:
          logger.warning(
            "because max_grad_norm is set, clip_grad_norm is enabled. consider set to 0 / max_grad_normが設定されているためclip_grad_normが有効になります。0に設定して無効にしたほうがいいかもしれません"
          )
        if optimizer_kwargs.get("clip_threshold", 1.0) != 1.0:
          logger.warning("clip_threshold=1.0 will be good / clip_thresholdは1.0が良いかもしれません")

      optimizer_class = transformers.optimization.Adafactor
      optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "AdamW".lower():
      logger.info(f"use AdamW optimizer | {optimizer_kwargs}")
      optimizer_class = torch.optim.AdamW
      optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    if optimizer is None:
      # 任意のoptimizerを使う
      case_sensitive_optimizer_type = cfg.optimizer_type  # not lower
      logger.info(f"use {case_sensitive_optimizer_type} | {optimizer_kwargs}")

      if "." not in case_sensitive_optimizer_type:  # from torch.optim
        optimizer_module = torch.optim
      else:  # from other library
        values = case_sensitive_optimizer_type.split(".")
        optimizer_module = importlib.import_module(".".join(values[:-1]))
        case_sensitive_optimizer_type = values[-1]

      optimizer_class = getattr(optimizer_module, case_sensitive_optimizer_type)
      optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    # for logging
    optimizer_name = optimizer_class.__module__ + "." + optimizer_class.__name__
    optimizer_args = ",".join([f"{k}={v}" for k, v in optimizer_kwargs.items()])

    # get train and eval functions
    if hasattr(optimizer, "train") and callable(optimizer.train):
      train_fn = optimizer.train
      eval_fn = optimizer.eval
    else:
      train_fn = lambda: None
      eval_fn = lambda: None

    return optimizer_name, optimizer_args, optimizer, train_fn, eval_fn

  def get_lr_scheduler(self, cfg, optimizer: torch.optim.Optimizer):
    from transformers.optimization import get_cosine_schedule_with_warmup

    num_training_steps = cfg.max_train_steps
    num_warmup_steps = int(cfg.lr_warmup_steps * num_training_steps) if isinstance(cfg.lr_warmup_steps, float) else cfg.lr_warmup_steps
    return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

  def get_bucketed_timestep(self) -> float:
    if self.num_timestep_buckets is None or self.num_timestep_buckets <= 1:
      return random.random()

    if len(self.timestep_range_pool) == 0:
      bucket_size = 1.0 / self.num_timestep_buckets
      for i in range(self.num_timestep_buckets):
        self.timestep_range_pool.append((i * bucket_size, (i + 1) * bucket_size))
      random.shuffle(self.timestep_range_pool)

    # print(f"timestep_range_pool: {self.timestep_range_pool}")
    a, b = self.timestep_range_pool.pop()
    return random.uniform(a, b)

  def get_noisy_model_input_and_timesteps(
    self,
    cfg: argparse.Namespace,
    noise: torch.Tensor,
    latents: torch.Tensor,
    timesteps: list[float] | None,
    noise_scheduler: FlowMatchDiscreteScheduler,
    device: torch.device,
    dtype: torch.dtype,
  ):
    batch_size = noise.shape[0]

    if timesteps is not None:
      timesteps = torch.tensor(timesteps, device=device)

    def uniform_to_normal_ppF(t_uniform: torch.Tensor) -> torch.Tensor:
      """Use `torch.erfinv` to compute the inverse CDF to generate values from a normal distribution."""
      # Clip small values to prevent inf in erfinv
      eps = 1e-7
      t_uniform = torch.clamp(t_uniform, eps, 1.0 - eps)

      # PPF of standard normal distribution: sqrt(2) * erfinv(2q - 1)
      term = 2.0 * t_uniform - 1.0
      x_normal = math.sqrt(2.0) * torch.erfinv(term)
      return x_normal

    def uniform_to_logsnr_ppF_pytorch(t_uniform: torch.Tensor, mean: float, std: float) -> torch.Tensor:
      """Use erfinv to compute the inverse CDF."""
      # Clip small values to prevent inf in erfinv
      eps = 1e-7
      t_uniform = torch.clamp(t_uniform, eps, 1.0 - eps)

      term = 2.0 * t_uniform - 1.0
      logsnr = mean + std * math.sqrt(2.0) * torch.erfinv(term)
      return logsnr

    if cfg.timestep_sampling in ("uniform", "sigmoid", "shift", "logsnr"):

      def compute_sampling_timesteps(org_timesteps: torch.Tensor | None) -> torch.Tensor:
        def rand(bs: int, org_ts: torch.Tensor | None = None) -> torch.Tensor:
          nonlocal device
          return torch.rand((bs,), device=device) if org_ts is None else org_ts

        def randn(bs: int, org_ts: torch.Tensor | None = None) -> torch.Tensor:
          nonlocal device
          return uniform_to_normal_ppF(org_ts) if org_ts is not None else torch.randn((bs,), device=device)

        def rand_logsnr(bs: int, mean: float, std: float, org_ts: torch.Tensor | None = None) -> torch.Tensor:
          nonlocal device
          logsnr = uniform_to_logsnr_ppF_pytorch(org_ts, mean, std) if org_ts is not None else torch.normal(mean=mean, std=std, size=(bs,), device=device)
          return logsnr

        if cfg.timestep_sampling == "uniform" or cfg.timestep_sampling == "sigmoid":
          # Simple random t-based noise sampling
          if cfg.timestep_sampling == "sigmoid":
            t = torch.sigmoid(cfg.sigmoid_scale * randn(batch_size, org_timesteps))
          else:
            t = rand(batch_size, org_timesteps)

        elif cfg.timestep_sampling == "shift":
          shift = cfg.discrete_flow_shift
          logits_norm = randn(batch_size, org_timesteps)
          logits_norm = logits_norm * cfg.sigmoid_scale  # larger scale for more uniform sampling
          t = logits_norm.sigmoid()
          t = (t * shift) / (1 + (shift - 1) * t)

        elif cfg.timestep_sampling == "logsnr":
          # https://arxiv.org/abs/2411.14793v3
          logsnr = rand_logsnr(batch_size, cfg.logit_mean, cfg.logit_std, org_timesteps)
          t = torch.sigmoid(-logsnr / 2)

        return t  # 0 to 1

      t_min = cfg.min_timestep if cfg.min_timestep is not None else 0
      t_max = cfg.max_timestep if cfg.max_timestep is not None else 1000.0
      t_min /= 1000.0
      t_max /= 1000.0

      if not cfg.preserve_distribution_shape:
        t = compute_sampling_timesteps(timesteps)
        t = t * (t_max - t_min) + t_min  # scale to [t_min, t_max], default [0, 1]
      else:
        max_loops = 1000
        available_t = []
        for i in range(max_loops):
          t = None
          if self.num_timestep_buckets is not None:
            t = torch.tensor([self.get_bucketed_timestep() for _ in range(batch_size)], device=device)
          t = compute_sampling_timesteps(t)
          for t_i in t:
            if t_min <= t_i <= t_max:
              available_t.append(t_i)
            if len(available_t) == batch_size:
              break
          if len(available_t) == batch_size:
            break
        if len(available_t) < batch_size:
          logger.warning(
            f"Could not sample {batch_size} valid timesteps in {max_loops} loops / {max_loops}ループで{batch_size}個の有効なタイムステップをサンプリングできませんでした"
          )
          available_t = compute_sampling_timesteps(timesteps)
        else:
          t = torch.stack(available_t, dim=0)  # [batch_size, ]

      timesteps = t * 1000.0
      t = t.view(-1, 1, 1, 1, 1) if latents.ndim == 5 else t.view(-1, 1, 1, 1)
      noisy_model_input = (1 - t) * latents + t * noise

      timesteps += 1  # 1 to 1000
    else:
      # Sample a random timestep for each image
      # for weighting schemes where we sample timesteps non-uniformly
      u = compute_density_for_timestep_sampling(batch_size=batch_size)
      # indices = (u * noise_scheduler.config.num_train_timesteps).long()
      t_min = cfg.min_timestep if cfg.min_timestep is not None else 0
      t_max = cfg.max_timestep if cfg.max_timestep is not None else 1000
      indices = (u * (t_max - t_min) + t_min).long()

      timesteps = noise_scheduler.timesteps[indices].to(device=device)  # 1 to 1000

      # Add noise according to flow matching.
      sigmas = get_sigmas(noise_scheduler, timesteps, device, n_dim=latents.ndim, dtype=dtype)
      noisy_model_input = sigmas * noise + (1.0 - sigmas) * latents

    # print(f"actual timesteps: {timesteps}")
    return noisy_model_input, timesteps

  def show_timesteps(self, cfg: argparse.Namespace):
    N_TRY = 100000
    BATCH_SIZE = 1000
    CONSOLE_WIDTH = 64
    N_TIMESTEPS_PER_LINE = 25

    noise_scheduler = FlowMatchDiscreteScheduler(shift=cfg.discrete_flow_shift, reverse=True, solver="euler")
    # print(f"Noise scheduler timesteps: {noise_scheduler.timesteps}")

    latents = torch.zeros(BATCH_SIZE, 1, 1, 1024 // 8, 1024 // 8, dtype=torch.float16)
    noise = torch.ones_like(latents)

    # sample timesteps
    sampled_timesteps = [0] * noise_scheduler.config.num_train_timesteps
    for i in tqdm(range(N_TRY // BATCH_SIZE)):
      bucketed_timesteps = None
      if cfg.num_timestep_buckets is not None and cfg.num_timestep_buckets > 1:
        self.num_timestep_buckets = cfg.num_timestep_buckets
        bucketed_timesteps = [self.get_bucketed_timestep() for _ in range(BATCH_SIZE)]

      # we use noise=1, so retured noisy_model_input is same as timestep, because `noisy_model_input = (1 - t) * latents + t * noise`
      actual_timesteps, _ = self.get_noisy_model_input_and_timesteps(cfg, noise, latents, bucketed_timesteps, noise_scheduler, "cpu", torch.float16)
      actual_timesteps = actual_timesteps[:, 0, 0, 0, 0] * 1000
      for t in actual_timesteps:
        t = int(t.item())
        sampled_timesteps[t] += 1

    # loss weighting is uniform (no SD3-style weighting), so all weights are 1.0
    sampled_weighting = [1.0] * noise_scheduler.config.num_train_timesteps

    # show results
    if cfg.show_timesteps == "image":
      # show timesteps with matplotlib
      import matplotlib.pyplot as plt

      plt.figure(figsize=(10, 5))
      plt.subplot(1, 2, 1)
      plt.bar(range(len(sampled_timesteps)), sampled_timesteps, width=1.0)
      plt.title("Sampled timesteps")
      plt.xlabel("Timestep")
      plt.ylabel("Count")

      plt.subplot(1, 2, 2)
      plt.bar(range(len(sampled_weighting)), sampled_weighting, width=1.0)
      plt.title("Sampled loss weighting")
      plt.xlabel("Timestep")
      plt.ylabel("Weighting")

      plt.tight_layout()
      plt.show()

    else:
      sampled_timesteps = np.array(sampled_timesteps)
      sampled_weighting = np.array(sampled_weighting)

      # average per line
      sampled_timesteps = sampled_timesteps.reshape(-1, N_TIMESTEPS_PER_LINE).mean(axis=1)
      sampled_weighting = sampled_weighting.reshape(-1, N_TIMESTEPS_PER_LINE).mean(axis=1)

      max_count = max(sampled_timesteps)
      print(f"Sampled timesteps: max count={max_count}")
      for i, t in enumerate(sampled_timesteps):
        line = f"{(i) * N_TIMESTEPS_PER_LINE:4d}-{(i + 1) * N_TIMESTEPS_PER_LINE - 1:4d}: "
        line += "#" * int(t / max_count * CONSOLE_WIDTH)
        print(line)

      max_weighting = max(sampled_weighting)
      print(f"Sampled loss weighting: max weighting={max_weighting}")
      for i, w in enumerate(sampled_weighting):
        line = f"{i * N_TIMESTEPS_PER_LINE:4d}-{(i + 1) * N_TIMESTEPS_PER_LINE - 1:4d}: {w:8.2f} "
        line += "#" * int(w / max_weighting * CONSOLE_WIDTH)
        print(line)

  # region Wan 2.2 I2V model implementation

  def handle_model_specific_args(self, cfg: argparse.Namespace):
    self.config = detect_wan_model_config(cfg.dit)
    self.dit_dtype = detect_wan_sd_dtype(cfg.dit)

    if self.dit_dtype == torch.float16:
      assert cfg.mixed_precision in ["fp16", "no"], "DiT weights are in fp16, mixed precision must be fp16 or no"
    elif self.dit_dtype == torch.bfloat16:
      assert cfg.mixed_precision in ["bf16", "no"], "DiT weights are in bf16, mixed precision must be bf16 or no"
    else:
      raise ValueError(f"Unsupported DiT dtype for training: {self.dit_dtype}. Use fp16 or bf16 weights.")

    cfg.dit_dtype = model_utils.dtype_to_str(self.dit_dtype)

  def load_transformer(
    self,
    accelerator: Accelerator,
    cfg: argparse.Namespace,
    dit_path: str,
    loading_device: str,
    dit_weight_dtype: torch.dtype | None,
  ):
    return load_wan_model(
      self.config,
      accelerator.device,
      dit_path,
      loading_device,
      dit_weight_dtype,
      disable_numpy_memmap=cfg.disable_numpy_memmap,
    )

  def scale_shift_latents(self, latents):
    return latents

  def call_dit(
    self,
    cfg: argparse.Namespace,
    accelerator: Accelerator,
    transformer_arg,
    latents: torch.Tensor,
    batch: dict[str, torch.Tensor],
    noise: torch.Tensor,
    noisy_model_input: torch.Tensor,
    timesteps: torch.Tensor,
    network_dtype: torch.dtype,
    **kwargs,
  ) -> DiTOutput:
    model: WanModel = transformer_arg
    image_latents = batch["latents_image"].to(device=accelerator.device, dtype=network_dtype)
    context = [t.to(device=accelerator.device, dtype=network_dtype) for t in batch["t5"]]

    if cfg.gradient_checkpointing:
      noisy_model_input.requires_grad_(True)
      for t in context:
        t.requires_grad_(True)
      image_latents.requires_grad_(True)

    lat_f, lat_h, lat_w = latents.shape[2:5]
    seq_len = lat_f * lat_h * lat_w // (1 * 2 * 2)  # Wan I2V patch_size=(1,2,2)
    latents = latents.to(device=accelerator.device, dtype=network_dtype)
    noisy_model_input = noisy_model_input.to(device=accelerator.device, dtype=network_dtype)
    with accelerator.autocast():
      model_pred = model(noisy_model_input, t=timesteps, context=context, seq_len=seq_len, y=image_latents)
    model_pred = torch.stack(model_pred, dim=0)

    target = noise - latents
    return DiTOutput(pred=model_pred, target=target)

  # endregion Wan 2.2 I2V model implementation

  # region extension seams
  # Internal extension points — no API stability guarantees.
  # Subclasses live in this repo; if you fork, expect breakage on updates.

  def process_batch(
    self,
    cfg: argparse.Namespace,
    accelerator: Accelerator,
    transformer,
    network,
    batch: dict[str, torch.Tensor],
    latents: torch.Tensor,
    noise: torch.Tensor,
    noise_scheduler,
    dit_dtype: torch.dtype,
    network_dtype: torch.dtype,
    global_step: int,
  ) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute scalar loss for one training batch (pre-backward).

    Default implementation: vanilla flow matching, delegating the loss
    formulation itself to ``compute_loss``. Override either method:
    ``process_batch`` to change what gets fed to the model (Self-Flow's
    dual-timestep dance, etc.), or ``compute_loss`` to swap the loss
    formulation while keeping the standard data flow.

    Returns ``(scalar_loss, loss_metrics)`` — ``loss_metrics`` is merged
    into the per-step log dict alongside ``extra_step_logs``.

    ``latents`` is already scale-shifted; ``noise`` is already sampled.
    """
    noisy_model_input, timesteps = self.get_noisy_model_input_and_timesteps(
      cfg, noise, latents, batch["timesteps"], noise_scheduler, accelerator.device, dit_dtype
    )

    output = self.call_dit(cfg, accelerator, transformer, latents, batch, noise, noisy_model_input, timesteps, network_dtype)

    return self.compute_loss(cfg, output, timesteps, noise_scheduler, dit_dtype, network_dtype)

  def compute_loss(
    self,
    cfg: argparse.Namespace,
    output: DiTOutput,
    timesteps: torch.Tensor,
    noise_scheduler,
    dit_dtype: torch.dtype,
    network_dtype: torch.dtype,
  ) -> tuple[torch.Tensor, dict[str, float]]:
    """Reduce a ``DiTOutput`` to a scalar loss + per-step metrics dict.

    Default implementation: MSE between ``output.pred`` and ``output.target``,
    then ``.mean()``. Override to swap the loss formulation entirely (e.g.
    Self-Flow's L_gen + gamma * L_rep).

    ``loss_metrics`` defaults to empty; populate with named scalars for
    loss-decomposition logging (e.g. ``{"loss/gen": ..., "loss/rep": ...}``).
    """
    loss = torch.nn.functional.mse_loss(output.pred.to(network_dtype), output.target, reduction="none")
    return loss.mean(), {}

  def on_transformer_loaded(
    self,
    cfg: argparse.Namespace,
    accelerator: Accelerator,
    transformer,
  ) -> None:
    """Called immediately after ``self.load_transformer(...)`` returns.

    At this point the transformer is on its loading device but not yet wrapped
    by the accelerator and not yet in eval mode. Use this hook for one-time
    post-load setup that needs the raw module (e.g. ``register_forward_hook``
    for feature extraction).
    """

  def on_train_start(
    self,
    cfg: argparse.Namespace,
    accelerator: Accelerator,
    network,
    transformer,
    optimizer,
  ) -> None:
    """Called once after accelerator.prepare and before the training loop starts.

    Use this for initializing extension state that depends on prepared models
    (EMA copies, decay schedulers, register_forward_hook on the transformer, etc.).
    """

  def on_post_optimizer_step(
    self,
    cfg: argparse.Namespace,
    accelerator: Accelerator,
    network,
    transformer,
    sync_gradients: bool,
    global_step: int,
  ) -> None:
    """Called after optimizer.step / lr_scheduler.step / zero_grad each inner step.

    ``sync_gradients`` mirrors ``accelerator.sync_gradients`` and is True only
    on steps where an actual optimizer update occurred (gradient accumulation aware).
    ``transformer`` is the accelerator-wrapped DiT — passed so subclasses doing
    non-network (full fine-tuning) bookkeeping or EMA on transformer weights can
    reach it without stashing a reference in ``on_train_start``.
    Use for EMA updates or any post-step bookkeeping.
    """

  def on_post_save(
    self,
    cfg: argparse.Namespace,
    accelerator: Accelerator,
    network,
    transformer,
    ckpt_name: str,
    save_dtype,
    metadata: dict,
  ) -> None:
    """Called after the main network checkpoint has been saved.

    ``ckpt_name`` is the basename written to ``cfg.output_dir``. Use this hook
    to write companion files (EMA weights, projection heads, etc.) alongside.
    ``transformer`` is the accelerator-wrapped DiT — provided so non-network
    (full fine-tuning) subclasses can save companion artifacts derived from it.
    """

  def extra_trainable_params(
    self,
    cfg: argparse.Namespace,
    accelerator: Accelerator,
    network,
    transformer,
    trainable_params: list,
  ) -> list:
    """Optionally augment the param-group list passed to the optimizer.

    Default: pass-through. Override to merge extra modules' parameters
    (e.g. a representation projection head) into ``trainable_params``.
    Subclasses are expected to stash any owned modules on ``self`` so
    ``on_train_start`` and later hooks can use them.
    """
    return trainable_params

  def extra_metadata(self, cfg: argparse.Namespace) -> dict:
    """Returns extra ``ss_*`` metadata keys to embed in saved safetensors.

    Default: empty dict. Override to add extension-specific metadata.
    """
    return {}

  def extra_step_logs(self, cfg: argparse.Namespace, logs: dict) -> dict:
    """Returns additional log entries to merge into the per-step log payload.

    Called just before ``accelerator.log`` on logging steps. The returned
    dict is merged into ``logs`` (existing keys are overwritten on collision).
    Default: empty dict.
    """
    return {}

  # endregion extension seams

  def train(self, cfg):
    if not self._validate_args_and_init(cfg):
      return

    session_id, training_started_at = self._init_session(cfg)
    train_dataset_group, collator, current_epoch = self._build_dataset(cfg)
    accelerator, weight_dtype, dit_dtype, dit_weight_dtype = self._prepare_accelerator_and_dtypes(cfg)
    transformer = self._load_dit_and_swap(cfg, accelerator, dit_weight_dtype)
    network = self._build_network(cfg, accelerator, transformer, weight_dtype)
    if network is None:
      return
    (
      optimizer,
      optimizer_name,
      optimizer_args,
      optimizer_train_fn,
      optimizer_eval_fn,
      lr_scheduler,
      lr_descriptions,
      train_dataloader,
    ) = self._build_optimizer_and_dataloader(cfg, accelerator, network, train_dataset_group, collator, transformer)
    (
      transformer,
      network,
      optimizer,
      train_dataloader,
      lr_scheduler,
      training_model,
      network_dtype,
    ) = self._prepare_with_accelerator(
      cfg,
      accelerator,
      transformer,
      network,
      optimizer,
      train_dataloader,
      lr_scheduler,
      weight_dtype,
      dit_dtype,
      dit_weight_dtype,
    )
    self._run_training_loop(
      cfg,
      accelerator,
      session_id,
      training_started_at,
      train_dataset_group,
      train_dataloader,
      current_epoch,
      transformer,
      network,
      training_model,
      optimizer,
      optimizer_name,
      optimizer_args,
      optimizer_train_fn,
      optimizer_eval_fn,
      lr_scheduler,
      lr_descriptions,
      dit_dtype,
      network_dtype,
    )

  def _validate_args_and_init(self, cfg) -> bool:
    """Validate required cfg, configure CUDA flags, handle `--show_timesteps`.

    Returns False if training should stop early (e.g. `--show_timesteps`).
    """
    if cfg.cuda_allow_tf32:
      torch.backends.cuda.matmul.allow_tf32 = True
      torch.backends.cudnn.allow_tf32 = True
      logger.info("Enabled TF32 on CUDA / CUDAでTF32を有効化しました")
    if cfg.cuda_cudnn_benchmark:
      torch.backends.cudnn.benchmark = True
      logger.info("Enabled cuDNN benchmark / cuDNNベンチマークを有効化しました")

    # check required arguments
    if cfg.dataset_config is None:
      raise ValueError("dataset_config is required / dataset_configが必要です")
    if cfg.dit is None:
      raise ValueError("path to DiT model is required / DiTモデルのパスが必要です")

    if cfg.disable_numpy_memmap:
      logger.info(
        "Disabling numpy memory mapping for model loading. This may lead to higher memory usage but can speed up loading in some cases."
      )

    # check model specific arguments
    self.handle_model_specific_args(cfg)

    # show timesteps for debugging
    if cfg.show_timesteps:
      self.show_timesteps(cfg)
      return False

    return True

  def _init_session(self, cfg):
    session_id = random.randint(0, 2**32)
    training_started_at = time.time()
    # setup_logging(cfg, reset=True)

    if cfg.seed is None:
      cfg.seed = random.randint(0, 2**32)
    set_seed(cfg.seed)
    return session_id, training_started_at

  def _build_dataset(self, cfg):
    # Load dataset config
    if cfg.num_timestep_buckets is not None:
      logger.info(f"Using timestep bucketing. Number of buckets: {cfg.num_timestep_buckets}")
    self.num_timestep_buckets = cfg.num_timestep_buckets  # None or int, None makes all the behavior same as before

    current_epoch = Value("i", 0)  # shared between processes

    logger.info(f"Load dataset config from {cfg.dataset_config}")
    user_config = config_utils.load_user_config(cfg.dataset_config)
    blueprint = config_utils.generate_blueprint(user_config, cfg)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(
      blueprint.dataset_group, training=True, num_timestep_buckets=self.num_timestep_buckets, shared_epoch=current_epoch
    )

    if train_dataset_group.num_train_items == 0:
      raise ValueError(
        "No training items found in the dataset. Please ensure that the latent/Text Encoder cache has been created beforehand."
        " / データセットに学習データがありません。latent/Text Encoderキャッシュを事前に作成したか確認してください"
      )

    ds_for_collator = train_dataset_group if cfg.max_data_loader_n_workers == 0 else None
    collator = collator_class(current_epoch, ds_for_collator)
    return train_dataset_group, collator, current_epoch

  def _prepare_accelerator_and_dtypes(self, cfg):
    # prepare accelerator
    logger.info("preparing accelerator")
    accelerator = prepare_accelerator(cfg)
    assert accelerator.num_processes == 1, (
      f"This fork only supports single-GPU training (got {accelerator.num_processes} processes). "
      "Drop `--num_processes` / `--multi_gpu` from `accelerate launch`."
    )
    if cfg.mixed_precision is None:
      cfg.mixed_precision = accelerator.mixed_precision
      logger.info(f"mixed precision set to {cfg.mixed_precision} / mixed precisionを{cfg.mixed_precision}に設定")

    # prepare dtype
    weight_dtype = torch.float32
    if cfg.mixed_precision == "fp16":
      weight_dtype = torch.float16
    elif cfg.mixed_precision == "bf16":
      weight_dtype = torch.bfloat16

    dit_dtype = torch.bfloat16 if cfg.dit_dtype is None else model_utils.str_to_dtype(cfg.dit_dtype)
    dit_weight_dtype = dit_dtype
    logger.info(f"DiT precision: {dit_dtype}, weight precision: {dit_weight_dtype}")

    return accelerator, weight_dtype, dit_dtype, dit_weight_dtype

  def _load_dit_and_swap(self, cfg, accelerator, dit_weight_dtype):
    logger.info(f"Loading DiT model from {cfg.dit}")
    transformer = self.load_transformer(accelerator, cfg, cfg.dit, accelerator.device, dit_weight_dtype)
    self.on_transformer_loaded(cfg, accelerator, transformer)
    transformer.eval()
    transformer.requires_grad_(False)
    return transformer

  def _build_network(self, cfg, accelerator, transformer, weight_dtype):
    accelerator.print("network module:", NETWORK_MODULE_NAME)
    network_module = lora_wan_module

    if cfg.base_weights is not None:
      # if base_weights is specified, merge the weights to DiT model
      for i, weight_path in enumerate(cfg.base_weights):
        if cfg.base_weights_multiplier is None or len(cfg.base_weights_multiplier) <= i:
          multiplier = 1.0
        else:
          multiplier = cfg.base_weights_multiplier[i]

        accelerator.print(f"merging module: {weight_path} with multiplier {multiplier}")

        weights_sd = load_file(weight_path)
        module = network_module.create_arch_network_from_weights(multiplier, weights_sd, unet=transformer, for_inference=True)
        module.merge_to(transformer, weights_sd, weight_dtype, "cpu")

      accelerator.print(f"all weights merged: {', '.join(cfg.base_weights)}")

    # prepare network
    net_kwargs = {}
    if cfg.network_args is not None:
      for net_arg in cfg.network_args:
        key, value = net_arg.split("=")
        net_kwargs[key] = value

    if cfg.dim_from_weights:
      logger.info(f"Loading network from weights: {cfg.dim_from_weights}")
      weights_sd = load_file(cfg.dim_from_weights)
      network, _ = network_module.create_arch_network_from_weights(1, weights_sd, unet=transformer)
    else:
      network = network_module.create_arch_network(
        1.0,
        cfg.network_dim,
        cfg.network_alpha,
        transformer,
        neuron_dropout=cfg.network_dropout,
        **net_kwargs,
      )
    if network is None:
      return None

    if hasattr(network_module, "prepare_network"):
      network.prepare_network(cfg)

    # apply network to DiT
    network.apply_to(transformer)

    if cfg.network_weights is not None:
      # FIXME consider alpha of weights: this assumes that the alpha is not changed
      info = network.load_weights(cfg.network_weights)
      accelerator.print(f"load network weights from {cfg.network_weights}: {info}")

    if cfg.gradient_checkpointing:
      transformer.enable_gradient_checkpointing()
      network.enable_gradient_checkpointing()  # may have no effect

    # net_kwargs is reconstructed in the metadata phase from cfg.network_args
    return network

  def _build_optimizer_and_dataloader(self, cfg, accelerator, network, train_dataset_group, collator, transformer):
    # prepare optimizer, data loader etc.
    accelerator.print("prepare optimizer, data loader etc.")

    trainable_params, lr_descriptions = network.prepare_optimizer_params(unet_lr=cfg.learning_rate)
    trainable_params = self.extra_trainable_params(cfg, accelerator, network, transformer, trainable_params)
    optimizer_name, optimizer_args, optimizer, optimizer_train_fn, optimizer_eval_fn = self.get_optimizer(cfg, trainable_params)

    # prepare dataloader

    # num workers for data loader: if 0, persistent_workers is not available
    n_workers = min(cfg.max_data_loader_n_workers, os.cpu_count())  # cpu_count or max_data_loader_n_workers

    train_dataloader = torch.utils.data.DataLoader(
      train_dataset_group,
      batch_size=1,
      shuffle=True,
      collate_fn=collator,
      num_workers=n_workers,
      persistent_workers=cfg.persistent_data_loader_workers,
    )

    # calculate max_train_steps
    if cfg.max_train_epochs is not None:
      cfg.max_train_steps = cfg.max_train_epochs * math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
      accelerator.print(f"override steps. steps for {cfg.max_train_epochs} epochs is / 指定エポックまでのステップ数: {cfg.max_train_steps}")

    # send max_train_steps to train_dataset_group
    train_dataset_group.set_max_train_steps(cfg.max_train_steps)

    # prepare lr_scheduler
    lr_scheduler = self.get_lr_scheduler(cfg, optimizer)

    return (
      optimizer,
      optimizer_name,
      optimizer_args,
      optimizer_train_fn,
      optimizer_eval_fn,
      lr_scheduler,
      lr_descriptions,
      train_dataloader,
    )

  def _prepare_with_accelerator(
    self,
    cfg,
    accelerator,
    transformer,
    network,
    optimizer,
    train_dataloader,
    lr_scheduler,
    weight_dtype,
    dit_dtype,
    dit_weight_dtype,
  ):
    # prepare training model. accelerator does some magic here
    network_dtype = torch.float32

    if dit_weight_dtype != dit_dtype and dit_weight_dtype is not None:
      logger.info(f"casting model to {dit_weight_dtype}")
      transformer.to(dit_weight_dtype)

    transformer = accelerator.prepare(transformer)

    network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(network, optimizer, train_dataloader, lr_scheduler)
    training_model = network

    if cfg.gradient_checkpointing:
      transformer.train()
    else:
      transformer.eval()

    accelerator.unwrap_model(network).prepare_grad_etc(transformer)

    return transformer, network, optimizer, train_dataloader, lr_scheduler, training_model, network_dtype

  def _run_training_loop(
    self,
    cfg,
    accelerator,
    session_id,
    training_started_at,
    train_dataset_group,
    train_dataloader,
    current_epoch,
    transformer,
    network,
    training_model,
    optimizer,
    optimizer_name,
    optimizer_args,
    optimizer_train_fn,
    optimizer_eval_fn,
    lr_scheduler,
    lr_descriptions,
    dit_dtype,
    network_dtype,
  ):
    self.on_train_start(cfg, accelerator, network, transformer, optimizer)

    # epoch数を計算する
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    num_train_epochs = math.ceil(cfg.max_train_steps / num_update_steps_per_epoch)

    accelerator.print("running training / 学習開始")
    accelerator.print(f"  num train items / 学習画像、動画数: {train_dataset_group.num_train_items}")
    accelerator.print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
    accelerator.print(f"  num epochs / epoch数: {num_train_epochs}")
    accelerator.print(f"  batch size per device / バッチサイズ: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}")
    # accelerator.print(f"  total train batch size (with parallel & distributed & accumulation) / 総バッチサイズ（並列学習、勾配合計含む）: {total_batch_size}")
    accelerator.print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {cfg.gradient_accumulation_steps}")
    accelerator.print(f"  total optimization steps / 学習ステップ数: {cfg.max_train_steps}")

    # reconstruct net_kwargs for metadata
    net_kwargs = {}
    if cfg.network_args is not None:
      for net_arg in cfg.network_args:
        key, value = net_arg.split("=")
        net_kwargs[key] = value

    # TODO refactor metadata creation and move to util
    metadata = {
      "ss_session_id": session_id,  # random integer indicating which group of epochs the model came from
      "ss_training_started_at": training_started_at,  # unix timestamp
      "ss_output_name": cfg.output_name,
      "ss_learning_rate": cfg.learning_rate,
      "ss_num_train_items": train_dataset_group.num_train_items,
      "ss_num_batches_per_epoch": len(train_dataloader),
      "ss_num_epochs": num_train_epochs,
      "ss_gradient_checkpointing": cfg.gradient_checkpointing,
      "ss_gradient_accumulation_steps": cfg.gradient_accumulation_steps,
      "ss_max_train_steps": cfg.max_train_steps,
      "ss_lr_warmup_steps": cfg.lr_warmup_steps,
      "ss_lr_scheduler": "cosine",
      SS_METADATA_KEY_BASE_MODEL_VERSION: ARCHITECTURE_WAN,
      SS_METADATA_KEY_NETWORK_MODULE: NETWORK_MODULE_NAME,
      SS_METADATA_KEY_NETWORK_DIM: cfg.network_dim,
      SS_METADATA_KEY_NETWORK_ALPHA: cfg.network_alpha,
      "ss_network_dropout": cfg.network_dropout,  # some networks may not have dropout
      "ss_mixed_precision": cfg.mixed_precision,
      "ss_seed": cfg.seed,
      "ss_training_comment": cfg.training_comment,  # will not be updated after training
      # "ss_sd_scripts_commit_hash": train_util.get_git_revision_hash(),
      "ss_optimizer": optimizer_name + (f"({optimizer_args})" if len(optimizer_args) > 0 else ""),
      "ss_max_grad_norm": cfg.max_grad_norm,
      "ss_full_fp16": False,
      "ss_full_bf16": False,
      "ss_logit_mean": cfg.logit_mean,
      "ss_logit_std": cfg.logit_std,
      "ss_timestep_sampling": cfg.timestep_sampling,
      "ss_sigmoid_scale": cfg.sigmoid_scale,
      "ss_discrete_flow_shift": cfg.discrete_flow_shift,
    }
    metadata.update(self.extra_metadata(cfg))

    datasets_metadata = []
    # tag_frequency = {}  # merge tag frequency for metadata editor # TODO support tag frequency
    for dataset in train_dataset_group.datasets:
      dataset_metadata = dataset.get_metadata()
      datasets_metadata.append(dataset_metadata)

    metadata["ss_datasets"] = json.dumps(datasets_metadata)

    # add extra cfg
    if cfg.network_args:
      # metadata["ss_network_args"] = json.dumps(net_kwargs)
      metadata[SS_METADATA_KEY_NETWORK_ARGS] = json.dumps(net_kwargs)

    # model name and hash
    # calculate hash takes time, so we omit it for now
    if cfg.dit is not None:
      # logger.info(f"calculate hash for DiT model: {cfg.dit}")
      logger.info(f"set DiT model name for metadata: {cfg.dit}")
      sd_model_name = cfg.dit
      if os.path.exists(sd_model_name):
        # metadata["ss_sd_model_hash"] = model_utils.model_hash(sd_model_name)
        # metadata["ss_new_sd_model_hash"] = model_utils.calculate_sha256(sd_model_name)
        sd_model_name = os.path.basename(sd_model_name)
      metadata["ss_sd_model_name"] = sd_model_name

    metadata = {k: str(v) for k, v in metadata.items()}

    # make minimum metadata for filtering
    minimum_metadata = {}
    for key in SS_METADATA_MINIMUM_KEYS:
      if key in metadata:
        minimum_metadata[key] = metadata[key]

    init_kwargs = {}
    if cfg.log_tracker_config is not None:
      init_kwargs = toml.load(cfg.log_tracker_config)
    accelerator.init_trackers(
      "network_train" if cfg.log_tracker_name is None else cfg.log_tracker_name,
      config=get_sanitized_config_or_none(cfg),
      init_kwargs=init_kwargs,
    )

    progress_bar = tqdm(range(cfg.max_train_steps), smoothing=0, desc="steps")

    epoch_to_start = 0
    global_step = 0
    noise_scheduler = FlowMatchDiscreteScheduler(shift=cfg.discrete_flow_shift, reverse=True, solver="euler")
    torch.cuda.reset_peak_memory_stats(accelerator.device)

    loss_recorder = LossRecorder()
    del train_dataset_group

    loop_start = time.time()
    max_training_seconds = getattr(cfg, "max_training_seconds", None)
    time_budget_reached = False

    # function for saving/removing
    save_dtype = dit_dtype

    def save_model(ckpt_name: str, unwrapped_nw, steps, epoch_no):
      os.makedirs(cfg.output_dir, exist_ok=True)
      ckpt_file = os.path.join(cfg.output_dir, ckpt_name)

      accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
      metadata["ss_training_finished_at"] = str(time.time())
      metadata["ss_steps"] = str(steps)
      metadata["ss_epoch"] = str(epoch_no)

      metadata_to_save = minimum_metadata if cfg.no_metadata else metadata

      if cfg.min_timestep is not None or cfg.max_timestep is not None:
        min_time_step = cfg.min_timestep if cfg.min_timestep is not None else 0
        max_time_step = cfg.max_timestep if cfg.max_timestep is not None else 1000
        md_timesteps = (min_time_step, max_time_step)
      else:
        md_timesteps = None

      sai_metadata = build_metadata(time.time(), timesteps=md_timesteps)
      metadata_to_save.update(sai_metadata)

      unwrapped_nw.save_weights(ckpt_file, save_dtype, metadata_to_save)
      self.on_post_save(cfg, accelerator, network, transformer, ckpt_name, save_dtype, metadata_to_save)

    def remove_model(old_ckpt_name):
      old_ckpt_file = os.path.join(cfg.output_dir, old_ckpt_name)
      if os.path.exists(old_ckpt_file):
        accelerator.print(f"removing old checkpoint: {old_ckpt_file}")
        os.remove(old_ckpt_file)

    if len(accelerator.trackers) > 0:
      accelerator.log({}, step=0)

    # training loop

    # log device and dtype for each model
    unwrapped_transformer = accelerator.unwrap_model(transformer)
    first_param = next(iter(unwrapped_transformer.parameters()), None)
    logger.info(
      f"DiT dtype: {first_param.dtype if first_param is not None else None}, device: {first_param.device if first_param is not None else accelerator.device}"
    )

    clean_memory_on_device(accelerator.device)

    optimizer_train_fn()  # Set training mode

    for epoch in range(epoch_to_start, num_train_epochs):
      accelerator.print(f"\nepoch {epoch + 1}/{num_train_epochs}")
      current_epoch.value = epoch + 1

      metadata["ss_epoch"] = str(epoch + 1)

      accelerator.unwrap_model(network).on_epoch_start(transformer)

      for step, batch in enumerate(train_dataloader):
        # torch.compiler.cudagraph_mark_step_begin() # for cudagraphs

        latents = batch["latents"]

        with accelerator.accumulate(training_model):
          accelerator.unwrap_model(network).on_step_start()

          latents = self.scale_shift_latents(latents)

          # Sample noise that we'll add to the latents
          noise = torch.randn_like(latents)

          loss, loss_metrics = self.process_batch(
            cfg,
            accelerator,
            transformer,
            network,
            batch,
            latents,
            noise,
            noise_scheduler,
            dit_dtype,
            network_dtype,
            global_step,
          )

          accelerator.backward(loss)
          if accelerator.sync_gradients and cfg.max_grad_norm != 0.0:
            params_to_clip = accelerator.unwrap_model(network).get_trainable_params()
            accelerator.clip_grad_norm_(params_to_clip, cfg.max_grad_norm)

          optimizer.step()
          lr_scheduler.step()
          optimizer.zero_grad(set_to_none=True)

          self.on_post_optimizer_step(cfg, accelerator, network, transformer, accelerator.sync_gradients, global_step)

        if cfg.scale_weight_norms:
          keys_scaled, mean_norm, maximum_norm = accelerator.unwrap_model(network).apply_max_norm_regularization(cfg.scale_weight_norms, accelerator.device)
          max_mean_logs = {"Keys Scaled": keys_scaled, "Average key norm": mean_norm}
        else:
          keys_scaled, mean_norm, maximum_norm = None, None, None

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
          if global_step == 0:
            progress_bar.reset()  # exclude first step from progress bar, because it may take long due to initializations
          progress_bar.update(1)
          global_step += 1
          if max_training_seconds is not None and time.time() - loop_start >= max_training_seconds:
            time_budget_reached = True

        current_loss = loss.detach().item()
        loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
        avr_loss: float = loss_recorder.moving_average
        logs = {"avr_loss": avr_loss, "gpu_C": torch.cuda.temperature(accelerator.device)}
        if cfg.scale_weight_norms:
          progress_bar.set_postfix(**{**max_mean_logs, **logs})
        else:
          progress_bar.set_postfix(**logs)

        if len(accelerator.trackers) > 0:
          logs = self.generate_step_logs(cfg, current_loss, avr_loss, lr_scheduler, lr_descriptions, optimizer, keys_scaled, mean_norm, maximum_norm)
          logs.update(loss_metrics)
          logs.update(self.extra_step_logs(cfg, logs))
          accelerator.log(logs, step=global_step)

        if global_step >= cfg.max_train_steps:
          break
        if time_budget_reached:
          break

      if time_budget_reached:
        break

      # save model at the end of epoch if needed
      optimizer_eval_fn()
      if cfg.save_every_n_epochs is not None:
        saving = (epoch + 1) % cfg.save_every_n_epochs == 0 and (epoch + 1) < num_train_epochs
        if saving:
          ckpt_name = get_epoch_ckpt_name(cfg.output_name, epoch + 1)
          save_model(ckpt_name, accelerator.unwrap_model(network), global_step, epoch + 1)

      optimizer_train_fn()

      # end of epoch

    metadata["ss_training_finished_at"] = str(time.time())

    network = accelerator.unwrap_model(network)

    accelerator.end_training()
    optimizer_eval_fn()

    ckpt_name = get_last_ckpt_name(cfg.output_name)
    save_model(ckpt_name, network, global_step, num_train_epochs)

    logger.info("model saved.")

    training_seconds = time.time() - loop_start
    total_seconds = time.time() - training_started_at
    peak_vram_mb = torch.cuda.max_memory_allocated(accelerator.device) / 1024 / 1024
    print("---")
    print(f"loss:             {loss_recorder.moving_average:.6f}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_steps:        {global_step}")
