"""Wan 2.2 I2V LoRA training entry point."""

import argparse
import logging
import pathlib

from musubi_tuner.training.trainer_base import NetworkTrainer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
  parser = argparse.ArgumentParser(description="Train Wan 2.2 I2V LoRA")
  parser.add_argument("--dit", type=str, required=True, help="DiT checkpoint path")
  parser.add_argument("--dataset_config", type=pathlib.Path, required=True, help="dataset config .toml")
  parser.add_argument("--output_dir", type=str, required=True, help="output directory")
  parser.add_argument("--output_name", type=str, required=True, help="output model name (no extension)")
  parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
  parser.add_argument("--min_timestep", type=int, default=0)
  parser.add_argument("--max_timestep", type=int, default=1000)
  parser.add_argument("--discrete_flow_shift", type=float, default=5.0)
  parser.add_argument("--max_train_epochs", type=int, default=300)
  parser.add_argument("--save_every_n_epochs", type=int, default=10)
  cfg = parser.parse_args()

  # model
  cfg.dit_dtype = None  # auto-detected from checkpoint

  # optimizer
  cfg.optimizer_type = "adamw8bit"
  cfg.optimizer_args = None
  cfg.learning_rate = 1e-4
  cfg.lr_warmup_steps = 100
  cfg.max_grad_norm = 1.0

  # training
  cfg.gradient_checkpointing = True
  cfg.gradient_accumulation_steps = 1
  cfg.max_train_steps = 999999  # effectively unlimited; time budget drives stopping
  cfg.max_training_seconds = 900  # 15 minutes
  cfg.max_data_loader_n_workers = 2
  cfg.persistent_data_loader_workers = True
  cfg.seed = 42

  # network / LoRA
  cfg.network_dim = 32
  cfg.network_alpha = 1
  cfg.network_dropout = None
  cfg.network_args = ["loraplus_lr_ratio=4"]
  cfg.network_weights = None
  cfg.dim_from_weights = False
  cfg.base_weights = None
  cfg.base_weights_multiplier = None
  cfg.scale_weight_norms = None

  # timestep sampling
  cfg.timestep_sampling = "shift"
  cfg.sigmoid_scale = 1.0
  cfg.logit_mean = 0.0
  cfg.logit_std = 1.0
  cfg.preserve_distribution_shape = True
  cfg.num_timestep_buckets = None
  cfg.show_timesteps = None

  cfg.cuda_allow_tf32 = True
  cfg.cuda_cudnn_benchmark = True

  # logging (set logging_dir to enable tensorboard)
  cfg.logging_dir = None
  cfg.log_with = None
  cfg.log_prefix = None
  cfg.log_tracker_name = None
  cfg.log_tracker_config = None
  cfg.log_config = False

  # model metadata
  cfg.no_metadata = False
  cfg.training_comment = None

  # misc
  cfg.disable_numpy_memmap = False

  # dynamo (disabled)
  cfg.dynamo_backend = "NO"
  cfg.dynamo_mode = None
  cfg.dynamo_fullgraph = False
  cfg.dynamo_dynamic = False

  NetworkTrainer().train(cfg)


if __name__ == "__main__":
  main()
