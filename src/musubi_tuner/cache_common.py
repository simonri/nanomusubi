import argparse
import logging
import os
from collections.abc import Callable
from typing import Any

import numpy as np
from tqdm import tqdm

from musubi_tuner.dataset.image_video_dataset import BaseDataset

logger = logging.getLogger(__name__)


def encode_datasets(datasets: list[BaseDataset], encode: Callable[..., Any], args: argparse.Namespace, supports_alpha: bool = False):
  num_workers = args.num_workers if args.num_workers is not None else max(1, os.cpu_count() - 1)
  for dataset_index, dataset in enumerate(datasets):
    logger.info(f"Encoding dataset [{dataset_index}]")
    all_latent_cache_paths = []
    for _, batch in tqdm(dataset.retrieve_latent_cache_batches(num_workers)):
      if not supports_alpha:
        for item in batch:
          if isinstance(item.content, np.ndarray):
            if item.content.shape[-1] == 4:
              item.content = item.content[..., :3]
          else:
            item.content = [img[..., :3] if img.shape[-1] == 4 else img for img in item.content]

      all_latent_cache_paths.extend([item.latent_cache_path for item in batch])

      if args.skip_existing:
        batch = [item for item in batch if not os.path.exists(item.latent_cache_path)]
        if len(batch) == 0:
          continue

      batch_size = args.batch_size if args.batch_size is not None else len(batch)
      for start in range(0, len(batch), batch_size):
        encode(batch[start : start + batch_size])

    all_latent_cache_paths = {os.path.normpath(path) for path in all_latent_cache_paths}
    for cache_file in dataset.get_all_latent_cache_files():
      if os.path.normpath(cache_file) not in all_latent_cache_paths:
        if args.keep_cache:
          logger.info(f"Keep cache file not in the dataset: {cache_file}")
        else:
          os.remove(cache_file)
          logger.info(f"Removed old cache file: {cache_file}")


def setup_latent_cache_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset_config", type=str, required=True, help="path to dataset config .toml file")
  parser.add_argument("--vae", type=str, required=False, default=None, help="path to vae checkpoint")
  parser.add_argument("--vae_dtype", type=str, default=None, help="data type for VAE, default depends on model, e.g., float16")
  parser.add_argument("--device", type=str, default=None, help="device to use, default is cuda if available")
  parser.add_argument("--batch_size", type=int, default=None, help="batch size, override dataset config if dataset batch size > this")
  parser.add_argument("--num_workers", type=int, default=None, help="number of workers for dataset. default is cpu count-1")
  parser.add_argument("--skip_existing", action="store_true", help="skip existing cache files")
  parser.add_argument("--keep_cache", action="store_true", help="keep cache files not in dataset")
  parser.add_argument("--debug_mode", type=str, default=None, choices=["image", "console", "video"], help="debug mode")
  parser.add_argument("--console_width", type=int, default=80, help="debug mode: console width")
  parser.add_argument("--console_back", type=str, default=None, help="debug mode: console background color, one of ascii_magic.Back")
  parser.add_argument(
    "--console_num_images",
    type=int,
    default=None,
    help="debug mode: not interactive, number of images to show for each dataset",
  )
  parser.add_argument("--disable_cudnn_backend", action="store_true", help="Disable CUDNN PyTorch backend. May be useful for AMD GPUs.")
  return parser


def prepare_cache_files_and_paths(datasets: list[BaseDataset]):
  all_cache_files_for_dataset = []
  all_cache_paths_for_dataset = []
  for dataset in datasets:
    all_cache_files_for_dataset.append({os.path.normpath(file) for file in dataset.get_all_text_encoder_output_cache_files()})
    all_cache_paths_for_dataset.append(set())
  return all_cache_files_for_dataset, all_cache_paths_for_dataset


def process_text_encoder_batches(
  num_workers: int | None,
  skip_existing: bool,
  batch_size: int,
  datasets: list[BaseDataset],
  all_cache_files_for_dataset: list[set],
  all_cache_paths_for_dataset: list[set],
  encode: Callable[..., Any],
  requires_content: bool | None = False,
):
  num_workers = num_workers if num_workers is not None else max(1, os.cpu_count() - 1)
  for dataset_index, dataset in enumerate(datasets):
    logger.info(f"Encoding dataset [{dataset_index}]")
    all_cache_files = all_cache_files_for_dataset[dataset_index]
    all_cache_paths = all_cache_paths_for_dataset[dataset_index]
    batches = dataset.retrieve_latent_cache_batches(num_workers) if requires_content else dataset.retrieve_text_encoder_output_cache_batches(num_workers)

    for batch in tqdm(batches):
      if requires_content:
        batch = batch[1]
      all_cache_paths.update([os.path.normpath(item.text_encoder_output_cache_path) for item in batch])

      if skip_existing:
        batch = [item for item in batch if os.path.normpath(item.text_encoder_output_cache_path) not in all_cache_files]
        if len(batch) == 0:
          continue

      effective_batch_size = batch_size if batch_size is not None else len(batch)
      for start in range(0, len(batch), effective_batch_size):
        encode(batch[start : start + effective_batch_size])


def post_process_cache_files(datasets: list[BaseDataset], all_cache_files_for_dataset: list[set], all_cache_paths_for_dataset: list[set], keep_cache: bool):
  for dataset_index, dataset in enumerate(datasets):
    all_cache_files = all_cache_files_for_dataset[dataset_index]
    all_cache_paths = all_cache_paths_for_dataset[dataset_index]
    for cache_file in all_cache_files:
      if cache_file not in all_cache_paths:
        if keep_cache:
          logger.info(f"Keep cache file not in the dataset: {cache_file}")
        else:
          os.remove(cache_file)
          logger.info(f"Removed old cache file: {cache_file}")


def setup_text_encoder_cache_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset_config", type=str, required=True, help="path to dataset config .toml file")
  parser.add_argument("--device", type=str, default=None, help="device to use, default is cuda if available")
  parser.add_argument("--batch_size", type=int, default=None, help="batch size, override dataset config if dataset batch size > this")
  parser.add_argument("--num_workers", type=int, default=None, help="number of workers for dataset. default is cpu count-1")
  parser.add_argument("--skip_existing", action="store_true", help="skip existing cache files")
  parser.add_argument("--keep_cache", action="store_true", help="keep cache files not in dataset")
  return parser
