import argparse
import logging

import torch

from musubi_tuner import cache_common
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.image_video_dataset import ItemInfo, save_text_encoder_output_cache_wan
from musubi_tuner.wan.modules.t5 import T5EncoderModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch(text_encoder: T5EncoderModel, batch: list[ItemInfo], device: torch.device):
  prompts = [item.caption for item in batch]
  with torch.no_grad():
    context = text_encoder(prompts, device)
  for item, ctx in zip(batch, context):
    save_text_encoder_output_cache_wan(item, ctx)


def main():
  parser = cache_common.setup_text_encoder_cache_parser()
  parser = wan_setup_parser(parser)

  args = parser.parse_args()

  device = args.device if args.device is not None else "cuda"
  device = torch.device(device)

  # Load dataset config
  logger.info(f"Load dataset config from {args.dataset_config}")
  user_config = config_utils.load_user_config(args.dataset_config)
  blueprint = config_utils.generate_blueprint(user_config, args)
  train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

  datasets = train_dataset_group.datasets
  # prepare cache files and paths: all_cache_files_for_dataset = existing cache files, all_cache_paths_for_dataset = all cache paths in the dataset
  all_cache_files_for_dataset, all_cache_paths_for_dataset = cache_common.prepare_cache_files_and_paths(datasets)

  # Load T5
  logger.info(f"Loading T5: {args.t5}")
  text_encoder = T5EncoderModel(text_len=512, dtype=torch.bfloat16, device=device, weight_path=args.t5)

  # Encode with T5
  logger.info("Encoding with T5")

  def encode_for_text_encoder(batch: list[ItemInfo]):
    nonlocal text_encoder, device
    encode_and_save_batch(text_encoder, batch, device)

  cache_common.process_text_encoder_batches(
    args.num_workers,
    args.skip_existing,
    args.batch_size,
    datasets,
    all_cache_files_for_dataset,
    all_cache_paths_for_dataset,
    encode_for_text_encoder,
  )
  del text_encoder

  # remove cache files not in dataset
  cache_common.post_process_cache_files(datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset, args.keep_cache)


def wan_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
  parser.add_argument("--t5", type=str, default=None, required=True, help="text encoder (T5) checkpoint path")
  return parser


if __name__ == "__main__":
  main()
