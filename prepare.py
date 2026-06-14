"""Pre-cache latents and text encoder outputs for Wan 2.2 I2V LoRA training."""

import argparse
import logging

import torch

from musubi_tuner import cache_common
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.image_video_dataset import ItemInfo, save_latent_cache_wan, save_text_encoder_output_cache_wan
from musubi_tuner.utils.model_utils import str_to_dtype
from musubi_tuner.wan.modules.t5 import T5EncoderModel
from musubi_tuner.wan.modules.vae import WanVAE

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_latent_batch(vae: WanVAE, batch: list[ItemInfo]):
  contents = torch.stack([torch.from_numpy(item.content) for item in batch])
  if len(contents.shape) == 4:
    contents = contents.unsqueeze(1)  # B, H, W, C -> B, F, H, W, C

  contents = contents.permute(0, 4, 1, 2, 3).contiguous()  # B, C, F, H, W
  contents = contents.to(vae.device, dtype=vae.dtype)
  contents = contents / 127.5 - 1.0

  h, w = contents.shape[3], contents.shape[4]
  if h < 8 or w < 8:
    item = batch[0]
    raise ValueError(f"Image or video size too small: {item.item_key} and {len(batch) - 1} more, size: {item.original_size}")

  with torch.amp.autocast(device_type=vae.device.type, dtype=vae.dtype), torch.no_grad():
    latent = vae.encode(contents)
  latent = torch.stack(latent, dim=0)
  latent = latent.to(vae.dtype)

  images = contents[:, :, 0:1, :, :]
  B, _, _, lat_h, lat_w = latent.shape
  F = contents.shape[2]

  msk = torch.ones(1, F, lat_h, lat_w, dtype=vae.dtype, device=vae.device)
  msk[:, 1:] = 0
  msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
  msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
  msk = msk.transpose(1, 2)
  msk = msk.repeat(B, 1, 1, 1, 1)

  padding_frames = F - 1
  images_resized = torch.concat([images, torch.zeros(B, 3, padding_frames, h, w, device=vae.device)], dim=2)
  with torch.amp.autocast(device_type=vae.device.type, dtype=vae.dtype), torch.no_grad():
    y = vae.encode(images_resized)
  y = torch.stack(y, dim=0)
  y = y[:, :, :F]
  y = y.to(vae.dtype)
  y = torch.concat([msk, y], dim=1)

  if batch[0].control_content is not None:
    if isinstance(batch[0].control_content, list):
      control_contents = torch.stack([torch.from_numpy(item.control_content[0]) for item in batch])
    else:
      control_contents = torch.stack([torch.from_numpy(item.control_content) for item in batch])
    if len(control_contents.shape) == 4:
      control_contents = control_contents.unsqueeze(1)
    control_contents = control_contents.permute(0, 4, 1, 2, 3).contiguous()
    control_contents = control_contents.to(vae.device, dtype=vae.dtype)
    control_contents = control_contents / 127.5 - 1.0
    with torch.amp.autocast(device_type=vae.device.type, dtype=vae.dtype), torch.no_grad():
      control_latent = vae.encode(control_contents)
    control_latent = torch.stack(control_latent, dim=0).to(vae.dtype)
  else:
    control_latent = None

  for i, item in enumerate(batch):
    save_latent_cache_wan(item, latent[i], None, y[i], control_latent[i] if control_latent is not None else None)


def encode_text_batch(text_encoder: T5EncoderModel, batch: list[ItemInfo], device: torch.device):
  prompts = [item.caption for item in batch]
  with torch.no_grad():
    context = text_encoder(prompts, device)
  for item, ctx in zip(batch, context):
    save_text_encoder_output_cache_wan(item, ctx)


def main():
  parser = argparse.ArgumentParser(description="Pre-cache latents and text encoder outputs")
  parser.add_argument("--dataset_config", type=str, required=True, help="path to dataset config .toml file")
  parser.add_argument("--vae", type=str, required=True, help="path to VAE checkpoint")
  parser.add_argument("--t5", type=str, required=True, help="path to T5 text encoder checkpoint")
  parser.add_argument("--vae_dtype", type=str, default=None, help="data type for VAE (default: bfloat16)")
  parser.add_argument("--vae_cache_cpu", action="store_true", help="cache VAE features on CPU")
  parser.add_argument("--device", type=str, default=None, help="device to use (default: cuda)")
  parser.add_argument("--batch_size", type=int, default=None, help="batch size override")
  parser.add_argument("--num_workers", type=int, default=None, help="number of dataset workers")
  parser.add_argument("--skip_existing", action="store_true", help="skip existing cache files")
  parser.add_argument("--keep_cache", action="store_true", help="keep cache files not in dataset")
  parser.add_argument("--disable_cudnn_backend", action="store_true", help="disable cuDNN backend")
  args = parser.parse_args()

  if args.disable_cudnn_backend:
    logger.info("Disabling cuDNN PyTorch backend.")
    torch.backends.cudnn.enabled = False

  device = torch.device(args.device if args.device is not None else "cuda")

  logger.info(f"Loading dataset config from {args.dataset_config}")
  user_config = config_utils.load_user_config(args.dataset_config)
  blueprint = config_utils.generate_blueprint(user_config, args)
  train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
  datasets = train_dataset_group.datasets

  # --- Latent caching ---
  logger.info("=== Caching latents ===")
  vae_dtype = torch.bfloat16 if args.vae_dtype is None else str_to_dtype(args.vae_dtype)
  cache_device = torch.device("cpu") if args.vae_cache_cpu else None
  vae = WanVAE(vae_path=args.vae, device=device, dtype=vae_dtype, cache_device=cache_device)
  cache_common.encode_datasets(datasets, lambda batch: encode_latent_batch(vae, batch), args)
  del vae

  # --- Text encoder caching ---
  logger.info("=== Caching text encoder outputs ===")
  all_cache_files, all_cache_paths = cache_common.prepare_cache_files_and_paths(datasets)
  text_encoder = T5EncoderModel(text_len=512, dtype=torch.bfloat16, device=device, weight_path=args.t5)
  cache_common.process_text_encoder_batches(
    args.num_workers,
    args.skip_existing,
    args.batch_size,
    datasets,
    all_cache_files,
    all_cache_paths,
    lambda batch: encode_text_batch(text_encoder, batch, device),
  )
  del text_encoder
  cache_common.post_process_cache_files(datasets, all_cache_files, all_cache_paths, args.keep_cache)


if __name__ == "__main__":
  main()
