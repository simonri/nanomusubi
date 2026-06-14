import argparse
import logging

import torch

from musubi_tuner import cache_common
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.image_video_dataset import ItemInfo, save_latent_cache_wan
from musubi_tuner.utils.model_utils import str_to_dtype
from musubi_tuner.wan.modules.vae import WanVAE

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch(vae: WanVAE, batch: list[ItemInfo]):
  contents = torch.stack([torch.from_numpy(item.content) for item in batch])
  if len(contents.shape) == 4:
    contents = contents.unsqueeze(1)  # B, H, W, C -> B, F, H, W, C

  contents = contents.permute(0, 4, 1, 2, 3).contiguous()  # B, C, F, H, W
  contents = contents.to(vae.device, dtype=vae.dtype)
  contents = contents / 127.5 - 1.0  # normalize to [-1, 1]

  h, w = contents.shape[3], contents.shape[4]
  if h < 8 or w < 8:
    item = batch[0]  # other items should have the same size
    raise ValueError(f"Image or video size too small: {item.item_key} and {len(batch) - 1} more, size: {item.original_size}")

  with torch.amp.autocast(device_type=vae.device.type, dtype=vae.dtype), torch.no_grad():
    latent = vae.encode(contents)  # list of Tensor[C, F, H, W]
  latent = torch.stack(latent, dim=0)  # B, C, F, H, W
  latent = latent.to(vae.dtype)  # convert to bfloat16, we are not sure if this is correct

  # I2V: encode the first frame as conditioning
  images = contents[:, :, 0:1, :, :]  # B, C, F, H, W

  B, _, _, lat_h, lat_w = latent.shape
  F = contents.shape[2]

  # Create mask for the required number of frames
  msk = torch.ones(1, F, lat_h, lat_w, dtype=vae.dtype, device=vae.device)
  msk[:, 1:] = 0
  msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
  msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
  msk = msk.transpose(1, 2)  # 1, F, 4, H, W -> 1, 4, F, H, W
  msk = msk.repeat(B, 1, 1, 1, 1)  # B, 4, F, H, W

  # Zero padding for the required number of frames only
  padding_frames = F - 1  # The first frame is the input image
  images_resized = torch.concat([images, torch.zeros(B, 3, padding_frames, h, w, device=vae.device)], dim=2)
  with torch.amp.autocast(device_type=vae.device.type, dtype=vae.dtype), torch.no_grad():
    y = vae.encode(images_resized)
  y = torch.stack(y, dim=0)  # B, C, F, H, W

  y = y[:, :, :F]  # may be not needed
  y = y.to(vae.dtype)  # convert to bfloat16
  y = torch.concat([msk, y], dim=1)  # B, 4 + C, F, H, W

  # control videos/images
  if batch[0].control_content is not None:
    # Check if control_content is a list (for images) or ndarray (for videos)
    if isinstance(batch[0].control_content, list):
      # For images with control images: control_content is list[np.ndarray]
      # We take the first control image from each item
      control_contents = torch.stack([torch.from_numpy(item.control_content[0]) for item in batch])
    else:
      # For videos with control videos: control_content is np.ndarray
      control_contents = torch.stack([torch.from_numpy(item.control_content) for item in batch])

    if len(control_contents.shape) == 4:
      control_contents = control_contents.unsqueeze(1)
    control_contents = control_contents.permute(0, 4, 1, 2, 3).contiguous()  # B, C, F, H, W
    control_contents = control_contents.to(vae.device, dtype=vae.dtype)
    control_contents = control_contents / 127.5 - 1.0  # normalize to [-1, 1]
    with torch.amp.autocast(device_type=vae.device.type, dtype=vae.dtype), torch.no_grad():
      control_latent = vae.encode(control_contents)  # list of Tensor[C, F, H, W]
    control_latent = torch.stack(control_latent, dim=0)  # B, C, F, H, W
    control_latent = control_latent.to(vae.dtype)  # convert to bfloat16
  else:
    control_latent = None

  for i, item in enumerate(batch):
    l = latent[i]
    y_i = y[i]
    control_latent_i = control_latent[i] if control_latent is not None else None
    save_latent_cache_wan(item, l, None, y_i, control_latent_i)


def main():
  parser = cache_common.setup_latent_cache_parser()
  parser = wan_setup_parser(parser)

  args = parser.parse_args()

  if args.disable_cudnn_backend:
    logger.info("Disabling cuDNN PyTorch backend.")
    torch.backends.cudnn.enabled = False

  device = args.device if args.device is not None else "cuda"
  device = torch.device(device)

  # Load dataset config
  logger.info(f"Load dataset config from {args.dataset_config}")
  user_config = config_utils.load_user_config(args.dataset_config)
  blueprint = config_utils.generate_blueprint(user_config, args)
  train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

  datasets = train_dataset_group.datasets

  assert args.vae is not None, "vae checkpoint is required"

  vae_path = args.vae

  logger.info(f"Loading VAE model from {vae_path}")
  vae_dtype = torch.bfloat16 if args.vae_dtype is None else str_to_dtype(args.vae_dtype)
  cache_device = torch.device("cpu") if args.vae_cache_cpu else None
  vae = WanVAE(vae_path=vae_path, device=device, dtype=vae_dtype, cache_device=cache_device)

  def encode(one_batch: list[ItemInfo]):
    encode_and_save_batch(vae, one_batch)

  cache_common.encode_datasets(datasets, encode, args)


def wan_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
  parser.add_argument("--vae_cache_cpu", action="store_true", help="cache features in VAE on CPU")
  return parser


if __name__ == "__main__":
  main()
