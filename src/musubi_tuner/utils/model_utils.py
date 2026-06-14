import hashlib
import logging
from io import BytesIO

import safetensors.torch
import torch

logger = logging.getLogger(__name__)


def addnet_hash_legacy(b):
  """Old model hash used by sd-webui-additional-networks for .safetensors format files"""
  m = hashlib.sha256()

  b.seek(0x100000)
  m.update(b.read(0x10000))
  return m.hexdigest()[0:8]


def addnet_hash_safetensors(b):
  """New model hash used by sd-webui-additional-networks for .safetensors format files"""
  hash_sha256 = hashlib.sha256()
  blksize = 1024 * 1024

  b.seek(0)
  header = b.read(8)
  n = int.from_bytes(header, "little")

  offset = n + 8
  b.seek(offset)
  for chunk in iter(lambda: b.read(blksize), b""):
    hash_sha256.update(chunk)

  return hash_sha256.hexdigest()


def precalculate_safetensors_hashes(tensors, metadata):
  """Precalculate the model hashes needed by sd-webui-additional-networks to
  save time on indexing the model later."""

  # Because writing user metadata to the file can change the result of
  # sd_models.model_hash(), only retain the training metadata for purposes of
  # calculating the hash, as they are meant to be immutable
  metadata = {k: v for k, v in metadata.items() if k.startswith("ss_")}

  bytes = safetensors.torch.save(tensors, metadata)
  b = BytesIO(bytes)

  model_hash = addnet_hash_safetensors(b)
  legacy_hash = addnet_hash_legacy(b)
  return model_hash, legacy_hash


def dtype_to_str(dtype: torch.dtype) -> str:
  # get name of the dtype
  dtype_name = str(dtype).split(".")[-1]
  return dtype_name


def str_to_dtype(s: str | None, default_dtype: torch.dtype | None = None) -> torch.dtype:
  """Convert a string to a torch.dtype (bf16/fp16/fp32 only)."""
  if s is None:
    return default_dtype
  if s in ["bf16", "bfloat16"]:
    return torch.bfloat16
  elif s in ["fp16", "float16"]:
    return torch.float16
  elif s in ["fp32", "float32", "float"]:
    return torch.float32
  else:
    raise ValueError(f"Unsupported dtype: {s}")



import gc


def clean_memory_on_device(device: str | torch.device | None):
  if device is None:
    return
  gc.collect()
  torch.cuda.empty_cache()


def synchronize_device(device: str | torch.device | None):
  if device is None:
    return
  torch.cuda.synchronize()
