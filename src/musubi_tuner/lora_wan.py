# LoRA module for Wan2.1

import ast
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import musubi_tuner.lora as lora

WAN_TARGET_REPLACE_MODULES = ["WanAttentionBlock"]


def create_arch_network(
  multiplier: float,
  network_dim: int | None,
  network_alpha: float | None,
  unet: nn.Module,
  neuron_dropout: float | None = None,
  **kwargs,
):
  # add default exclude patterns
  exclude_patterns = kwargs.get("exclude_patterns", None)
  if exclude_patterns is None:
    exclude_patterns = []
  else:
    exclude_patterns = ast.literal_eval(exclude_patterns)

  exclude_patterns.append(r".*(patch_embedding|text_embedding|time_embedding|time_projection|norm|head).*")
  kwargs["exclude_patterns"] = exclude_patterns

  return lora.create_network(
    WAN_TARGET_REPLACE_MODULES,
    "lora_unet",
    multiplier,
    network_dim,
    network_alpha,
    unet,
    neuron_dropout=neuron_dropout,
    **kwargs,
  )


def create_arch_network_from_weights(
  multiplier: float,
  weights_sd: dict[str, torch.Tensor],
  unet: nn.Module | None = None,
  for_inference: bool = False,
  **kwargs,
) -> lora.LoRANetwork:
  return lora.create_network_from_weights(WAN_TARGET_REPLACE_MODULES, multiplier, weights_sd, unet, for_inference, **kwargs)
