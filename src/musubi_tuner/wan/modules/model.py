# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging
import math
import os

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from musubi_tuner.utils.model_utils import synchronize_device
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen, TensorWeightAdapter, WeightTransformHooks, get_split_weight_filenames

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from musubi_tuner.wan.modules.attention import flash_attention


def load_safetensors_with_lora(
  model_files: str | list[str],
  lora_weights_list: list[dict[str, torch.Tensor]] | None,
  lora_multipliers: list[float] | None,
  calc_device: torch.device,
  move_to_device: bool = False,
  dit_weight_dtype: torch.dtype | None = None,
  disable_numpy_memmap: bool = False,
  weight_transform_hooks: WeightTransformHooks | None = None,
) -> dict[str, torch.Tensor]:
  if isinstance(model_files, str):
    model_files = [model_files]

  extended_model_files = []
  for model_file in model_files:
    split_filenames = get_split_weight_filenames(model_file)
    if split_filenames is not None:
      extended_model_files.extend(split_filenames)
    else:
      extended_model_files.append(model_file)
  model_files = extended_model_files
  logger.info(f"Loading model files: {model_files}")

  weight_hook = None
  if lora_weights_list is None or len(lora_weights_list) == 0:
    lora_weights_list = []
    lora_multipliers = []
    list_of_lora_weight_keys = []
  else:
    list_of_lora_weight_keys = []
    for lora_sd in lora_weights_list:
      list_of_lora_weight_keys.append(set(lora_sd.keys()))

    if lora_multipliers is None:
      lora_multipliers = [1.0] * len(lora_weights_list)
    while len(lora_multipliers) < len(lora_weights_list):
      lora_multipliers.append(1.0)
    if len(lora_multipliers) > len(lora_weights_list):
      lora_multipliers = lora_multipliers[: len(lora_weights_list)]

    logger.info(f"Merging LoRA weights into state dict. multipliers: {lora_multipliers}")

    def weight_hook_func(model_weight_key, model_weight: torch.Tensor, keep_on_calc_device=False):
      nonlocal list_of_lora_weight_keys, lora_weights_list, lora_multipliers, calc_device

      if not model_weight_key.endswith(".weight"):
        return model_weight

      original_device = model_weight.device
      if original_device != calc_device:
        model_weight = model_weight.to(calc_device)

      for lora_weight_keys, lora_sd, multiplier in zip(list_of_lora_weight_keys, lora_weights_list, lora_multipliers):
        lora_name = "lora_unet_" + model_weight_key.rsplit(".", 1)[0].replace(".", "_")
        down_key = lora_name + ".lora_down.weight"
        up_key = lora_name + ".lora_up.weight"
        alpha_key = lora_name + ".alpha"
        if down_key not in lora_weight_keys or up_key not in lora_weight_keys:
          continue

        down_weight = lora_sd[down_key].to(calc_device)
        up_weight = lora_sd[up_key].to(calc_device)
        dim = down_weight.size()[0]
        alpha = lora_sd.get(alpha_key, dim)
        scale = alpha / dim

        if len(model_weight.size()) == 2:
          if len(up_weight.size()) == 4:
            up_weight = up_weight.squeeze(3).squeeze(2)
            down_weight = down_weight.squeeze(3).squeeze(2)
          model_weight = model_weight + multiplier * (up_weight @ down_weight) * scale
        elif down_weight.size()[2:4] == (1, 1):
          model_weight = model_weight + multiplier * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3) * scale
        else:
          conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
          model_weight = model_weight + multiplier * conved * scale

        lora_weight_keys.remove(down_key)
        lora_weight_keys.remove(up_key)
        if alpha_key in lora_weight_keys:
          lora_weight_keys.remove(alpha_key)

      if not keep_on_calc_device and original_device != calc_device:
        model_weight = model_weight.to(original_device)
      return model_weight

    weight_hook = weight_hook_func

  state_dict = {}
  logger.info(f"Loading state dict. Dtype of weight: {dit_weight_dtype}, hook enabled: {weight_hook is not None}")
  for model_file in model_files:
    with MemoryEfficientSafeOpen(model_file, disable_numpy_memmap=disable_numpy_memmap) as original_f:
      f = TensorWeightAdapter(weight_transform_hooks, original_f) if weight_transform_hooks is not None else original_f
      for key in tqdm(f.keys(), desc=f"Loading {os.path.basename(model_file)}", leave=False):
        if weight_hook is None and move_to_device:
          value = f.get_tensor(key, device=calc_device, dtype=dit_weight_dtype)
        else:
          value = f.get_tensor(key)
          if weight_hook is not None:
            value = weight_hook(key, value, keep_on_calc_device=move_to_device)
          if move_to_device:
            value = value.to(calc_device, dtype=dit_weight_dtype, non_blocking=True)
          elif dit_weight_dtype is not None:
            value = value.to(dit_weight_dtype)
        state_dict[key] = value
  if move_to_device:
    synchronize_device(calc_device)

  for lora_weight_keys in list_of_lora_weight_keys:
    if len(lora_weight_keys) > 0:
      logger.warning(f"Warning: not all LoRA keys are used: {', '.join(lora_weight_keys)}")

  return state_dict

__all__ = ["WanModel", "detect_wan_model_config"]


def sinusoidal_embedding_1d(dim, position):
  # preprocess
  assert dim % 2 == 0
  half = dim // 2
  position = position.type(torch.float64)

  # calculation
  sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
  x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
  return x


# @amp.autocast(enabled=False)
# no autocast is needed for rope_apply, because it is already in float64
def rope_params(max_seq_len, dim, theta=10000):
  assert dim % 2 == 0
  freqs = torch.outer(torch.arange(max_seq_len), 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)))
  freqs = torch.polar(torch.ones_like(freqs), freqs)
  return freqs


# @amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
  device_type = x.device.type
  with torch.amp.autocast(device_type=device_type, enabled=False):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
      seq_len = f * h * w

      # precompute multipliers
      x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
      freqs_i = torch.cat(
        [
          freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
          freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
          freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
      ).reshape(seq_len, 1, -1)

      # apply rotary embedding
      x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
      x_i = torch.cat([x_i, x[i, seq_len:]])

      # append to collection
      output.append(x_i)
    return torch.stack(output).float()


def calculate_freqs_i(fhw, c, freqs, f_indices=None):
  """f_indices is used to select specific frames for rotary embedding. e.g. [0,8] (with start image) or [0,8,20] (with start and end images)"""
  f, h, w = fhw[:3]
  freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

  if f_indices is None:
    freqs_f = freqs[0][:f]
  else:
    logger.info(f"Using f_indices: {f_indices} for rotary embedding. fhw: {fhw}")
    freqs_f = freqs[0][f_indices]

  freqs_i = torch.cat(
    [
      freqs_f.view(f, 1, 1, -1).expand(f, h, w, -1),
      freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
      freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ],
    dim=-1,
  ).reshape(f * h * w, 1, -1)
  return freqs_i


# inplace version of rope_apply
def rope_apply_inplace_cached(x, grid_sizes, freqs_list):
  # with torch.amp.autocast(device_type=device_type, enabled=False):
  rope_dtype = torch.float64  # float32 does not reduce memory usage significantly

  n, c = x.size(2), x.size(3) // 2

  # loop over samples
  for i, (f, h, w) in enumerate(grid_sizes.tolist()):
    seq_len = f * h * w

    # precompute multipliers
    x_i = torch.view_as_complex(x[i, :seq_len].to(rope_dtype).reshape(seq_len, n, -1, 2))
    freqs_i = freqs_list[i]

    # apply rotary embedding
    x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
    # x_i = torch.cat([x_i, x[i, seq_len:]])

    # inplace update
    x[i, :seq_len] = x_i.to(x.dtype)

  return x


class WanRMSNorm(nn.Module):
  def __init__(self, dim, eps=1e-5):
    super().__init__()
    self.dim = dim
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))

  def forward(self, x):
    r"""
    Args:
        x(Tensor): Shape [B, L, C]
    """
    return self._norm(x.float()).type_as(x) * self.weight

  def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
  def __init__(self, dim, eps=1e-6, elementwise_affine=False):
    super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

  def forward(self, x):
    r"""
    Args:
        x(Tensor): Shape [B, L, C]
    """
    return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):
  def __init__(self, dim, num_heads, eps=1e-6):
    assert dim % num_heads == 0
    super().__init__()
    self.dim = dim
    self.num_heads = num_heads
    self.head_dim = dim // num_heads
    self.eps = eps

    # layers
    self.q = nn.Linear(dim, dim)
    self.k = nn.Linear(dim, dim)
    self.v = nn.Linear(dim, dim)
    self.o = nn.Linear(dim, dim)
    self.norm_q = WanRMSNorm(dim, eps=eps)
    self.norm_k = WanRMSNorm(dim, eps=eps)

  def forward(self, x, seq_lens, grid_sizes, freqs):
    r"""
    Args:
        x(Tensor): Shape [B, L, num_heads, C / num_heads]
        seq_lens(Tensor): Shape [B]
        grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
        freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
    """
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

    # # query, key, value function
    # def qkv_fn(x):
    #     q = self.norm_q(self.q(x)).view(b, s, n, d)
    #     k = self.norm_k(self.k(x)).view(b, s, n, d)
    #     v = self.v(x).view(b, s, n, d)
    #     return q, k, v
    # q, k, v = qkv_fn(x)
    # del x
    # query, key, value function

    q = self.q(x)
    k = self.k(x)
    v = self.v(x)
    del x
    q = self.norm_q(q)
    k = self.norm_k(k)
    q = q.view(b, s, n, d)
    k = k.view(b, s, n, d)
    v = v.view(b, s, n, d)

    rope_apply_inplace_cached(q, grid_sizes, freqs)
    rope_apply_inplace_cached(k, grid_sizes, freqs)
    qkv = [q, k, v]
    del q, k, v
    x = flash_attention(qkv, k_lens=seq_lens)

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x


class WanCrossAttention(WanSelfAttention):
  def forward(self, x, context, context_lens):
    r"""
    Args:
        x(Tensor): Shape [B, L1, C]
        context(Tensor): Shape [B, L2, C]
        context_lens(Tensor): Shape [B]
    """
    b, n, d = x.size(0), self.num_heads, self.head_dim

    # compute query, key, value
    # q = self.norm_q(self.q(x)).view(b, -1, n, d)
    # k = self.norm_k(self.k(context)).view(b, -1, n, d)
    # v = self.v(context).view(b, -1, n, d)
    q = self.q(x)
    del x
    k = self.k(context)
    v = self.v(context)
    del context
    q = self.norm_q(q)
    k = self.norm_k(k)
    q = q.view(b, -1, n, d)
    k = k.view(b, -1, n, d)
    v = v.view(b, -1, n, d)

    # compute attention
    qkv = [q, k, v]
    del q, k, v
    x = flash_attention(qkv, k_lens=context_lens)

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x


class WanAttentionBlock(nn.Module):
  def __init__(
    self,
    dim,
    ffn_dim,
    num_heads,
    eps=1e-6,
  ):
    super().__init__()
    self.dim = dim
    self.ffn_dim = ffn_dim
    self.num_heads = num_heads
    self.eps = eps

    self.norm1 = WanLayerNorm(dim, eps)
    self.self_attn = WanSelfAttention(dim, num_heads, eps)
    self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True)
    self.cross_attn = WanCrossAttention(dim, num_heads, eps)
    self.norm2 = WanLayerNorm(dim, eps)
    self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))

    # modulation
    self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    self.gradient_checkpointing = False

  def enable_gradient_checkpointing(self):
    self.gradient_checkpointing = True

  def disable_gradient_checkpointing(self):
    self.gradient_checkpointing = False

  def _forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens):
    r"""
    Args:
        x(Tensor): Shape [B, L, C]
        e(Tensor): Shape [B, L, 6, C]
        seq_lens(Tensor): Shape [B], length of each sequence in batch
        grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
        freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
    """
    org_dtype = x.dtype
    assert e.dtype == torch.float32
    e = self.modulation.to(torch.float32) + e
    e = e.chunk(6, dim=2)
    assert e[0].dtype == torch.float32

    # self-attention
    y = self.self_attn(torch.addcmul(e[0].squeeze(2), self.norm1(x).float(), (1 + e[1].squeeze(2))).to(org_dtype), seq_lens, grid_sizes, freqs)
    x = torch.addcmul(x, y.to(torch.float32), e[2].squeeze(2)).to(org_dtype)
    del y

    # cross-attention & ffn
    x = x + self.cross_attn(self.norm3(x), context, context_lens)
    del context
    y = self.ffn(torch.addcmul(e[3].squeeze(2), self.norm2(x).float(), (1 + e[4].squeeze(2))).to(org_dtype))
    x = torch.addcmul(x, y.to(torch.float32), e[5].squeeze(2)).to(org_dtype)
    del y

    return x

  def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens):
    if self.training and self.gradient_checkpointing:
      return checkpoint(self._forward, x, e, seq_lens, grid_sizes, freqs, context, context_lens, use_reentrant=False)
    return self._forward(x, e, seq_lens, grid_sizes, freqs, context, context_lens)


class Head(nn.Module):
  def __init__(self, dim, out_dim, patch_size, eps=1e-6):
    super().__init__()
    self.dim = dim
    self.out_dim = out_dim
    self.patch_size = patch_size
    self.eps = eps

    # layers
    out_dim = math.prod(patch_size) * out_dim
    self.norm = WanLayerNorm(dim, eps)
    self.head = nn.Linear(dim, out_dim)

    # modulation
    self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

  def forward(self, x, e):
    r"""
    Args:
        x(Tensor): Shape [B, L, C]
        e(Tensor): Shape [B, L, 6, C]
    """
    assert e.dtype == torch.float32
    e = (self.modulation.unsqueeze(0).to(torch.float32) + e.unsqueeze(2)).chunk(2, dim=2)
    x = self.head(torch.addcmul(e[0].squeeze(2), self.norm(x), (1 + e[1].squeeze(2))))
    return x


class WanModel(nn.Module):  # ModelMixin, ConfigMixin):
  r"""
  Wan diffusion backbone supporting both text-to-video and image-to-video.
  """

  ignore_for_config = ["patch_size", "text_dim"]
  _no_split_modules = ["WanAttentionBlock"]

  # @register_to_config
  def __init__(
    self,
    patch_size=(1, 2, 2),
    text_len=512,
    in_dim=16,
    dim=2048,
    ffn_dim=8192,
    freq_dim=256,
    text_dim=4096,
    out_dim=16,
    num_heads=16,
    num_layers=32,
    eps=1e-6,
  ):
    r"""
    Initialize the diffusion model backbone.

    Args:
        patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
        text_len (`int`, *optional*, defaults to 512):
            Fixed length for text embeddings
        in_dim (`int`, *optional*, defaults to 16):
            Input video channels (C_in)
        dim (`int`, *optional*, defaults to 2048):
            Hidden dimension of the transformer
        ffn_dim (`int`, *optional*, defaults to 8192):
            Intermediate dimension in feed-forward network
        freq_dim (`int`, *optional*, defaults to 256):
            Dimension for sinusoidal time embeddings
        text_dim (`int`, *optional*, defaults to 4096):
            Input dimension for text embeddings
        out_dim (`int`, *optional*, defaults to 16):
            Output video channels (C_out)
        num_heads (`int`, *optional*, defaults to 16):
            Number of attention heads
        num_layers (`int`, *optional*, defaults to 32):
            Number of transformer blocks
        eps (`float`, *optional*, defaults to 1e-6):
            Epsilon value for normalization layers
    """

    super().__init__()

    self.patch_size = patch_size
    self.text_len = text_len
    self.in_dim = in_dim
    self.dim = dim
    self.ffn_dim = ffn_dim
    self.freq_dim = freq_dim
    self.text_dim = text_dim
    self.out_dim = out_dim
    self.num_heads = num_heads
    self.num_layers = num_layers
    self.eps = eps

    # embeddings
    self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
    self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))

    self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
    self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

    # blocks
    self.blocks = nn.ModuleList(
      [
        WanAttentionBlock(
          dim,
          ffn_dim,
          num_heads,
          eps,
        )
        for _ in range(num_layers)
      ]
    )

    # head
    self.head = Head(dim, out_dim, patch_size, eps)

    # buffers (don't use register_buffer otherwise dtype will be changed in to())
    assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
    d = dim // num_heads
    self.freqs = torch.cat([rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))], dim=1)
    self.freqs_fhw = {}

    # initialize weights
    self.init_weights()

    self.gradient_checkpointing = False

  @property
  def dtype(self):
    return self.patch_embedding.weight.dtype

  @property
  def device(self):
    return self.patch_embedding.weight.device

  def enable_gradient_checkpointing(self):
    self.gradient_checkpointing = True
    for block in self.blocks:
      block.enable_gradient_checkpointing()

  def disable_gradient_checkpointing(self):
    self.gradient_checkpointing = False
    for block in self.blocks:
      block.disable_gradient_checkpointing()

  def forward(self, x, t, context, seq_len, y=None, skip_block_indices=None, f_indices=None):
    r"""
    Forward pass through the diffusion model

    Args:
        x (List[Tensor]):
            List of input video tensors, each with shape [C_in, F, H, W]
        t (Tensor):
            Diffusion timesteps tensor of shape [B]
        context (List[Tensor]):
            List of text embeddings each with shape [L, C]
        seq_len (`int`):
            Maximum sequence length for positional encoding
        y (List[Tensor], *optional*):
            Conditional video inputs for image-to-video mode, same shape as x
        skip_block_indices (List[int], *optional*):
            Indices of blocks to skip during forward pass
        f_indices (List[List[int]], *optional*):
            Indices of frames used for rotary embeddings, list of lists for each video in the batch

    Returns:
        List[Tensor]:
            List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
    """
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
      self.freqs = self.freqs.to(device)

    if y is not None:
      x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
      y = None

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]  # x[0].shape = [1, 5120, F, H, W]
    grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])  # list of [F, H, W]

    freqs_list = []
    for i, fhw in enumerate(grid_sizes):
      fhw = tuple(fhw.tolist())
      if f_indices is not None:
        fhw = tuple(list(fhw) + f_indices[i])  # add f_indices to fhw for cache key
      if fhw not in self.freqs_fhw:
        c = self.dim // self.num_heads // 2
        self.freqs_fhw[fhw] = calculate_freqs_i(fhw, c, self.freqs, None if f_indices is None else f_indices[i])
      freqs_list.append(self.freqs_fhw[fhw])

    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len, f"Sequence length exceeds maximum allowed length {seq_len}. Got {seq_lens.max()}"
    x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])

    # time embeddings
    with torch.amp.autocast(device_type=device.type, dtype=torch.float32):
      if t.dim() == 1:
        t = t.unsqueeze(1).expand(-1, seq_len)
      bt = t.size(0)
      t = t.flatten()
      e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).unflatten(0, (bt, seq_len)).float())
      e0 = self.time_projection(e).unflatten(2, (6, self.dim))

    assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    if type(context) is list:
      context = torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
    context = self.text_embedding(context)

    # arguments
    kwargs = dict(e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=freqs_list, context=context, context_lens=context_lens)

    # print(f"x: {x.shape}, e: {e0.shape}, context: {context.shape}, seq_lens: {seq_lens}")
    for block_idx, block in enumerate(self.blocks):
      if skip_block_indices is None or block_idx not in skip_block_indices:
        x = block(x, **kwargs)

    # head
    x = self.head(x, e)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return [u.float() for u in x]

  def unpatchify(self, x, grid_sizes):
    r"""
    Reconstruct video tensors from patch embeddings.

    Args:
        x (List[Tensor]):
            List of patchified features, each with shape [L, C_out * prod(patch_size)]
        grid_sizes (Tensor):
            Original spatial-temporal grid dimensions before patching,
                shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

    Returns:
        List[Tensor]:
            Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
    """

    c = self.out_dim
    out = []
    for u, v in zip(x, grid_sizes.tolist()):
      u = u[: math.prod(v)].view(*v, *self.patch_size, c)
      u = torch.einsum("fhwpqrc->cfphqwr", u)
      u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
      out.append(u)
    return out

  def init_weights(self):
    r"""
    Initialize model parameters using Xavier initialization.
    """

    # basic init
    for m in self.modules():
      if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
          nn.init.zeros_(m.bias)

    # init embeddings
    nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
    for m in self.text_embedding.modules():
      if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.02)
    for m in self.time_embedding.modules():
      if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.02)

    # init output layer
    nn.init.zeros_(self.head.head.weight)


def detect_wan_sd_dtype(path: str) -> torch.dtype:
  # get dtype from model weights
  with MemoryEfficientSafeOpen(path) as f:
    keys = set(f.keys())
    key1 = "model.diffusion_model.blocks.0.cross_attn.k.weight"  # 1.3B
    key2 = "blocks.0.cross_attn.k.weight"  # 14B
    if key1 in keys:
      dit_dtype = f.get_tensor(key1).dtype
    elif key2 in keys:
      dit_dtype = f.get_tensor(key2).dtype
    else:
      raise ValueError(f"Could not find the dtype in the model weights: {path}")
  logger.info(f"Detected DiT dtype: {dit_dtype}")
  return dit_dtype


def detect_wan_model_config(path: str):
  """Infer WanModel constructor kwargs from safetensors weight shapes."""
  import types
  with MemoryEfficientSafeOpen(path) as f:
    keys = set(f.keys())
    prefix = "model.diffusion_model." if "model.diffusion_model.patch_embedding.weight" in keys else ""

    def shape(key):
      return f.get_tensor(prefix + key).shape

    pe = shape("patch_embedding.weight")   # [dim, in_dim, t, h, w]
    dim, in_dim = int(pe[0]), int(pe[1])

    ffn0 = shape("blocks.0.ffn.0.weight")  # [ffn_dim, dim]
    ffn_dim = int(ffn0[0])

    te = shape("time_embedding.0.weight")   # [dim, freq_dim]
    freq_dim = int(te[1])

    block_indices = {
      int(k[len(prefix):].split(".")[1])
      for k in keys
      if k[len(prefix):].startswith("blocks.") and k[len(prefix):].split(".")[1].isdigit()
    }
    num_layers = max(block_indices) + 1

    head_w = shape("head.head.weight")      # [out_dim * prod(patch_size), dim]
    patch_size = (int(pe[2]), int(pe[3]), int(pe[4]))
    out_dim = int(head_w[0]) // math.prod(patch_size)

    num_heads = dim // 128  # all Wan models use head_dim=128

  cfg = types.SimpleNamespace(
    dim=dim, in_dim=in_dim, out_dim=out_dim, ffn_dim=ffn_dim,
    freq_dim=freq_dim, num_heads=num_heads, num_layers=num_layers,
    eps=1e-6, text_len=512,
  )
  logger.info(
    f"Detected WanModel config: dim={dim}, in_dim={in_dim}, ffn_dim={ffn_dim}, "
    f"num_heads={num_heads}, num_layers={num_layers}, out_dim={out_dim}"
  )
  return cfg


def load_wan_model(
  config: any,
  device: str | torch.device,
  dit_path: str,
  loading_device: str | torch.device,
  dit_weight_dtype: torch.dtype,
  lora_weights_list: dict[str, torch.Tensor] | None = None,
  lora_multipliers: list[float] | None = None,
  disable_numpy_memmap: bool = False,
) -> WanModel:
  """
  Load a WAN model from the specified checkpoint.

  Args:
      config (any): Configuration object containing model parameters.
      device (Union[str, torch.device]): Device to load the model on.
      dit_path (str): Path to the DiT model checkpoint.
      loading_device (Union[str, torch.device]): Device to load the model weights on.
      dit_weight_dtype (torch.dtype): Data type to cast the DiT weights to.
      lora_weights_list (Optional[Dict[str, torch.Tensor]]): LoRA weights to apply, if any.
      lora_multipliers (Optional[List[float]]): LoRA multipliers for the weights, if any.
      disable_numpy_memmap (bool): Whether to disable numpy memmap when loading weights.
  """
  device = torch.device(device)
  loading_device = torch.device(loading_device)

  with init_empty_weights():
    logger.info(f"Creating WanModel. device: {device}, loading_device: {loading_device}")
    model = WanModel(
      dim=config.dim,
      eps=config.eps,
      ffn_dim=config.ffn_dim,
      freq_dim=config.freq_dim,
      in_dim=config.in_dim,
      num_heads=config.num_heads,
      num_layers=config.num_layers,
      out_dim=config.out_dim,
      text_len=config.text_len,
    )
    model.to(dit_weight_dtype)

  logger.info(f"Loading DiT model from {dit_path}, device={loading_device}")

  sd = load_safetensors_with_lora(
    model_files=dit_path,
    lora_weights_list=lora_weights_list,
    lora_multipliers=lora_multipliers,
    calc_device=device,
    move_to_device=(loading_device == device),
    disable_numpy_memmap=disable_numpy_memmap,
  )

  # remove "model.diffusion_model." prefix: 1.3B model has this prefix
  for key in list(sd.keys()):
    if key.startswith("model.diffusion_model."):
      sd[key[22:]] = sd.pop(key)

  info = model.load_state_dict(sd, strict=True, assign=True)
  logger.info(f"Casting model weights to {dit_weight_dtype}")
  model = model.to(dit_weight_dtype)
  logger.info(f"Loaded DiT model from {dit_path}, info={info}")

  return model
