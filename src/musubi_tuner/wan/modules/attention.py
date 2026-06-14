# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Flash Attention 3-only attention helper used by Wan 2.2 modules."""

import flash_attn_interface
import torch

__all__ = ["flash_attention"]


def flash_attention(
  qkv,
  q_lens=None,
  k_lens=None,
  softmax_scale=None,
  q_scale=None,
  causal=False,
  deterministic=False,
  dtype=torch.bfloat16,
):
  """Flash Attention 3 wrapper.

  q: [B, Lq, Nq, C1]
  k: [B, Lk, Nk, C1]
  v: [B, Lk, Nk, C2]
  Nq must be divisible by Nk.
  """
  q, k, v = qkv
  qkv.clear()

  half_dtypes = (torch.float16, torch.bfloat16)
  assert dtype in half_dtypes

  b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

  def half(x):
    return x if x.dtype in half_dtypes else x.to(dtype)

  # preprocess query
  if q_lens is None:
    q = half(q.flatten(0, 1))
    q_lens = torch.tensor([lq] * b, dtype=torch.int32).to(device=q.device, non_blocking=True)
  else:
    q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

  # preprocess key, value
  if k_lens is None:
    k = half(k.flatten(0, 1))
    v = half(v.flatten(0, 1))
    k_lens = torch.tensor([lk] * b, dtype=torch.int32).to(device=k.device, non_blocking=True)
  elif min(k_lens) == max(k_lens) and k.shape[1] == k_lens[0]:
    # all k_lens equal — fast flatten path
    k = half(k.flatten(0, 1))
    v = half(v.flatten(0, 1))
  else:
    k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
    v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

  q = q.to(v.dtype)
  k = k.to(v.dtype)

  if q_scale is not None:
    q = q * q_scale

  x = flash_attn_interface.flash_attn_varlen_func(
    q=q,
    k=k,
    v=v,
    cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
    cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
    seqused_q=None,
    seqused_k=None,
    max_seqlen_q=lq,
    max_seqlen_k=lk,
    softmax_scale=softmax_scale,
    causal=causal,
    deterministic=deterministic,
  ).unflatten(0, (b, lq))

  return x.type(out_dtype)
