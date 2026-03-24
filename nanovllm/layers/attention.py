import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context

# attention.py — The Core (Paged KV Cache + Flash Attention)

@triton.jit
def store_kvcache_kernel(
    key_ptr,          # pointer to K tensor in memory
    key_stride,       # stride between rows in K
    value_ptr,        # pointer to V tensor in memory
    value_stride,     # stride between rows in V
    k_cache_ptr,      # pointer to the KV cache for keys
    v_cache_ptr,      # pointer to the KV cache for values
    slot_mapping_ptr, # maps token index → cache slot
    D: tl.constexpr,  # num_heads × head_dim (compile-time constant)
):
    idx = tl.program_id(0) # which token am I? (one GPU thread block per token)
    slot = tl.load(slot_mapping_ptr + idx) # where does this token go in the cache?
    if slot == -1: return                  # -1 means "skip" (padding)
    key_offsets = idx * key_stride + tl.arange(0, D) # read K for this token
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)       # write to cache at the assigned slot
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)

# Visualize the KV cache as a giant hotel:
"""
Cache (hotel):
  Room 0: [K values for some token] [V values for some token]
  Room 1: [K values for some token] [V values for some token]
  Room 2: [empty]
  Room 3: [K values for some token] [V values for some token]
  ...

slot_mapping = [3, 0, -1, 1]
  Token 0 → Room 3
  Token 1 → Room 0
  Token 2 → skip (padding)
  Token 3 → Room 1

This kernel runs N thread blocks in parallel (one per token), and each writes D values (all heads concatenated).
"""

def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    # ... assert stride correctness (memory layout checks) ...
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([]) # Empty until model_runner assigns them

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context() # Global metadata: prefill vs decode, block tables, etc.
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            # First, store K,V into cache
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None: # If prefix caching is active
                k, v = k_cache, v_cache          # Read from cache (includes shared prefix)
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
