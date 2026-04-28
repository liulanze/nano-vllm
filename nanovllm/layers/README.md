# Layers — nano-vllm

## Token Flow Overview

A token flows through these layers from top to bottom:

```
Token IDs
  │
  ▼
┌─────────────────────┐
│  embed_head.py      │  "Look up what this word means" (token → vector)
└─────────┬───────────┘
          │
          ▼  (repeat for each decoder layer)
  ┌───────────────────────────────────────────┐
  │  layernorm.py    → Normalize the vector   │
  │  linear.py       → Project to Q, K, V     │
  │  rotary_embedding.py → Add position info  │
  │  attention.py    → Tokens look at each other │
  │  linear.py       → Project output back    │
  │  layernorm.py    → Normalize again        │
  │  linear.py       → Expand (gate + up)     │
  │  activation.py   → Non-linear gating      │
  │  linear.py       → Compress back down     │
  └───────────────────────────────────────────┘
          │
          ▼
┌─────────────────────┐
│  embed_head.py      │  "What word does this vector mean?" (vector → logits)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  sampler.py         │  "Pick the next token" (logits → token ID)
└─────────────────────┘
```

---

## How It All Connects — One Token's Journey

Trace a single forward pass through `Qwen3DecoderLayer` (see `qwen3.py:198-211`):

**Input:** `hidden_states [total_tokens, 1024]`, `residual`

### 1. Input LayerNorm

```
hidden_states, residual = input_layernorm(hidden_states, residual)
→ layernorm.py: Fuse residual add + RMSNorm
```

### 2. Self-Attention

```
hidden_states = self_attn(positions, hidden_states)
```

Inside `self_attn`:

| Step | Operation | Layer | Shape |
|------|-----------|-------|-------|
| a | `qkv = qkv_proj(hidden_states)` | `linear.py: QKVParallelLinear` | `[tokens, 1024] → [tokens, 2048]` |
| b | Split into q, k, v | — | — |
| c | `q_norm(q)`, `k_norm(k)` | `layernorm.py: RMSNorm` | per head |
| d | `q, k = rotary_emb(positions, q, k)` | `rotary_embedding.py` | Rotate by position |
| e | `o = attn(q, k, v)` | `attention.py` | Store KV cache + Flash Attention |
| f | `output = o_proj(o.flatten())` | `linear.py: RowParallelLinear` | `[tokens, 1024] → [tokens, 1024]` |

### 3. Post-Attention LayerNorm

```
hidden_states, residual = post_attention_layernorm(hidden_states, residual)
→ layernorm.py: Fuse residual add + RMSNorm
```

### 4. MLP

```
hidden_states = mlp(hidden_states)
```

Inside `mlp`:

| Step | Operation | Layer | Shape |
|------|-----------|-------|-------|
| a | `gate_up = gate_up_proj(x)` | `linear.py: MergedColumnParallel` | `[tokens, 1024] → [tokens, 5632]` |
| b | `x = SiluAndMul(gate_up)` | `activation.py` | Split, SiLU gate, multiply → `[tokens, 2816]` |
| c | `x = down_proj(x)` | `linear.py: RowParallelLinear` | `[tokens, 2816] → [tokens, 1024]` |

**Output:** `hidden_states`, `residual`

---

## File-by-File Cheat Sheet

| File | Purpose | Key Concept |
|------|---------|-------------|
| `activation.py` | Gated non-linearity | `SiLU(gate) * up` — the "dimmer switch" |
| `layernorm.py` | Keep numbers stable | `RMSNorm = x / sqrt(mean(x²)) * weight` |
| `rotary_embedding.py` | Position encoding | Rotate Q,K vectors — nearby tokens have similar rotations |
| `sampler.py` | Pick next token | Gumbel-max trick — divide probs by exponential noise, take argmax |
| `embed_head.py` | Token ID ↔ vector | Entry: lookup table. Exit: similarity search against all vocab |
| `linear.py` | Matrix multiplication | Column/Row parallel splits matrices across GPUs |
| `attention.py` | Token interaction | Triton kernel writes paged KV cache; Flash Attention reads it |

## Big Picture — Full Architecture

```
Token IDs  [batch]                          ← integers like [1542, 389, 7201]
    │
    ▼
┌─────────────────────────────────────────────┐
│  VocabParallelEmbedding                     │  embed_head.py
│  Lookup table: token ID → 1024-dim vector   │
└─────────────────────────────────────────────┘
    │  hidden_states: [tokens, 1024]
    ▼
┌─────────────────────────────────────────────┐
│  Decoder Layer × N  (N=28 for Qwen3-0.6B)  │
│  ┌───────────────────────────────────┐      │
│  │ RMSNorm                           │      │  layernorm.py
│  │ Attention (QKV → RoPE → FlashAttn)│      │  linear.py, rotary_embedding.py, attention.py
│  │ RMSNorm                           │      │  layernorm.py
│  │ MLP (gate+up → SiLU → down)       │      │  linear.py, activation.py
│  └───────────────────────────────────┘      │
└─────────────────────────────────────────────┘
    │  hidden_states: [tokens, 1024]
    ▼
┌─────────────────────────────────────────────┐
│  Final RMSNorm                              │  layernorm.py
│  ParallelLMHead                             │  embed_head.py
│  Score every vocab word → [batch, 151936]   │
└─────────────────────────────────────────────┘
    │  logits: [batch, 151936]
    ▼
┌─────────────────────────────────────────────┐
│  Sampler (Gumbel-max trick)                 │  sampler.py
│  Pick one token ID from the distribution    │
└─────────────────────────────────────────────┘
    │
    ▼
  Output token ID  [batch]                     ← integer like 4821
```

---

## Tensor Parallelism — Column + Row Pattern

Each transformer layer has **two** Column→Row pairs (attention + MLP).
Only **2 all_reduce calls per layer** — the minimum possible.

```
ColumnParallel (no communication)
  GPU 0: computes output[:512]      ← each GPU works on its slice independently
  GPU 1: computes output[512:]
         │
         ▼ (no data exchange needed)
RowParallel (one all_reduce)
  GPU 0: partial_0 = x_left @ W_left
  GPU 1: partial_1 = x_right @ W_right
         │
         ▼ all_reduce (sum across GPUs)
  Both GPUs: y = partial_0 + partial_1   ← identical result everywhere
```

**Why all_reduce, not all_gather?**

Matrix multiplication with a column-split input decomposes into a sum:

```
y = x_full @ W_full
  = [x_left | x_right] @ [W_left ]  =  x_left @ W_left  +  x_right @ W_right
                          [W_right]      ───── GPU 0 ────    ───── GPU 1 ─────
```

Each GPU computes a partial result with half the work, then `all_reduce(sum)` combines them.

---

## Embedding Tensor Parallelism

Vocabulary is split by rows across GPUs. Each GPU embeds only its tokens, zeros out the rest,
then `all_reduce(sum)` combines (since non-owners contribute zeros, sum = gather).

```
GPU 0 (vocab 0-75967):     [REAL,  0,     REAL,  0,     REAL ]
GPU 1 (vocab 75968-151935): [0,     REAL,  0,     REAL,  0    ]
─────────────────────────────────────────────────────────────────
all_reduce(sum):            [REAL,  REAL,  REAL,  REAL,  REAL ]
```

---

## Complete Forward Pass Trace

```
input_ids = [1542, 389, 7201]    positions = [0, 1, 2]

① Embedding:     [3] → [3, 1024]        (lookup table)

② × 28 Decoder Layers:
   ├── RMSNorm:      normalize + residual add
   ├── QKV Proj:     [3, 1024] → [3, 2048]    (fused matmul)
   ├── Split:        Q[3,16,64]  K[3,8,64]  V[3,8,64]
   ├── Q/K Norm:     RMSNorm on head dim
   ├── RoPE:         rotate Q,K by position-dependent angles
   ├── Store KV:     write K,V to paged cache (Triton kernel)
   ├── Flash Attn:   softmax(Q@K^T/√d) @ V → [3,16,64]
   ├── O Proj:       [3, 1024] → [3, 1024]    (Row parallel + all_reduce)
   ├── RMSNorm:      normalize + residual add
   ├── Gate+Up:      [3, 1024] → [3, 5632]    (fused matmul)
   ├── SiLU Gate:    [3, 5632] → [3, 2816]    (activation)
   └── Down Proj:    [3, 2816] → [3, 1024]    (Row parallel + all_reduce)

③ Final Norm:    [3, 1024] → [3, 1024]

④ LM Head:       [1, 1024] → [1, 151936]   (only last token during prefill)

⑤ Sampler:       [1, 151936] → [1]          (Gumbel-max → token ID 4821)
```

---

## Key Concepts Quick Reference

| Concept | What It Means | Where |
|---------|---------------|-------|
| **GQA** | Q has more heads than K,V (16 vs 8). Every 2 Q heads share 1 KV head. Saves KV cache memory. | `attention.py` |
| **RoPE** | Rotates Q,K by position-dependent angles. Attention score naturally encodes relative distance. | `rotary_embedding.py` |
| **Paged KV Cache** | KV cache is a "hotel" — BlockManager assigns rooms (slots). Triton kernel writes K,V to assigned slots. | `attention.py` |
| **Prefill vs Decode** | Prefill: process all prompt tokens at once. Decode: one new token at a time, reads KV cache. | `attention.py` |
| **Residual Stream** | The "highway" — raw signal skips layers. Each layer adds its contribution. Prevents vanishing gradients. | `layernorm.py` |
| **Gumbel-max** | Fast GPU sampling: `probs / exponential_noise → argmax`. Equivalent to multinomial but ~10x faster. | `sampler.py` |
| **o_proj** | Output projection — learned matrix that combines all attention heads into one unified representation. | `qwen3.py` |
