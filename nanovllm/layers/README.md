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

How It All Connects — One Token's Journey
Let's trace a single forward pass through Qwen3DecoderLayer (from your annotated qwen3.py:198-211):

Input: hidden_states [total_tokens, 1024], residual

1. hidden_states, residual = input_layernorm(hidden_states, residual)
   → layernorm.py: Fuse residual add + RMSNorm

2. hidden_states = self_attn(positions, hidden_states)
   Inside self_attn:
   a. qkv = qkv_proj(hidden_states)              → linear.py: QKVParallelLinear [tokens,1024] → [tokens,2048]
   b. Split into q, k, v
   c. q_norm(q), k_norm(k)                        → layernorm.py: RMSNorm each head
   d. q, k = rotary_emb(positions, q, k)          → rotary_embedding.py: Rotate by position
   e. o = attn(q, k, v)                           → attention.py: Store KV cache + Flash Attention
   f. output = o_proj(o.flatten())                 → linear.py: RowParallelLinear [tokens,1024] → [tokens,1024]

3. hidden_states, residual = post_attention_layernorm(hidden_states, residual)
   → layernorm.py: Fuse residual add + RMSNorm

4. hidden_states = mlp(hidden_states)
   Inside mlp:
   a. gate_up = gate_up_proj(x)                   → linear.py: MergedColumnParallel [tokens,1024] → [tokens,5632]
   b. x = SiluAndMul(gate_up)                     → activation.py: Split, SiLU gate, multiply [tokens,2816]
   c. x = down_proj(x)                            → linear.py: RowParallelLinear [tokens,2816] → [tokens,1024]

Output: hidden_states, residual

Summary — File-by-File Cheat Sheet
File	Purpose	Key Concept
activation.py	Gated non-linearity	SiLU(gate) * up — the "dimmer switch"
layernorm.py	Keep numbers stable	RMSNorm = x / sqrt(mean(x²)) * weight
rotary_embedding.py	Position encoding	Rotate Q,K vectors — nearby tokens have similar rotations
sampler.py	Pick next token	Gumbel-max trick — divide probs by exponential noise, take argmax
embed_head.py	Token ID ↔ vector	Entry: lookup table. Exit: similarity search against all vocab
linear.py	Matrix multiplication	Column/Row parallel splits matrices across GPUs
attention.py	Token interaction	Triton kernel writes paged KV cache; Flash Attention reads it
