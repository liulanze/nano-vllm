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
