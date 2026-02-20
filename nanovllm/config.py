import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    # Max total num of tokens that can be processed in a single batch, just for
    # prefill, since decode's seq_len = 1.
    max_num_batched_tokens: int = 16384
    # Max num of sequences that can be in-flight at the same time, both prefill
    # and decode. For example, during decode, each sequence generates one token
    # per step, so a batch of 512 sequences will generate 512 tokens per step,
    # which means there are 512 forward passes fused into one.
    max_num_seqs: int = 512
    # Total length/tokens of a sequence/request (prompt + generated tokens) cannot exceed this value.
    max_model_len: int = 4096
    # Fraction of GPU memory allowed to use. The remaining 90% is reserved and
    # split between model weights (fixed) and KV cache (dynamic, grows as generation goes on).
    gpu_memory_utilization: float = 0.9
    # Number of GPUs to shard the model across.
    tensor_parallel_size: int = 1
    # if True, skip CUDA graph capture and always run eagerly. If False,
    # pre-capture CUDA graphs for decode.
    enforce_eager: bool = False
    # The rest of the system needs architecture details such as
    # num_hidden_layers, hidden_size, num_attention_heads, num_key_value_heads,
    # vocab_size, torch_dtype, etc. Rather than passing dozens of individual
    # fields, the whole hugging face config object is stored here.
    hf_config: AutoConfig | None = None
    # The end of sequence token id, init default to -1, which will set to the
    # actual value after tokenizer loads.
    eos: int = -1
    # Number of tokens in each KV cache block.
    kvcache_block_size: int = 256
    # Total number of KV cache blocks to allocate.
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        # KV cache block size must be multiple of 256, align to Flash Attention.
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
