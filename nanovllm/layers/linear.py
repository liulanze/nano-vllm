import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


# The foundation
class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim # Which dimension to split for tensor parallelism. 0=output(rows), 1=input(cols)
        self.tp_rank = dist.get_rank() # Which GPU am I
        self.tp_size = dist.get_world_size() # Total number of GPUs
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        # weight_loader is a function attached to each parameter. When the
        # weight loader (in utils/loader.py) loads a checkpoint, it calls
        # param.weight_loader(param, loaded_weight) to let each layer decide how
        # to slice and load its portion of the weight.
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# Same weight on every GPU
class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight) # Just copy the whole thing

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias) # Standard matmul


# Split the output
class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim) # my portion size
        start_idx = self.tp_rank * shard_size     # where my slice starts
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size) # slice
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


# Fused Gate + Up Projection

# In Qwen3's MLP, there are two separate projections — gate_proj and up_proj —
# both with the same input but different learned weights. Instead of two
# separate matrix multiplications, this class fuses them into one big matrix:

"""
Instead of:
  gate = x @ W_gate    (1024 → 2816)
  up   = x @ W_up      (1024 → 2816)

We do:
  gate_up = x @ [W_gate; W_up]   (1024 → 5632)
"""
class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    # The weight_loader here is more complex because it loads gate and up separately.
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        # loaded_shard_id=0 → gate_proj, loaded_shard_id=1 → up_proj
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)        # Find my slot
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank] # My shard
        param_data.copy_(loaded_weight)


# Fused Q, K, V

# Same idea as MergedColumn, but for attention's Q, K, V projections. The
# wrinkle is that Q and KV can have different numbers of heads (GQA): K and V
# have fewer heads than Q — this is Grouped-Query Attention (GQA). Every 2 Q
# heads share 1 KV head. Saves memory (less KV cache) with minimal quality loss.
"""
With 16 Q heads, 8 KV heads, head_dim=64:
  Q: 16 × 64 = 1024
  K:  8 × 64 =  512
  V:  8 × 64 =  512
  Total: 2048

Fused layout: [QQQQQQQQQQQQQQQQ | KKKKKKKK | VVVVVVVV]
               1024 dims          512 dims    512 dims
"""
class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


# Split the Input, Then Sum

# The complement of ColumnParallel. Splits the weight horizontally:
"""
Full weight [output=1024, input=4096]:

GPU 0 gets: [output=1024, input=2048]  (left half of columns)
GPU 1 gets: [output=1024, input=2048]  (right half of columns)

Each GPU gets a different slice of the input and produces a partial output:

GPU 0: y_partial = x_left  @ W_left   → [tokens, 1024]
GPU 1: y_partial = x_right @ W_right  → [tokens, 1024]
"""
class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if param_data.ndim == 1:
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    # But these are partial sums! The true answer is y = y_partial_0 +
    # y_partial_1. That's what all_reduce does:
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y) # Sum partial results across GPUs!!
        return y

# Column + Row always pair together: In the transformer, this is the pattern:
"""
ColumnParallel (split output, no communication)
     → each GPU works on its slice
RowParallel (sum partials, all_reduce)
     → result is identical on every GPU
"""
# This minimizes communication — only one all_reduce per pair.
