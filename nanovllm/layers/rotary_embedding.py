from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    x1 = [x0, x1, x2]
    x2 = [x3, x4, x5]
    """
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)

    """
    cos = [cosθ0, cosθ1, cosθ2]
    sin = [sinθ0, sinθ1, sinθ2]
    """
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    """
    y1 = [x0 * cosθ0 - x3 * sinθ0, x1 * cosθ1 - x4 * sinθ1, x2 * cosθ2 - x5 * sinθ2]
    y2 = [x3 * cosθ0 + x0 * sinθ0, x4 * cosθ1 + x1 * sinθ1, x5 * cosθ2 + x2 * sinθ2]
    """

    # Assemble to one tensor again:
    """
    y = [
        x0*cosθ0 - x3*sinθ0,
        x1*cosθ1 - x4*sinθ1,
        x2*cosθ2 - x5*sinθ2,
        x3*cosθ0 + x0*sinθ0,
        x4*cosθ1 + x1*sinθ1,
        x5*cosθ2 + x2*sinθ2
    ]
    """
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


# RoPE (Rotary Positional Embedding) injects position by rotating the Q and K
# vectors.
class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        """
        这里讲这些 θ 是从哪里来的？为什么每一对不一样？
        inv_freq 是一个向量，
        长度 = 二维向量的“对数”的个数，
        每一项表示：
        “这一对二维向量，位置每 +1，要转多少角度”
        前面的维度 → 转得快
        后面的维度 → 转得慢
        """
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        # This precomputes cos and sin for every possible position (up to 131K).
        # The cache shape [131072, 1, 64] means: for each position, a single row
        # of cos+sin values covering the full head dimension. ONLY cache
        # cos/sin, because the SAME cos/sin can be applied to any query/key
        # vector at that position.
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb

"""
对 同一个 token，RoPE 会做：
(x0, x3)  →  旋转一个角度 θ(pos)

所以：

token A 的 Q：
(x0, x3) 旋转 θ(p)


token B 的 K：
(k0, k3) 旋转 θ(q)

现在就是Q和K什么时候碰到一起，答案就是在 attention score 里：
score = Q ⋅ K

点积会出现 θ_p - θ_q 的差值，这个差值就是相对位置，就是在这里“自然出现的”。
但在 QK 点积时，绝对角度会相互抵消，只剩下相对角度差。

在 decode 阶段，只有当前 token 产生新的 Query，
它在和 KV cache 里的 Key 做点积时，
attention 分数天然就包含了它与每个历史 token 的相对位置。
"""
