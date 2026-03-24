import torch
from torch import nn

"""
As vectors pass through many layers, their values can drift — getting very large
or very small. This causes training instability and numerical issues.
Normalization rescales vectors to a consistent magnitude after each layer.
"""

"""
RMSNorm(x) = x / RMS(x) * weight
where RMS(x) = sqrt(mean(x²))
"""
class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        # PyTorch will update this weight during training. Initialized to all 1s
        self.weight = nn.Parameter(torch.ones(hidden_size))
        """
        Think of RMSNorm like:
        Step 1: make all vectors same loudness 🔊
        Step 2: learnable weight says:
        “But feature 42 should always be louder than feature 17”
        """

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype # Remember original type (e.g. float16)
        x = x.float()        # Convert to float32 for precision
        var = x.pow(2).mean(dim=-1, keepdim=True) # mean(x²) per vector
        x.mul_(torch.rsqrt(var + self.eps))       # x / sqrt(mean(x²) + eps)
        # A learned parameter (initialized to all 1s). After normalization makes
        # all vectors the same magnitude, this learned weight lets the model
        # re-scale each feature dimension as needed. It's like saying "feature
        # #42 should always be 2x louder than feature #43."
        x = x.to(orig_dtype).mul_(self.weight)    # Back to fp16, scale by learned weight
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
