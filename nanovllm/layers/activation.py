import torch
from torch import nn
import torch.nn.functional as F

"""
Neural networks need non-linearity. If every layer is just matrix multiplication
(which is linear), stacking 100 layers would be equivalent to a single matrix
multiplication. Non-linearity is what gives deep networks their power.
"""

class SiluAndMul(nn.Module):

    # @torch.compile tells PyTorch to fuse these operations into a single GPU
    # kernel for speed.
    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Splits the input tensor in half along the last dimension. If input is
        # [tokens, 5632], you get two tensors of [tokens, 2816] each. The first
        # half is the "gate", the second half is the "up projection".
        x, y = x.chunk(2, -1)

        # SiLU (Sigmoid Linear Unit) = x * sigmoid(x). It's a smooth activation
        # function. Unlike ReLU which harshly clips negatives to 0, SiLU gently
        # curves:
        """
        SiLU(x) = x * σ(x)
        When x = -5  → SiLU ≈ -0.03  (almost zero, but not exactly)
        When x =  0  → SiLU = 0
        When x =  5  → SiLU ≈ 4.97   (almost identity)
        """

        # F.silu(x) * y — The gate (x after SiLU) element-wise multiplies the up
        # projection (y). This is called gated activation — the gate controls
        # how much of each feature passes through. Think of it as a dimmer
        # switch for each feature.
        return F.silu(x) * y

        # Why gating? The model learns that some features should be suppressed
        # and others amplified. The gate half learns which features matter, and
        # the up half carries the content.
