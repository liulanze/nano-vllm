import torch
from torch import nn


# Turning Logits into Token IDs.
class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # logits shape: [batch, vocab_size], e.g. [batch, 151936] Devided by
        # temperature. Higher temperature → more random/creative, lower temperature →
        # more greedy.
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        # Converts raw scores into probabilities that sum to 1
        probs = torch.softmax(logits, dim=-1)

        """
        This is a mathematically equivalent alternative to torch.multinomial()
        (random weighted sampling), but faster on GPU.

        The Gumbel-max trick works like this:
        1. torch.empty_like(probs).exponential_(1) — Generate random noise from an
        exponential distribution. Each vocab entry gets a random number.
        2. .clamp_min_(1e-10) — Prevent division by zero. 
        3. probs.div_(noise) — Divide probabilities by the random noise. 
        4. .argmax(dim=-1) — Pick the index with the highest value.

        Why does this work? Mathematically, if you divide probabilities by
        exponential random variables, then taking argmax gives you a sample from
        the original probability distribution. A token with probability 0.6 will
        "win" the argmax 60% of the time because its higher probability makes it
        harder for the random noise to knock it down.
        """
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
