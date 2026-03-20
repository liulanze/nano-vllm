from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


# One sequence is one continuous conversation context processed by the model.
class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter) # Unique ID, e.g. 0, 1, 2, ... global auto-increment
        """
        self.status — A sequence moves through three states:
        WAITING → RUNNING → FINISHED
            ↑  ↓
            └──┘  (can be preempted back to WAITING)
        """
        self.status = SequenceStatus.WAITING # starts as WAITING
        self.token_ids = copy(token_ids)     # full token list (prompt + generated)
        self.last_token = token_ids[-1]      # most recent token, for decode input.
        self.num_tokens = len(self.token_ids) # total length so far
        # This is how the system knows where the prompt ends and the completion begins.
        # fixed, how many tokens in the original prompt.
        self.num_prompt_tokens = len(token_ids)
        # how many tokens were found in prefix cache.
        self.num_cached_tokens = 0    # tokens that don't need prefill
        self.num_scheduled_tokens = 0
        # This is the page table, just like in an operating system. It maps
        # logical blocks to physical KV cache blocks on the GPU:
        """
        block_table = [5, 12, 3]
        Means:
        Logical block 0 (tokens 0-255)   → physical block 5 on GPU
        Logical block 1 (tokens 256-511) → physical block 12 on GPU
        Logical block 2 (tokens 512-...)  → physical block 3 on GPU
        """
        self.block_table = [] # list of physical block IDs.
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    # e.g. 300 tokens, 2 blocks: 300 - 1*256 = 44 tokens in the last block
    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    # returns the actual token IDs in block i, used for prefix cache hashing
    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    # Called after each decode step
    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        last_state = self.token_ids if self.num_completion_tokens == 0 or self.num_cached_tokens < self.num_tokens else self.last_token
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
