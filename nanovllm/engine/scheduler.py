from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs # 512
        self.max_num_batched_tokens = config.max_num_batched_tokens # 16384
        self.eos = config.eos # end of sequence token id
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque() # Sequences that need prefill (haven't started generating yet)
        self.running: deque[Sequence] = deque() # Sequences currently generating tokens (in decode phase)

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill: try prefill first.
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]            # peek at first waiting sequence
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq) # assign KV cache blocks
            num_batched_tokens += len(seq) - seq.num_cached_tokens # only count non-cached tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()           # remove from waiting
            self.running.append(seq)         # add to running
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True      # True = is_prefill

        # decode: fallback to try decode next.
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop()) # evict the LAST sequence (most recent)
                else:
                    self.preempt(seq)                # no one else to evict, evict ourselves
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq) # free all its KV cache blocks
        self.waiting.appendleft(seq)       # put back at FRONT of waiting queue

    """
    For each sequence, append the new token, then check two stopping conditions:
    1. EOS token: The model generated the end-of-sequence token (unless ignore_eos=True)
    2. max_tokens reached: The user's requested limit is hit (e.g., generated 64 tokens)
    """
    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
