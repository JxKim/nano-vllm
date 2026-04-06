from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        # 这两个属性，是什么意思？
        self.max_num_seqs = config.max_num_seqs
        # prefill阶段一次性计算整段prompt，计算量和显存占用都跟token总数强相关。
        # 如果不设置上限，就可能出现：一次塞太多token，显存爆掉  / attention计算过重，延迟很差 / 一轮batch太大，系统不稳定
        # 在推理系统中，每条请求长度可能都不一样，请求是动态进来的，不是所有请求都同时处于同一个阶段，prefill和decode的开销差异很大，所以它不能只靠 每条序列最大长度 这一个限制来关注系统
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        # 此处传入的config，已经是经过了Model Runner当中计算而得到的，kvcache_blocks
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        # 等待队列，运行队列，都是双端队列
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        # 等待队列和运行队列都为空时，调度器完成
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        在LLMEngine的step中调用。
        优先把等待中的请求做prefill；如果这轮没法做prefill，就继续让运行中的请求做decode；如果decode时显存不够，就抢占一部分请求腾空间
        本质上是在回答一个问题：
            当前这一轮，GPU应该跑哪些序列

        如果能接新请求，就优先做prefill，否则推进老请求做decode，如果decode空间不够，就抢占一部分请求
        """
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            # 做两项检查：1. 批量大小是否超过最大限制；2. 是否有足够的内存分配给该序列
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            # 从等待队列中移除该序列，添加到运行队列
            self.waiting.popleft()
            self.running.append(seq)
            # 将该序列添加到已调度序列列中
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """
        抢占一个序列，将其从运行队列中移除，添加到等待队列中
        """
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
