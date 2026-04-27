"""
这个文件不是在分配 GPU K/V tensor 本体，而是在维护一套“block 元数据管理系统”：

哪些 block 空闲
哪些 block 被用了
哪些 block 可被 prefix cache 复用
每个请求对应哪些 block
decode 时怎么按 block 扩展
请求结束时怎么按引用计数回收
"""

from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        # 有多少序列正在引用这个block，主要是为了prefix cache复用
        self.ref_count = 0
        # 这个block对应的token内容的哈希值
        self.hash = -1
        # 这个block对应的token列表，用来进一步校验哈希命中是不是假阳性，所以它更像是“页表项”，不是“页内容”
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    # 命中 prefix cache 时，seq间的 block 是共享关系，未命中时，seq间竞争使用free block
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # dict的key就是通过compute_hash算出来的哈希值，value是对应的block_id
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """
        计算token_ids的哈希值，考虑前缀prefix。
        此处的prefix是上一个block的哈希值，用于实现前缀链式哈希。
        """
        h = xxhash.xxh64()
        if prefix != -1:
            # 计算哈希值时，需要将前缀也考虑进去，从而形成前缀链式哈希
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        """
        如果一个新序列需要seq.num_blocks个block，那空闲的block数必须足够，这是prefill前的准入检查
        """
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """
        为一个序列分配block，将token_ids写入block中
        如果token_ids在block中，直接增加引用计数
        如果token_ids不在block中，创建新的block
        """
        assert not seq.block_table
        h = -1
        cache_miss = False
        # 假设seq.num_blocks=3
        # seq.token_ids= [1,2,3,4,5,6], block_size=2
        for i in range(seq.num_blocks):
            # 获取到第i个block所对应的token_ids
            # 第一次i = 0,
            token_ids = seq.block(i)
            # 计算token_ids的哈希值，考虑前缀prefix。
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            # 如果哈希值命中，且token_ids也匹配，那么就认为是当前seq的token_ids，命中了prefix cache
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else: # 之前命中了cache
                # 更新seq的num_cached_tokens
                seq.num_cached_tokens += self.block_size
                # 如果block_id在used_block_ids中，说明这个block已经被其他序列引用了，需要增加引用计数
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    # 当前这个block被共享的次数
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        """
        释放一个序列占用的block，将引用计数减1，如果引用计数为0，则将block归还给空闲池
        """
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """
        decode时的追加是“页式扩展”，而不是“整体重拷贝”
        当一个序列准备继续生成下一个token时，该方法会判断：
        需不需要给它新开一个block，或者当前block是否刚好已经填满
        """
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1: # 刚跨进一个新的block，需要分配新的block
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0: # 当前block刚好填满，需要计算hash，加入prefix cache索引
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1
