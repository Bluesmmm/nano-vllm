# 这个文件是 nano-vllm 的"KV Cache 块管理器"。
#
# 什么是 KV Cache？ 大语言模型（LLM）在生成文本时，每一步都要用到"注意力机制"，
# 而注意力机制需要用到之前所有 token 的 Key 和 Value 向量。为了避免重复计算，
# 我们把这些向量缓存起来，这就是 KV Cache。
#
# 为什么需要"块管理"？ 想象一下操作系统的虚拟内存——内存被分成固定大小的"页"来管理。
# 这里做的事情完全一样：把 KV Cache 分成固定大小的"块"（默认 256 个 token 为一个块），
# 这样可以灵活地分配、回收、共享。
#
# 为什么要共享？ 如果两个用户发来的请求有相同的 prompt 前缀（比如相同的 system prompt），
# 那它们的 KV Cache 前面部分是一模一样的，不需要算两遍！
# 就像两个人看同一本书的前三章，只需要买一本，大家一起看就行。
# 这就是 vLLM 论文中最核心的优化之一。

from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    """
    一个"块"——KV Cache 存储的最小单位。

    你可以把它想象成一个"容器"：
    - 每个容器有编号（block_id）
    - 容器里装的是一段 token 的 KV Cache
    - 用"引用计数"（ref_count）来跟踪有多少个序列在用这个容器
      （就像 Python 的垃圾回收原理：没人用了就回收）
    - 用"哈希值"（hash）来判断这个块里装的内容是否和别的块一样
      （一样就可以共享，不用重复计算）
    """

    def __init__(self, block_id):
        self.block_id = block_id     # 这个块的编号，相当于"门牌号"
        self.ref_count = 0           # 引用计数：有多少个序列正在使用这个块（0 表示空闲）
        self.hash = -1               # 这个块内容的哈希值（-1 表示"还没算完/还没填满"）
        self.token_ids = []          # 这个块里存的是哪些 token（只在块填满时有意义）

    def update(self, hash: int, token_ids: list[int]):
        """当一个块被填满时，更新它的哈希值和内容记录。"""
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """重置一个块，让它可以被重新使用（清空内容，引用计数设为 1 表示即将被一个序列使用）。"""
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """
    块管理器——负责管理所有 KV Cache 块的分配、回收和共享。

    它就像一个"仓库管理员"：
    - 手里有一堆容器（blocks），每个容器有编号
    - 知道哪些容器是空闲的（free_block_ids）
    - 知道哪些容器正在被使用（used_block_ids）
    - 还有一本"内容目录"（hash_to_block_id），通过内容的哈希值就能找到对应的容器
      这样当新来的请求需要的 KV Cache 和某个已有块一样时，可以直接复用！
    """

    def __init__(self, num_blocks: int, block_size: int):
        """
        初始化块管理器。

        参数：
            num_blocks: 总共有多少个块（由 GPU 显存大小决定）
            block_size: 每个块能存多少个 token 的 KV Cache（默认 256）
        """
        self.block_size = block_size
        # 创建所有块，编号从 0 到 num_blocks-1
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # "内容目录"：哈希值 -> 块编号的映射。用来快速查找"有没有内容和这个一样的块"
        self.hash_to_block_id: dict[int, int] = dict()
        # 空闲块的编号队列（先进先出，方便分配）
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # 正在使用的块的编号集合
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """
        计算一段 token 的哈希值。

        哈希的作用：给每个块的内容算一个"指纹"，如果两个块的指纹相同，
        说明内容一样，就可以共享（不用重复计算 KV Cache）。

        参数：
            token_ids: 这段 token 的 id 列表
            prefix: 前一个块的哈希值（用于链式哈希，保证顺序不同但 token 相同的块哈希不同）

        工作原理：类似于区块链的思想——每个块的哈希值包含了前一个块的哈希值，
        这样即使两组 token 内容一样，如果它们出现在不同位置，哈希值也会不同。
        这保证了只有真正相同的前缀才能共享 KV Cache。
        """
        h = xxhash.xxh64()  # 使用 xxhash 做快速哈希（比 MD5/SHA 快很多，这里不需要加密安全性）
        if prefix != -1:
            # 把前一个块的哈希值也算进来（链式哈希）
            h.update(prefix.to_bytes(8, "little"))
        # 把 token id 列表转成字节数组后计算哈希
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        """
        分配一个空闲块（内部方法）。

        如果这个空闲块还保留着旧哈希，需要先从哈希索引里移除，避免复用到即将被覆盖的内容。
        """
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        """回收一个引用计数已经归零的块。"""
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        """
        检查能否为序列分配块，并返回可复用的完整缓存块数量。

        返回 -1 表示空闲块不足；非负数表示前缀缓存命中的块数。
        """
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)

            # 如果哈希没找到，或者哈希找到了但实际 token 内容不匹配（哈希冲突的极端情况）
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        """
        为序列分配 KV Cache 块。

        前 num_cached_blocks 个块复用已有前缀缓存；后续块从空闲池分配。
        """
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        """
        回收一个序列的所有块。

        当一个序列生成完毕后，需要把它占用的块释放掉，让别的序列可以用。

        做法很简单：
        1. 从后往前遍历序列的块表
        2. 每个块的引用计数 -1
        3. 如果引用计数变成 0，说明没人用了，回收这个块

        为什么要从后往前？没有特殊原因，从前往后也可以，但从后往前在逻辑上
        更符合"释放"的直觉（像栈一样，后分配的先释放）。
        """
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1      # 引用计数减 1
            if block.ref_count == 0:  # 没人用了 → 回收
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0     # 清零缓存命中数
        seq.block_table.clear()       # 清空块表

    def can_append(self, seq: Sequence) -> bool:
        """
        检查：序列新增一个 token 时，空闲块够不够？

        为什么可能需要新块？因为当一个块被填满后（达到 block_size 个 token），
        下一个 token 就需要一个新的空块来存放它的 KV Cache。

        len(seq) % self.block_size == 1 的含义：
        - 序列长度对 block_size 取模等于 1
        - 意思是：添加这个 token 后，刚好是某个新块的第一个 token
        - 也就是说，上一个块刚好被填满了，需要开一个新块

        表达式的结果是 1（需要新块）或 0（不需要），用来和空闲块数量比较。
        """
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """在 decode 过程中按需为新块分配空间。"""
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        """
        对本轮已经完成调度的完整块计算哈希，并写入哈希索引。

        chunked prefill 会分多轮处理 prompt；只有调度完成的完整块才可以进入复用索引。
        """
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
