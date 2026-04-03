"""
sequence.py —— nano-vllm 中最核心的数据结构：序列（Sequence）

【这个文件在做什么？】
想象你向 ChatGPT 提了一个问题："请给我讲个故事"。
从你提交问题到收到完整回答，整个过程可以用一条"序列"来表示：

    序列 = 你输入的文字（prompt） + 模型一个字一个字生成的回答（completion）

这个文件就是定义这条"序列"的数据结构。

具体来说：
  - SequenceStatus：序列的三种状态（等待中 / 生成中 / 已完成）
  - Sequence：序列本身，记录了：
      · 所有 token（可以把 token 理解为"分词"，一个汉字或一个单词片段就是一个 token）
      · 生成参数（温度、最大长度等）
      · KV Cache 块表（模型"记忆"的显存管理信息）

【什么是 token？】
大语言模型不是逐字处理文字，而是先把文字切分成一个个"token"。
比如 "Hello world" 可能被切分为 [15496, 995] 两个 token。
每个 token 对应一个整数 ID，模型内部只认识这些数字。
"""


from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """
    序列的状态枚举 —— 一条序列从出生到死亡只有三种状态：

    ┌──────────┐    被调度器选中     ┌──────────┐    生成完毕      ┌──────────┐
    │  WAITING  │ ──────────────► │  RUNNING  │ ────────────► │ FINISHED  │
    │  （等待）  │                  │  （运行）  │                │ （完成）   │
    └──────────┘                   └──────────┘                └──────────┘
         ▲                              │
         │        显存不够，被踢回         │
         └──────────────────────────────┘

    WAITING  = 刚提交的请求，排队等着被处理
    RUNNING  = 正在被模型处理（每一步生成一个新 token）
    FINISHED = 生成完毕（遇到了结束标记，或者达到了最大长度）

    auto() 会自动给每个状态分配一个递增的数字（1, 2, 3...），
    具体值不重要，重要的是名字本身。
    """
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    """
    序列 —— 代表一次完整的文本生成请求。

    你可以把它理解为一个"容器"，里面装着：
      1. 用户输入的所有 token（prompt 部分）
      2. 模型生成的所有 token（completion 部分）
      3. 控制生成行为的参数（温度、最大长度等）
      4. KV Cache 块表（管理模型"记忆"占用的显存）

    整个 LLM 推理引擎围绕 Sequence 工作：
      - 调度器（Scheduler）决定哪些序列先处理
      - 块管理器（BlockManager）为每个序列分配显存
      - 模型引擎（LLMEngine）对序列执行推理计算
    """

    # block_size 是 KV Cache 的"块大小"—— 模型的"记忆"按这个大小分块存储
    # 比如设成 256，就是每 256 个 token 共享一块显存
    # 这个值是类变量（所有 Sequence 实例共享同一个值）
    block_size = 256

    # counter 是一个无限递增的计数器，用来给每个新序列分配唯一的 ID
    # 比如：第 1 个序列 ID=0，第 2 个 ID=1，第 3 个 ID=2 ...
    # count() 返回的迭代器每次调用 next() 就会产出下一个整数
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        """
        创建一条新序列。

        参数：
          token_ids：用户输入文本的 token ID 列表。
                     比如 "你好" 可能被分词为 [345, 678]
          sampling_params：生成参数（温度、最大生成长度等），有默认值

        刚创建的序列状态为 WAITING（等待处理）。
        """

        # 给这条序列分配一个唯一的 ID
        self.seq_id = next(Sequence.counter)

        # 刚创建，还没开始处理，状态设为"等待"
        self.status = SequenceStatus.WAITING

        # 用 copy() 复制一份 token_ids，避免修改原始列表
        # （因为后续生成的新 token 会不断追加到这个列表里）
        self.token_ids = copy(token_ids)

        # 记录最后一个 token 的 ID（生成时需要知道"当前位置"）
        self.last_token = token_ids[-1]

        # 当前总共的 token 数量（prompt + 已生成的 completion）
        self.num_tokens = len(self.token_ids)

        # 用户输入的 prompt 有多少个 token（这个值创建后就不会变了）
        self.num_prompt_tokens = len(token_ids)

        # 已经缓存过的 token 数量（KV Cache 命中的部分可以跳过计算）
        # 比如 prompt 和之前的 prompt 有相同的开头，那开头部分就不需要重新计算
        self.num_cached_tokens = 0

        # 本轮已经被调度执行的 token 数，chunked prefill 会用它记录分块进度
        self.num_scheduled_tokens = 0

        # 是否仍处于 prefill 阶段；进入 decode 后序列化时只需要传最后一个 token
        self.is_prefill = True

        # 块表：记录这条序列的 KV Cache 存在显存的哪些"块"里
        # 比如 block_table = [3, 7, 12] 表示这条序列用了第 3、7、12 号块
        # 这个列表由 BlockManager 在分配显存时填充
        self.block_table = []

        # 以下是从采样参数中提取的控制变量：

        # temperature（温度）：控制生成的随机性
        #   温度高 → 更随机、更有创意
        #   温度低 → 更确定、更保守
        self.temperature = sampling_params.temperature

        # max_tokens：最多生成多少个新 token
        # 比如设成 64，模型最多生成 64 个 token 就必须停下来
        self.max_tokens = sampling_params.max_tokens

        # ignore_eos：是否忽略结束标记（EOS）
        # 正常情况下，模型生成了 EOS token 就会停止
        # 设为 True 则忽略它，一直生成到 max_tokens 才停
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        """返回序列的 token 总数，这样就可以用 len(seq) 来获取长度。"""
        return self.num_tokens

    def __getitem__(self, key):
        """
        支持用 seq[i] 来访问第 i 个 token。
        也支持切片，比如 seq[0:5] 获取前 5 个 token。
        """
        return self.token_ids[key]

    @property
    def is_finished(self):
        """判断序列是否已经生成完毕。用 seq.is_finished 就能直接访问，不需要加括号。"""
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """
        已经生成了多少个新 token（即 completion 部分的长度）。

        总 token 数 = prompt token 数 + completion token 数
        所以 completion token 数 = 总数 - prompt 数
        """
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        """返回用户输入的 prompt 部分（前 num_prompt_tokens 个 token）。"""
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """返回模型生成的 completion 部分（从 prompt 之后的所有 token）。"""
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        """
        这条序列总共需要多少个块来存储 KV Cache。

        计算公式：(token总数 + 块大小 - 1) // 块大小
        这是一个经典的"向上取整除法"技巧：
          (a + b - 1) // b 等价于 ceil(a / b)

        举个例子：block_size = 256
          256 个 token → (256 + 255) // 256 = 1 块
          257 个 token → (257 + 255) // 256 = 2 块
          512 个 token → (512 + 255) // 256 = 2 块
        """
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """
        最后一个块里实际有多少个 token。

        通常最后一个块是不满的（除非 token 总数恰好是 block_size 的倍数）。
        比如 300 个 token，block_size = 256：
          总共 2 个块
          最后一个块有 300 - (2-1)*256 = 300 - 256 = 44 个 token
        """
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """
        获取第 i 个块里的所有 token ID。

        举个例子，block_size = 256：
          block(0) → token_ids[0:256]   （第 0 个块：第 0~255 个 token）
          block(1) → token_ids[256:512] （第 1 个块：第 256~511 个 token）
          block(2) → token_ids[512:768] （第 2 个块：第 512~767 个 token，可能不满 256 个）

        这个方法被 BlockManager 用来计算块的哈希值（判断 KV Cache 是否可以复用）。
        """
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        """
        把模型新生成的一个 token 追加到序列末尾。

        这是序列最重要的操作之一 —— 模型每一步推理只生成一个新 token，
        然后调用这个方法把它加到序列里。

        比如：
          序列当前 token：[1, 2, 3, 4, 5]
          模型生成了 token_id = 88
          调用 append_token(88) 后：
          序列变成：[1, 2, 3, 4, 5, 88]
        """
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        """
        序列化（pickle）时调用的钩子 —— 决定"保存什么数据"。

        为什么要特殊处理？因为 token_ids 列表可能很长（几千个 token），
        传输大量数据很浪费。所以做了优化：
          - 如果序列还在 prefill 阶段（还没开始生成），保存完整 token_ids
          - 如果已经在 decode 阶段（已经开始生成），只保存最后一个 token
            （前面的 token 可以从 prompt 和之前的传输结果中恢复）

        返回一个元组，包含序列的关键状态信息。
        """
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
                self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        """
        反序列化（unpickle）时调用的钩子 —— 决定"如何恢复数据"。

        和 __getstate__ 配对使用：保存了什么，就按照相同的格式恢复。

        恢复逻辑：
          - 如果还没生成任何 completion token（刚 prefill 完），直接恢复完整 token_ids
          - 否则只恢复 last_token（完整的 token_ids 需要从其他地方重建）
        """
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, \
            self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
