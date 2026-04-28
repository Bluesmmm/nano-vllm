# Chunked Prefill 深度解析

> 本文系统梳理 vLLM 中 **Chunked Prefill** 的研发动机、实现原理、数学推导与官方源码实现。内容基于 vLLM v0.8.x 及 Sarathi-Serve 原始论文，并结合 2024-2025 年最新研究进展。

---

## 目录

1. [引言](#1-引言)
2. [研发动机：为什么要做 Chunked Prefill](#2-研发动机为什么要做-chunked-prefill)
3. [核心原理：它是如何工作的](#3-核心原理它是如何工作的)
4. [数学推导：FLOPs 与开销分析](#4-数学推导flops-与开销分析)
5. [vLLM 官方实现详解](#5-vllm-官方实现详解)
6. [参数调优与性能权衡](#6-参数调优与性能权衡)
7. [局限性与演进方向](#7-局限性与演进方向)
8. [参考资料](#8-参考资料)

---

## 1. 引言

在 LLM 推理服务中，**Prefill**（首次处理 prompt）和 **Decode**（逐 token 生成）是两种截然不同的计算模式：

- **Prefill** 是 **compute-bound**（计算密集型）：一次性处理整个 prompt 的所有 token，大量矩阵乘法可以充分占满 GPU 的 Tensor Core。
- **Decode** 是 **memory-bandwidth-bound**（显存带宽密集型）：每步只处理 1 个新 token，计算量极小，大部分时间花在从 HBM 读取 KV Cache 上。

传统的调度策略把 prefill 和 decode **完全隔离**——要么先跑完所有 prefill，再跑 decode；要么每轮只跑一种。这导致一个致命问题：**长 prompt 的 prefill 会像一堵墙，挡住后面所有 decode 请求的去路**，造成严重的队头阻塞（Head-of-Line Blocking）。

**Chunked Prefill** 的核心思想是：

> 把长 prompt 切成多个小块（chunk），每轮只处理一小块 prefill，剩下的计算资源用来跑 decode。这样既不让长 prefill 独占 GPU，又能把 compute-bound 和 memory-bound 的计算混合到同一批中，提升整体 GPU 利用率。

这一技术最早由 **Sarathi-Serve**（Microsoft Research, 2024）提出，随后被 vLLM、SGLang、TGI、TensorRT-LLM 等主流推理框架广泛采用。在 vLLM V1 引擎中，chunked prefill 已成为**默认行为**。

---

## 2. 研发动机：为什么要做 Chunked Prefill

### 2.1 队头阻塞（Head-of-Line Blocking）

假设服务同时收到两类请求：

- **Request A**：prompt 长度 8192 tokens（比如一篇长文档）
- **Request B~K**：prompt 长度 32 tokens，正在 decode 阶段（每步产出 1 个 token）

在没有 chunked prefill 的传统调度器（如 nano-vllm 的 `Scheduler.schedule()`）中，调度策略通常是 **Prefill-First**：

```
Iteration 1: 处理 Request A 的 8192 tokens prefill  → 耗时 500ms
Iteration 2~N: Request B~K 的 decode 恢复             → 每步 15ms
```

在这 500ms 内，Request B~K 的所有 decode 完全停滞。对于流式输出场景，用户会感受到明显的"卡顿"——这是无法接受的体验。

### 2.2 GPU 利用率低下

即使不考虑延迟，单纯从硬件利用率看：

| 阶段 | 瓶颈资源 | GPU 利用率 |
|------|----------|-----------|
| Prefill | Tensor Core（计算） | 高 |
| Decode | HBM 带宽（显存读取） | 低 |

如果把它们分开跑，prefill 时 HBM 带宽空闲，decode 时 Tensor Core 空闲。两者互补，却无法同时利用。

### 2.3 TTFT vs ITL 的根本矛盾

| 指标 | 含义 | 优化方向 |
|------|------|----------|
| **TTFT** (Time To First Token) | 用户发送请求到收到第一个生成 token 的延迟 | Prefill-First 策略更优 |
| **ITL** (Inter-Token Latency) / **TBT** (Time Between Tokens) | 生成过程中相邻 token 的间隔延迟 | Decode-First 策略更优 |

传统调度器必须二选一：
- 优先 prefill → TTFT 好，但长 prefill 会饿死 decode，ITL 极差。
- 优先 decode → ITL 好，但新请求要等很久才排上 prefill，TTFT 差。

**Chunked Prefill 的破局之道**：不再把 prefill 当作原子操作，而是切成可中断的 chunk。这样每轮调度可以：
1. 先保证所有 decode 请求都跑一步（ITL 最优）。
2. 用剩余的 token budget 跑一小段 prefill（TTFT 渐进式优化）。
3. 把 compute-bound 和 memory-bound 混合到同一次 forward 中，GPU 两边资源都占满。

---

## 3. 核心原理：它是如何工作的

### 3.1 宏观流程

Chunked Prefill 把调度器的工作方式从：

```
[Prefill Request A] → [Prefill Request B] → [Decode All] → [Decode All] ...
```

变成了：

```
[Decode B~K + Prefill chunk A₁] → [Decode B~K + Prefill chunk A₂] → ...
```

其中每个 `chunk A_i` 只处理 prompt 的一部分（比如 512 或 2048 个 token）。

### 3.2 Token Budget 机制

Chunked Prefill 的核心是一个 **Token Budget（令牌预算）**，由参数 `max_num_batched_tokens` 控制：

```
每轮调度的总 token 数 = Σ(decode tokens) + Σ(prefill chunk tokens) ≤ max_num_batched_tokens
```

调度流程如下：

```
1. 计算当前轮次的 token budget = max_num_batched_tokens

2. 【第一阶段】优先调度所有 RUNNING decode 请求
   - 每个 decode 请求消耗 1 个 token budget
   - 这些请求从 running 队列取出，保留在 running 中

3. 【第二阶段】用剩余 budget 调度 prefill 请求
   - 从 waiting 队列取新请求，或从 running 取未完成的 chunked prefill
   - 如果请求的剩余 prefill 长度 > 剩余 budget，只切出一个 chunk
   - 该请求状态变为 RUNNING（但 prefill 未完成），下轮继续处理

4. 【第三阶段】如果 budget 仍不足且显存不够，触发抢占（preemption）
   - 把低优先级的 running 请求踢回 waiting，释放 KV Cache
```

### 3.3 调度优先级（vLLM 实现）

启用 chunked prefill 后，vLLM 调度器的优先级彻底反转：

| 优先级 | 请求类型 | 说明 |
|--------|----------|------|
| 1（最高）| Running decode | 已经在生成中的请求，必须每步都跑，否则用户会感知卡顿 |
| 2 | Running chunked prefill | 上一轮没跑完的 prefill chunk，继续处理 |
| 3 | Swapped | 之前被抢占出去、需要恢复的请求 |
| 4（最低）| New prefill | 全新的请求，首次进入 prefill |

这与 nano-vllm 中的 **Prefill-First** 策略（`waiting` 优先于 `running`）完全相反。

### 3.4 Attention 计算的特殊性

Chunked Prefill 在 Attention 计算上有一个关键细节：

> 第 `i` 个 chunk 的 Attention 需要 **读取之前所有 chunk 的 KV Cache**，而不仅仅是当前 chunk。

这意味着：
- Chunk 0：Q × K₀V₀（只涉及自己的 token）
- Chunk 1：Q × K₀₋₁V₀₋₁（需要加载 Chunk 0 的 KV Cache）
- Chunk 2：Q × K₀₋₂V₀₋₂（需要加载 Chunk 0 和 1 的 KV Cache）
- ...

这正是 chunked prefill **额外开销的来源**——不是额外计算，而是**重复读取 KV Cache 的显存带宽开销**。

---

## 4. 数学推导：FLOPs 与开销分析

### 4.1 符号定义

| 符号 | 含义 |
|------|------|
| N | Prompt 总长度（tokens） |
| c | Chunk size（每个 chunk 的 token 数） |
| n = N / c | Chunk 数量 |
| d | Head dimension / hidden dimension |
| h | Attention heads 数量 |
| l | Transformer layers 数量 |

### 4.2 Attention FLOPs 推导

对于标准的 causal dot-product attention，每个 chunk 的计算包括：

**Chunk i（0-indexed，包含 c 个 token）**：
- Query 长度：c
- Key/Value 长度：(i + 1) × c（当前 chunk + 所有历史 chunk）
- Attention FLOPs ≈ 2 × c × (i + 1)c × d（QK^T 矩阵乘法）
- Weighted Sum FLOPs ≈ 2 × c × (i + 1)c × d（Softmax(QK^T) × V）

因此，Chunk i 的 Attention FLOPs：

```
FLOPs_attention(i) = 4 × (i + 1) × c² × d
```

**所有 Chunk 的总 Attention FLOPs**：

```
FLOPs_attention(total) = Σ(i=0 to n-1) 4 × (i + 1) × c² × d
                       = 4c²d × Σ(i=0 to n-1)(i + 1)
                       = 4c²d × n(n + 1) / 2
                       ≈ 2c²d × n²          （当 n 较大时）
```

代入 n = N / c：

```
FLOPs_attention(total) ≈ 2c²d × (N/c)² = 2N²d
```

**对比完整 Prefill**：

完整 prefill 的 Attention FLOPs（causal mask）：

```
FLOPs_attention(full) = 2 × N × N × d = 2N²d
```

**结论**：

> **Chunked Prefill 的总 Attention FLOPs 与完整 Prefill 完全相同，都是 2N²d。**
> Chunk size `c` 在总 FLOPs 计算中被完全约掉了。

这意味着：**从纯计算量角度，chunking 不会增加任何额外开销。**

### 4.3 Linear Layer FLOPs 推导

Linear layers（Q/K/V/O projection, FFN 等）的计算与历史无关，**每个 chunk 的 FLOPs 是恒定的**：

```
FLOPs_linear_per_chunk ≈ l × (12 × c × d²)   （标准 Transformer 每层约 12hd²）
```

总 Linear FLOPs：

```
FLOPs_linear(total) = n × FLOPs_linear_per_chunk
                    = (N/c) × l × 12cd²
                    = l × 12Nd²
```

这也与完整 prefill 的 Linear FLOPs 完全一致。

### 4.4 那么开销到底来自哪里？

既然总 FLOPs 不变，为什么 Sarathi-Serve 论文指出 chunking 有 overhead？

**答案：显存带宽开销（Memory Bandwidth），而非计算开销。**

每处理 Chunk i，Attention kernel 必须从 HBM 加载所有历史 KV Cache：

```
Chunk 0: 读取 0   tokens 的 KV Cache
Chunk 1: 读取 c   tokens 的 KV Cache
Chunk 2: 读取 2c  tokens 的 KV Cache
...
Chunk n-1: 读取 (n-1)c tokens 的 KV Cache
```

总 KV Cache 读取量：

```
KV_read_total = 2 × d × (0 + c + 2c + ... + (n-1)c) × sizeof(dtype)
              = 2 × d × c × n(n-1)/2 × sizeof(dtype)
              ≈ N²d × sizeof(dtype)        （当 n 较大时）
```

而完整 prefill 只需要读取一次 KV Cache（因为一次性算完，内部缓存复用）。

**因此，chunk size 越小**：
- n 越大（chunk 数量越多）
- KV Cache 重复读取次数越多
- Memory bandwidth 开销越大

Sarathi-Serve 论文的实验表明（Figure 11, Yi-34B）：
- **chunk size ≥ 2048** 时，linear operator 的 overhead 接近 0。
- Attention 和 all-reduce 仍有少量 overhead（来自 memory traffic）。

### 4.5 每 Chunk 耗时恒定的谜题

理论上，Chunk i 的 Attention FLOPs 随 `i` 线性增长：

```
FLOPs(i) ∝ (i + 1) × c²
```

但实际观察中，很多 chunk 的 wall-clock 时间却几乎恒定。原因在于：

1. **Linear layers 占主导**：对于常见模型和中等长度（N < 16K），linear layer 的 FLOPs 远大于 attention。Linear layer 每 chunk 恒定，因此总时间被"拉平"。
2. **Memory-bound 而非 Compute-bound**：当 chunk size 较小时，attention kernel 的瓶颈在读取 KV Cache 的带宽，而非 Tensor Core 的计算能力。带宽需求的增长被 memory-level parallelism 掩盖。
3. **GPU Tensor Core 利用率**：现代 GPU 的 Tensor Core 吞吐量远高于 HBM 带宽，计算密集型任务往往等数据，而非数据等计算。

---

## 5. vLLM 官方实现详解

### 5.1 启用方式

在 vLLM v0.4.0+ 中：

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_chunked_prefill=True,      # 启用 chunked prefill
    max_num_batched_tokens=2048,      # 控制 chunk / batch 大小
)
```

或在命令行：

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --enable-chunked-prefill \
    --max-num-batched-tokens 2048
```

### 5.2 关键参数 `max_num_batched_tokens`

这是 Chunked Prefill 的唯一调参入口，它同时控制三件事情：

| 作用 | 说明 |
|------|------|
| **Prefill chunk size 上限** | 单个请求的 prefill 最多切出这么长的一段 |
| **Batch token 上限** | 整批所有请求的 token 总数（decode + prefill chunk）不超过此值 |
| **显存压力调节器** | 越大则单批越"重"，越接近 OOM；越小则调度越频繁 |

默认值演变：

| vLLM 版本 | 默认值 | 说明 |
|-----------|--------|------|
| v0.4.x / v0.5.x | 512 | 偏向低 ITL，但吞吐可能受限 |
| v0.8.2+ | 2048 | 更均衡的默认值 |
| V1 engine | 动态 / 更大 | 默认启用，自动调节 |

### 5.3 Scheduler 调度策略对比

#### 未启用 Chunked Prefill（`_schedule_default`）

```python
# 伪代码逻辑
if waiting_queue:
    # 优先 prefill：把 waiting 里的请求尽可能塞满一批
    schedule_prefills(waiting_queue)
else:
    # 没有新请求，才跑 decode
    schedule_decodes(running_queue)
```

特征：
- **Prefill-First**
- 一批里全是 prefill 或全是 decode，不会混合
- 新请求响应快（TTFT 好），但长 prefill 会饿死 decode

#### 启用 Chunked Prefill（`_schedule_chunked_prefill`）

```python
# 伪代码逻辑
token_budget = max_num_batched_tokens

# 1. 先跑所有 running decode（每个占 1 token）
for seq in running_decode_queue:
    if token_budget > 0:
        schedule(seq)
        token_budget -= 1

# 2. 再跑 running chunked prefill（未完成的 prefill）
for seq in running_chunked_prefill_queue:
    chunk_size = min(remaining_prefill_len(seq), token_budget)
    if chunk_size > 0:
        schedule(seq, chunk_size)
        token_budget -= chunk_size

# 3. 再恢复 swapped 请求
for seq in swapped_queue:
    if can_allocate(seq) and token_budget > 0:
        schedule(seq)
        token_budget -= len(seq)

# 4. 最后处理新 prefill 请求
for seq in waiting_queue:
    chunk_size = min(len(seq), token_budget)
    if chunk_size > 0 and can_allocate(seq):
        schedule(seq, chunk_size)
        token_budget -= chunk_size
```

特征：
- **Decode-First**：decode 请求优先级最高
- **混合批次**：同一轮 forward 里同时有 decode token 和 prefill chunk
- **Chunk 切分**：长 prefill 被切成多块，分散到多轮执行

### 5.4 状态流转变化

Chunked Prefill 引入了一个新的状态概念：**一个请求可以处于 RUNNING 状态，但 prefill 尚未完成。**

```
【传统调度】
WAITING ──prefill──► RUNNING ──decode──► FINISHED

【Chunked Prefill】
WAITING ──chunk1──► RUNNING(chunked) ──chunk2──► RUNNING(chunked)
                                              ──chunkN──► RUNNING(complete)
                                                           ──decode──► FINISHED
```

这意味着：
- `running` 队列里同时有两种请求：正在 decode 的和正在 chunked prefill 的。
- 被抢占的 chunked prefill 请求，如果 KV Cache 被释放，回到 `waiting` 后**必须重新从第一个 chunk 开始**（因为 KV Cache 丢了）。

### 5.5 与 Prefix Caching 的交互限制

vLLM 的 **Automatic Prefix Caching (APC)** 可以和 Chunked Prefill 同时开启，但存在已知限制：

> **只有第一个 chunk 能享受 prefix caching。**

原因：
- 第一个 chunk 调度时，请求还在 `waiting` 状态，scheduler 会检查 prefix cache。
- 一旦第一个 chunk 开始执行，请求进入 `running` 状态。
- 后续 chunk 被当作 running 请求处理，scheduler **不再重新评估 prefix cache**。

这意味着：如果一个长 prompt 的前 50% 命中了 prefix cache，但被切成多个 chunk，只有第一个 chunk 能跳过计算，后面的 chunk 仍需完整执行。

这是 vLLM 社区中已知的待优化项（相关 issue: #7883）。

### 5.6 V1 Engine 的演进

在 vLLM 最新的 V1 引擎中：
- Chunked Prefill **默认开启**，无需手动配置。
- `max_num_batched_tokens` 的默认值更大，调度更激进。
- 调度器与 CUDA kernel 的深度集成进一步优化了混合批次的效率。

---

## 6. 参数调优与性能权衡

### 6.1 `max_num_batched_tokens` 的选择

这是唯一需要手动权衡的参数，它直接影响三个指标：

| 指标 | 小值（如 512） | 大值（如 8192） |
|------|---------------|----------------|
| **TTFT** | 差：长 prompt 需要更多轮次才 prefill 完 | 好：prefill 完成更快 |
| **ITL / TBT** | 好：decode 被 prefill 打断的次数少 | 差：长 prefill chunk 会阻塞 decode |
| **吞吐量** | 可能差：kernel launch 开销占比高 | 好：每批更饱满，GPU 利用率更高 |
| **OOM 风险** | 低 | 高 |

**推荐策略**：

| 场景 | 推荐值 | 理由 |
|------|--------|------|
| 流式聊天（低延迟优先）| 512 ~ 1024 | ITL 敏感，decode 不能被长时间打断 |
| 离线批处理（吞吐优先）| 4096 ~ 8192 | TTFT 不敏感，最大化 GPU 利用率 |
| 通用在线服务 | 2048 | vLLM 官方推荐默认值，均衡 |
| A100/H100 大模型 | ≥ 2048 | 大 GPU 显存足，可以承受更大 batch |

### 6.2 Chunk Size 与开销的量化关系

根据 Sarathi-Serve 论文实验（Yi-34B）：

| Chunk Size | Attention Overhead | Linear Overhead | 总 Overhead |
|-----------|-------------------|-----------------|-------------|
| 256 | 较高 | 显著 | 明显 |
| 512 | 中等 | 较小 | 可接受 |
| 1024 | 低 | 接近 0 | 很低 |
| 2048 | 很低 | ~0 | 极低 |
| 4096 | 极低 | ~0 | 可忽略 |

**经验法则**：chunk size 不应小于 512，推荐 ≥ 2048。

### 6.3 与模型尺寸的关联

更大的模型（更多参数、更大 hidden dimension）对 chunked prefill 的收益更明显：

- **大模型**：prefill 的绝对耗时更长，队头阻塞问题更严重。chunked prefill 的收益最大。
- **小模型**：prefill 本身很快，chunking 的相对收益较小，但仍有帮助。

---

## 7. 局限性与演进方向

### 7.1 当前局限性

| 局限性 | 说明 |
|--------|------|
| **Prefix Caching 利用率低** | 只有第一个 chunk 能命中 prefix cache |
| **混合批次干扰** | 同一 batch 中 prefill 和 decode kernel 互相干扰，decode 可能慢 8~10 倍 |
| **固定 chunk 粒度** | chunk size 由 `max_num_batched_tokens` 统一控制，无法针对不同请求动态调整 |
| **多模态挑战** | 图像/视频 prefill 的 token 数远超文本，固定 chunk 策略难以适配 |
| **抢占代价高** | chunked prefill 请求被抢占后需从头重算（KV Cache 丢失） |

### 7.2 2024-2025 研究演进

| 方向 | 代表工作 | 核心思想 |
|------|----------|----------|
| **动态粒度** | FlowPrefill (2025) | 解耦抢占粒度和 chunk 边界，减少控制面开销 |
| **PD 分离** | DistServe, TaiChi (2024) | 把 prefill 和 decode 放到不同 GPU 上，彻底消除干扰 |
| **层内交错** | POD-Attention (ASPLOS 2025) | 在同一个 batch 内让 prefill 和 decode 的 attention 完全重叠 |
| **模态感知** | Rocks, Pebbles and Sand (2025) | 针对多模态的不同 prefill 大小做差异化调度 |

**趋势判断**：Chunked Prefill 是当前单节点推理的**标准基线**，但业界正在向 **PD 分离（Prefill-Decode Disaggregation）** 和 **更细粒度的混合调度** 演进。

---

## 8. 参考资料

### 原始论文

1. **Sarathi: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills**
   - 作者：Agrawal et al.
   - 发表：arXiv:2308.16369, 2023
   - [链接](https://arxiv.org/abs/2308.16369)

2. **Sarathi-Serve: Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve**
   - 作者：Agrawal et al., Microsoft Research
   - 发表：arXiv:2403.02310, 2024
   - [链接](https://arxiv.org/abs/2403.02310)
   - [论文精读](https://www.abhimanyutalwar.com/paper_summaries/20240729_sarathi.html)

### vLLM 官方文档

3. **vLLM Optimization and Tuning Guide**
   - [vLLM Docs v0.8.2](https://docs.vllm.ai/en/v0.8.2/performance/optimization.html)

4. **Inside vLLM: Anatomy of a High-Throughput LLM Inference System**
   - 作者：Aleksa Gordić
   - [博客](https://www.aleksagordic.com/blog/vllm)

### 进阶研究

5. **POD-Attention: Unlocking Full Prefill-Decode Overlap for LLM Serving**
   - 作者：Microsoft Research
   - 发表：ASPLOS 2025
   - [PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2025/03/POD-Attention-ASPLOS25.pdf)

6. **Proactive Intra-GPU Disaggregation of Prefill and Decode in LLM Serving**
   - 发表：arXiv:2507.06608, 2025
   - [链接](https://arxiv.org/abs/2507.06608)

7. **FlowPrefill: Decoupling Preemption from Prefill Scheduling Granularity**
   - 发表：arXiv:2602.16603, 2025
   - [链接](https://arxiv.org/abs/2602.16603)

### 中文技术文章

8. **vLLM 源码解析之 chunked prefill**
   - [知乎专栏](https://zhuanlan.zhihu.com/p/1914830989265437343)

9. **大模型分块预填充 Chunked Prefill**
   - [微信公众号](http://mp.weixin.qq.com/s/YPBEPlTBht2BWNlvc-XbdA)

10. **Chunked-Prefills 分块预填充机制详解**
    - [掘金](https://juejin.cn/post/7526472026291994624)

---

> 本文档最后更新于 2026-04-18。vLLM 版本迭代较快，部分实现细节可能随版本更新而变化，建议以 [vLLM 官方文档](https://docs.vllm.ai) 和 [GitHub 源码](https://github.com/vllm-project/vllm) 为准。
