# Nano-vLLM 源码阅读计划

> 目标：通过阅读 ~1,200 行代码，深入理解 vLLM 的核心原理

---

## 📋 前置知识（建议先掌握）

| 知识点 | 重要性 | 学习资源 |
|--------|--------|----------|
| Transformer 架构 | ⭐⭐⭐ | Attention Is All You Need 论文 |
| PyTorch 基础 | ⭐⭐⭐ | 官方文档 |
| Flash Attention | ⭐⭐ | Dao-AILab/flash-attention |
| Triton Kernel | ⭐ | OpenAI Triton 教程 |

---

## 🎯 总体阅读策略

```
总时长建议：3-4 小时（可分 2-3 次完成）
阅读方式：先整体后细节，带着问题读代码
```

---

## 📖 第一阶段：项目概览（15 分钟）

### 1.1 目录结构
```
nanovllm/
├── __init__.py          # 入口：导出 LLM 和 SamplingParams
├── llm.py               # 用户 API（仅 5 行）
├── sampling_params.py   # 采样参数配置
├── config.py            # 全局配置
├── engine/              # 核心引擎
│   ├── llm_engine.py    # 主引擎（93 行）
│   ├── scheduler.py     # 调度器（71 行）⭐
│   ├── block_manager.py # PagedAttention（112 行）⭐
│   ├── model_runner.py  # 模型执行
│   └── sequence.py      # 序列状态管理
├── layers/              # 模型层实现
│   ├── attention.py     # Attention + KV Cache（75 行）⭐
│   └── ...              # 其他层
└── models/              # 模型定义
    └── qwen3.py         # Qwen3 模型实现
```

### 1.2 核心数据流
```
用户输入 → LLM.generate() → LLMEngine.add_request() 
    → Scheduler.schedule() → ModelRunner.run()
    → 返回 token → Scheduler.postprocess()
```

---

## 📖 第二阶段：核心模块精读（2-2.5 小时）

### 2.1 入口与配置（20 分钟）

**阅读文件：**
- `example.py` - 了解 API 使用方式
- `nanovllm/llm.py` - 入口类（极简）
- `nanovllm/sampling_params.py` - 采样参数
- `nanovllm/config.py` - 配置定义

**关键问题：**
- [ ] LLM 的构造函数接收哪些参数？
- [ ] `SamplingParams` 支持哪些采样策略？
- [ ] `Config` 如何管理 GPU 内存分配？

---

### 2.2 引擎层 - 调度系统（40 分钟）⭐⭐⭐

**阅读文件：**
- `nanovllm/engine/llm_engine.py`
- `nanovllm/engine/scheduler.py`

**重点理解：**

#### A. Continuous Batching 机制
```python
# scheduler.py 中的调度逻辑
# 1. 优先处理 prefill（新请求）
# 2. 然后处理 decode（生成中请求）
# 3. 内存不足时触发抢占(preempt)
```

**关键问题：**
- [ ] `max_num_seqs` 和 `max_num_batched_tokens` 如何限制 batch 大小？
- [ ] prefill 和 decode 为什么不能混合在同一个 batch？
- [ ] 抢占(preempt)发生时，被抢占的请求会怎样？

#### B. 阅读技巧
1. 先读 `schedule()` 方法的整体流程（约 30 行）
2. 理解 `waiting` 和 `running` 两个队列的状态流转
3. 关注 `SequenceStatus` 的状态变化

---

### 2.3 内存管理 - PagedAttention（40 分钟）⭐⭐⭐

**阅读文件：**
- `nanovllm/engine/block_manager.py`
- `nanovllm/engine/sequence.py`

**重点理解：**

#### A. Block 管理
```python
# Block 结构：物理内存块
- block_id: 块编号
- ref_count: 引用计数（用于共享）
- hash: 内容哈希（用于 prefix caching）
- token_ids: 包含的 token
```

#### B. 核心算法
| 方法 | 作用 |
|------|------|
| `can_allocate()` | 检查是否有足够空闲块 |
| `allocate()` | 为新序列分配 blocks，支持 prefix cache |
| `deallocate()` | 释放 blocks，处理引用计数 |
| `compute_hash()` | 计算 token 块的哈希值 |

**关键问题：**
- [ ] Prefix Caching 是如何工作的？`compute_hash` 为什么要传入 prefix？
- [ ] 两个序列共享同一个 block 时，`ref_count` 如何变化？
- [ ] 为什么 block 大小通常是 16？（考虑 GPU 内存对齐）

---

### 2.4 Attention 与 KV Cache（30 分钟）⭐⭐⭐

**阅读文件：**
- `nanovllm/layers/attention.py`

**重点理解：**

#### A. KV Cache 存储
```python
# Triton kernel 将 KV 写入缓存
store_kvcache_kernel[(N,)](...)
```

#### B. Prefill vs Decode
```python
if context.is_prefill:
    # 使用 flash_attn_varlen_func 处理变长序列
    # 支持 prefix cache（block_tables 不为 None 时）
else:
    # 使用 flash_attn_with_kvcache 逐个 token 解码
```

**关键问题：**
- [ ] Prefill 阶段为什么要用 `varlen_func`？和 `flash_attn_func` 有什么区别？
- [ ] `slot_mapping` 的作用是什么？它如何映射逻辑位置到物理 block？
- [ ] `block_tables` 在什么情况下为 `None`？什么时候不为 `None`？

---

### 2.5 模型执行（20 分钟）

**阅读文件：**
- `nanovllm/engine/model_runner.py`
- `nanovllm/models/qwen3.py`（选读）

**重点理解：**
- [ ] `ModelRunner` 如何支持 Tensor Parallelism？
- [ ] `run()` 方法中 `input_metadata` 包含哪些信息？
- [ ] CUDA Graph 是如何应用的？

---

## 📖 第三阶段：整合与思考（30 分钟）

### 3.1 画一张架构图
用自己的理解画出：
- 数据流：从请求进入到输出生成
- 控制流：各个模块如何协作
- 状态流：Sequence 的状态变化

### 3.2 回答核心问题

**关于调度：**
1. 为什么 vLLM 比 HF Transformers 快？（Continuous Batching）
2. 调度器如何保证公平性？（FCFS + 抢占机制）

**关于内存：**
1. PagedAttention 相比传统 KV Cache 的优势？（减少内存碎片、支持共享）
2. Prefix Caching 在什么场景下最有效？（多轮对话、系统提示词复用）

**关于性能：**
1. 为什么 decode 阶段是瓶颈？（memory-bound）
2. Tensor Parallelism 适用于什么场景？（大模型、多卡）

---

## 🛠️ 第四阶段：动手实践（可选，1 小时）

### 4.1 添加调试日志
在关键位置添加 `print()` 或 `logging`，观察运行时的状态：

```python
# 推荐添加日志的位置：
1. Scheduler.schedule() - 观察 batch 组成
2. BlockManager.allocate() - 观察 block 分配
3. Attention.forward() - 观察 prefill/decode 切换
```

### 4.2 修改参数观察影响
```python
# 在 example.py 中尝试修改：
1. max_num_seqs - 观察吞吐量变化
2. block_size - 观察内存使用
3. 开启/关闭 prefix caching
```

### 4.3 尝试添加功能
- [ ] 实现贪心解码（temperature=0）
- [ ] 添加推理时间的详细统计
- [ ] 支持更多的采样参数（top_p、repetition_penalty）

---

## 📚 第五阶段：拓展学习（持续）

### 5.1 对比官方 vLLM
阅读 vLLM 源码时关注：
- 多 GPU 支持的完整实现
- 更复杂的调度策略（chunked prefill）
- Speculative Decoding
- 模型并行（Pipeline Parallelism）

### 5.2 推荐资源

| 类型 | 资源 | 说明 |
|------|------|------|
| 论文 | vLLM (SOSP 2023) | PagedAttention 原始论文 |
| 博客 | vLLM Blog | 官方技术博客 |
| 视频 | CUDA Mode | 深入的 GPU 优化讲解 |
| 项目 | SGLang | 另一个高效的 LLM 推理框架 |

---

## ✅ 检查清单

完成阅读后，确认你能解释：

- [ ] 一个请求从进入到输出的完整生命周期
- [ ] PagedAttention 如何解决内存碎片问题
- [ ] Prefix Caching 的哈希计算方式
- [ ] Continuous Batching 的工作机制
- [ ] Tensor Parallelism 的通信方式

---

## 💡 学习技巧

1. **带着问题读**：不要一行一行读，先看函数签名和注释
2. **画图辅助**：边读边画类图、时序图、状态图
3. **断点调试**：如果条件允许，用 pdb 或 IDE 调试跟踪
4. **做笔记**：记录不理解的地方，后续查证
