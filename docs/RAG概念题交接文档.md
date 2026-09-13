# RAG 概念题通路 交接文档

> 给接手「概念题 / RAG」后续工作的人。目标：说清楚**这条通路现在做到哪、卡在哪、下一步该谁做什么**。
> 本文件是最终状态汇报，详细技术过程见 `docs/概念题评测报告.md`、`docs/概念题交接文档.md`。

---

## 一、一句话现状

**RAG 检索层已完成、可用；概念「理解正确率」卡在 76%，瓶颈是本地 Qwen3-8B 模型能力，不是检索。**

---

## 二、三个指标是什么意思（先看懂这个，再看结论）

| 指标 | 白话 | 测谁 | 当前值(v3) |
|---|---|---|---|
| grounded 有出处 | 答案引用的论文段落**真对得上**、不瞎编 | 检索准不准 | **98%** ✅ |
| relevance 答到点 | 答案**到底答没答对** | 模型理解强弱 | **76%** ⚠️ |
| honesty 诚实 | 不知道时**老实说查不到** | 模型诚不诚实 | **96%** ✅ |

> 关键区分：**grounded/honesty 是检索的功劳；relevance 是模型（8B）的理解能力。** 前者完成了，后者没到位。

---

## 三、四个版本（v1→v4）各做了什么

| 版本 | 改动 | grounded | relevance | honesty | 状态 |
|---|---|---|---|---|---|
| v1 | 原始流程，20 题 | 95% | 75% | 90% | 基线 |
| v2 | 扩库 50 题 + RAG 调参（cutoff+消歧 prompt） | 100% | 74% | 96% | 检索干净 |
| **v3** | **+注入 31 条权威词条** | 98% | 76% | 96% | **最终采用 ✅** |
| v4 | 关摘要直读原文 | 14% ❌ | 84% | 14% ❌ | **已弃用** |

**核心结论（重要，接手人务必理解）：**

1. **v3 是当前可用版**。它 = PaperQA2 开摘要 + 注入权威词条 + 消歧 prompt。
2. **v4 证明了「关摘要」能救 relevance（84%）但会毁 grounded（14%）**——因为摘要环节同时承担「压缩」+「相关度过滤」两个功能，关掉压缩也把过滤关掉了。**这条路线已经探过、判定不可用，别重走。**

---

## 四、现在的问题（要交接给别人的部分）

### 问题 1：relevance 卡 76%，根因是 8B 理解能力

- 24% 的题「答不到点」，典型三类失败：
  1. **成对概念语义答反**：Precision/Recall、漏检/误报 这类映射关系，8B 会写反（cq39/cq40）。
  2. **多关键点覆盖不全**：金标准 3 个点只答中 1–2 个（cq03/06/13 等 9 题）。
  3. **个别问题理解偏**：数据增强题去讲配准（cq34）。

- **这不是检索能再修的**。证据：v2/v3 换了检索策略、注入权威定义、调 prompt，relevance 都停在 74–76%，没突破。

### 问题 2（已知、未修）：权威词条是「双刃剑」

- 注入 31 条权威词条后，relevance 净 +2%（74→76），**提升有限**。
- 原因：权威词条能「提供定义」，但不能「强制 8B 正确使用定义」。8B 读到 precision/recall 两条词条，依然把关系答反。

### 问题 3（技术债，接手人注意）

- **关摘要方向已探明不可用**（v4），但「关摘要 + 补一个非 LLM 的 embedding 相似度重排过滤」这个**组合方案还没做**——理论上能把 relevance 84% 和 grounded 干净两者兼得，但需要改 paperqa 流程（自定义 context_serializer 或加 rerank 步骤），本会话未实现，留作待办。

---

## 五、下一步建议（给接手人）

按优先级，前两条是「提 relevance」的正路，第三条是「换模型」的兜底：

1. **「关摘要 + embedding 重排过滤」组合**（还没做，最可能把 relevance 提到 84% 且保引用干净）—— 技术点在自定义 paperqa 的 `context_serializer`，在拼上下文前用 bge-m3 相似度过滤不相关 chunk。
2. **换更强的本地模型做 summary + 答题**（如 Qwen3-14B/32B、Qwen2.5-72B 量化版）—— 直接治「8B 理解能力」这个根，代价是资源占用。
3. **接受 76% 作为现状**，把精力转到别的模块（数字题/导师/蒸馏训练已经另有负责人）。

> 注意：**概念题「理解」这最后一公里，本质是模型能力问题，不是 RAG 通路问题。** 接手人若目标是「把 relevance 提上去」，应优先 1 或 2，而不是继续调 RAG。

---

## 六、复现环境（跑起来的关键）

- 脚本（本仓库 `src/agent/`）：
  - `paperqa_concept_v3.py` —— **最终采用版**（摘要 + 权威词条 + 消歧）
  - `paperqa_concept_v4.py` —— 关摘要实验（已弃用，只留教训）
  - `eval_concept_judge.py` —— LLM-as-judge 评分（API 问卷，key 走环境变量）
  - `concept_questions.jsonl` —— 50 题题目集
  - `concepts_authoritative.json` —— 31 条权威词条
- 服务器（见 `docs/概念题交接文档.md`）：
  - vLLM：本地 Qwen3-8B `:8000`（`hosted_vllm/Qwen3-8B`，`enable_thinking=False`）
  - paperqa venv：`/media/sdb1/gzj/support/llm/envs/paperqa/`
  - 语料：`/media/sdb1/gzj/data/rscd/corpus/fulltext/*.json`（109 篇，`pages_text` 字段）
- 评分：API `qwen3.8-max`（仅评测用，key 在 `/media/sdb1/gzj/support/llm/rs cd/.mentor_env`，不提交 git）

---

## 七、已知的坑（接手人别重踩，详见 `概念题评测报告.md` 第 7 节）

1. aadd 要传文件路径，不能传文本（Errno36）
2. aadd 要设 `summary_llm`（默认 gpt-4o 报错）
3. vLLM 要 `enable_thinking=False`
4. contexts 提取字段：id/context/text/score
5. judge 正则要兼容 `authoritative_concepts chunk N`
6. 关摘要会毁 grounded，别单独用
7. pkill -f 自杀、SSH 抽风带重试